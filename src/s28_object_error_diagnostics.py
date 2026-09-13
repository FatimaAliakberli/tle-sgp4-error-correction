"""
object_error_diagnostics.py

PER-OBJECT (not pooled/aggregated) diagnostic pass over
output/error_attribution_dataset.csv. Where stats_summary.csv and
evaluation_summary.csv summarize error statistics ACROSS objects, this
script looks at each debris object's error timeline individually and asks:
"what else was true at the moment of this object's worst prediction
errors?"

METHOD
------
1. Catastrophic-sample selection (per object, not global)
   -----------------------------------------------------
   For each object with at least config.OBJECT_DIAG_MIN_SAMPLES_FOR_OWN_THRESHOLD
   samples, "catastrophic" = position_error_km >= that OBJECT's own
   config.OBJECT_DIAG_CATASTROPHIC_QUANTILE (default: 99th percentile).
   This is object-relative: an object whose SGP4 error is generally larger
   or smaller than the population isn't judged against a population-wide
   cutoff, it's judged against its own history.

   For objects with too few samples to trust a percentile computed purely
   from their own data, we fall back to the existing dataset-wide
   `outlier_flag` column (99th percentile of position_error_km computed
   PER HORIZON across the whole dataset in build_error_attribution_dataset.py).

2. Per-sample heuristic classification
   ------------------------------------
   Each catastrophic sample is checked against three independent,
   non-exclusive heuristics (documented with exact thresholds in
   config.py, prefixed OBJECT_DIAG_*). A sample can match more than one
   category simultaneously (e.g. a tracking gap that happens to coincide
   with a storm) -- we do NOT force a single label.

     possible_tracking_gap:
         hours_since_last_tle > config.OBJECT_DIAG_TRACKING_GAP_HOURS
         (an unusually stale TLE was used for this prediction)

     possible_maneuver_or_breakup:
         ANY of the following exceeds its threshold in absolute value,
         over EITHER the last-24h or last-72h window:
           |bstar_change|            > config.OBJECT_DIAG_BSTAR_CHANGE_THRESHOLD
           |mean_motion_change|      > config.OBJECT_DIAG_MEAN_MOTION_CHANGE_THRESHOLD_REV_PER_DAY
           |semi_major_axis_change|  > config.OBJECT_DIAG_SEMI_MAJOR_AXIS_CHANGE_THRESHOLD_KM
           |perigee_change|          > config.OBJECT_DIAG_PERIGEE_CHANGE_THRESHOLD_KM
         (a jump in orbital elements not explained by ordinary drag decay)

     possible_space_weather_driven:
         sw_current_geomagnetic_storm_flag OR sw_target_geomagnetic_storm_flag
         OR sw_current_high_flux_flag OR sw_target_high_flux_flag
         OR sw_current_high_sn_flag OR sw_target_high_sn_flag is set
         (a geomagnetic storm or high solar-flux/sunspot period was active
         near either the prediction epoch or the target epoch)

     unexplained:
         none of the above matched.

   If a dataset was built before the sn/daily_ap space-weather extension,
   the sw_current_/sw_target_ *_flag columns simply won't be present; this
   script treats any missing flag column as all-zero rather than failing.

3. Outputs
   -------
     output/object_error_diagnostics.csv
         one row per object: counts per category, worst error, threshold
         used, etc. Sorted by n_catastrophic_samples descending.

     output/plots/object_diagnostics/<object_id>_timeline.png
         per-object position_error_km vs. time, catastrophic samples
         color-coded by (primary) classification, with geomagnetic-storm
         periods shaded, so it's visually obvious whether spikes line up
         with storms, gaps, or orbital-element jumps.

Follows the logging/try-except conventions used elsewhere in this project
(see build_error_attribution_dataset.py / train_sequence_model.py): one
object failing does not stop the run, and progress is logged per object
with no long silent gaps.
"""

import logging
import re
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("s28_object_error_diagnostics")

LOG_EVERY_N_OBJECTS = 25

CATEGORIES = [
    "possible_tracking_gap",
    "possible_maneuver_or_breakup",
    "possible_space_weather_driven",
    "unexplained",
]

# Priority order used ONLY to pick a single "primary" category for plot
# color-coding when a sample matches more than one heuristic. The full,
# non-exclusive set of matched categories is still counted correctly in
# the per-object summary counts.
PLOT_COLOR_PRIORITY = [
    "possible_maneuver_or_breakup",
    "possible_space_weather_driven",
    "possible_tracking_gap",
    "unexplained",
]
PLOT_COLORS = {
    "possible_maneuver_or_breakup": "#d62728",   # red
    "possible_space_weather_driven": "#ff7f0e",  # orange
    "possible_tracking_gap": "#1f77b4",          # blue
    "unexplained": "#7f7f7f",                    # gray
}


def _sanitize_filename(name):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(name))


def _safe_col(df, col):
    """Return df[col] as float array if present, else zeros -- keeps this
    script working on datasets built before a given column existed."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0).values
    return np.zeros(len(df))


def select_catastrophic_mask(g):
    """
    Returns (mask: np.ndarray[bool], threshold_km: float, method: str)
    for one object's rows `g` (a DataFrame slice of error_attribution_dataset).
    """
    errors = pd.to_numeric(g["position_error_km"], errors="coerce")
    n_valid = errors.notna().sum()

    if n_valid >= config.OBJECT_DIAG_MIN_SAMPLES_FOR_OWN_THRESHOLD:
        threshold = float(errors.quantile(config.OBJECT_DIAG_CATASTROPHIC_QUANTILE))
        mask = (errors >= threshold).fillna(False).values
        method = f"own_p{int(config.OBJECT_DIAG_CATASTROPHIC_QUANTILE * 100)}"
        return mask, threshold, method

    # Fallback: not enough samples to trust an object-specific percentile.
    if "outlier_flag" in g.columns:
        mask = g["outlier_flag"].astype(bool).values
        valid_errors = errors[mask]
        threshold = float(valid_errors.min()) if len(valid_errors) else float(errors.max() if n_valid else np.nan)
        return mask, threshold, "dataset_wide_outlier_flag_fallback"

    # Last resort: no outlier_flag column at all -- nothing flagged.
    return np.zeros(len(g), dtype=bool), np.nan, "no_threshold_available"


def classify_sample(row):
    """
    Returns a set of category strings (may contain multiple entries).
    `row` is a pandas Series for one catastrophic sample.
    """
    matched = set()

    # --- possible_tracking_gap ---
    hours_since_last = row.get("hours_since_last_tle", np.nan)
    if pd.notna(hours_since_last) and hours_since_last > config.OBJECT_DIAG_TRACKING_GAP_HOURS:
        matched.add("possible_tracking_gap")

    # --- possible_maneuver_or_breakup ---
    change_checks = [
        ("bstar_change_last_24h", config.OBJECT_DIAG_BSTAR_CHANGE_THRESHOLD),
        ("bstar_change_last_72h", config.OBJECT_DIAG_BSTAR_CHANGE_THRESHOLD),
        ("mean_motion_change_last_24h", config.OBJECT_DIAG_MEAN_MOTION_CHANGE_THRESHOLD_REV_PER_DAY),
        ("mean_motion_change_last_72h", config.OBJECT_DIAG_MEAN_MOTION_CHANGE_THRESHOLD_REV_PER_DAY),
        ("semi_major_axis_change_last_24h", config.OBJECT_DIAG_SEMI_MAJOR_AXIS_CHANGE_THRESHOLD_KM),
        ("semi_major_axis_change_last_72h", config.OBJECT_DIAG_SEMI_MAJOR_AXIS_CHANGE_THRESHOLD_KM),
        ("perigee_change_last_24h", config.OBJECT_DIAG_PERIGEE_CHANGE_THRESHOLD_KM),
        ("perigee_change_last_72h", config.OBJECT_DIAG_PERIGEE_CHANGE_THRESHOLD_KM),
    ]
    for col, thresh in change_checks:
        val = row.get(col, np.nan)
        if pd.notna(val) and abs(val) > thresh:
            matched.add("possible_maneuver_or_breakup")
            break

    # --- possible_space_weather_driven ---
    sw_flag_cols = [
        "sw_current_geomagnetic_storm_flag", "sw_target_geomagnetic_storm_flag",
        "sw_current_high_flux_flag", "sw_target_high_flux_flag",
        "sw_current_high_sn_flag", "sw_target_high_sn_flag",
    ]
    for col in sw_flag_cols:
        val = row.get(col, 0)
        if pd.notna(val) and bool(val):
            matched.add("possible_space_weather_driven")
            break

    if not matched:
        matched.add("unexplained")

    return matched


def primary_category(matched):
    for cat in PLOT_COLOR_PRIORITY:
        if cat in matched:
            return cat
    return "unexplained"


def plot_object_timeline(object_id, g, catastrophic_idx, sample_categories):
    """
    g: full (sorted by current_epoch_utc) DataFrame slice for this object.
    catastrophic_idx: index labels (within g) of catastrophic samples.
    sample_categories: dict {index_label: set_of_categories}
    """
    safe_name = _sanitize_filename(object_id)
    fig, ax = plt.subplots(figsize=(11, 5))

    x_all = g["current_epoch_utc"]
    y_all = g["position_error_km"]
    ax.plot(x_all, y_all, linewidth=0.5, alpha=0.5, color="black", zorder=1,
            label="position_error_km (all samples)")

    # Shade geomagnetic storm periods (using sw_current_geomagnetic_storm_flag
    # as a proxy for "storm active near this sample's current epoch").
    if "sw_current_geomagnetic_storm_flag" in g.columns:
        storm_mask = g["sw_current_geomagnetic_storm_flag"].fillna(0).astype(bool).values
        x_vals = x_all.values
        in_storm = False
        start = None
        for i, is_storm in enumerate(storm_mask):
            if is_storm and not in_storm:
                start = x_vals[i]
                in_storm = True
            elif not is_storm and in_storm:
                ax.axvspan(start, x_vals[i], color="#ffcc00", alpha=0.15, zorder=0)
                in_storm = False
        if in_storm:
            ax.axvspan(start, x_vals[-1], color="#ffcc00", alpha=0.15, zorder=0)

    plotted_labels = set()
    for cat in PLOT_COLOR_PRIORITY:
        idxs = [i for i in catastrophic_idx if cat in sample_categories[i] and primary_category(sample_categories[i]) == cat]
        if not idxs:
            continue
        sub = g.loc[idxs]
        label = cat if cat not in plotted_labels else None
        ax.scatter(sub["current_epoch_utc"], sub["position_error_km"],
                    color=PLOT_COLORS[cat], s=35, zorder=3, label=cat, edgecolor="black", linewidth=0.4)
        plotted_labels.add(cat)

    ax.set_title(f"Per-object error timeline: {object_id}")
    ax.set_xlabel("Epoch (UTC)")
    ax.set_ylabel("Position error (km)")
    ax.legend(loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path = f"{config.OBJECT_DIAGNOSTICS_PLOTS_DIR}/{safe_name}_timeline.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def process_object(object_id, g):
    """
    g: DataFrame slice for one object_id (any horizon mix), NOT yet sorted.
    Returns (summary_row: dict, n_catastrophic: int)
    """
    g = g.sort_values("current_epoch_utc").reset_index(drop=True)

    mask, threshold_km, method = select_catastrophic_mask(g)
    catastrophic_idx = list(np.where(mask)[0])

    category_counts = {cat: 0 for cat in CATEGORIES}
    sample_categories = {}
    for idx in catastrophic_idx:
        row = g.iloc[idx]
        matched = classify_sample(row)
        sample_categories[idx] = matched
        for cat in matched:
            category_counts[cat] += 1

    worst_error_km = float(g["position_error_km"].max()) if len(g) else np.nan
    worst_row_idx = int(g["position_error_km"].idxmax()) if len(g) and g["position_error_km"].notna().any() else None
    worst_epoch = g.loc[worst_row_idx, "current_epoch_utc"] if worst_row_idx is not None else pd.NaT

    plot_path = None
    if catastrophic_idx:
        try:
            plot_path = plot_object_timeline(object_id, g, catastrophic_idx, sample_categories)
        except Exception as exc:
            logger.warning("Plotting failed for %s: %s", object_id, exc)

    summary_row = {
        "object_id": object_id,
        "n_samples": len(g),
        "n_catastrophic_samples": len(catastrophic_idx),
        "catastrophic_threshold_km": threshold_km,
        "catastrophic_threshold_method": method,
        "worst_error_km": worst_error_km,
        "worst_error_epoch_utc": worst_epoch,
        "n_possible_tracking_gap": category_counts["possible_tracking_gap"],
        "n_possible_maneuver_or_breakup": category_counts["possible_maneuver_or_breakup"],
        "n_possible_space_weather_driven": category_counts["possible_space_weather_driven"],
        "n_unexplained": category_counts["unexplained"],
        "plot_path": plot_path,
    }
    return summary_row, len(catastrophic_idx)


def run():
    config.ensure_dirs()
    pipeline_start = time.time()

    dataset_path = f"{config.OUTPUT_DIR}/error_attribution_dataset.csv"
    logger.info("Loading %s ...", dataset_path)
    t0 = time.time()
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        logger.warning("%s not found. Run build_error_attribution_dataset.py first. Nothing to do.",
                        dataset_path)
        pd.DataFrame().to_csv(f"{config.OUTPUT_DIR}/object_error_diagnostics.csv", index=False)
        return
    logger.info("Loaded %d rows in %.1fs.", len(df), time.time() - t0)

    if df.empty:
        logger.warning("Error attribution dataset is empty. Nothing to do.")
        pd.DataFrame().to_csv(f"{config.OUTPUT_DIR}/object_error_diagnostics.csv", index=False)
        return

    df["current_epoch_utc"] = pd.to_datetime(df["current_epoch_utc"], utc=True, format="ISO8601")

    object_ids = sorted(df["object_id"].astype(str).unique())
    n_objects_total = len(object_ids)
    logger.info("Running per-object diagnostics for %d objects...", n_objects_total)

    summary_rows = []
    n_ok, n_failed, n_total_catastrophic = 0, 0, 0

    for obj_idx, object_id in enumerate(object_ids, start=1):
        t_obj = time.time()
        try:
            g = df[df["object_id"].astype(str) == object_id]
            if g.empty:
                continue
            summary_row, n_catastrophic = process_object(object_id, g)
            summary_rows.append(summary_row)
            n_ok += 1
            n_total_catastrophic += n_catastrophic

            if obj_idx % LOG_EVERY_N_OBJECTS == 0 or obj_idx == n_objects_total:
                elapsed_total = time.time() - pipeline_start
                logger.info(
                    "[%d/%d] processed (running total: %d objects ok, %d catastrophic samples "
                    "found, %.1fs elapsed).",
                    obj_idx, n_objects_total, n_ok, n_total_catastrophic, elapsed_total,
                )
            else:
                logger.debug("[%d/%d] %s: done in %.2fs -- %d catastrophic samples.",
                             obj_idx, n_objects_total, object_id, time.time() - t_obj, n_catastrophic)
        except Exception as exc:
            logger.warning("[%d/%d] %s: FAILED after %.1fs: %s",
                            obj_idx, n_objects_total, object_id, time.time() - t_obj, exc)
            n_failed += 1
            continue

    logger.info("Finished per-object diagnostics in %.1fs total. %d objects succeeded, %d failed.",
                time.time() - pipeline_start, n_ok, n_failed)

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["n_catastrophic_samples", "worst_error_km"], ascending=[False, False]
        ).reset_index(drop=True)

    out_path = f"{config.OUTPUT_DIR}/object_error_diagnostics.csv"
    summary_df.to_csv(out_path, index=False)
    logger.info("Wrote %d per-object summary rows to %s.", len(summary_df), out_path)

    if not summary_df.empty:
        print("\n=== OBJECT ERROR DIAGNOSTICS (top 15 by catastrophic sample count) ===")
        print(summary_df.head(15).to_string(index=False))

    logger.info("object_error_diagnostics complete.")


if __name__ == "__main__":
    run()
