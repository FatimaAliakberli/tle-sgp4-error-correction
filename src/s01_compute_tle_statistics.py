"""
statistics_analysis.py

Computes:
    - TLE update-interval statistics (overall + per time range)
    - SGP4 one-step propagation error statistics (overall + per time range)
    - Fixed-horizon SGP4 baseline error statistics (12h/24h/36h)

Outputs:
    output/stats_summary.csv
    output/time_range_stats.csv
    output/fixed_horizon_stats.csv
    output/plots/<object_name>_one_step_error.png
    output/plots/<object_name>_fixed_horizon_errors.png
"""

import logging
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import tle_utils
import feature_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("s01_compute_tle_statistics")


def _sanitize_filename(name):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def _percentile(values, p):
    if len(values) == 0:
        return np.nan
    return float(np.percentile(values, p))


def _interval_stats(epochs):
    """epochs: sorted list of pandas.Timestamp"""
    if len(epochs) < 2:
        return {
            "total_tle_records": len(epochs),
            "mean_interval_hours": np.nan,
            "std_interval_hours": np.nan,
            "min_interval_hours": np.nan,
            "max_interval_hours": np.nan,
            "median_interval_hours": np.nan,
        }
    diffs_h = np.diff([e.timestamp() for e in epochs]) / 3600.0
    diffs_h = diffs_h[diffs_h >= config.MIN_INTERVAL_HOURS]
    if len(diffs_h) == 0:
        return {
            "total_tle_records": len(epochs),
            "mean_interval_hours": np.nan,
            "std_interval_hours": np.nan,
            "min_interval_hours": np.nan,
            "max_interval_hours": np.nan,
            "median_interval_hours": np.nan,
        }
    return {
        "total_tle_records": len(epochs),
        "mean_interval_hours": float(np.mean(diffs_h)),
        "std_interval_hours": float(np.std(diffs_h)),
        "min_interval_hours": float(np.min(diffs_h)),
        "max_interval_hours": float(np.max(diffs_h)),
        "median_interval_hours": float(np.median(diffs_h)),
    }


def _error_stats(errors_km):
    errors_km = np.asarray([e for e in errors_km if e is not None and not np.isnan(e)])
    if len(errors_km) == 0:
        return {
            "n_samples": 0, "mean_error_km": np.nan, "median_error_km": np.nan,
            "p90_error_km": np.nan, "p95_error_km": np.nan, "max_error_km": np.nan,
        }
    return {
        "n_samples": int(len(errors_km)),
        "mean_error_km": float(np.mean(errors_km)),
        "median_error_km": float(np.median(errors_km)),
        "p90_error_km": _percentile(errors_km, 90),
        "p95_error_km": _percentile(errors_km, 95),
        "max_error_km": float(np.max(errors_km)),
    }


def compute_one_step_errors(sats, ts):
    """
    For consecutive TLE pairs (sat[i], sat[i+1]) propagate sat[i] to
    sat[i+1]'s epoch and compare against sat[i+1]'s own epoch position
    (near-zero-propagation "truth").

    Returns list of dicts: {epoch, error_km}
    """
    results = []
    for i in range(len(sats) - 1):
        cur = sats[i]
        nxt = sats[i + 1]
        try:
            cur_epoch_dt = tle_utils.epoch_datetime(cur)
            nxt_epoch_dt = tle_utils.epoch_datetime(nxt)

            pred_pos, _ = feature_utils.propagate_state(cur, ts, nxt_epoch_dt)
            truth_pos, _ = feature_utils.propagate_state(nxt, ts, nxt_epoch_dt)

            err = float(np.linalg.norm(pred_pos - truth_pos))
            results.append({"epoch": nxt_epoch_dt, "error_km": err})
        except Exception as exc:
            logger.debug("One-step error failed at index %d: %s", i, exc)
            continue
    return results


def compute_fixed_horizon_errors(object_name, sats, ts, horizons_hours):
    """
    For every TLE and each horizon, find an approximate-truth future TLE
    within MAX_TRUTH_OFFSET_HOURS and compute the position error.

    Returns list of dicts, one per accepted sample.
    """
    epochs = [tle_utils.epoch_datetime(s) for s in sats]
    records = []

    for i, cur in enumerate(sats):
        cur_epoch = epochs[i]
        for horizon_h in horizons_hours:
            target_time = cur_epoch + pd.Timedelta(hours=horizon_h)

            # Find nearest future/near TLE to target_time (search forward from i)
            best_j = None
            best_offset = None
            for j in range(i + 1, len(sats)):
                offset_h = abs((epochs[j] - target_time).total_seconds()) / 3600.0
                if offset_h <= config.MAX_TRUTH_OFFSET_HOURS:
                    if best_offset is None or offset_h < best_offset:
                        best_offset = offset_h
                        best_j = j
                # once we're far past target_time in the future, stop searching
                if epochs[j] > target_time + pd.Timedelta(hours=config.MAX_TRUTH_OFFSET_HOURS):
                    break

            if best_j is None:
                continue

            truth_sat = sats[best_j]
            try:
                pred_pos, _ = feature_utils.propagate_state(cur, ts, target_time)
                truth_pos, _ = feature_utils.propagate_state(truth_sat, ts, target_time)
                err = float(np.linalg.norm(pred_pos - truth_pos))
                records.append({
                    "object_name": object_name,
                    "current_epoch": cur_epoch,
                    "horizon_hours": horizon_h,
                    "truth_offset_hours": best_offset,
                    "error_km": err,
                })
            except Exception as exc:
                logger.debug("Fixed-horizon propagation failed for %s: %s", object_name, exc)
                continue

    return records


def run():
    config.ensure_dirs()
    ts = tle_utils.get_timescale()

    logger.info("Loading TLE dataset from %s", config.TLE_DATASET_DIR)
    satellites = tle_utils.load_all_tle_history(config.TLE_DATASET_DIR)

    satellites, dropped_objects = tle_utils.filter_high_gap_objects(
        satellites,
        max_mean_interval_hours=config.MAX_OBJECT_MEAN_INTERVAL_HOURS,
        min_tle_count=config.MIN_OBJECT_TLE_COUNT,
        min_interval_hours=config.MIN_INTERVAL_HOURS,
    )
    if dropped_objects:
        dropped_df = pd.DataFrame([
            {"object_name": name, **info} for name, info in dropped_objects.items()
        ]).sort_values("mean_interval_hours", ascending=False)
        dropped_df.to_csv(f"{config.OUTPUT_DIR}/filtered_out_objects.csv", index=False)
        logger.info("Wrote %d filtered-out objects to output/filtered_out_objects.csv", len(dropped_df))

    if not satellites:
        logger.warning("No satellites loaded. Nothing to do.")
        pd.DataFrame().to_csv(f"{config.OUTPUT_DIR}/stats_summary.csv", index=False)
        pd.DataFrame().to_csv(f"{config.OUTPUT_DIR}/time_range_stats.csv", index=False)
        pd.DataFrame().to_csv(f"{config.OUTPUT_DIR}/fixed_horizon_stats.csv", index=False)
        return

    stats_summary_rows = []
    time_range_rows = []
    fixed_horizon_rows_all = []

    for object_name, sats in satellites.items():
        try:
            if len(sats) < 2:
                logger.info("Skipping %s: fewer than 2 TLEs.", object_name)
                continue

            epochs = [tle_utils.epoch_datetime(s) for s in sats]

            # --- interval statistics (overall) ---
            interval_stats = _interval_stats(epochs)

            # --- one-step SGP4 error statistics (overall) ---
            one_step = compute_one_step_errors(sats, ts)
            one_step_errors = [r["error_km"] for r in one_step]
            err_stats = _error_stats(one_step_errors)

            row = {"object_name": object_name}
            row.update(interval_stats)
            row.update(err_stats)
            stats_summary_rows.append(row)

            # --- per time-range statistics ---
            for range_name, (start_str, end_str) in config.TIME_RANGES.items():
                start = pd.Timestamp(start_str, tz="UTC")
                end = pd.Timestamp(end_str, tz="UTC")

                range_epochs = [e for e in epochs if start <= e <= end]
                range_interval_stats = _interval_stats(range_epochs)

                range_one_step_errors = [
                    r["error_km"] for r in one_step if start <= r["epoch"] <= end
                ]
                range_err_stats = _error_stats(range_one_step_errors)

                trow = {"object_name": object_name, "time_range": range_name}
                trow.update(range_interval_stats)
                trow.update(range_err_stats)
                time_range_rows.append(trow)

            # --- fixed horizon statistics ---
            fh_records = compute_fixed_horizon_errors(object_name, sats, ts, config.HORIZONS_HOURS)
            fixed_horizon_rows_all.extend(fh_records)

            for horizon_h in config.HORIZONS_HOURS:
                horizon_errors = [r["error_km"] for r in fh_records if r["horizon_hours"] == horizon_h]
                hstats = _error_stats(horizon_errors)
                stats_summary_rows[-1][f"horizon_{horizon_h}h_n_samples"] = hstats["n_samples"]
                stats_summary_rows[-1][f"horizon_{horizon_h}h_mean_error_km"] = hstats["mean_error_km"]

            # --- plots ---
            safe_name = _sanitize_filename(object_name)
            if one_step:
                try:
                    fig, ax = plt.subplots(figsize=(9, 4))
                    xs = [r["epoch"] for r in one_step]
                    ys = [r["error_km"] for r in one_step]
                    ax.plot(xs, ys, marker=".", linestyle="-", linewidth=0.7, markersize=3)
                    ax.set_title(f"One-step SGP4 error: {object_name}")
                    ax.set_xlabel("Epoch (UTC)")
                    ax.set_ylabel("Position error (km)")
                    fig.autofmt_xdate()
                    fig.tight_layout()
                    fig.savefig(f"{config.PLOTS_DIR}/{safe_name}_one_step_error.png", dpi=120)
                    plt.close(fig)
                except Exception as exc:
                    logger.warning("Plotting one-step error failed for %s: %s", object_name, exc)

            if fh_records:
                try:
                    fig, ax = plt.subplots(figsize=(7, 4))
                    for horizon_h in config.HORIZONS_HOURS:
                        vals = [r["error_km"] for r in fh_records if r["horizon_hours"] == horizon_h]
                        if vals:
                            ax.boxplot(vals, positions=[horizon_h], widths=4, showfliers=False)
                    ax.set_title(f"Fixed-horizon SGP4 errors: {object_name}")
                    ax.set_xlabel("Horizon (hours)")
                    ax.set_ylabel("Position error (km)")
                    ax.set_xticks(config.HORIZONS_HOURS)
                    fig.tight_layout()
                    fig.savefig(f"{config.PLOTS_DIR}/{safe_name}_fixed_horizon_errors.png", dpi=120)
                    plt.close(fig)
                except Exception as exc:
                    logger.warning("Plotting fixed-horizon error failed for %s: %s", object_name, exc)

        except Exception as exc:
            logger.warning("Failed processing object %s: %s", object_name, exc)
            continue

    # --- fixed horizon summary table (aggregated per object x horizon) ---
    fh_df = pd.DataFrame(fixed_horizon_rows_all)
    fh_summary_rows = []
    if not fh_df.empty:
        for (object_name, horizon_h), group in fh_df.groupby(["object_name", "horizon_hours"]):
            errs = group["error_km"].values
            stats = _error_stats(errs)
            fh_summary_rows.append({
                "object_name": object_name,
                "horizon_hours": horizon_h,
                "number_of_valid_samples": stats["n_samples"],
                "mean_error_km": stats["mean_error_km"],
                "median_error_km": stats["median_error_km"],
                "p90_error_km": stats["p90_error_km"],
                "p95_error_km": stats["p95_error_km"],
                "max_error_km": stats["max_error_km"],
            })

    stats_summary_df = pd.DataFrame(stats_summary_rows)
    time_range_df = pd.DataFrame(time_range_rows)
    fixed_horizon_df = pd.DataFrame(fh_summary_rows)

    stats_summary_df.to_csv(f"{config.OUTPUT_DIR}/stats_summary.csv", index=False)
    time_range_df.to_csv(f"{config.OUTPUT_DIR}/time_range_stats.csv", index=False)
    fixed_horizon_df.to_csv(f"{config.OUTPUT_DIR}/fixed_horizon_stats.csv", index=False)

    # Combined comparison plot across horizons (all objects pooled)
    if not fh_df.empty:
        try:
            fig, ax = plt.subplots(figsize=(7, 4))
            data = [fh_df.loc[fh_df["horizon_hours"] == h, "error_km"].values for h in config.HORIZONS_HOURS]
            ax.boxplot(data, labels=[f"{h}h" for h in config.HORIZONS_HOURS], showfliers=False)
            ax.set_title("SGP4 baseline error by horizon (all objects)")
            ax.set_ylabel("Position error (km)")
            fig.tight_layout()
            fig.savefig(f"{config.PLOTS_DIR}/horizon_error_comparison.png", dpi=120)
            plt.close(fig)
        except Exception as exc:
            logger.warning("Failed to save horizon comparison plot: %s", exc)

    logger.info("statistics_analysis complete. %d objects processed.", len(stats_summary_rows))


if __name__ == "__main__":
    run()
