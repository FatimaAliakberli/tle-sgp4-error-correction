"""
check_solar_cycle_mismatch.py

Diagnostic (not a modeling script): checks whether the TRAIN_END_DATE /
TEST_START_DATE split in config.py happens to straddle a real change in
solar activity -- i.e. whether the "solar cycle mismatch" hypothesis
(train captures Cycle 25's rising phase, test sits near/at solar maximum)
is actually present in this dataset, before spending effort on an
augmentation or retraining strategy aimed at fixing it.

What this does, using space-weather columns already in the dataset
(no re-download, no new features):
    1. Loads train/val/test splits, dedupes the training set down to
       "original" (non-augmented) rows so oversampling/noise augmentation
       doesn't distort the comparison.
    2. Reports descriptive stats (mean/median/std/min/max/quantiles) of
       sw_current_f107, sw_current_kp, sw_current_sn, sw_current_ap for
       each split.
    3. Reports the % of samples in each split flagged as geomagnetic
       storm / high-flux / high-sn days (using the existing threshold
       flags, or the raw thresholds in config.py if the flag columns are
       absent).
    4. Runs a two-sample Kolmogorov-Smirnov test (train vs test) on
       sw_current_f107 and sw_current_kp to quantify whether the
       difference is statistically real, not just eyeballed.
    5. Plots a daily-mean time series of f107 and kp across the FULL
       combined dataset (train+val+test), with vertical lines marking the
       train/val/test date boundaries from config.py, so you can see
       exactly where the split falls relative to the solar cycle.
    6. Plots overlaid histograms of f107 and kp for train vs test.
    7. As a secondary "does this actually matter" check: within the TEST
       set only, splits samples into high-flux vs normal-flux days (using
       the existing sw_current_high_flux_flag) and compares
       position_error_km between the two groups. If elevated solar
       activity is really driving worse SGP4 error, this group should
       show meaningfully higher error than the rest of the test set.

Outputs:
    output/solar_cycle_mismatch_stats.csv
    output/solar_cycle_mismatch_summary.txt
    output/plots/solar_cycle_timeseries.png
    output/plots/solar_cycle_train_vs_test_histograms.png
"""

import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("s10_check_solar_cycle_mismatch")

try:
    from scipy.stats import ks_2samp
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Space-weather columns to compare across splits. Only "current" epoch
# values are used (never sw_target_*/sw_mid_*), since those describe
# conditions the correction models are never allowed to see as inputs
# anyway, and current-epoch values are what actually varies with calendar
# time / solar cycle phase in the way this hypothesis is about.
SW_COLUMNS = ["sw_current_f107", "sw_current_kp", "sw_current_sn", "sw_current_ap"]

THRESHOLD_FLAG_COLUMNS = {
    "sw_current_geomagnetic_storm_flag": ("sw_current_kp", config.GEOMAGNETIC_STORM_KP_THRESHOLD, ">="),
    "sw_current_high_flux_flag": ("sw_current_f107", config.HIGH_FLUX_F107_THRESHOLD, ">="),
    "sw_current_high_sn_flag": ("sw_current_sn", config.HIGH_SN_THRESHOLD, ">="),
}


def _load_split(path, split_name, dedupe_original_only=False):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        logger.warning("%s not found. Skipping '%s' split.", path, split_name)
        return pd.DataFrame()

    if df.empty:
        logger.warning("%s is empty. Skipping '%s' split.", path, split_name)
        return df

    if dedupe_original_only and "augmentation_type" in df.columns:
        before = len(df)
        df = df[df["augmentation_type"] == "original"].copy()
        logger.info(
            "Split '%s': kept %d/%d rows after filtering to augmentation_type == 'original'.",
            split_name, len(df), before,
        )

    df["current_epoch_utc"] = pd.to_datetime(df["current_epoch_utc"], utc=True, format="ISO8601")
    df["split"] = split_name
    return df


def _ensure_flag_columns(df):
    """Compute threshold flags from raw values if the flag columns aren't
    already present (keeps this script usable even on older datasets)."""
    for flag_col, (source_col, threshold, op) in THRESHOLD_FLAG_COLUMNS.items():
        if flag_col in df.columns or source_col not in df.columns:
            continue
        vals = pd.to_numeric(df[source_col], errors="coerce")
        df[flag_col] = (vals >= threshold) if op == ">=" else (vals <= threshold)
    return df


def _descriptive_stats(df, split_name):
    rows = []
    for col in SW_COLUMNS:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        rows.append({
            "split": split_name,
            "column": col,
            "n_samples": int(len(vals)),
            "mean": float(vals.mean()),
            "median": float(vals.median()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "p25": float(vals.quantile(0.25)),
            "p75": float(vals.quantile(0.75)),
            "max": float(vals.max()),
        })

    for flag_col in THRESHOLD_FLAG_COLUMNS:
        if flag_col not in df.columns:
            continue
        pct = float(pd.to_numeric(df[flag_col], errors="coerce").fillna(0).mean() * 100)
        rows.append({
            "split": split_name,
            "column": flag_col,
            "n_samples": int(len(df)),
            "mean": pct,  # reused as "% of samples flagged" for this row type
            "median": np.nan, "std": np.nan, "min": np.nan, "p25": np.nan, "p75": np.nan, "max": np.nan,
        })

    return rows


def _plot_timeseries(combined_df, output_path):
    daily = (
        combined_df.set_index("current_epoch_utc")[["sw_current_f107", "sw_current_kp"]]
        .apply(pd.to_numeric, errors="coerce")
        .resample("1D").mean()
        .dropna(how="all")
    )
    if daily.empty:
        logger.warning("No data available to plot the solar-activity time series. Skipping.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(daily.index, daily["sw_current_f107"], color="tab:orange", linewidth=0.8)
    axes[0].axhline(config.HIGH_FLUX_F107_THRESHOLD, color="red", linestyle="--", linewidth=1,
                     label=f"high-flux threshold ({config.HIGH_FLUX_F107_THRESHOLD:.0f})")
    axes[0].set_ylabel("F10.7 (daily mean)")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].set_title("Daily-mean space weather across the full dataset, with train/val/test boundaries")

    axes[1].plot(daily.index, daily["sw_current_kp"], color="tab:blue", linewidth=0.8)
    axes[1].axhline(config.GEOMAGNETIC_STORM_KP_THRESHOLD, color="red", linestyle="--", linewidth=1,
                     label=f"storm threshold ({config.GEOMAGNETIC_STORM_KP_THRESHOLD:.0f})")
    axes[1].set_ylabel("Kp (daily mean)")
    axes[1].set_xlabel("Date")
    axes[1].legend(loc="upper left", fontsize=8)

    boundaries = [
        ("train end", config.TRAIN_END_DATE),
        ("val start", config.VALIDATION_START_DATE),
        ("val end", config.VALIDATION_END_DATE),
        ("test start", config.TEST_START_DATE),
    ]
    for ax in axes:
        for label, date_str in boundaries:
            ts = pd.Timestamp(date_str, tz="UTC")
            if daily.index.min() <= ts <= daily.index.max():
                ax.axvline(ts, color="gray", linestyle=":", linewidth=1)
        # annotate boundaries once, on the top subplot only
    for label, date_str in boundaries:
        ts = pd.Timestamp(date_str, tz="UTC")
        if daily.index.min() <= ts <= daily.index.max():
            axes[0].text(ts, axes[0].get_ylim()[1], label, rotation=90, fontsize=7,
                         va="top", ha="right", color="gray")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info("Saved plot: %s", output_path)


def _plot_train_vs_test_histograms(train_df, test_df, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, col, xlabel in [
        (axes[0], "sw_current_f107", "F10.7 (sfu)"),
        (axes[1], "sw_current_kp", "Kp index"),
    ]:
        train_vals = pd.to_numeric(train_df[col], errors="coerce").dropna() if col in train_df.columns else pd.Series(dtype=float)
        test_vals = pd.to_numeric(test_df[col], errors="coerce").dropna() if col in test_df.columns else pd.Series(dtype=float)
        if len(train_vals) == 0 and len(test_vals) == 0:
            continue
        ax.hist(train_vals, bins=40, alpha=0.5, density=True, label="train", color="tab:blue")
        ax.hist(test_vals, bins=40, alpha=0.5, density=True, label="test", color="tab:orange")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.set_title(f"{xlabel}: train vs test")
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info("Saved plot: %s", output_path)


def _error_by_flux_level(test_df):
    """Secondary check: within the test set only, does high-flux actually
    coincide with worse SGP4 error? If sw_current_high_flux_flag or
    position_error_km aren't present, this is skipped gracefully."""
    if "sw_current_high_flux_flag" not in test_df.columns or "position_error_km" not in test_df.columns:
        return None

    flag = pd.to_numeric(test_df["sw_current_high_flux_flag"], errors="coerce").fillna(0).astype(bool)
    errors = pd.to_numeric(test_df["position_error_km"], errors="coerce")

    high_flux = errors[flag]
    normal_flux = errors[~flag]
    if len(high_flux) == 0 or len(normal_flux) == 0:
        return None

    return {
        "n_high_flux_samples": int(len(high_flux)),
        "n_normal_flux_samples": int(len(normal_flux)),
        "mean_error_high_flux_km": float(high_flux.mean()),
        "median_error_high_flux_km": float(high_flux.median()),
        "mean_error_normal_flux_km": float(normal_flux.mean()),
        "median_error_normal_flux_km": float(normal_flux.median()),
    }


def run():
    config.ensure_dirs()

    logger.info("Loading train/val/test splits for solar-cycle mismatch check...")
    train_df = _load_split(f"{config.OUTPUT_DIR}/augmented_train_dataset.csv", "train", dedupe_original_only=True)
    val_df = _load_split(f"{config.OUTPUT_DIR}/val_dataset.csv", "val")
    test_df = _load_split(f"{config.OUTPUT_DIR}/test_dataset.csv", "test")

    if train_df.empty or test_df.empty:
        logger.warning("Train and/or test split unavailable. Cannot check solar-cycle mismatch.")
        return

    for df in (train_df, val_df, test_df):
        if not df.empty:
            _ensure_flag_columns(df)

    # --- 1. Descriptive stats per split ---
    logger.info("Computing descriptive space-weather statistics per split...")
    stats_rows = []
    for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        if not df.empty:
            stats_rows.extend(_descriptive_stats(df, name))
    stats_df = pd.DataFrame(stats_rows)
    stats_path = f"{config.OUTPUT_DIR}/solar_cycle_mismatch_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    logger.info("Saved %s", stats_path)

    # --- 2. KS test: train vs test, for f107 and kp ---
    ks_results = {}
    if SCIPY_AVAILABLE:
        logger.info("Running two-sample KS tests (train vs test)...")
        for col in ["sw_current_f107", "sw_current_kp"]:
            if col not in train_df.columns or col not in test_df.columns:
                continue
            train_vals = pd.to_numeric(train_df[col], errors="coerce").dropna()
            test_vals = pd.to_numeric(test_df[col], errors="coerce").dropna()
            if len(train_vals) == 0 or len(test_vals) == 0:
                continue
            stat, p_value = ks_2samp(train_vals, test_vals)
            ks_results[col] = {"ks_statistic": float(stat), "p_value": float(p_value)}
            logger.info("[%s] KS statistic=%.4f, p_value=%.3g", col, stat, p_value)
    else:
        logger.warning("scipy not installed; skipping the KS significance test (descriptive stats and plots still run).")

    # --- 3. Plots ---
    combined_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True) if not val_df.empty else pd.concat([train_df, test_df], axis=0, ignore_index=True)
    _plot_timeseries(combined_df, f"{config.PLOTS_DIR}/solar_cycle_timeseries.png")
    _plot_train_vs_test_histograms(train_df, test_df, f"{config.PLOTS_DIR}/solar_cycle_train_vs_test_histograms.png")

    # --- 4. Secondary check: does high-flux actually coincide with worse error in the test set? ---
    flux_error_check = _error_by_flux_level(test_df)

    # --- 5. Plain-language summary ---
    lines = []
    lines.append("Solar-Cycle Mismatch Diagnostic")
    lines.append("=" * 32)
    lines.append("")
    lines.append(f"Train split: {config.TRAIN_END_DATE.split()[0]} and earlier ({len(train_df)} rows, original-only)")
    if not val_df.empty:
        lines.append(f"Val split:   {config.VALIDATION_START_DATE.split()[0]} to {config.VALIDATION_END_DATE.split()[0]} ({len(val_df)} rows)")
    lines.append(f"Test split:  {config.TEST_START_DATE.split()[0]} to {config.TEST_END_DATE.split()[0]} ({len(test_df)} rows)")
    lines.append("")

    lines.append("Mean space-weather levels by split:")
    lines.append("-" * 36)
    for col in SW_COLUMNS:
        col_stats = stats_df[stats_df["column"] == col]
        if col_stats.empty:
            continue
        parts = []
        for _, row in col_stats.iterrows():
            parts.append(f"{row['split']}: mean={row['mean']:.2f} (median={row['median']:.2f})")
        lines.append(f"  {col}: " + " | ".join(parts))
    lines.append("")

    lines.append("Percent of samples flagged as storm / high-flux / high-SN, by split:")
    lines.append("-" * 70)
    for flag_col in THRESHOLD_FLAG_COLUMNS:
        col_stats = stats_df[stats_df["column"] == flag_col]
        if col_stats.empty:
            continue
        parts = [f"{row['split']}: {row['mean']:.1f}%" for _, row in col_stats.iterrows()]
        lines.append(f"  {flag_col}: " + " | ".join(parts))
    lines.append("")

    if ks_results:
        lines.append("Train vs test distribution difference (two-sample KS test):")
        lines.append("-" * 61)
        for col, result in ks_results.items():
            verdict = "STATISTICALLY SIGNIFICANT" if result["p_value"] < 0.01 else "not significant at p<0.01"
            lines.append(
                f"  {col}: KS statistic={result['ks_statistic']:.4f}, p_value={result['p_value']:.3g} ({verdict})"
            )
        lines.append("")
        lines.append(
            "  A significant KS result means train and test are drawn from measurably different"
        )
        lines.append(
            "  f107/Kp distributions -- i.e. the solar-cycle mismatch hypothesis is REAL in this data,"
        )
        lines.append(
            "  not just plausible. It does not by itself prove this is the cause of the sequence/baseline"
        )
        lines.append("  model's test-set error, only that the input distribution genuinely shifted.")
        lines.append("")

    if flux_error_check is not None:
        lines.append("Does high solar flux actually coincide with worse SGP4 error in the test set?")
        lines.append("-" * 78)
        lines.append(
            f"  High-flux test samples (n={flux_error_check['n_high_flux_samples']}): "
            f"mean position_error_km={flux_error_check['mean_error_high_flux_km']:.4f}, "
            f"median={flux_error_check['median_error_high_flux_km']:.4f}"
        )
        lines.append(
            f"  Normal-flux test samples (n={flux_error_check['n_normal_flux_samples']}): "
            f"mean position_error_km={flux_error_check['mean_error_normal_flux_km']:.4f}, "
            f"median={flux_error_check['median_error_normal_flux_km']:.4f}"
        )
        ratio = (
            flux_error_check["mean_error_high_flux_km"] / flux_error_check["mean_error_normal_flux_km"]
            if flux_error_check["mean_error_normal_flux_km"] > 0 else np.nan
        )
        if not np.isnan(ratio):
            lines.append(f"  => High-flux samples have {ratio:.2f}x the mean SGP4 error of normal-flux samples.")
        lines.append("")
    else:
        lines.append("(Could not run the high-flux-vs-error check: missing sw_current_high_flux_flag or position_error_km.)")
        lines.append("")

    lines.append("(Generated by check_solar_cycle_mismatch.py. See solar_cycle_mismatch_stats.csv for full")
    lines.append(" per-column statistics, and plots/solar_cycle_timeseries.png / ")
    lines.append(" plots/solar_cycle_train_vs_test_histograms.png for the visual check.)")

    summary_path = f"{config.OUTPUT_DIR}/solar_cycle_mismatch_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Wrote %s", summary_path)

    print("\n".join(lines))
    logger.info("check_solar_cycle_mismatch complete.")


if __name__ == "__main__":
    run()
