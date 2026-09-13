"""
diagnose_outliers.py

Standalone diagnostic script (Part D.2 companion, but self-contained and
read-only with respect to every existing file in the pipeline).

Purpose
-------
For the raw SGP4 baseline and each of the 7 rolling-CV-selected sequence
model variants (Part D.1 artifacts, "<variant>_rolling_cv_v2"), quantify how
"tail-heavy" each model's error distribution on the TEST SET is:

    1. Fraction of samples exceeding fixed error thresholds
       (5 / 10 / 20 / 50 / 100 km).
    2. What share of that model's TOTAL summed absolute error is
       concentrated in its own worst 0.1% and worst 1% of samples.
    3. Overall mean / median error, as a sanity cross-check against
       output/ablation_summary_v2.csv's "overall" rows for the same models.

It also dumps each variant's 20 worst rows to their own CSV for manual
inspection, and a single combined summary CSV across all 8 "models"
(raw_sgp4 + 7 sequence variants).

This script:
    - NEVER modifies any existing file.
    - NEVER retrains anything -- it reuses
      evaluate_sequence_ablation_v2.add_sequence_variant_predictions_v2
      exactly as-is (import only), which itself only does a forward pass
      (model.eval() + torch.no_grad() internally).
    - Is fully independent of ablation_summary_v2.csv; it recomputes the
      corrected-error columns itself via the reused function so that (c)
      above is a genuine, from-scratch cross-check rather than just
      re-reading the same file.

Outputs:
    new-output/diagnostics_top_outliers_<variant_name>.csv   (7 files)
    new-output/diagnostics_outlier_summary.csv               (1 file, 8 rows)
"""

import logging

import numpy as np
import pandas as pd

import config
from evaluate_sequence_ablation_v2_predup import add_sequence_variant_predictions_v2
from s06_train_sequence_model import FEATURE_SET_VARIANTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("s29_diagnose_outliers")

RAW_COLUMN = "position_error_km"
MODEL_SUFFIX = "_rolling_cv_v2"

ERROR_THRESHOLDS_KM = [5, 10, 20, 50, 100]
TAIL_QUANTILES = {
    "pct_total_abs_error_from_worst_0_1pct": 0.001,
    "pct_total_abs_error_from_worst_1pct": 0.01,
}

TOP_N_OUTLIERS = 20

OUTLIER_EXPORT_COLUMNS_TEMPLATE = [
    "object_id",
    "horizon_hours",
    "current_epoch_utc",
    RAW_COLUMN,
    None,  # placeholder for the variant's own corrected-error column
]


def compute_tail_stats(errors: np.ndarray) -> dict:
    """
    Given a 1D array of per-sample (non-negative) errors for one model,
    compute the threshold-exceedance fractions, worst-0.1%/1% total-error
    shares, and mean/median. Returns a dict of stat_name -> value.
    """
    errors = np.asarray(errors, dtype=float)
    errors = errors[~np.isnan(errors)]

    n = errors.size
    if n == 0:
        stats = {f"pct_over_{t}km": np.nan for t in ERROR_THRESHOLDS_KM}
        stats.update({k: np.nan for k in TAIL_QUANTILES})
        stats["mean_error_km"] = np.nan
        stats["median_error_km"] = np.nan
        return stats

    stats = {}
    for t in ERROR_THRESHOLDS_KM:
        stats[f"pct_over_{t}km"] = float(np.mean(errors > t) * 100.0)

    total_abs_error = float(np.sum(errors))
    sorted_errors = np.sort(errors)[::-1]  # descending: worst first
    for stat_name, quantile in TAIL_QUANTILES.items():
        k = max(1, int(np.ceil(quantile * n)))
        worst_k_sum = float(np.sum(sorted_errors[:k]))
        stats[stat_name] = (
            float(worst_k_sum / total_abs_error * 100.0) if total_abs_error > 0 else np.nan
        )

    stats["mean_error_km"] = float(np.mean(errors))
    stats["median_error_km"] = float(np.median(errors))
    return stats


def run():
    config.ensure_dirs()

    test_path = f"{config.OUTPUT_DIR}/test_dataset.csv"
    logger.info("Loading test dataset from %s ...", test_path)
    test_df = pd.read_csv(test_path)
    if test_df.empty:
        logger.warning("Test dataset is empty. Nothing to diagnose.")
        pd.DataFrame().to_csv(f"{config.OUTPUT_DIR}/diagnostics_outlier_summary.csv", index=False)
        return

    test_df["current_epoch_utc"] = pd.to_datetime(
        test_df["current_epoch_utc"], utc=True, format="ISO8601"
    )

    summary_rows = []

    # --- 1. Raw SGP4 (no correction) ------------------------------------
    logger.info("Computing tail diagnostics for raw_sgp4 (%s)...", RAW_COLUMN)
    if RAW_COLUMN not in test_df.columns:
        logger.error(
            "Column '%s' not found in test dataset; cannot compute raw_sgp4 diagnostics. Skipping.",
            RAW_COLUMN,
        )
    else:
        try:
            raw_stats = compute_tail_stats(test_df[RAW_COLUMN].to_numpy())
            raw_stats["model"] = "raw_sgp4"
            summary_rows.append(raw_stats)
            logger.info(
                "raw_sgp4: mean=%.3f km, median=%.3f km, worst-0.1%%=%.1f%%, worst-1%%=%.1f%%",
                raw_stats["mean_error_km"],
                raw_stats["median_error_km"],
                raw_stats["pct_total_abs_error_from_worst_0_1pct"],
                raw_stats["pct_total_abs_error_from_worst_1pct"],
            )
        except Exception:
            logger.exception("Failed computing raw_sgp4 diagnostics. Skipping raw_sgp4 row.")

    # --- 2-8. Each of the 7 sequence-model variants ---------------------
    for variant_name in FEATURE_SET_VARIANTS:
        display_name = f"{variant_name}{MODEL_SUFFIX}"
        logger.info("Processing variant '%s' (%s)...", variant_name, display_name)
        try:
            test_df, ok, error_col = add_sequence_variant_predictions_v2(test_df, variant_name)
            if not ok:
                logger.warning(
                    "Variant '%s' unavailable (model/scaler missing or sequences could not be "
                    "built); skipping its diagnostics entirely.",
                    variant_name,
                )
                continue

            variant_errors = test_df[error_col]
            n_valid = int(variant_errors.notna().sum())
            n_total = len(test_df)
            if n_valid < n_total:
                logger.warning(
                    "Variant '%s': %d/%d test rows have no corrected-error value after merge "
                    "(likely rows that couldn't form a full sequence window); these are excluded "
                    "from this variant's stats.",
                    variant_name, n_total - n_valid, n_total,
                )

            variant_stats = compute_tail_stats(variant_errors.to_numpy())
            variant_stats["model"] = display_name
            summary_rows.append(variant_stats)
            logger.info(
                "%s: mean=%.3f km, median=%.3f km, worst-0.1%%=%.1f%%, worst-1%%=%.1f%%",
                display_name,
                variant_stats["mean_error_km"],
                variant_stats["median_error_km"],
                variant_stats["pct_total_abs_error_from_worst_0_1pct"],
                variant_stats["pct_total_abs_error_from_worst_1pct"],
            )

            # --- top-20 worst rows for this variant ---
            export_cols = [
                "object_id", "horizon_hours", "current_epoch_utc", RAW_COLUMN, error_col,
            ]
            missing_export_cols = [c for c in export_cols if c not in test_df.columns]
            if missing_export_cols:
                logger.warning(
                    "Variant '%s': missing expected export columns %s; skipping top-outlier CSV.",
                    variant_name, missing_export_cols,
                )
                continue

            worst_rows = (
                test_df.dropna(subset=[error_col])
                .sort_values(error_col, ascending=False)
                .head(TOP_N_OUTLIERS)[export_cols]
            )
            outlier_path = f"{config.OUTPUT_DIR}/diagnostics_top_outliers_{variant_name}.csv"
            worst_rows.to_csv(outlier_path, index=False)
            logger.info(
                "Saved top %d outlier rows for '%s' to %s",
                len(worst_rows), variant_name, outlier_path,
            )

        except Exception:
            logger.exception(
                "Unhandled error while processing variant '%s'; skipping this variant and "
                "continuing with the rest.",
                variant_name,
            )
            continue

    # --- combined summary table ------------------------------------------
    summary_cols = [
        "model",
        "pct_over_5km", "pct_over_10km", "pct_over_20km", "pct_over_50km", "pct_over_100km",
        "pct_total_abs_error_from_worst_0_1pct", "pct_total_abs_error_from_worst_1pct",
        "mean_error_km", "median_error_km",
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df[[c for c in summary_cols if c in summary_df.columns]]

    summary_path = f"{config.OUTPUT_DIR}/diagnostics_outlier_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved combined outlier summary to %s (%d rows).", summary_path, len(summary_df))

    # --- final console comparison ----------------------------------------
    print("\n=== OUTLIER / TAIL-CONCENTRATION DIAGNOSTICS (test set) ===")
    if summary_df.empty:
        print("No models could be evaluated -- see log warnings above.")
        return

    print(summary_df.to_string(index=False))

    if "raw_sgp4" in summary_df["model"].values:
        raw_row = summary_df[summary_df["model"] == "raw_sgp4"].iloc[0]
        print(
            "\n--- Tail share vs. raw SGP4 (is each variant's MAE gap concentrated "
            "in a few samples, or spread evenly?) ---"
        )
        print(
            f"{'model':45s} {'worst 0.1% share':>18s} {'worst 1% share':>16s} "
            f"{'mean err (km)':>14s}"
        )
        print(
            f"{'raw_sgp4':45s} "
            f"{raw_row['pct_total_abs_error_from_worst_0_1pct']:18.1f} "
            f"{raw_row['pct_total_abs_error_from_worst_1pct']:16.1f} "
            f"{raw_row['mean_error_km']:14.3f}"
        )
        for _, row in summary_df[summary_df["model"] != "raw_sgp4"].iterrows():
            print(
                f"{row['model']:45s} "
                f"{row['pct_total_abs_error_from_worst_0_1pct']:18.1f} "
                f"{row['pct_total_abs_error_from_worst_1pct']:16.1f} "
                f"{row['mean_error_km']:14.3f}"
            )
    else:
        print(
            "\nraw_sgp4 row unavailable -- cannot print the side-by-side tail-share "
            "comparison, only the raw per-model table above."
        )

    logger.info("diagnose_outliers complete.")


if __name__ == "__main__":
    run()
