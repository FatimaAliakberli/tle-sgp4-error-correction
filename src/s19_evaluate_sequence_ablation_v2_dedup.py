"""
evaluate_sequence_ablation_v2_dedup.py

PART D.2 (DEDUP-FIXED): Evaluates, on the TEST SET ONLY, ALL of the
following side by side:

    1. Raw SGP4 baseline (position_error_km, already in the dataset)
    2. SGP4 + baseline (HistGradientBoosting) residual model -- UNCHANGED,
       NOT retrained
    3-9. SGP4 + sequence (GRU) residual model, OLD (duplicate-timestamp
       -contaminated) "_rolling_cv_v2" artifacts, one per
       FEATURE_SET_VARIANTS entry -- UNCHANGED, NOT retrained here, loaded
       purely for comparison
    10-16. SGP4 + sequence (GRU) residual model, NEW dedup-fixed
       "_rolling_cv_v2_dedup" artifacts (from
       train_sequence_model_rolling_cv_final_dedup.py), one per
       FEATURE_SET_VARIANTS entry

This reuses the existing evaluation building blocks rather than
reimplementing them:
    - evaluate_model._metrics / add_baseline_predictions
    - evaluate_sequence_ablation._build_row / summarize_variant /
      _plot_grouped_bar_mae
    - evaluate_sequence_ablation_v2.add_sequence_variant_predictions_v2
      (for the OLD "_rolling_cv_v2" rows -- unchanged, not retrained)
    - train_sequence_model.GRUResidualModel / build_sequences /
      FEATURE_SET_VARIANTS

Only add_sequence_variant_predictions_v2_dedup (below) is new, and it is a
copy of add_sequence_variant_predictions_v2's exact pattern pointed at the
"_rolling_cv_v2_dedup" artifact paths instead -- nothing in
evaluate_sequence_ablation_v2.py, evaluate_sequence_ablation.py,
evaluate_model.py, or train_sequence_model.py is modified.

Neither ablation_summary.csv (v1) nor ablation_summary_v2.csv is ever read
from or written to by this script, and no existing model artifact
(v1, or "_rolling_cv_v2") is ever touched, overwritten, or retrained here.

Outputs:
    new-output/ablation_summary_v2_dedup.csv
    new-output/plots/ablation_comparison_v2_dedup.png                 (raw + baseline + all 14 sequence rows)
    new-output/plots/ablation_comparison_sequence_only_v2_dedup.png   (14 sequence variants only: 7 old + 7 dedup)
"""

import logging

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")

import config
from s05_evaluate_baseline_and_raw_sgp4 import _metrics, add_baseline_predictions
from evaluate_sequence_ablation_v1 import _build_row, summarize_variant, _plot_grouped_bar_mae
from s06_train_sequence_model import FEATURE_SET_VARIANTS, build_sequences
from evaluate_sequence_ablation_v2_predup import add_sequence_variant_predictions_v2, VARIANT_DISPLAY_NAMES_V2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("s19_evaluate_sequence_ablation_v2_dedup")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

RAW_COLUMN = "position_error_km"

# Matches train_sequence_model_rolling_cv_final_dedup.py's DEDUP_MODEL_SUFFIX.
DEDUP_MODEL_SUFFIX = "_rolling_cv_v2_dedup"

VARIANT_DISPLAY_NAMES_DEDUP = {
    variant: f"sequence_{variant}{DEDUP_MODEL_SUFFIX}" for variant in FEATURE_SET_VARIANTS
}

OLD_SEQUENCE_ORDER = list(VARIANT_DISPLAY_NAMES_V2.values())
DEDUP_SEQUENCE_ORDER = list(VARIANT_DISPLAY_NAMES_DEDUP.values())
FULL_MODEL_ORDER = ["raw_sgp4", "baseline_tree"] + OLD_SEQUENCE_ORDER + DEDUP_SEQUENCE_ORDER


def add_sequence_variant_predictions_v2_dedup(df, variant_name):
    """
    Exact same load-checkpoint / build-sequences / normalize / forward-pass
    pattern as evaluate_sequence_ablation_v2.add_sequence_variant_predictions_v2,
    pointed at the dedup-fixed "_rolling_cv_v2_dedup" artifact paths
    (train_sequence_model_rolling_cv_final_dedup.py's outputs) instead.
    Eval-mode, no_grad, forward pass only -- no training happens here.
    """
    if not TORCH_AVAILABLE:
        return df, False, None

    from s06_train_sequence_model import GRUResidualModel  # deferred: torch-dependent class

    model_path = f"{config.MODELS_DIR}/sequence_model_{variant_name}{DEDUP_MODEL_SUFFIX}.pt"
    scaler_path = f"{config.MODELS_DIR}/sequence_scaler_{variant_name}{DEDUP_MODEL_SUFFIX}.joblib"

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        norm_stats = joblib.load(scaler_path)
    except FileNotFoundError:
        logger.warning(
            "Dedup-fixed sequence model artifacts for variant '%s' not found (expected %s and %s). "
            "Did you run train_sequence_model_rolling_cv_final_dedup.py first? Skipping.",
            variant_name, model_path, scaler_path,
        )
        return df, False, None

    model = GRUResidualModel(
        n_features=checkpoint["n_features"],
        static_dim=checkpoint["static_dim"],
        n_targets=len(checkpoint["target_columns"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    data = build_sequences(df, checkpoint["window_size"], feature_cols=checkpoint["feature_cols"])
    if data is None:
        logger.warning("Could not build evaluation sequences for dedup variant '%s'. Skipping.", variant_name)
        return df, False, None

    seq_n = (data["sequences"] - norm_stats["seq_mean"]) / norm_stats["seq_std"]
    static_n = (data["statics"] - norm_stats["static_mean"]) / norm_stats["static_std"]

    with torch.no_grad():
        preds = model(torch.from_numpy(seq_n.astype(np.float32)), torch.from_numpy(static_n.astype(np.float32)))
    preds = preds.numpy()

    diff = data["targets"] - preds
    seq_errors = np.linalg.norm(diff, axis=1)

    error_col = f"{variant_name}{DEDUP_MODEL_SUFFIX}_corrected_error_km"
    meta_df = pd.DataFrame(data["meta"], columns=["object_id", "horizon_hours", "current_epoch_utc"])
    meta_df[error_col] = seq_errors

    df = df.merge(meta_df, on=["object_id", "horizon_hours", "current_epoch_utc"], how="left")
    return df, True, error_col


def run():
    config.ensure_dirs()

    test_df = pd.read_csv(f"{config.OUTPUT_DIR}/test_dataset.csv")
    if test_df.empty:
        logger.warning("Test dataset is empty. Nothing to evaluate.")
        pd.DataFrame().to_csv(f"{config.OUTPUT_DIR}/ablation_summary_v2_dedup.csv", index=False)
        return
    test_df["current_epoch_utc"] = pd.to_datetime(test_df["current_epoch_utc"], utc=True, format="ISO8601")

    all_rows = []

    # 1. Raw SGP4 -----------------------------------------------------
    logger.info("Evaluating raw SGP4 (no correction)...")
    try:
        all_rows.extend(summarize_variant(test_df, RAW_COLUMN, "raw_sgp4"))
    except Exception:
        logger.exception("Failed evaluating raw_sgp4; row will be absent from the summary.")

    # 2. Baseline tree model -- UNCHANGED, NOT retrained ----------------
    logger.info("Evaluating baseline (HistGradientBoosting) model (unchanged)...")
    try:
        test_df, has_baseline = add_baseline_predictions(test_df)
        if has_baseline:
            all_rows.extend(summarize_variant(test_df, "baseline_corrected_error_km", "baseline_tree"))
        else:
            logger.warning("Baseline model artifacts unavailable; 'baseline_tree' will be absent from the summary.")
    except Exception:
        logger.exception("Failed evaluating baseline_tree model; row will be absent from the summary.")

    if not TORCH_AVAILABLE:
        logger.warning("PyTorch is not installed; all sequence-model variants (old and dedup) will be skipped.")

    # 3-9. OLD (duplicate-timestamp-contaminated) v2 sequence models -----
    # Loaded purely for side-by-side comparison; NOT retrained here.
    for variant_name in FEATURE_SET_VARIANTS:
        display_name = VARIANT_DISPLAY_NAMES_V2[variant_name]
        logger.info("Evaluating OLD (contaminated) variant '%s' (%s)...", variant_name, display_name)
        try:
            test_df, ok, error_col = add_sequence_variant_predictions_v2(test_df, variant_name)
            if ok:
                all_rows.extend(summarize_variant(test_df, error_col, display_name))
            else:
                logger.warning("OLD variant '%s' unavailable; '%s' will be absent from the summary.",
                                variant_name, display_name)
        except Exception:
            logger.exception("Failed evaluating OLD variant '%s'; row will be absent from the summary.", variant_name)

    # 10-16. NEW dedup-fixed v2 sequence models --------------------------
    for variant_name in FEATURE_SET_VARIANTS:
        display_name = VARIANT_DISPLAY_NAMES_DEDUP[variant_name]
        logger.info("Evaluating NEW dedup-fixed variant '%s' (%s)...", variant_name, display_name)
        try:
            test_df, ok, error_col = add_sequence_variant_predictions_v2_dedup(test_df, variant_name)
            if ok:
                all_rows.extend(summarize_variant(test_df, error_col, display_name))
            else:
                logger.warning("NEW variant '%s' unavailable; '%s' will be absent from the summary. "
                                "Did you run train_sequence_model_rolling_cv_final_dedup.py first?",
                                variant_name, display_name)
        except Exception:
            logger.exception("Failed evaluating NEW variant '%s'; row will be absent from the summary.", variant_name)

    summary_df = pd.DataFrame(all_rows)
    if summary_df.empty:
        logger.warning("No rows could be evaluated at all. Writing an empty summary.")
        pd.DataFrame().to_csv(f"{config.OUTPUT_DIR}/ablation_summary_v2_dedup.csv", index=False)
        return

    # Order rows: model_variant (fixed order: raw, baseline, all 7 OLD,
    # then all 7 NEW), then group_type (by_horizon before overall), then
    # horizon_hours ascending within by_horizon.
    variant_rank = {name: i for i, name in enumerate(FULL_MODEL_ORDER)}
    group_rank = {"by_horizon": 0, "overall": 1}
    summary_df["_variant_rank"] = summary_df["model_variant"].map(variant_rank).fillna(len(FULL_MODEL_ORDER))
    summary_df["_group_rank"] = summary_df["group_type"].map(group_rank).fillna(2)
    summary_df["_horizon_rank"] = pd.to_numeric(summary_df["horizon_hours"], errors="coerce")
    summary_df = summary_df.sort_values(
        ["_variant_rank", "_group_rank", "_horizon_rank"]
    ).drop(columns=["_variant_rank", "_group_rank", "_horizon_rank"])

    ordered_cols = [
        "model_variant", "group_type", "horizon_hours",
        "n_samples", "mae_km", "median_ae_km", "rmse_km", "p90_km", "p95_km", "max_km",
        "pct_samples_improved", "mean_pct_improvement",
    ]
    summary_df = summary_df[[c for c in ordered_cols if c in summary_df.columns]]

    summary_path = f"{config.OUTPUT_DIR}/ablation_summary_v2_dedup.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(
        "Saved dedup ablation summary to %s (%d rows). ablation_summary.csv and ablation_summary_v2.csv "
        "were NOT touched.", summary_path, len(summary_df),
    )

    # --- plots ---
    try:
        _plot_grouped_bar_mae(
            summary_df, FULL_MODEL_ORDER,
            "MAE by horizon: raw SGP4 vs baseline vs OLD (contaminated) vs NEW (dedup-fixed) sequence variants",
            f"{config.PLOTS_DIR}/ablation_comparison_v2_dedup.png",
        )
    except Exception:
        logger.exception("Failed generating the full comparison plot.")
    try:
        _plot_grouped_bar_mae(
            summary_df, OLD_SEQUENCE_ORDER + DEDUP_SEQUENCE_ORDER,
            "MAE by horizon: sequence-model variants only -- OLD (contaminated) vs NEW (dedup-fixed)",
            f"{config.PLOTS_DIR}/ablation_comparison_sequence_only_v2_dedup.png",
        )
    except Exception:
        logger.exception("Failed generating the sequence-only comparison plot.")

    # Console summary
    print("\n=== SEQUENCE ABLATION EVALUATION SUMMARY (dedup-fixed retrain, test set) ===")
    print(summary_df[summary_df["group_type"] == "overall"].to_string(index=False))
    print("\nBy horizon:")
    print(summary_df[summary_df["group_type"] == "by_horizon"].to_string(index=False))

    # --- per-variant side-by-side verdict: did dedup fix each variant? ---
    print("\n=== PER-VARIANT VERDICT: did removing duplicate-timestamp training rows help? ===")
    overall = summary_df[summary_df["group_type"] == "overall"].set_index("model_variant")
    raw_mae = overall.loc["raw_sgp4", "mae_km"] if "raw_sgp4" in overall.index else np.nan
    print(f"{'variant':45s} {'old_v2 mae_km':>15s} {'dedup mae_km':>15s} {'fixed?':>8s} {'beats raw?':>11s}")
    for variant_name in FEATURE_SET_VARIANTS:
        old_name = VARIANT_DISPLAY_NAMES_V2[variant_name]
        new_name = VARIANT_DISPLAY_NAMES_DEDUP[variant_name]
        old_mae = overall.loc[old_name, "mae_km"] if old_name in overall.index else np.nan
        new_mae = overall.loc[new_name, "mae_km"] if new_name in overall.index else np.nan

        if np.isnan(old_mae) or np.isnan(new_mae):
            fixed_label = "N/A"
        elif new_mae < old_mae:
            fixed_label = "YES"
        else:
            fixed_label = "NO"

        if np.isnan(new_mae) or np.isnan(raw_mae):
            beats_raw_label = "N/A"
        elif new_mae <= raw_mae:
            beats_raw_label = "YES"
        else:
            beats_raw_label = "NO"

        print(f"{variant_name:45s} {old_mae:15.3f} {new_mae:15.3f} {fixed_label:>8s} {beats_raw_label:>11s}")

    logger.info("evaluate_sequence_ablation_v2_dedup complete.")


if __name__ == "__main__":
    run()
