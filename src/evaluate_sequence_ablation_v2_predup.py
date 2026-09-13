"""
evaluate_sequence_ablation_v2.py

PART D.2: Evaluates, on the TEST SET ONLY (touched exactly once across all
of Parts A-D), every model in the rolling-origin-CV-selected ablation
study side by side:

    1. Raw SGP4 baseline (position_error_km, already in the dataset)
    2. SGP4 + baseline (HistGradientBoosting) residual model -- UNCHANGED,
       NOT retrained (same model artifact as v1's evaluation; re-evaluated
       here only so this file is fully self-contained and directly
       comparable to ablation_summary.csv row-for-row)
    3-9. SGP4 + sequence (GRU) residual model, one per FEATURE_SET_VARIANTS
       entry (all 7, including the 2 new cycle-phase variants from Part B),
       using the FINAL models retrained in Part D.1
       (train_sequence_model_rolling_cv_final.py) at their rolling-CV-
       selected fixed epoch counts.

This reuses the existing evaluation building blocks rather than
reimplementing them:
    - evaluate_model._metrics / add_baseline_predictions
    - evaluate_sequence_ablation._build_row / summarize_variant /
      _plot_grouped_bar_mae   (all generic -- no v1-specific behavior)
    - train_sequence_model.GRUResidualModel / build_sequences /
      FEATURE_SET_VARIANTS

Because RTN is an orthonormal basis, the magnitude of the residual
correction error (||actual_RTN - predicted_RTN||) equals the true 3D
Cartesian position error after correction -- same reasoning as
evaluate_model.py / evaluate_sequence_ablation.py.

Schema: output/ablation_summary_v2.csv uses EXACTLY the same column names,
in the same order, as v1's output/ablation_summary.csv, so the two files
can be diffed directly. Row LABELS differ where needed (v2's sequence rows
are suffixed "_rolling_cv_v2" to distinguish them from v1's rows within
this file, and so the two are never confused if concatenated).

output/ablation_summary.csv (v1) is NEVER read from or written to by this
script, and the v1 model files under output/models/ are never touched.

Outputs:
    output/ablation_summary_v2.csv
    output/plots/ablation_comparison_v2.png                 (all 9 models)
    output/plots/ablation_comparison_sequence_only_v2.png   (7 sequence variants only)
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluate_sequence_ablation_v2_predup")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

RAW_COLUMN = "position_error_km"

# Suffix distinguishing the v2 (rolling-CV-selected) model artifacts,
# matching train_sequence_model_rolling_cv_final.py's MODEL_SUFFIX.
MODEL_SUFFIX = "_rolling_cv_v2"

# Display names for the model_variant column and plot legends. Suffixed
# with "_rolling_cv_v2" so these rows are unambiguous even if someone later
# concatenates this file with v1's ablation_summary.csv for analysis.
VARIANT_DISPLAY_NAMES_V2 = {
    "original": "sequence_original_rolling_cv_v2",
    "plus_orbital_jump": "sequence_plus_orbital_jump_rolling_cv_v2",
    "plus_space_weather_full": "sequence_plus_space_weather_full_rolling_cv_v2",
    "plus_tracking_cadence": "sequence_plus_tracking_cadence_rolling_cv_v2",
    "all_features": "sequence_all_features_rolling_cv_v2",
    "plus_space_weather_full_and_cycle_phase": "sequence_plus_space_weather_full_and_cycle_phase_rolling_cv_v2",
    "all_features_plus_cycle_phase": "sequence_all_features_plus_cycle_phase_rolling_cv_v2",
}

FULL_MODEL_ORDER_V2 = ["raw_sgp4", "baseline_tree"] + list(VARIANT_DISPLAY_NAMES_V2.values())
SEQUENCE_MODEL_ORDER_V2 = list(VARIANT_DISPLAY_NAMES_V2.values())


def add_sequence_variant_predictions_v2(df, variant_name):
    """
    Same logic as evaluate_sequence_ablation.add_sequence_variant_predictions,
    but points at the Part D.1 (rolling_cv_v2) model/scaler paths instead of
    the v1 ablation paths.
    """
    if not TORCH_AVAILABLE:
        return df, False, None

    from s06_train_sequence_model import GRUResidualModel  # deferred: torch-dependent class

    model_path = f"{config.MODELS_DIR}/sequence_model_{variant_name}{MODEL_SUFFIX}.pt"
    scaler_path = f"{config.MODELS_DIR}/sequence_scaler_{variant_name}{MODEL_SUFFIX}.joblib"

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        norm_stats = joblib.load(scaler_path)
    except FileNotFoundError:
        logger.warning(
            "v2 sequence model artifacts for variant '%s' not found (expected %s and %s). "
            "Did you run train_sequence_model_rolling_cv_final.py (Part D.1) first? Skipping.",
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
        logger.warning("Could not build evaluation sequences for v2 variant '%s'. Skipping.", variant_name)
        return df, False, None

    seq_n = (data["sequences"] - norm_stats["seq_mean"]) / norm_stats["seq_std"]
    static_n = (data["statics"] - norm_stats["static_mean"]) / norm_stats["static_std"]

    with torch.no_grad():
        preds = model(torch.from_numpy(seq_n.astype(np.float32)), torch.from_numpy(static_n.astype(np.float32)))
    preds = preds.numpy()

    diff = data["targets"] - preds
    seq_errors = np.linalg.norm(diff, axis=1)

    error_col = f"{variant_name}{MODEL_SUFFIX}_corrected_error_km"
    meta_df = pd.DataFrame(data["meta"], columns=["object_id", "horizon_hours", "current_epoch_utc"])
    meta_df[error_col] = seq_errors

    df = df.merge(meta_df, on=["object_id", "horizon_hours", "current_epoch_utc"], how="left")
    return df, True, error_col


def run():
    config.ensure_dirs()

    test_df = pd.read_csv(f"{config.OUTPUT_DIR}/test_dataset.csv")
    if test_df.empty:
        logger.warning("Test dataset is empty. Nothing to evaluate.")
        pd.DataFrame().to_csv(f"{config.OUTPUT_DIR}/ablation_summary_v2.csv", index=False)
        return
    test_df["current_epoch_utc"] = pd.to_datetime(test_df["current_epoch_utc"], utc=True, format="ISO8601")

    all_rows = []

    # 1. Raw SGP4 -----------------------------------------------------
    logger.info("Evaluating raw SGP4 (no correction)...")
    all_rows.extend(summarize_variant(test_df, RAW_COLUMN, "raw_sgp4"))

    # 2. Baseline tree model -- UNCHANGED, NOT retrained ----------------
    logger.info("Evaluating baseline (HistGradientBoosting) model (unchanged, same artifact as v1)...")
    test_df, has_baseline = add_baseline_predictions(test_df)
    if has_baseline:
        all_rows.extend(summarize_variant(test_df, "baseline_corrected_error_km", "baseline_tree"))
    else:
        logger.warning("Baseline model artifacts unavailable; 'baseline_tree' will be absent from the summary.")

    # 3-9. Sequence model variants (v2, rolling-CV-selected) -------------
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch is not installed; all sequence-model variants will be skipped.")
    for variant_name in FEATURE_SET_VARIANTS:
        display_name = VARIANT_DISPLAY_NAMES_V2[variant_name]
        logger.info("Evaluating v2 sequence model variant '%s' (%s)...", variant_name, display_name)
        test_df, ok, error_col = add_sequence_variant_predictions_v2(test_df, variant_name)
        if ok:
            all_rows.extend(summarize_variant(test_df, error_col, display_name))
        else:
            logger.warning("Variant '%s' unavailable; '%s' will be absent from the summary.", variant_name, display_name)

    summary_df = pd.DataFrame(all_rows)

    # Order rows: model_variant (fixed order), then group_type (by_horizon
    # before overall), then horizon_hours ascending within by_horizon --
    # identical convention to v1's evaluate_sequence_ablation.py.
    variant_rank = {name: i for i, name in enumerate(FULL_MODEL_ORDER_V2)}
    group_rank = {"by_horizon": 0, "overall": 1}
    summary_df["_variant_rank"] = summary_df["model_variant"].map(variant_rank).fillna(len(FULL_MODEL_ORDER_V2))
    summary_df["_group_rank"] = summary_df["group_type"].map(group_rank).fillna(2)
    summary_df["_horizon_rank"] = pd.to_numeric(summary_df["horizon_hours"], errors="coerce")
    summary_df = summary_df.sort_values(
        ["_variant_rank", "_group_rank", "_horizon_rank"]
    ).drop(columns=["_variant_rank", "_group_rank", "_horizon_rank"])

    # EXACT same column names/order as v1's ablation_summary.csv, so the
    # two files can be diffed directly.
    ordered_cols = [
        "model_variant", "group_type", "horizon_hours",
        "n_samples", "mae_km", "median_ae_km", "rmse_km", "p90_km", "p95_km", "max_km",
        "pct_samples_improved", "mean_pct_improvement",
    ]
    summary_df = summary_df[[c for c in ordered_cols if c in summary_df.columns]]

    summary_path = f"{config.OUTPUT_DIR}/ablation_summary_v2.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved v2 ablation summary to %s (%d rows). v1's ablation_summary.csv was NOT touched.", summary_path, len(summary_df))

    # --- plots ---
    _plot_grouped_bar_mae(
        summary_df, FULL_MODEL_ORDER_V2,
        "MAE by horizon: raw SGP4 vs baseline vs rolling-CV-selected sequence-model variants (v2)",
        f"{config.PLOTS_DIR}/ablation_comparison_v2.png",
    )
    _plot_grouped_bar_mae(
        summary_df, SEQUENCE_MODEL_ORDER_V2,
        "MAE by horizon: sequence-model feature-set variants only (v2, rolling-CV-selected)",
        f"{config.PLOTS_DIR}/ablation_comparison_sequence_only_v2.png",
    )

    # Console summary
    print("\n=== SEQUENCE ABLATION EVALUATION SUMMARY v2 (test set, rolling-CV-selected) ===")
    print(summary_df[summary_df["group_type"] == "overall"].to_string(index=False))
    print("\nBy horizon:")
    print(summary_df[summary_df["group_type"] == "by_horizon"].to_string(index=False))

    logger.info("evaluate_sequence_ablation_v2 complete.")


if __name__ == "__main__":
    run()
