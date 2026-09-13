"""
evaluate_dedup_vs_filtered_all_variants.py

STANDALONE, EVAL-ONLY SCRIPT. Trains nothing. Loads whatever
"<variant>_rolling_cv_v2_dedup" (new-output/models/) and
"<variant>_postwindow_oversample_filtered_test"
(new-output/test_output/models/) artifacts already exist on disk, scores
all of them plus raw SGP4 on new-output/test_dataset.csv, and prints the
complete dedup-only vs. flag-filtered-oversampling comparison per variant.

WHY THIS EXISTS
---------------
train_missing_dedup_baselines.py just filled in the three
"<variant>_rolling_cv_v2_dedup" baselines that were missing
(all_features, plus_space_weather_full_and_cycle_phase,
all_features_plus_cycle_phase). Re-running the full
train_and_eval_postwindow_oversample_filtered_all_variants.py script would
needlessly retrain all 7 filtered models again (each already exists under
new-output/test_output/models/ from the earlier run). This script only
does the evaluation half, against whatever is already on disk, so you get
the completed comparison without any retraining.

For any variant missing one or both of its two artifacts (e.g.
"all_features_plus_cycle_phase" -- its filtered model never finished
training due to an earlier out-of-memory error), that variant's row(s)
are simply skipped with a warning rather than failing the whole run.

REUSED, UNCHANGED:
    - evaluate_sequence_ablation_v2_dedup.py: add_sequence_variant_predictions_v2_dedup
    - train_sequence_model.py: FEATURE_SET_VARIANTS, GRUResidualModel, build_sequences
    - evaluate_model.py: _metrics

Outputs:
    new-output/test_output/diagnostics_dedup_vs_filtered_all_variants_summary.csv
"""

import logging
import os

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")

import config
import s06_train_sequence_model as tsm
from s19_evaluate_sequence_ablation_v2_dedup import add_sequence_variant_predictions_v2_dedup
from s05_evaluate_baseline_and_raw_sgp4 import _metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("s24_evaluate_dedup_vs_filtered_all_variants")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

RAW_COLUMN = "position_error_km"
DEDUP_MODEL_SUFFIX = "_rolling_cv_v2_dedup"
FILTERED_MODEL_SUFFIX = "_postwindow_oversample_filtered_test"

TEST_OUTPUT_DIR = f"{config.OUTPUT_DIR}/test_output"
TEST_MODELS_DIR = f"{TEST_OUTPUT_DIR}/models"


def add_checkpoint_predictions(df, model_path, scaler_path, display_name):
    """
    Generic load-checkpoint / build-sequences / normalize / forward-pass
    pattern (same as add_sequence_variant_predictions_v2_dedup's). Eval
    mode, no_grad, forward pass only -- no training happens here.
    """
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    norm_stats = joblib.load(scaler_path)

    model = tsm.GRUResidualModel(
        n_features=checkpoint["n_features"],
        static_dim=checkpoint["static_dim"],
        n_targets=len(checkpoint["target_columns"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    data = tsm.build_sequences(df, checkpoint["window_size"], feature_cols=checkpoint["feature_cols"])
    if data is None:
        logger.warning("Could not build evaluation sequences for '%s'. Skipping.", display_name)
        return df, False, None

    seq_n = (data["sequences"] - norm_stats["seq_mean"]) / norm_stats["seq_std"]
    static_n = (data["statics"] - norm_stats["static_mean"]) / norm_stats["static_std"]

    with torch.no_grad():
        preds = model(torch.from_numpy(seq_n.astype(np.float32)), torch.from_numpy(static_n.astype(np.float32)))
    preds = preds.numpy()

    diff = data["targets"] - preds
    seq_errors = np.linalg.norm(diff, axis=1)

    error_col = f"{display_name}_corrected_error_km"
    meta_df = pd.DataFrame(data["meta"], columns=["object_id", "horizon_hours", "current_epoch_utc"])
    meta_df[error_col] = seq_errors

    df = df.merge(meta_df, on=["object_id", "horizon_hours", "current_epoch_utc"], how="left")
    return df, True, error_col


def _improvement_stats(raw_series, corrected_series):
    """
    Same convention as evaluate_sequence_ablation.py's _build_row:
      pct_samples_improved  = fraction of rows where corrected < raw (as a %)
      mean_pct_improvement  = mean of (raw - corrected) / raw * 100,
                              over rows where both are present and raw != 0
    """
    mask = raw_series.notna() & corrected_series.notna()
    raw_m = raw_series[mask]
    corr_m = corrected_series[mask]
    if len(raw_m) == 0:
        return np.nan, np.nan
    pct_samples_improved = float((corr_m < raw_m).mean() * 100.0)
    nonzero_mask = raw_m != 0
    if nonzero_mask.any():
        pct_improvement = ((raw_m[nonzero_mask] - corr_m[nonzero_mask]) / raw_m[nonzero_mask]) * 100.0
        mean_pct_improvement = float(pct_improvement.mean())
    else:
        mean_pct_improvement = np.nan
    return pct_samples_improved, mean_pct_improvement


def run():
    if not TORCH_AVAILABLE:
        logger.error("PyTorch is not installed. Cannot run this evaluation.")
        return

    test_path = f"{config.OUTPUT_DIR}/test_dataset.csv"
    logger.info("Loading test dataset from %s ...", test_path)
    test_df = pd.read_csv(test_path)
    if test_df.empty:
        logger.warning("Test dataset is empty. Nothing to evaluate.")
        return
    test_df["current_epoch_utc"] = pd.to_datetime(test_df["current_epoch_utc"], utc=True, format="ISO8601")

    rows = []

    try:
        raw_metrics = _metrics(test_df[RAW_COLUMN])
        rows.append({"model": "raw_sgp4", **raw_metrics, "pct_samples_improved": np.nan, "mean_pct_improvement": np.nan})
    except Exception:
        logger.exception("Failed computing metrics for raw_sgp4.")

    per_variant_maes = {}

    for variant_name in tsm.FEATURE_SET_VARIANTS:
        per_variant_maes[variant_name] = {"dedup": np.nan, "filtered": np.nan}

        # dedup-only baseline
        dedup_display = f"{variant_name}{DEDUP_MODEL_SUFFIX}"
        try:
            test_df, ok, dedup_error_col = add_sequence_variant_predictions_v2_dedup(test_df, variant_name)
            if ok:
                dedup_metrics = _metrics(test_df[dedup_error_col].dropna())
                pct_improved, mean_improve = _improvement_stats(test_df[RAW_COLUMN], test_df[dedup_error_col])
                rows.append({
                    "model": dedup_display, **dedup_metrics,
                    "pct_samples_improved": pct_improved, "mean_pct_improvement": mean_improve,
                })
                per_variant_maes[variant_name]["dedup"] = dedup_metrics["mae_km"]
            else:
                logger.warning("'%s' unavailable; its row will be absent from the summary.", dedup_display)
        except Exception:
            logger.exception("Failed evaluating '%s'.", dedup_display)

        # flag-filtered post-window oversample model
        filtered_display = f"{variant_name}{FILTERED_MODEL_SUFFIX}"
        model_path = f"{TEST_MODELS_DIR}/sequence_model_{filtered_display}.pt"
        scaler_path = f"{TEST_MODELS_DIR}/sequence_scaler_{filtered_display}.joblib"
        if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
            logger.warning(
                "'%s' artifacts not found (expected %s, %s). Skipping -- its row will be absent from the summary.",
                filtered_display, model_path, scaler_path,
            )
        else:
            try:
                test_df, ok, filtered_error_col = add_checkpoint_predictions(
                    test_df, model_path, scaler_path, filtered_display
                )
                if ok:
                    filtered_metrics = _metrics(test_df[filtered_error_col].dropna())
                    pct_improved, mean_improve = _improvement_stats(test_df[RAW_COLUMN], test_df[filtered_error_col])
                    rows.append({
                        "model": filtered_display, **filtered_metrics,
                        "pct_samples_improved": pct_improved, "mean_pct_improvement": mean_improve,
                    })
                    per_variant_maes[variant_name]["filtered"] = filtered_metrics["mae_km"]
                else:
                    logger.warning("'%s' unavailable; its row will be absent from the summary.", filtered_display)
            except Exception:
                logger.exception("Failed evaluating '%s'.", filtered_display)

    summary_df = pd.DataFrame(rows)
    ordered_cols = [
        "model", "n_samples", "mae_km", "median_ae_km", "rmse_km", "p90_km", "p95_km", "max_km",
        "pct_samples_improved", "mean_pct_improvement",
    ]
    summary_df = summary_df[[c for c in ordered_cols if c in summary_df.columns]]

    summary_path = f"{TEST_OUTPUT_DIR}/diagnostics_dedup_vs_filtered_all_variants_summary.csv"
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved comparison table to %s (%d rows).", summary_path, len(summary_df))

    print("\n=== DEDUP-ONLY vs. FLAG-FILTERED OVERSAMPLING -- ALL VARIANTS (test set) ===")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    else:
        print("No rows could be evaluated.")

    print("\n=== PER-VARIANT VERDICT ===")
    by_model = summary_df.set_index("model") if not summary_df.empty else pd.DataFrame()
    raw_mae = by_model.loc["raw_sgp4", "mae_km"] if "raw_sgp4" in by_model.index else np.nan
    print(f"{'variant':45s} {'dedup mae_km':>15s} {'filtered mae_km':>17s} {'helped?':>9s} {'beats raw?':>11s}")
    for variant_name in tsm.FEATURE_SET_VARIANTS:
        dedup_mae = per_variant_maes[variant_name]["dedup"]
        new_mae = per_variant_maes[variant_name]["filtered"]

        if np.isnan(dedup_mae) or np.isnan(new_mae):
            helped_label = "N/A"
        elif new_mae < dedup_mae:
            helped_label = "YES"
        else:
            helped_label = "NO"

        if np.isnan(new_mae) or np.isnan(raw_mae):
            beats_raw_label = "N/A"
        elif new_mae <= raw_mae:
            beats_raw_label = "YES"
        else:
            beats_raw_label = "NO"

        dedup_str = f"{dedup_mae:15.3f}" if not np.isnan(dedup_mae) else f"{'n/a':>15s}"
        new_str = f"{new_mae:17.3f}" if not np.isnan(new_mae) else f"{'n/a':>17s}"
        print(f"{variant_name:45s} {dedup_str} {new_str} {helped_label:>9s} {beats_raw_label:>11s}")

    logger.info("evaluate_dedup_vs_filtered_all_variants complete.")


if __name__ == "__main__":
    run()
