"""
evaluate_sequence_ablation.py

Evaluates, on the TEST SET ONLY, every model in the feature-set ablation
study side by side:

    1. Raw SGP4 baseline (position_error_km, already in the dataset)
    2. SGP4 + baseline (HistGradientBoosting) residual model
    3. SGP4 + sequence (GRU) residual model, "original" feature set
    4. SGP4 + sequence (GRU) residual model, "plus_orbital_jump"
    5. SGP4 + sequence (GRU) residual model, "plus_space_weather_full"
    6. SGP4 + sequence (GRU) residual model, "plus_tracking_cadence"
    7. SGP4 + sequence (GRU) residual model, "all_features"

This reuses the existing evaluation building blocks rather than
reimplementing them:
    - evaluate_model._metrics              (mae/median/rmse/p90/p95/max)
    - evaluate_model.add_baseline_predictions
    - train_sequence_model.GRUResidualModel / build_sequences
    - train_sequence_model.FEATURE_SET_VARIANTS

Because RTN is an orthonormal basis, the magnitude of the residual
correction error (||actual_RTN - predicted_RTN||) equals the true 3D
Cartesian position error after correction -- no need to reconstruct
Cartesian vectors (same reasoning as evaluate_model.py).

Outputs:
    output/ablation_summary.csv
    output/plots/ablation_comparison.png              (all 7 variants)
    output/plots/ablation_comparison_sequence_only.png (5 sequence variants only)
"""

import logging

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from s05_evaluate_baseline_and_raw_sgp4 import _metrics, add_baseline_predictions
# GRUResidualModel is defined inside train_sequence_model's `if TORCH_AVAILABLE:`
# block, so it's only importable when torch is installed. FEATURE_SET_VARIANTS
# and build_sequences have no torch dependency and are always importable.
from s06_train_sequence_model import FEATURE_SET_VARIANTS, build_sequences

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluate_sequence_ablation_v1")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

RAW_COLUMN = "position_error_km"

# Display names for the model_variant column in ablation_summary.csv and
# the plot legends. Keys match train_sequence_model.FEATURE_SET_VARIANTS.
VARIANT_DISPLAY_NAMES = {
    "original": "sequence_original",
    "plus_orbital_jump": "sequence_plus_orbital_jump",
    "plus_space_weather_full": "sequence_plus_space_weather_full",
    "plus_tracking_cadence": "sequence_plus_tracking_cadence",
    "all_features": "sequence_all_features",
}

# Fixed ordering for the "all models" comparison (row order in the CSV and
# bar order in the full comparison plot).
FULL_MODEL_ORDER = ["raw_sgp4", "baseline_tree"] + list(VARIANT_DISPLAY_NAMES.values())

# Sequence-only ordering for the second, zoomed-in plot.
SEQUENCE_MODEL_ORDER = list(VARIANT_DISPLAY_NAMES.values())


def add_sequence_variant_predictions(df, variant_name):
    """
    Adds a column f"{variant_name}_corrected_error_km" to df using the
    trained sequence model for that variant, if its artifacts exist.

    Mirrors evaluate_model.add_sequence_predictions, but is parameterized
    by variant name (variant-specific model/scaler paths) and passes each
    checkpoint's own `feature_cols` through to build_sequences() -- this is
    the piece that must be variant-aware, since different variants were
    trained on different per-step feature sets.

    Returns
    -------
    df : pd.DataFrame (with the new column merged in, if successful)
    success : bool
    error_col : str or None
    """
    if not TORCH_AVAILABLE:
        return df, False, None

    from s06_train_sequence_model import GRUResidualModel  # deferred: torch-dependent class

    model_path = f"{config.MODELS_DIR}/sequence_model_{variant_name}.pt"
    scaler_path = f"{config.MODELS_DIR}/sequence_scaler_{variant_name}.joblib"

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        norm_stats = joblib.load(scaler_path)
    except FileNotFoundError:
        logger.warning(
            "Sequence model artifacts for variant '%s' not found (expected %s and %s). Skipping.",
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
        logger.warning(
            "Could not build evaluation sequences for variant '%s'. Skipping.", variant_name
        )
        return df, False, None

    seq_n = (data["sequences"] - norm_stats["seq_mean"]) / norm_stats["seq_std"]
    static_n = (data["statics"] - norm_stats["static_mean"]) / norm_stats["static_std"]

    with torch.no_grad():
        preds = model(torch.from_numpy(seq_n.astype(np.float32)), torch.from_numpy(static_n.astype(np.float32)))
    preds = preds.numpy()

    diff = data["targets"] - preds
    seq_errors = np.linalg.norm(diff, axis=1)

    error_col = f"{variant_name}_corrected_error_km"
    meta_df = pd.DataFrame(data["meta"], columns=["object_id", "horizon_hours", "current_epoch_utc"])
    meta_df[error_col] = seq_errors

    df = df.merge(meta_df, on=["object_id", "horizon_hours", "current_epoch_utc"], how="left")
    return df, True, error_col


def _build_row(g, error_col, raw_col, model_variant, group_type, horizon_value):
    """Build one summary row for a (model_variant, group) slice, matching
    the metric-naming convention used in evaluate_model.summarize()."""
    row = {"model_variant": model_variant, "group_type": group_type, "horizon_hours": horizon_value}
    row.update(_metrics(g[error_col]))

    if model_variant == "raw_sgp4":
        # "Improvement vs raw SGP4" is undefined for raw SGP4 itself.
        row["pct_samples_improved"] = np.nan
        row["mean_pct_improvement"] = np.nan
    else:
        valid = g[[error_col, raw_col]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(valid) > 0:
            row["pct_samples_improved"] = (valid[error_col] < valid[raw_col]).mean() * 100
            row["mean_pct_improvement"] = (
                (valid[raw_col] - valid[error_col]) / valid[raw_col].replace(0, np.nan)
            ).mean() * 100
        else:
            row["pct_samples_improved"] = np.nan
            row["mean_pct_improvement"] = np.nan

    return row


def summarize_variant(df, error_col, model_variant, raw_col=RAW_COLUMN):
    """Produce by_horizon rows (one per horizon_hours) plus one overall row
    for a single model variant, matching evaluate_model.py's group_type
    convention (LEAKY_OR_ID_COLUMNS / summarize() pattern: group by
    horizon_hours for the "by_horizon" breakdown, plus a single "overall"
    row with no groupby key)."""
    rows = []
    for horizon_hours, g in df.groupby("horizon_hours"):
        rows.append(_build_row(g, error_col, raw_col, model_variant, "by_horizon", horizon_hours))
    rows.append(_build_row(df, error_col, raw_col, model_variant, "overall", "all"))
    return rows


def _plot_grouped_bar_mae(summary_df, model_variants, title, output_path):
    """Grouped bar chart of mae_km by horizon_hours, one bar group per
    horizon and one bar per model_variant present in model_variants."""
    horizon_df = summary_df[summary_df["group_type"] == "by_horizon"].copy()
    horizons = sorted(horizon_df["horizon_hours"].unique())
    present_variants = [v for v in model_variants if v in set(horizon_df["model_variant"])]

    if not horizons or not present_variants:
        logger.warning("Not enough data to plot '%s'. Skipping.", title)
        return

    n_variants = len(present_variants)
    x = np.arange(len(horizons))
    width = 0.8 / n_variants

    fig, ax = plt.subplots(figsize=(max(8.0, 1.8 * len(horizons) + 0.6 * n_variants), 5))
    for i, variant in enumerate(present_variants):
        values = []
        for h in horizons:
            match = horizon_df[(horizon_df["model_variant"] == variant) & (horizon_df["horizon_hours"] == h)]
            values.append(match["mae_km"].iloc[0] if len(match) > 0 else np.nan)
        offset = (i - (n_variants - 1) / 2) * width
        ax.bar(x + offset, values, width, label=variant)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(h)}h" for h in horizons])
    ax.set_xlabel("Horizon")
    ax.set_ylabel("MAE (km)")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info("Saved plot: %s", output_path)


def run():
    config.ensure_dirs()

    test_df = pd.read_csv(f"{config.OUTPUT_DIR}/test_dataset.csv")
    if test_df.empty:
        logger.warning("Test dataset is empty. Nothing to evaluate.")
        pd.DataFrame().to_csv(f"{config.OUTPUT_DIR}/ablation_summary.csv", index=False)
        return
    test_df["current_epoch_utc"] = pd.to_datetime(test_df["current_epoch_utc"], utc=True, format="ISO8601")

    all_rows = []

    # 1. Raw SGP4 -----------------------------------------------------
    logger.info("Evaluating raw SGP4 (no correction)...")
    all_rows.extend(summarize_variant(test_df, RAW_COLUMN, "raw_sgp4"))

    # 2. Baseline tree model -------------------------------------------
    logger.info("Evaluating baseline (HistGradientBoosting) model...")
    test_df, has_baseline = add_baseline_predictions(test_df)
    if has_baseline:
        all_rows.extend(summarize_variant(test_df, "baseline_corrected_error_km", "baseline_tree"))
    else:
        logger.warning("Baseline model artifacts unavailable; 'baseline_tree' will be absent from the summary.")

    # 3-7. Sequence model variants --------------------------------------
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch is not installed; all sequence-model variants will be skipped.")
    for variant_name in FEATURE_SET_VARIANTS:
        display_name = VARIANT_DISPLAY_NAMES[variant_name]
        logger.info("Evaluating sequence model variant '%s' (%s)...", variant_name, display_name)
        test_df, ok, error_col = add_sequence_variant_predictions(test_df, variant_name)
        if ok:
            all_rows.extend(summarize_variant(test_df, error_col, display_name))
        else:
            logger.warning("Variant '%s' unavailable; '%s' will be absent from the summary.", variant_name, display_name)

    summary_df = pd.DataFrame(all_rows)

    # Order rows: model_variant (fixed order), then group_type (by_horizon
    # before overall), then horizon_hours ascending within by_horizon.
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

    summary_path = f"{config.OUTPUT_DIR}/ablation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved ablation summary to %s (%d rows).", summary_path, len(summary_df))

    # --- plots ---
    _plot_grouped_bar_mae(
        summary_df, FULL_MODEL_ORDER,
        "MAE by horizon: raw SGP4 vs baseline vs sequence-model variants",
        f"{config.PLOTS_DIR}/ablation_comparison.png",
    )
    _plot_grouped_bar_mae(
        summary_df, SEQUENCE_MODEL_ORDER,
        "MAE by horizon: sequence-model feature-set variants only",
        f"{config.PLOTS_DIR}/ablation_comparison_sequence_only.png",
    )

    # Console summary
    print("\n=== SEQUENCE ABLATION EVALUATION SUMMARY (test set) ===")
    print(summary_df[summary_df["group_type"] == "overall"].to_string(index=False))
    print("\nBy horizon:")
    print(summary_df[summary_df["group_type"] == "by_horizon"].to_string(index=False))

    logger.info("evaluate_sequence_ablation complete.")


if __name__ == "__main__":
    run()
