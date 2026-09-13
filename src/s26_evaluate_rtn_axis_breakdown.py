"""
evaluate_rtn_axis_breakdown.py

STANDALONE, EVAL-ONLY, READ-ONLY-INPUT script.

Every existing evaluation script in this project (evaluate_model.py,
evaluate_sequence_ablation*.py, evaluate_dedup_vs_filtered_all_variants.py)
collapses the three RTN residual components -- radial_residual_km,
along_track_residual_km, cross_track_residual_km -- into a single scalar
via np.linalg.norm(diff, axis=1) immediately after computing the
prediction error, and never looks at the three components separately
again. This script fills that gap: it evaluates raw SGP4 and every
available corrected model PER AXIS, so we can answer:

    1. Which axis dominates raw SGP4's own error (checking the paper claim
       that "atmospheric drag primarily perturbs an object's radial
       distance rather than its orbital plane geometry")?
    2. For our best sequence-model configurations, is the improvement over
       raw SGP4 concentrated in one axis or spread across all three?

WHAT THIS SCRIPT DOES NOT DO
-----------------------------
- It does not modify any existing file.
- It does not modify, overwrite, retrain, or delete any existing model
  artifact, new-output/test_dataset.csv, or
  new-output/augmented_train_dataset.csv. new-output/test_dataset.csv is
  only ever read (pd.read_csv), never written.
- It does not train or retrain anything -- eval mode / no_grad / forward
  pass only, for already-trained checkpoints.

REUSED, UNCHANGED (imported, not reimplemented):
    - train_sequence_model.py: FEATURE_SET_VARIANTS, GRUResidualModel,
      build_sequences, TARGET_COLUMNS
    - evaluate_model.py: _metrics
    - config.py: OUTPUT_DIR, MODELS_DIR, PLOTS_DIR, RESIDUAL_TARGETS,
      ensure_dirs

NEW IN THIS SCRIPT:
    - add_axis_predictions_dedup / add_axis_predictions_filtered: follow
      the identical load-checkpoint / build-sequences / normalize /
      forward-pass pattern used by
      evaluate_sequence_ablation_v2_dedup.add_sequence_variant_predictions_v2_dedup,
      but PRESERVE the per-axis signed diff (actual - predicted) instead
      of immediately collapsing it via np.linalg.norm.

Outputs:
    new-output/rtn_axis_breakdown_summary.csv
    new-output/plots/rtn_axis_breakdown_comparison_dedup.png
    new-output/plots/rtn_axis_breakdown_comparison_filtered.png
"""

import logging
import os

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from s05_evaluate_baseline_and_raw_sgp4 import _metrics
from s06_train_sequence_model import FEATURE_SET_VARIANTS, TARGET_COLUMNS, build_sequences

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("s26_evaluate_rtn_axis_breakdown")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

RAW_MODEL_NAME = "raw_sgp4"
DEDUP_MODEL_SUFFIX = "_rolling_cv_v2_dedup"
FILTERED_MODEL_SUFFIX = "_postwindow_oversample_filtered_test"

TEST_MODELS_DIR = f"{config.OUTPUT_DIR}/test_output/models"  # filtered/oversampled models live here

# --------------------------------------------------------------------------
# Confirm (not assume) that config.RESIDUAL_TARGETS and
# train_sequence_model.TARGET_COLUMNS are the exact same list in the exact
# same order. The GRU's output vector's column order must match this for
# the per-axis diff below to be meaningful.
# --------------------------------------------------------------------------
if list(config.RESIDUAL_TARGETS) != list(TARGET_COLUMNS):
    raise RuntimeError(
        "config.RESIDUAL_TARGETS does not match train_sequence_model.TARGET_COLUMNS "
        f"(config.RESIDUAL_TARGETS={config.RESIDUAL_TARGETS!r}, "
        f"train_sequence_model.TARGET_COLUMNS={TARGET_COLUMNS!r}). "
        "Per-axis alignment between raw SGP4's residual columns and the model's "
        "output columns cannot be guaranteed -- refusing to proceed rather than "
        "silently comparing mismatched axes."
    )

AXES = list(config.RESIDUAL_TARGETS)  # ["radial_residual_km", "along_track_residual_km", "cross_track_residual_km"]
AXIS_SHORT_NAMES = {
    "radial_residual_km": "radial",
    "along_track_residual_km": "along_track",
    "cross_track_residual_km": "cross_track",
}


def _load_checkpoint_and_predict(df, model_path, scaler_path, display_name):
    """
    Shared load-checkpoint / build-sequences / normalize / forward-pass
    pattern (identical to
    evaluate_sequence_ablation_v2_dedup.add_sequence_variant_predictions_v2_dedup's),
    but returns the raw per-axis SIGNED diff array (actual - predicted)
    instead of collapsing it into a norm.

    Eval mode, no_grad, forward pass only -- no training happens here, and
    the checkpoint file itself is only ever read (torch.load), never
    written back to.

    Returns (meta_df, ok) where meta_df has columns
    ["object_id", "horizon_hours", "current_epoch_utc"] plus one signed
    per-axis diff column per entry in AXES, or (None, False) on failure.
    """
    if not TORCH_AVAILABLE:
        return None, False

    from s06_train_sequence_model import GRUResidualModel  # deferred: torch-dependent class

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        norm_stats = joblib.load(scaler_path)
    except FileNotFoundError:
        logger.warning(
            "Artifacts for '%s' not found (expected %s and %s). Skipping.",
            display_name, model_path, scaler_path,
        )
        return None, False

    checkpoint_targets = list(checkpoint.get("target_columns", TARGET_COLUMNS))
    if checkpoint_targets != TARGET_COLUMNS:
        logger.warning(
            "'%s' checkpoint's target_columns %r does not match "
            "train_sequence_model.TARGET_COLUMNS %r. Skipping this model rather "
            "than risk misaligned per-axis comparisons.",
            display_name, checkpoint_targets, TARGET_COLUMNS,
        )
        return None, False

    model = GRUResidualModel(
        n_features=checkpoint["n_features"],
        static_dim=checkpoint["static_dim"],
        n_targets=len(checkpoint["target_columns"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    data = build_sequences(df, checkpoint["window_size"], feature_cols=checkpoint["feature_cols"])
    if data is None:
        logger.warning("Could not build evaluation sequences for '%s'. Skipping.", display_name)
        return None, False

    seq_n = (data["sequences"] - norm_stats["seq_mean"]) / norm_stats["seq_std"]
    static_n = (data["statics"] - norm_stats["static_mean"]) / norm_stats["static_std"]

    with torch.no_grad():
        preds = model(torch.from_numpy(seq_n.astype(np.float32)), torch.from_numpy(static_n.astype(np.float32)))
    preds = preds.numpy()

    diff = data["targets"] - preds  # per-axis SIGNED error, columns in TARGET_COLUMNS order

    meta_df = pd.DataFrame(data["meta"], columns=["object_id", "horizon_hours", "current_epoch_utc"])
    for axis_idx, axis_name in enumerate(AXES):
        meta_df[f"{display_name}__{axis_name}_signed_err_km"] = diff[:, axis_idx]

    return meta_df, True


def add_axis_predictions_dedup(df, variant_name):
    """
    Per-axis version of add_sequence_variant_predictions_v2_dedup: loads
    the existing "<variant>_rolling_cv_v2_dedup" checkpoint (UNCHANGED,
    NOT retrained) from new-output/models/ and merges per-axis signed-error
    columns onto `df`.
    """
    display_name = f"{variant_name}{DEDUP_MODEL_SUFFIX}"
    model_path = f"{config.MODELS_DIR}/sequence_model_{display_name}.pt"
    scaler_path = f"{config.MODELS_DIR}/sequence_scaler_{display_name}.joblib"

    meta_df, ok = _load_checkpoint_and_predict(df, model_path, scaler_path, display_name)
    if not ok:
        logger.warning(
            "Dedup-only model '%s' unavailable; it will be absent from the summary.", display_name,
        )
        return df, False, display_name

    df = df.merge(meta_df, on=["object_id", "horizon_hours", "current_epoch_utc"], how="left")
    return df, True, display_name


def add_axis_predictions_filtered(df, variant_name):
    """
    Per-axis version of the flag-filtered / postwindow-oversample model
    loader: loads the existing "<variant>_postwindow_oversample_filtered_test"
    checkpoint (UNCHANGED, NOT retrained) from new-output/test_output/models/
    and merges per-axis signed-error columns onto `df`.
    """
    display_name = f"{variant_name}{FILTERED_MODEL_SUFFIX}"
    model_path = f"{TEST_MODELS_DIR}/sequence_model_{display_name}.pt"
    scaler_path = f"{TEST_MODELS_DIR}/sequence_scaler_{display_name}.joblib"

    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        logger.warning(
            "Filtered model '%s' artifacts not found (expected %s, %s). "
            "Skipping -- it will be absent from the summary.",
            display_name, model_path, scaler_path,
        )
        return df, False, display_name

    meta_df, ok = _load_checkpoint_and_predict(df, model_path, scaler_path, display_name)
    if not ok:
        logger.warning(
            "Filtered model '%s' unavailable; it will be absent from the summary.", display_name,
        )
        return df, False, display_name

    df = df.merge(meta_df, on=["object_id", "horizon_hours", "current_epoch_utc"], how="left")
    return df, True, display_name


def _axis_improvement_stats(raw_abs_series, corrected_abs_series):
    """
    Same convention used elsewhere in this project (see
    evaluate_dedup_vs_filtered_all_variants._improvement_stats /
    evaluate_sequence_ablation._build_row):
      pct_samples_improved  = fraction of rows where |corrected| < |raw| (as a %)
      mean_pct_improvement  = mean of (|raw| - |corrected|) / |raw| * 100,
                              over rows where both are present and raw != 0
    Applied here per axis rather than to the pooled L2 norm.
    """
    mask = raw_abs_series.notna() & corrected_abs_series.notna()
    raw_m = raw_abs_series[mask]
    corr_m = corrected_abs_series[mask]
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


def _summarize_axis_rows(df, model_name, axis_to_abs_col, raw_axis_to_abs_col):
    """
    Builds one summary row per axis for `model_name`, calling _metrics() on
    that axis's absolute-error array and computing pct_samples_improved /
    mean_pct_improvement vs. raw SGP4's error in that SAME axis (skipped --
    left NaN -- for the raw model itself, since it has nothing to compare
    against).
    """
    rows = []
    for axis_name in AXES:
        abs_col = axis_to_abs_col[axis_name]
        metrics = _metrics(df[abs_col].dropna())

        if model_name == RAW_MODEL_NAME:
            pct_improved, mean_improve = np.nan, np.nan
        else:
            raw_abs_col = raw_axis_to_abs_col[axis_name]
            pct_improved, mean_improve = _axis_improvement_stats(df[raw_abs_col], df[abs_col])

        rows.append({
            "model": model_name,
            "axis": AXIS_SHORT_NAMES[axis_name],
            **metrics,
            "pct_samples_improved": pct_improved,
            "mean_pct_improvement": mean_improve,
        })
    return rows


def _plot_grouped_bar_by_axis(summary_df, model_order, title, output_path, y_max=None):
    """
    Grouped bar chart: one group of bars per model, one bar per axis
    (radial / along-track / cross-track) within each group, MAE (km) on
    the y-axis.
    """
    pivot = summary_df[summary_df["model"].isin(model_order)].pivot(
        index="model", columns="axis", values="mae_km"
    )
    pivot = pivot.reindex(model_order)
    axis_order = [AXIS_SHORT_NAMES[a] for a in AXES]
    pivot = pivot.reindex(columns=axis_order)

    n_models = len(pivot)
    n_axes = len(axis_order)
    x = np.arange(n_models)
    bar_width = 0.8 / n_axes

    fig_width = max(8, 1.1 * n_models)
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    for i, axis_label in enumerate(axis_order):
        offset = (i - (n_axes - 1) / 2) * bar_width
        ax.bar(x + offset, pivot[axis_label].values, width=bar_width, label=axis_label)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=35, ha="right")
    ax.set_ylabel("MAE (km)")
    ax.set_title(title)
    ax.legend(title="RTN axis")
    if y_max is not None:
        ax.set_ylim(0, y_max)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def run():
    config.ensure_dirs()

    test_path = f"{config.OUTPUT_DIR}/test_dataset.csv"
    logger.info("Loading test dataset (READ-ONLY) from %s ...", test_path)
    test_df = pd.read_csv(test_path)
    if test_df.empty:
        logger.warning("Test dataset is empty. Nothing to evaluate.")
        pd.DataFrame().to_csv(f"{config.OUTPUT_DIR}/rtn_axis_breakdown_summary.csv", index=False)
        return
    test_df["current_epoch_utc"] = pd.to_datetime(test_df["current_epoch_utc"], utc=True, format="ISO8601")

    missing_axis_cols = [a for a in AXES if a not in test_df.columns]
    if missing_axis_cols:
        raise RuntimeError(
            f"test_dataset.csv is missing expected RTN residual column(s): {missing_axis_cols}. "
            "Cannot compute a per-axis breakdown without them."
        )

    if not TORCH_AVAILABLE:
        logger.warning("PyTorch is not installed; all sequence-model variants (dedup and filtered) will be skipped.")

    all_rows = []

    # -----------------------------------------------------------------
    # 1. Raw SGP4 -- for raw SGP4, the per-axis "error" for a given row IS
    #    simply the (signed) value of that row's own residual column,
    #    since these already represent SGP4's own raw offset from the
    #    RTN-projected truth position, before any correction. Absolute
    #    value is taken only when feeding into _metrics()/improvement math.
    # -----------------------------------------------------------------
    logger.info("Evaluating raw SGP4 (no correction), per RTN axis...")
    raw_axis_to_abs_col = {}
    for axis_name in AXES:
        abs_col = f"{RAW_MODEL_NAME}__{axis_name}_abs_err_km"
        test_df[abs_col] = pd.to_numeric(test_df[axis_name], errors="coerce").abs()
        raw_axis_to_abs_col[axis_name] = abs_col
    try:
        all_rows.extend(_summarize_axis_rows(test_df, RAW_MODEL_NAME, raw_axis_to_abs_col, raw_axis_to_abs_col))
    except Exception:
        logger.exception("Failed evaluating raw_sgp4; its rows will be absent from the summary.")

    model_order = [RAW_MODEL_NAME]
    dedup_model_order = [RAW_MODEL_NAME]
    filtered_model_order = [RAW_MODEL_NAME]

    # -----------------------------------------------------------------
    # 2. Dedup-only models: "<variant>_rolling_cv_v2_dedup" under
    #    new-output/models/. Loaded purely for evaluation -- UNCHANGED,
    #    NOT retrained.
    # -----------------------------------------------------------------
    for variant_name in FEATURE_SET_VARIANTS:
        logger.info("Evaluating dedup-only variant '%s' (RTN axis breakdown)...", variant_name)
        try:
            test_df, ok, display_name = add_axis_predictions_dedup(test_df, variant_name)
            if ok:
                axis_to_abs_col = {}
                for axis_name in AXES:
                    signed_col = f"{display_name}__{axis_name}_signed_err_km"
                    abs_col = f"{display_name}__{axis_name}_abs_err_km"
                    test_df[abs_col] = test_df[signed_col].abs()
                    axis_to_abs_col[axis_name] = abs_col
                all_rows.extend(_summarize_axis_rows(test_df, display_name, axis_to_abs_col, raw_axis_to_abs_col))
                model_order.append(display_name)
                dedup_model_order.append(display_name)
        except Exception:
            logger.exception("Failed evaluating dedup-only variant '%s'; its rows will be absent.", variant_name)

    # -----------------------------------------------------------------
    # 3. Flag-filtered / postwindow-oversample models:
    #    "<variant>_postwindow_oversample_filtered_test" under
    #    new-output/test_output/models/. Loaded purely for evaluation --
    #    UNCHANGED, NOT retrained. Not every variant has one.
    # -----------------------------------------------------------------
    for variant_name in FEATURE_SET_VARIANTS:
        logger.info("Evaluating filtered/oversampled variant '%s' (RTN axis breakdown)...", variant_name)
        try:
            test_df, ok, display_name = add_axis_predictions_filtered(test_df, variant_name)
            if ok:
                axis_to_abs_col = {}
                for axis_name in AXES:
                    signed_col = f"{display_name}__{axis_name}_signed_err_km"
                    abs_col = f"{display_name}__{axis_name}_abs_err_km"
                    test_df[abs_col] = test_df[signed_col].abs()
                    axis_to_abs_col[axis_name] = abs_col
                all_rows.extend(_summarize_axis_rows(test_df, display_name, axis_to_abs_col, raw_axis_to_abs_col))
                model_order.append(display_name)
                filtered_model_order.append(display_name)
        except Exception:
            logger.exception("Failed evaluating filtered variant '%s'; its rows will be absent.", variant_name)

    summary_df = pd.DataFrame(all_rows)
    if summary_df.empty:
        logger.warning("No rows could be evaluated at all. Writing an empty summary.")
        pd.DataFrame().to_csv(f"{config.OUTPUT_DIR}/rtn_axis_breakdown_summary.csv", index=False)
        return

    model_rank = {name: i for i, name in enumerate(model_order)}
    axis_rank = {AXIS_SHORT_NAMES[a]: i for i, a in enumerate(AXES)}
    summary_df["_model_rank"] = summary_df["model"].map(model_rank)
    summary_df["_axis_rank"] = summary_df["axis"].map(axis_rank)
    summary_df = summary_df.sort_values(["_model_rank", "_axis_rank"]).drop(columns=["_model_rank", "_axis_rank"])

    ordered_cols = [
        "model", "axis", "n_samples", "mae_km", "median_ae_km", "rmse_km", "p90_km", "p95_km", "max_km",
        "pct_samples_improved", "mean_pct_improvement",
    ]
    summary_df = summary_df[[c for c in ordered_cols if c in summary_df.columns]]

    summary_path = f"{config.OUTPUT_DIR}/rtn_axis_breakdown_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(
        "Saved RTN axis breakdown summary to %s (%d rows). test_dataset.csv was only read, never modified.",
        summary_path, len(summary_df),
    )

    # -----------------------------------------------------------------
    # Grouped bar charts. Same y-axis scale on both for visual comparability.
    # -----------------------------------------------------------------
    global_max_mae = summary_df["mae_km"].max()
    y_max = float(global_max_mae) * 1.1 if pd.notna(global_max_mae) else None

    try:
        _plot_grouped_bar_by_axis(
            summary_df, dedup_model_order,
            "Per-axis MAE: raw SGP4 vs dedup-only sequence-model variants",
            f"{config.PLOTS_DIR}/rtn_axis_breakdown_comparison_dedup.png",
            y_max=y_max,
        )
    except Exception:
        logger.exception("Failed generating the dedup-only comparison plot.")

    if len(filtered_model_order) > 1:
        try:
            _plot_grouped_bar_by_axis(
                summary_df, filtered_model_order,
                "Per-axis MAE: raw SGP4 vs flag-filtered/oversampled sequence-model variants",
                f"{config.PLOTS_DIR}/rtn_axis_breakdown_comparison_filtered.png",
                y_max=y_max,
            )
        except Exception:
            logger.exception("Failed generating the filtered comparison plot.")
    else:
        logger.warning("No filtered/oversampled models were available; skipping that comparison plot.")

    # -----------------------------------------------------------------
    # Raw-SGP4-only conclusion: which axis has the largest mean absolute
    # error overall? Directly checks/refutes the "drag primarily perturbs
    # radial distance" claim.
    # -----------------------------------------------------------------
    raw_rows = summary_df[summary_df["model"] == RAW_MODEL_NAME].set_index("axis")
    if not raw_rows.empty:
        worst_axis = raw_rows["mae_km"].idxmax()
        worst_mae = raw_rows.loc[worst_axis, "mae_km"]
        print("\n=== RAW SGP4 PER-AXIS MAE (checking the 'drag primarily perturbs radial distance' claim) ===")
        for axis_label in [AXIS_SHORT_NAMES[a] for a in AXES]:
            if axis_label in raw_rows.index:
                print(f"  {axis_label:15s} MAE = {raw_rows.loc[axis_label, 'mae_km']:.4f} km")
        print(
            f"\nCONCLUSION: Raw SGP4's LARGEST mean absolute error is in the "
            f"'{worst_axis}' axis ({worst_mae:.4f} km). "
            + (
                "This is CONSISTENT with the paper's claim that drag primarily "
                "perturbs radial distance."
                if worst_axis == "radial"
                else
                "This CONTRADICTS the paper's claim that drag primarily perturbs "
                "radial distance rather than orbital plane geometry -- the claim "
                "should be revisited."
            )
        )
    else:
        logger.warning("Could not determine raw SGP4's worst axis; raw_sgp4 rows are missing from the summary.")

    # -----------------------------------------------------------------
    # Final console summary table: one row per model, one column per axis's MAE.
    # -----------------------------------------------------------------
    print("\n=== FINAL SUMMARY: MAE (km) BY MODEL x RTN AXIS ===")
    pivot = summary_df.pivot(index="model", columns="axis", values="mae_km")
    axis_order = [AXIS_SHORT_NAMES[a] for a in AXES]
    pivot = pivot.reindex(index=[m for m in model_order if m in pivot.index], columns=axis_order)
    print(pivot.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\n=== FINAL SUMMARY: pct_samples_improved (%) BY MODEL x RTN AXIS (vs raw SGP4, same axis) ===")
    pivot_improve = summary_df.pivot(index="model", columns="axis", values="pct_samples_improved")
    pivot_improve = pivot_improve.reindex(
        index=[m for m in model_order if m != RAW_MODEL_NAME and m in pivot_improve.index],
        columns=axis_order,
    )
    if not pivot_improve.empty:
        print(pivot_improve.to_string(float_format=lambda v: f"{v:.2f}"))
    else:
        print("(no corrected models were available)")

    logger.info("evaluate_rtn_axis_breakdown complete.")


if __name__ == "__main__":
    run()
