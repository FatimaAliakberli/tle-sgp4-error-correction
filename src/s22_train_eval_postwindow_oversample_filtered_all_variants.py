"""
train_and_eval_postwindow_oversample_filtered_all_variants.py

STANDALONE EXPERIMENT SCRIPT -- does not modify any existing file, does not
modify or overwrite new-output/augmented_train_dataset.csv, and does not
touch anything under new-output/models or new-output/plots. All outputs go
under new-output/test_output/ only.

WHAT THIS DOES
--------------
Runs the flag-filtered post-window oversampling augmentation (validated for
the "original" variant in
train_and_eval_postwindow_oversample_filtered_test.py) across EVERY entry
of FEATURE_SET_VARIANTS, one model per variant:

For each variant:
    1. Load new-output/augmented_train_dataset.csv READ-ONLY, filter
       in-memory to augmentation_type == "original" only (shared across all
       variants -- this filter does not depend on feature_cols).
    2. build_sequences() using that variant's FEATURE_SET_VARIANTS entry.
    3. Determine per-window oversampling eligibility: a window is eligible
       unless its source row (the row whose label the window predicts --
       g.iloc[k] inside build_sequences(), identified via the same
       (object_id, horizon_hours, current_epoch_utc) key build_sequences()
       records in `meta`) has outlier_flag, truth_offset_flag, or
       huge_gap_flag set to True. Ineligible windows remain in the base
       training set; they are simply not candidates for duplication.
    4. Among eligible windows only, duplicate those at or above the
       OVERSAMPLE_HIGH_ERROR_QUANTILE quantile of target L2-norm magnitude,
       OVERSAMPLE_FACTOR - 1 additional times, onto the full base tensors.
    5. _normalize() on the full (base + oversampled) tensor set.
    6. Train ONE GRUResidualModel (SmoothL1Loss / Adam lr=1e-3 / batch size
       64 / torch.manual_seed(config.RANDOM_SEED), no early stopping, no
       validation split) for that variant's fixed epoch count, read from
       new-output/rolling_origin_cv_variant_summary.csv.
    7. Save artifacts under new-output/test_output/{models,plots}/, named
       sequence_model_<variant>_postwindow_oversample_filtered_test.pt /
       .joblib / training-loss plot.

Then, on new-output/test_dataset.csv:
    8. Evaluates raw SGP4 (position_error_km) once.
    9. For each variant: evaluates the existing, UNCHANGED
       "<variant>_rolling_cv_v2_dedup" model (via
       add_sequence_variant_predictions_v2_dedup, not retrained) AND this
       run's new "<variant>_postwindow_oversample_filtered_test" model.
   10. Computes _metrics() plus pct_samples_improved / mean_pct_improvement
       (vs. raw SGP4) for every non-raw row.
   11. Saves the full comparison table and prints a per-variant verdict:
       did flag-filtered oversampling help vs. the dedup-only baseline for
       that variant, and does it beat raw SGP4.

A per-variant training failure does not stop the other variants (matches
train_sequence_model_rolling_cv_final_dedup.py's loop behavior) -- failures
are recorded in the training summary and that variant's rows are simply
absent from the final comparison table.

REUSED, UNCHANGED (per project convention -- nothing in these files is
modified or reimplemented):
    - train_sequence_model.py: FEATURE_SET_VARIANTS, GRUResidualModel,
      build_sequences, _normalize, SequenceDataset, TARGET_COLUMNS
    - evaluate_sequence_ablation_v2_dedup.py:
      add_sequence_variant_predictions_v2_dedup
    - evaluate_model.py: _metrics

Outputs:
    new-output/test_output/models/sequence_model_<variant>_postwindow_oversample_filtered_test.pt
    new-output/test_output/models/sequence_scaler_<variant>_postwindow_oversample_filtered_test.joblib
    new-output/test_output/plots/sequence_training_loss_<variant>_postwindow_oversample_filtered_test.png
    new-output/test_output/postwindow_oversample_filtered_all_variants_training_summary.csv
    new-output/test_output/diagnostics_postwindow_oversample_filtered_all_variants_summary.csv
"""

import logging
import os
import time
import traceback

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
logger = logging.getLogger("s22_train_eval_postwindow_oversample_filtered_all_variants")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# --------------------------------------------------------------------------
# Module-level configuration. Reuses config's existing augmentation
# constants if present, otherwise falls back to the stated defaults.
# --------------------------------------------------------------------------
OVERSAMPLE_HIGH_ERROR_QUANTILE = getattr(config, "AUGMENTATION_HIGH_ERROR_QUANTILE", 0.95)
OVERSAMPLE_FACTOR = getattr(config, "AUGMENTATION_OVERSAMPLE_FACTOR", 3)

# Windows whose source row has ANY of these flags set to True are excluded
# from the pool of oversampling CANDIDATES (they remain in the base
# training set normally -- only the oversampling step skips them).
FLAG_COLUMNS = ["outlier_flag", "truth_offset_flag", "huge_gap_flag"]

TEST_OUTPUT_DIR = f"{config.OUTPUT_DIR}/test_output"
TEST_MODELS_DIR = f"{TEST_OUTPUT_DIR}/models"
TEST_PLOTS_DIR = f"{TEST_OUTPUT_DIR}/plots"

TRAIN_BATCH_SIZE = 64
LEARNING_RATE = 1e-3

RAW_COLUMN = "position_error_km"
DEDUP_MODEL_SUFFIX = "_rolling_cv_v2_dedup"
NEW_MODEL_SUFFIX = "_postwindow_oversample_filtered_test"


def _ensure_test_output_dirs():
    os.makedirs(TEST_MODELS_DIR, exist_ok=True)
    os.makedirs(TEST_PLOTS_DIR, exist_ok=True)


def _load_fixed_epoch_counts():
    """
    Reads fixed_epoch_count for every FEATURE_SET_VARIANTS entry from
    new-output/rolling_origin_cv_variant_summary.csv, so each variant's new
    model trains for exactly the same number of epochs as that variant's
    existing "<variant>_rolling_cv_v2_dedup" model.
    """
    summary_path = f"{config.OUTPUT_DIR}/rolling_origin_cv_variant_summary.csv"
    summary_df = pd.read_csv(summary_path)
    counts = {}
    for _, row in summary_df.iterrows():
        variant = row.get("model_variant")
        if variant in tsm.FEATURE_SET_VARIANTS and pd.notna(row.get("fixed_epoch_count")):
            counts[variant] = int(row["fixed_epoch_count"])
    logger.info("Loaded fixed_epoch_count for %d/%d variants from %s: %s",
                len(counts), len(tsm.FEATURE_SET_VARIANTS), summary_path, counts)
    return counts


def build_deduplicated_train_df(raw_train_df: pd.DataFrame) -> pd.DataFrame:
    """
    In-memory-only filter: keep augmentation_type == "original" rows.
    Never writes back to disk. Shared across all variants -- does not
    depend on feature_cols.
    """
    n_before = len(raw_train_df)
    logger.info("Full augmented_train_dataset.csv row count (in memory, unmodified on disk): %d", n_before)

    if "augmentation_type" not in raw_train_df.columns:
        raise RuntimeError("'augmentation_type' column not found in augmented_train_dataset.csv.")

    dedup_df = raw_train_df[raw_train_df["augmentation_type"] == "original"].copy()
    n_after = len(dedup_df)
    logger.info(
        "Filtered to augmentation_type == 'original' only: %d -> %d rows (%.2f%% retained). "
        "This is an in-memory pandas filter only -- the CSV on disk is untouched.",
        n_before, n_after, (n_after / n_before * 100.0) if n_before > 0 else float("nan"),
    )
    if n_after == 0:
        raise RuntimeError("No rows remain after filtering to augmentation_type == 'original'.")

    return dedup_df


def _coerce_bool_column(series: pd.Series) -> pd.Series:
    """
    Robustly coerces a column that should be boolean into actual booleans,
    whether it was read as native bool or as the strings "True"/"False".
    Missing values are treated as False.
    """
    if series.dtype == bool:
        return series.fillna(False)
    return series.map(
        lambda v: str(v).strip().lower() in ("true", "1", "1.0", "yes")
        if pd.notna(v) else False
    )


def compute_window_eligibility(meta, dedup_train_df: pd.DataFrame, variant_name: str) -> np.ndarray:
    """
    For every window in `meta`, determines oversampling eligibility by
    looking up its source row's outlier_flag / truth_offset_flag /
    huge_gap_flag via the (object_id, horizon_hours, current_epoch_utc) key
    build_sequences() records -- the row = g.iloc[k] each window's target
    came from. Returns a boolean array (True = eligible).
    """
    missing_flag_cols = [c for c in FLAG_COLUMNS if c not in dedup_train_df.columns]
    if missing_flag_cols:
        raise RuntimeError(
            f"Expected flag column(s) missing from augmented_train_dataset.csv: {missing_flag_cols}"
        )

    key_cols = ["object_id", "horizon_hours", "current_epoch_utc"]
    lookup = dedup_train_df[key_cols + FLAG_COLUMNS].copy()
    for c in FLAG_COLUMNS:
        lookup[c] = _coerce_bool_column(lookup[c])
    lookup = lookup.drop_duplicates(subset=key_cols, keep="first")

    meta_df = pd.DataFrame(meta, columns=key_cols)
    merged = meta_df.merge(lookup, on=key_cols, how="left")

    n_unmatched = int(merged[FLAG_COLUMNS[0]].isna().sum())
    if n_unmatched > 0:
        logger.warning(
            "[%s] %d / %d window(s) could not be matched back to a source row for flag lookup "
            "(treating as eligible by default).", variant_name, n_unmatched, len(merged),
        )
    for c in FLAG_COLUMNS:
        merged[c] = merged[c].fillna(False).astype(bool)

    ineligible_mask = merged[FLAG_COLUMNS[0]] | merged[FLAG_COLUMNS[1]] | merged[FLAG_COLUMNS[2]]
    eligible_mask = (~ineligible_mask).values

    n_total = len(eligible_mask)
    n_eligible = int(eligible_mask.sum())
    logger.info(
        "[%s] Oversampling-eligibility filter: %d / %d windows (%.2f%%) eligible, %d excluded "
        "(outlier_flag/truth_offset_flag/huge_gap_flag) -- excluded windows remain in the base "
        "training set, they are only ineligible as oversampling candidates.",
        variant_name, n_eligible, n_total, (n_eligible / n_total * 100.0) if n_total else float("nan"),
        n_total - n_eligible,
    )
    return eligible_mask


def apply_postwindow_oversampling_filtered(train_data, eligible_mask: np.ndarray, variant_name: str):
    """
    Duplicates eligible high-error windows OVERSAMPLE_FACTOR - 1 additional
    times onto the full base tensors. Threshold quantile is computed only
    over eligible windows' target-magnitude distribution.
    """
    sequences = train_data["sequences"]
    statics = train_data["statics"]
    targets = train_data["targets"]
    meta = train_data["meta"]

    n_before = len(sequences)
    magnitudes = np.linalg.norm(targets, axis=1)

    eligible_magnitudes = magnitudes[eligible_mask]
    if len(eligible_magnitudes) == 0:
        raise RuntimeError(f"[{variant_name}] No eligible windows available to compute an oversampling threshold from.")

    threshold = float(np.quantile(eligible_magnitudes, OVERSAMPLE_HIGH_ERROR_QUANTILE))
    high_error_mask = eligible_mask & (magnitudes >= threshold)
    n_high_error = int(high_error_mask.sum())

    logger.info(
        "[%s] Post-window oversampling (flag-filtered candidates): quantile=%.4f over %d eligible "
        "window(s) -> threshold=%.6f km. %d / %d total windows (%.2f%%) identified as eligible high-error.",
        variant_name, OVERSAMPLE_HIGH_ERROR_QUANTILE, len(eligible_magnitudes), threshold,
        n_high_error, n_before, (n_high_error / n_before * 100.0) if n_before > 0 else float("nan"),
    )

    n_extra_copies = max(OVERSAMPLE_FACTOR - 1, 0)
    if n_high_error == 0 or n_extra_copies == 0:
        logger.info("[%s] No post-window oversampling applied. Training-sequence count remains %d.",
                     variant_name, n_before)
        return {
            "sequences": sequences, "statics": statics, "targets": targets,
            "meta": list(meta), "feature_cols": train_data["feature_cols"],
            "n_before": n_before, "n_after": n_before, "n_high_error": n_high_error, "threshold": threshold,
        }

    high_error_indices = np.nonzero(high_error_mask)[0]
    dup_indices = np.tile(high_error_indices, n_extra_copies)

    sequences_out = np.concatenate([sequences, sequences[dup_indices]], axis=0)
    statics_out = np.concatenate([statics, statics[dup_indices]], axis=0)
    targets_out = np.concatenate([targets, targets[dup_indices]], axis=0)
    meta_out = list(meta) + [meta[i] for i in dup_indices]

    n_after = len(sequences_out)
    logger.info(
        "[%s] Post-window oversampling applied: %d eligible high-error window(s) duplicated %d extra "
        "time(s) each (OVERSAMPLE_FACTOR=%d). Total training-sequence count: %d -> %d.",
        variant_name, n_high_error, n_extra_copies, OVERSAMPLE_FACTOR, n_before, n_after,
    )

    return {
        "sequences": sequences_out, "statics": statics_out, "targets": targets_out,
        "meta": meta_out, "feature_cols": train_data["feature_cols"],
        "n_before": n_before, "n_after": n_after, "n_high_error": n_high_error, "threshold": threshold,
    }


def _train_one_variant(oversampled_data, variant_name, n_epochs):
    """
    Same training-loop pattern as
    train_sequence_model_rolling_cv_final_dedup._train_one_variant_final_dedup:
    GRUResidualModel / SmoothL1Loss / Adam lr=1e-3 / batch size 64 /
    torch.manual_seed(config.RANDOM_SEED) / same device-selection logic,
    fixed epoch count, no validation split, no early stopping. Saves to
    new-output/test_output/{models,plots}/ only.
    """
    display_name = f"{variant_name}{NEW_MODEL_SUFFIX}"

    (train_seq_n, train_static_n), norm_stats = tsm._normalize(
        oversampled_data["sequences"], oversampled_data["statics"]
    )

    torch.manual_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_features = train_seq_n.shape[-1]
    model = tsm.GRUResidualModel(n_features=n_features, static_dim=train_static_n.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.SmoothL1Loss()

    train_dataset = tsm.SequenceDataset(train_seq_n, train_static_n, oversampled_data["targets"])
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

    train_losses = []
    for epoch in range(n_epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for seq, static, tgt in train_loader:
            seq, static, tgt = seq.to(device), static.to(device), tgt.to(device)
            optimizer.zero_grad()
            pred = model(seq, static)
            loss = loss_fn(pred, tgt)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(train_loss)
        logger.info("[%s] Epoch %d/%d: train_loss=%.4f", display_name, epoch + 1, n_epochs, train_loss)

    model_path = f"{TEST_MODELS_DIR}/sequence_model_{display_name}.pt"
    scaler_path = f"{TEST_MODELS_DIR}/sequence_scaler_{display_name}.joblib"
    plot_path = f"{TEST_PLOTS_DIR}/sequence_training_loss_{display_name}.png"

    torch.save({
        "state_dict": model.state_dict(),
        "n_features": n_features,
        "static_dim": train_static_n.shape[-1],
        "feature_cols": oversampled_data["feature_cols"],
        "window_size": config.SEQUENCE_WINDOW_SIZE,
        "target_columns": tsm.TARGET_COLUMNS,
        "variant_name": variant_name,
        "n_epochs_trained": n_epochs,
        "trained_via": "postwindow_oversample_filtered_test_all_variants",
        "oversample_high_error_quantile": OVERSAMPLE_HIGH_ERROR_QUANTILE,
        "oversample_factor": OVERSAMPLE_FACTOR,
        "oversample_candidate_flag_columns": FLAG_COLUMNS,
    }, model_path)
    joblib.dump(norm_stats, scaler_path)

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(train_losses, label="train_loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Smooth L1 loss")
        ax.set_title(
            f"Post-window oversample (flag-filtered) test: {variant_name} "
            f"(fixed {n_epochs} epochs, quantile={OVERSAMPLE_HIGH_ERROR_QUANTILE}, factor={OVERSAMPLE_FACTOR})"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
    except Exception as exc:
        logger.warning("[%s] Failed to plot training loss: %s", display_name, exc)

    return {
        "model_path": model_path,
        "scaler_path": scaler_path,
        "plot_path": plot_path,
        "final_train_loss": train_losses[-1] if train_losses else np.nan,
    }


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
    _ensure_test_output_dirs()

    if not TORCH_AVAILABLE:
        logger.error("PyTorch is not installed. Cannot run this experiment.")
        return

    fixed_epoch_counts = _load_fixed_epoch_counts()
    variant_names = [v for v in tsm.FEATURE_SET_VARIANTS if v in fixed_epoch_counts]
    skipped = [v for v in tsm.FEATURE_SET_VARIANTS if v not in fixed_epoch_counts]
    if skipped:
        logger.warning("Skipping variant(s) with no usable fixed_epoch_count: %s", skipped)
    logger.info("Variants to train (%d): %s", len(variant_names), variant_names)

    # --- Load + dedup-filter training data once (shared across variants) ----
    train_path = f"{config.OUTPUT_DIR}/augmented_train_dataset.csv"
    logger.info("Loading (READ-ONLY) training dataset from %s ...", train_path)
    raw_train_df = pd.read_csv(train_path)
    if raw_train_df.empty:
        raise RuntimeError(f"{train_path} is empty.")
    raw_train_df["current_epoch_utc"] = pd.to_datetime(raw_train_df["current_epoch_utc"], utc=True, format="ISO8601")
    logger.info(
        "Loaded FULL training set: %d rows (augmentation types present: %s).",
        len(raw_train_df),
        sorted(raw_train_df["augmentation_type"].unique()) if "augmentation_type" in raw_train_df.columns else "n/a",
    )
    dedup_train_df = build_deduplicated_train_df(raw_train_df)

    # --- Train one model per variant -----------------------------------------
    training_rows = []
    trained_models = {}  # variant_name -> {"model_path":..., "scaler_path":...}
    run_start = time.monotonic()

    for i, variant_name in enumerate(variant_names, start=1):
        feature_cols = tsm.FEATURE_SET_VARIANTS[variant_name]
        n_epochs = fixed_epoch_counts[variant_name]
        variant_start = time.monotonic()
        status, error_message = "success", ""
        n_before = n_after = n_high_error = np.nan
        final_train_loss = np.nan

        logger.info(
            "--- [%d/%d] Starting variant '%s' (%d feature columns, fixed %d epochs) ---",
            i, len(variant_names), variant_name, len(feature_cols), n_epochs,
        )
        try:
            base_train_data = tsm.build_sequences(dedup_train_df, config.SEQUENCE_WINDOW_SIZE, feature_cols=feature_cols)
            if base_train_data is None:
                raise RuntimeError("Could not build any training sequences.")

            eligible_mask = compute_window_eligibility(base_train_data["meta"], dedup_train_df, variant_name)
            oversampled_data = apply_postwindow_oversampling_filtered(base_train_data, eligible_mask, variant_name)
            n_before = oversampled_data["n_before"]
            n_after = oversampled_data["n_after"]
            n_high_error = oversampled_data["n_high_error"]

            result = _train_one_variant(oversampled_data, variant_name, n_epochs)
            final_train_loss = result["final_train_loss"]
            trained_models[variant_name] = {
                "model_path": result["model_path"],
                "scaler_path": result["scaler_path"],
            }
        except Exception as exc:
            status = "failed"
            error_message = str(exc)
            logger.error("--- [%d/%d] variant '%s' FAILED: %s ---", i, len(variant_names), variant_name, exc)
            logger.error("Full traceback:\n%s", traceback.format_exc())
        else:
            logger.info(
                "--- [%d/%d] variant '%s' complete in %.1fs: final_train_loss=%.4f ---",
                i, len(variant_names), variant_name, time.monotonic() - variant_start, final_train_loss,
            )

        training_rows.append({
            "model_variant": variant_name,
            "n_epochs_trained": n_epochs,
            "n_train_sequences_before_oversample": n_before,
            "n_train_sequences_after_oversample": n_after,
            "n_eligible_high_error_windows": n_high_error,
            "final_train_loss": final_train_loss,
            "status": status,
            "elapsed_seconds": round(time.monotonic() - variant_start, 2),
            "error_message": error_message,
        })

    training_summary_df = pd.DataFrame(training_rows)
    training_summary_path = f"{TEST_OUTPUT_DIR}/postwindow_oversample_filtered_all_variants_training_summary.csv"
    training_summary_df.to_csv(training_summary_path, index=False)
    logger.info(
        "Training phase complete in %.1fs total. %d/%d variants succeeded. Summary: %s",
        time.monotonic() - run_start, len(trained_models), len(variant_names), training_summary_path,
    )

    # --- Evaluate everything on the test set ---------------------------------
    test_path = f"{config.OUTPUT_DIR}/test_dataset.csv"
    logger.info("Loading test dataset from %s ...", test_path)
    test_df = pd.read_csv(test_path)
    if test_df.empty:
        logger.warning("Test dataset is empty. Nothing to evaluate.")
        pd.DataFrame().to_csv(f"{TEST_OUTPUT_DIR}/diagnostics_postwindow_oversample_filtered_all_variants_summary.csv", index=False)
        return
    test_df["current_epoch_utc"] = pd.to_datetime(test_df["current_epoch_utc"], utc=True, format="ISO8601")

    rows = []

    # Raw SGP4, once.
    try:
        raw_metrics = _metrics(test_df[RAW_COLUMN])
        rows.append({"model": "raw_sgp4", **raw_metrics, "pct_samples_improved": np.nan, "mean_pct_improvement": np.nan})
    except Exception:
        logger.exception("Failed computing metrics for raw_sgp4.")

    per_variant_maes = {}  # variant_name -> {"dedup": mae, "filtered": mae}

    for variant_name in variant_names:
        per_variant_maes[variant_name] = {"dedup": np.nan, "filtered": np.nan}

        # Existing dedup-only baseline, unchanged.
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

        # New flag-filtered post-window oversample model, if trained successfully.
        new_display = f"{variant_name}{NEW_MODEL_SUFFIX}"
        if variant_name in trained_models:
            try:
                paths = trained_models[variant_name]
                test_df, ok, new_error_col = add_checkpoint_predictions(
                    test_df, paths["model_path"], paths["scaler_path"], new_display
                )
                if ok:
                    new_metrics = _metrics(test_df[new_error_col].dropna())
                    pct_improved, mean_improve = _improvement_stats(test_df[RAW_COLUMN], test_df[new_error_col])
                    rows.append({
                        "model": new_display, **new_metrics,
                        "pct_samples_improved": pct_improved, "mean_pct_improvement": mean_improve,
                    })
                    per_variant_maes[variant_name]["filtered"] = new_metrics["mae_km"]
                else:
                    logger.warning("'%s' unavailable; its row will be absent from the summary.", new_display)
            except Exception:
                logger.exception("Failed evaluating '%s'.", new_display)
        else:
            logger.warning("'%s' was not trained successfully; its row will be absent from the summary.", new_display)

    summary_df = pd.DataFrame(rows)
    ordered_cols = [
        "model", "n_samples", "mae_km", "median_ae_km", "rmse_km", "p90_km", "p95_km", "max_km",
        "pct_samples_improved", "mean_pct_improvement",
    ]
    summary_df = summary_df[[c for c in ordered_cols if c in summary_df.columns]]

    summary_path = f"{TEST_OUTPUT_DIR}/diagnostics_postwindow_oversample_filtered_all_variants_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved comparison table to %s (%d rows).", summary_path, len(summary_df))

    # --- Console summary + per-variant verdict --------------------------------
    print("\n=== FLAG-FILTERED POST-WINDOW OVERSAMPLING -- ALL VARIANTS (test set) ===")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    else:
        print("No rows could be evaluated.")

    print("\n=== PER-VARIANT VERDICT ===")
    by_model = summary_df.set_index("model") if not summary_df.empty else pd.DataFrame()
    raw_mae = by_model.loc["raw_sgp4", "mae_km"] if "raw_sgp4" in by_model.index else np.nan
    print(f"{'variant':45s} {'dedup mae_km':>15s} {'filtered mae_km':>17s} {'helped?':>9s} {'beats raw?':>11s}")
    for variant_name in variant_names:
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

    logger.info("train_and_eval_postwindow_oversample_filtered_all_variants complete.")


if __name__ == "__main__":
    run()
