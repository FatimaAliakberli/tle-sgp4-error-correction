"""
train_and_eval_postwindow_oversample_test.py

STANDALONE EXPERIMENT SCRIPT -- does not modify any existing file, does not
modify or overwrite new-output/augmented_train_dataset.csv, and does not
touch anything under new-output/models or new-output/plots. All outputs
from this script go under new-output/test_output/ only.

BACKGROUND
----------
train_sequence_model_rolling_cv_final_dedup.py already fixed the
duplicate-current_epoch_utc contamination in
new-output/augmented_train_dataset.csv by filtering training rows to
augmentation_type == "original" before calling build_sequences(). That
throws away the row-level "oversampled_high_error" augmentation entirely
(since that augmentation lives at the row level and shares timestamps
with its source row).

This script tests whether the REGULARIZATION BENEFIT of that old
row-level oversampling can be recovered safely by oversampling AFTER
build_sequences() has already formed (sequence, static, target) window
tuples. Because those tuples no longer carry a current_epoch_utc column,
duplicating them cannot reintroduce duplicate-timestamp groups by
construction -- the sliding-window sort/ordering step has already
happened before any duplication occurs.

PIPELINE
--------
1. Load new-output/augmented_train_dataset.csv READ-ONLY, filter in-memory
   to augmentation_type == "original" only (matches
   train_sequence_model_rolling_cv_final_dedup.py's build_deduplicated_train_df
   pattern exactly).
2. build_sequences() on the filtered data using
   FEATURE_SET_VARIANTS["original"] -- no augmentation yet, matching the
   already-validated dedup approach.
3. POST-WINDOW OVERSAMPLING (the new part): compute each window's target
   magnitude as the L2 norm across target columns, identify windows at or
   above the OVERSAMPLE_HIGH_ERROR_QUANTILE quantile of that magnitude,
   and duplicate those (sequence, static, target, meta) tuples
   OVERSAMPLE_FACTOR - 1 additional times, concatenated onto the base
   tensors.
4. _normalize() on the full (base + oversampled) tensor set (train-stats
   only, no validation split, matching the dedup pattern).
5. Train ONE model (GRUResidualModel / SmoothL1Loss / Adam lr=1e-3 /
   batch size 64 / torch.manual_seed(config.RANDOM_SEED)) for the SAME
   fixed epoch count used for the existing "original_rolling_cv_v2_dedup"
   model, read from new-output/rolling_origin_cv_variant_summary.csv.
   No early stopping, no validation split.
6. Save artifacts under new-output/test_output/{models,plots}/ only.
7. Evaluate the new model against new-output/test_dataset.csv.
8. Also load and score the existing "original_rolling_cv_v2_dedup" model
   on the same test set (unchanged, not retrained) via
   add_sequence_variant_predictions_v2_dedup.
9. Compute _metrics() for raw SGP4, the existing dedup model, and this new
   post-window-oversample model, plus pct_samples_improved /
   mean_pct_improvement for the two corrected-error columns.
10. Save a 3-row comparison table to
    new-output/test_output/diagnostics_postwindow_oversample_summary.csv.
11. Print a final console verdict.

REUSED, UNCHANGED (per project convention -- nothing in these files is
modified or reimplemented):
    - train_sequence_model.py: FEATURE_SET_VARIANTS, GRUResidualModel,
      build_sequences, _normalize, SequenceDataset, TARGET_COLUMNS
    - evaluate_sequence_ablation_v2_dedup.py:
      add_sequence_variant_predictions_v2_dedup
    - evaluate_model.py: _metrics
"""

import logging

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
logger = logging.getLogger("s21_train_eval_postwindow_oversample_naive")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# --------------------------------------------------------------------------
# Module-level configuration for the post-window oversampling step. Mirrors
# the spirit of config.AUGMENTATION_HIGH_ERROR_QUANTILE /
# config.AUGMENTATION_OVERSAMPLE_FACTOR, reusing those exact values if
# present in config.py, otherwise falling back to the stated defaults.
# --------------------------------------------------------------------------
OVERSAMPLE_HIGH_ERROR_QUANTILE = getattr(config, "AUGMENTATION_HIGH_ERROR_QUANTILE", 0.95)
OVERSAMPLE_FACTOR = getattr(config, "AUGMENTATION_OVERSAMPLE_FACTOR", 3)

VARIANT_NAME = "original"

# All outputs from this experiment live under this subdirectory -- nothing
# mixes with existing new-output/models or new-output/plots artifacts.
TEST_OUTPUT_DIR = f"{config.OUTPUT_DIR}/test_output"
TEST_MODELS_DIR = f"{TEST_OUTPUT_DIR}/models"
TEST_PLOTS_DIR = f"{TEST_OUTPUT_DIR}/plots"

NEW_MODEL_SUFFIX = "_postwindow_oversample_test"

TRAIN_BATCH_SIZE = 64
LEARNING_RATE = 1e-3

RAW_COLUMN = "position_error_km"
DEDUP_MODEL_DISPLAY_NAME = f"{VARIANT_NAME}_rolling_cv_v2_dedup"
NEW_MODEL_DISPLAY_NAME = f"{VARIANT_NAME}{NEW_MODEL_SUFFIX}"


def _ensure_test_output_dirs():
    import os
    os.makedirs(TEST_MODELS_DIR, exist_ok=True)
    os.makedirs(TEST_PLOTS_DIR, exist_ok=True)


def _load_fixed_epoch_count(variant_name=VARIANT_NAME):
    """
    Reads the fixed epoch count for `variant_name` from
    new-output/rolling_origin_cv_variant_summary.csv (column
    fixed_epoch_count), so this new model trains for exactly the same
    number of epochs as the existing "original_rolling_cv_v2_dedup" model
    -- a clean apples-to-apples comparison.
    """
    summary_path = f"{config.OUTPUT_DIR}/rolling_origin_cv_variant_summary.csv"
    summary_df = pd.read_csv(summary_path)
    row = summary_df[summary_df["model_variant"] == variant_name]
    if row.empty:
        raise RuntimeError(
            f"No row with model_variant == '{variant_name}' found in {summary_path}."
        )
    n_epochs = int(row.iloc[0]["fixed_epoch_count"])
    logger.info("Loaded fixed_epoch_count=%d for variant '%s' from %s.", n_epochs, variant_name, summary_path)
    return n_epochs


def build_deduplicated_train_df(raw_train_df: pd.DataFrame) -> pd.DataFrame:
    """
    In-memory-only filter: keep augmentation_type == "original" rows.
    Never writes back to disk. Matches
    train_sequence_model_rolling_cv_final_dedup.build_deduplicated_train_df's
    exact approach.
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


def apply_postwindow_oversampling(train_data):
    """
    Given the dict returned by build_sequences() (keys: sequences, statics,
    targets, meta, feature_cols), identifies windows whose target L2-norm
    magnitude is at or above OVERSAMPLE_HIGH_ERROR_QUANTILE and duplicates
    those (sequence, static, target, meta) tuples OVERSAMPLE_FACTOR - 1
    additional times, concatenated onto the base tensors along axis 0.

    Because this operates entirely on already-built windows (which carry
    no current_epoch_utc column), it cannot create duplicate-timestamp
    training rows by construction.

    Returns a new dict with the same keys, oversampled.
    """
    sequences = train_data["sequences"]
    statics = train_data["statics"]
    targets = train_data["targets"]
    meta = train_data["meta"]

    n_before = len(sequences)
    magnitudes = np.linalg.norm(targets, axis=1)
    threshold = float(np.quantile(magnitudes, OVERSAMPLE_HIGH_ERROR_QUANTILE))
    high_error_mask = magnitudes >= threshold
    n_high_error = int(high_error_mask.sum())

    logger.info(
        "Post-window oversampling: quantile=%.4f -> magnitude threshold=%.6f km. "
        "%d / %d windows (%.2f%%) identified as high-error.",
        OVERSAMPLE_HIGH_ERROR_QUANTILE, threshold, n_high_error, n_before,
        (n_high_error / n_before * 100.0) if n_before > 0 else float("nan"),
    )

    n_extra_copies = max(OVERSAMPLE_FACTOR - 1, 0)
    if n_high_error == 0 or n_extra_copies == 0:
        logger.info(
            "No post-window oversampling applied (n_high_error=%d, OVERSAMPLE_FACTOR=%d). "
            "Training-sequence count remains %d.",
            n_high_error, OVERSAMPLE_FACTOR, n_before,
        )
        return {
            "sequences": sequences,
            "statics": statics,
            "targets": targets,
            "meta": list(meta),
            "feature_cols": train_data["feature_cols"],
        }

    high_error_indices = np.nonzero(high_error_mask)[0]
    dup_indices = np.tile(high_error_indices, n_extra_copies)

    seq_dup = sequences[dup_indices]
    static_dup = statics[dup_indices]
    target_dup = targets[dup_indices]
    meta_dup = [meta[i] for i in dup_indices]

    sequences_out = np.concatenate([sequences, seq_dup], axis=0)
    statics_out = np.concatenate([statics, static_dup], axis=0)
    targets_out = np.concatenate([targets, target_dup], axis=0)
    meta_out = list(meta) + meta_dup

    n_after = len(sequences_out)
    logger.info(
        "Post-window oversampling applied: %d high-error window(s) duplicated %d extra time(s) each "
        "(OVERSAMPLE_FACTOR=%d). Total training-sequence count: %d -> %d.",
        n_high_error, n_extra_copies, OVERSAMPLE_FACTOR, n_before, n_after,
    )

    return {
        "sequences": sequences_out,
        "statics": statics_out,
        "targets": targets_out,
        "meta": meta_out,
        "feature_cols": train_data["feature_cols"],
    }


def _train_postwindow_oversample_model(oversampled_data, n_epochs):
    """
    Same training-loop pattern as
    train_sequence_model_rolling_cv_final_dedup._train_one_variant_final_dedup:
    GRUResidualModel / SmoothL1Loss / Adam lr=1e-3 / batch size 64 /
    torch.manual_seed(config.RANDOM_SEED) / same device-selection logic,
    fixed epoch count, no validation split, no early stopping. Saves to
    new-output/test_output/{models,plots}/ only.
    """
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
        logger.info("[%s] Epoch %d/%d: train_loss=%.4f", NEW_MODEL_DISPLAY_NAME, epoch + 1, n_epochs, train_loss)

    model_path = f"{TEST_MODELS_DIR}/sequence_model_{NEW_MODEL_DISPLAY_NAME}.pt"
    scaler_path = f"{TEST_MODELS_DIR}/sequence_scaler_{NEW_MODEL_DISPLAY_NAME}.joblib"
    plot_path = f"{TEST_PLOTS_DIR}/sequence_training_loss_{NEW_MODEL_DISPLAY_NAME}.png"

    torch.save({
        "state_dict": model.state_dict(),
        "n_features": n_features,
        "static_dim": train_static_n.shape[-1],
        "feature_cols": oversampled_data["feature_cols"],
        "window_size": config.SEQUENCE_WINDOW_SIZE,
        "target_columns": tsm.TARGET_COLUMNS,
        "variant_name": VARIANT_NAME,
        "n_epochs_trained": n_epochs,
        "trained_via": "postwindow_oversample_test",
        "oversample_high_error_quantile": OVERSAMPLE_HIGH_ERROR_QUANTILE,
        "oversample_factor": OVERSAMPLE_FACTOR,
    }, model_path)
    joblib.dump(norm_stats, scaler_path)

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(train_losses, label="train_loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Smooth L1 loss")
        ax.set_title(
            f"Post-window oversample test: {VARIANT_NAME} "
            f"(fixed {n_epochs} epochs, quantile={OVERSAMPLE_HIGH_ERROR_QUANTILE}, factor={OVERSAMPLE_FACTOR})"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
    except Exception as exc:
        logger.warning("Failed to plot training loss: %s", exc)

    return {
        "model_path": model_path,
        "scaler_path": scaler_path,
        "plot_path": plot_path,
        "norm_stats": norm_stats,
        "n_features": n_features,
        "static_dim": train_static_n.shape[-1],
        "final_train_loss": train_losses[-1] if train_losses else np.nan,
    }


def add_postwindow_oversample_predictions(df, model_path, scaler_path):
    """
    Same load-checkpoint / build-sequences / normalize / forward-pass
    pattern used elsewhere in this project (e.g.
    add_sequence_variant_predictions_v2_dedup). Eval-mode, no_grad,
    forward pass only -- no training happens here.
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
        logger.warning("Could not build evaluation sequences for '%s'. Skipping.", NEW_MODEL_DISPLAY_NAME)
        return df, False, None

    seq_n = (data["sequences"] - norm_stats["seq_mean"]) / norm_stats["seq_std"]
    static_n = (data["statics"] - norm_stats["static_mean"]) / norm_stats["static_std"]

    with torch.no_grad():
        preds = model(torch.from_numpy(seq_n.astype(np.float32)), torch.from_numpy(static_n.astype(np.float32)))
    preds = preds.numpy()

    diff = data["targets"] - preds
    seq_errors = np.linalg.norm(diff, axis=1)

    error_col = f"{NEW_MODEL_DISPLAY_NAME}_corrected_error_km"
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
        logger.error("PyTorch is not installed. Cannot run the post-window oversampling test.")
        return

    n_epochs = _load_fixed_epoch_count(VARIANT_NAME)

    # --- 1. Load training data READ-ONLY, filter to "original" only -----
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

    # --- 2. build_sequences() -- no augmentation yet ---------------------
    feature_cols = tsm.FEATURE_SET_VARIANTS[VARIANT_NAME]
    logger.info(
        "Building base training sequences for variant '%s' (window_size=%d, %d feature columns)...",
        VARIANT_NAME, config.SEQUENCE_WINDOW_SIZE, len(feature_cols),
    )
    base_train_data = tsm.build_sequences(dedup_train_df, config.SEQUENCE_WINDOW_SIZE, feature_cols=feature_cols)
    if base_train_data is None:
        raise RuntimeError("Could not build any training sequences from the deduplicated training set.")
    logger.info("Base training sequences (no augmentation): %d", len(base_train_data["sequences"]))

    # --- 3. Post-window oversampling (the new part) ----------------------
    oversampled_data = apply_postwindow_oversampling(base_train_data)

    # --- 4 & 5. Normalize + train -----------------------------------------
    training_result = None
    try:
        logger.info(
            "Training '%s' for %d fixed epoch(s) (matching '%s')...",
            NEW_MODEL_DISPLAY_NAME, n_epochs, DEDUP_MODEL_DISPLAY_NAME,
        )
        training_result = _train_postwindow_oversample_model(oversampled_data, n_epochs)
        logger.info(
            "Training complete. final_train_loss=%.4f. Saved: %s, %s, %s",
            training_result["final_train_loss"], training_result["model_path"],
            training_result["scaler_path"], training_result["plot_path"],
        )
    except Exception:
        logger.exception("Training the post-window oversample model failed.")

    # --- 6/7. Evaluate on the test set -----------------------------------
    test_path = f"{config.OUTPUT_DIR}/test_dataset.csv"
    logger.info("Loading test dataset from %s ...", test_path)
    test_df = pd.read_csv(test_path)
    if test_df.empty:
        logger.warning("Test dataset is empty. Nothing to evaluate.")
        pd.DataFrame().to_csv(f"{TEST_OUTPUT_DIR}/diagnostics_postwindow_oversample_summary.csv", index=False)
        return
    test_df["current_epoch_utc"] = pd.to_datetime(test_df["current_epoch_utc"], utc=True, format="ISO8601")

    new_model_error_col = None
    if training_result is not None:
        try:
            test_df, ok, new_model_error_col = add_postwindow_oversample_predictions(
                test_df, training_result["model_path"], training_result["scaler_path"]
            )
            if not ok:
                new_model_error_col = None
        except Exception:
            logger.exception("Evaluating the new post-window oversample model failed.")
            new_model_error_col = None
    else:
        logger.warning("Skipping evaluation of the new model since training did not complete.")

    # --- 8. Also score the existing dedup model, unchanged ---------------
    dedup_error_col = None
    try:
        test_df, ok, dedup_error_col = add_sequence_variant_predictions_v2_dedup(test_df, VARIANT_NAME)
        if not ok:
            dedup_error_col = None
    except Exception:
        logger.exception("Evaluating the existing '%s' model failed.", DEDUP_MODEL_DISPLAY_NAME)
        dedup_error_col = None

    # --- 9. Metrics for all three rows ------------------------------------
    rows = []

    try:
        raw_metrics = _metrics(test_df[RAW_COLUMN])
        rows.append({
            "model": "raw_sgp4",
            **raw_metrics,
            "pct_samples_improved": np.nan,
            "mean_pct_improvement": np.nan,
        })
    except Exception:
        logger.exception("Failed computing metrics for raw_sgp4.")

    if dedup_error_col is not None:
        try:
            dedup_metrics = _metrics(test_df[dedup_error_col].dropna())
            pct_improved, mean_improve = _improvement_stats(test_df[RAW_COLUMN], test_df[dedup_error_col])
            rows.append({
                "model": DEDUP_MODEL_DISPLAY_NAME,
                **dedup_metrics,
                "pct_samples_improved": pct_improved,
                "mean_pct_improvement": mean_improve,
            })
        except Exception:
            logger.exception("Failed computing metrics for '%s'.", DEDUP_MODEL_DISPLAY_NAME)
    else:
        logger.warning("'%s' unavailable; its row will be absent from the summary.", DEDUP_MODEL_DISPLAY_NAME)

    if new_model_error_col is not None:
        try:
            new_metrics = _metrics(test_df[new_model_error_col].dropna())
            pct_improved, mean_improve = _improvement_stats(test_df[RAW_COLUMN], test_df[new_model_error_col])
            rows.append({
                "model": NEW_MODEL_DISPLAY_NAME,
                **new_metrics,
                "pct_samples_improved": pct_improved,
                "mean_pct_improvement": mean_improve,
            })
        except Exception:
            logger.exception("Failed computing metrics for '%s'.", NEW_MODEL_DISPLAY_NAME)
    else:
        logger.warning("'%s' unavailable; its row will be absent from the summary.", NEW_MODEL_DISPLAY_NAME)

    summary_df = pd.DataFrame(rows)
    ordered_cols = [
        "model", "n_samples", "mae_km", "median_ae_km", "rmse_km", "p90_km", "p95_km", "max_km",
        "pct_samples_improved", "mean_pct_improvement",
    ]
    summary_df = summary_df[[c for c in ordered_cols if c in summary_df.columns]]

    summary_path = f"{TEST_OUTPUT_DIR}/diagnostics_postwindow_oversample_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved comparison table to %s (%d rows).", summary_path, len(summary_df))

    # --- 10/11. Console verdict --------------------------------------------
    print("\n=== POST-WINDOW OVERSAMPLING TEST -- SUMMARY (test set) ===")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    else:
        print("No rows could be evaluated.")

    print("\n=== VERDICT ===")
    by_model = summary_df.set_index("model") if not summary_df.empty else pd.DataFrame()

    raw_mae = by_model.loc["raw_sgp4", "mae_km"] if "raw_sgp4" in by_model.index else np.nan
    dedup_mae = by_model.loc[DEDUP_MODEL_DISPLAY_NAME, "mae_km"] if DEDUP_MODEL_DISPLAY_NAME in by_model.index else np.nan
    new_mae = by_model.loc[NEW_MODEL_DISPLAY_NAME, "mae_km"] if NEW_MODEL_DISPLAY_NAME in by_model.index else np.nan

    if np.isnan(new_mae) or np.isnan(dedup_mae):
        print(
            "Could not compute a verdict: one or both of the dedup-only model and the new "
            "post-window-oversample model are missing from the summary (see warnings above)."
        )
    else:
        diff = dedup_mae - new_mae
        rel_diff_pct = (diff / dedup_mae * 100.0) if dedup_mae != 0 else np.nan
        if abs(rel_diff_pct) < 1.0:
            direction = "makes no meaningful difference relative to"
        elif new_mae < dedup_mae:
            direction = "HELPS relative to"
        else:
            direction = "HURTS relative to"

        print(
            f"Post-window oversampling {direction} the existing dedup-only model: "
            f"MAE {dedup_mae:.4f} km ({DEDUP_MODEL_DISPLAY_NAME}) vs. "
            f"{new_mae:.4f} km ({NEW_MODEL_DISPLAY_NAME}) "
            f"[{rel_diff_pct:+.2f}% relative change]."
        )

        if not np.isnan(raw_mae):
            beats_raw = "beats" if new_mae <= raw_mae else "does NOT beat"
            print(
                f"The new post-window-oversample model {beats_raw} raw SGP4: "
                f"{new_mae:.4f} km vs. {raw_mae:.4f} km (raw_sgp4)."
            )
        else:
            print("Could not compare against raw SGP4 (raw_sgp4 row missing).")

    logger.info("train_and_eval_postwindow_oversample_test complete.")


if __name__ == "__main__":
    run()
