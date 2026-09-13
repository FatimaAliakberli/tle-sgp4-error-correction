"""
train_sequence_model_rolling_cv_final.py

PART D.1: Final retraining of every FEATURE_SET_VARIANTS entry (all 7,
including the 2 new cycle-phase variants from Part B) on the FULL
output/augmented_train_dataset.csv -- all pre-2024 rows, ALL augmentation
types (no augmentation_type filtering; same row-selection convention Part
C's fold training used in run_rolling_origin_cv.py) -- for a FIXED number
of epochs taken from Part C's rolling-origin cross-validation
(output/rolling_origin_cv_variant_summary.csv's fixed_epoch_count column).

NO early stopping and NO held-out validation split is used here: Part C's
cross-validation already did model selection (both the feature-set choice
AND the epoch count), so this step just fits each final model at its
pre-chosen epoch count on all available pre-2024 data.

Architecture / loss (SmoothL1Loss) / optimizer (Adam, lr=1e-3) / batch
size (64) / RANDOM_SEED are IDENTICAL to train_sequence_model.run() and
run_rolling_origin_cv.py. The ONLY things that differ per variant are the
feature-set and the fixed epoch count. GRUResidualModel / build_sequences
/ _normalize / SequenceDataset are reused unchanged from
train_sequence_model.py -- nothing in that file is modified or rewritten.

Outputs (per variant, saved to paths DISTINCT from the existing v1
ablation artifacts -- v1 model files under output/models/ are never
touched or overwritten):
    output/models/sequence_model_<variant_name>_rolling_cv_v2.pt
    output/models/sequence_scaler_<variant_name>_rolling_cv_v2.joblib
    output/plots/sequence_training_loss_<variant_name>_rolling_cv_v2.png

Also writes:
    output/rolling_cv_final_training_run_summary.csv
        One row per variant: status, elapsed seconds, epoch count trained,
        final train loss, number of training sequences, and (on failure)
        the exception message.
"""

import logging
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib

import config
import s06_train_sequence_model as tsm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train_sequence_model_rolling_cv_final_predup")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

TRAIN_BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# Suffix distinguishing every artifact this script writes from the
# existing v1 ablation model files -- per the constraint not to overwrite
# output/models/sequence_model_<variant>.pt / sequence_scaler_<variant>.joblib.
MODEL_SUFFIX = "_rolling_cv_v2"


def _now_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _fmt_elapsed(seconds):
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}m{secs:04.1f}s"


def _load_fixed_epoch_counts():
    path = f"{config.OUTPUT_DIR}/rolling_origin_cv_variant_summary.csv"
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"{path} is empty or missing. Run run_rolling_origin_cv.py (Part C) first.")

    counts = {}
    for _, row in df.iterrows():
        variant = row["model_variant"]
        fixed = row["fixed_epoch_count"]
        if pd.isna(fixed):
            logger.warning(
                "Variant '%s' has no fixed_epoch_count in %s (all its CV folds failed?). "
                "It will be SKIPPED in final retraining.", variant, path,
            )
            continue
        counts[variant] = int(fixed)
    return counts


def _train_one_variant_final(full_train_df, feature_cols, variant_name, n_epochs):
    train_data = tsm.build_sequences(full_train_df, config.SEQUENCE_WINDOW_SIZE, feature_cols=feature_cols)
    if train_data is None:
        raise RuntimeError("Could not build any training sequences from the full training set.")

    missing = [c for c in feature_cols if c not in train_data["feature_cols"]]
    if missing:
        logger.warning(
            "[%s] %d requested feature column(s) not found in training data and were dropped: %s",
            variant_name, len(missing), missing,
        )

    # No validation split for the final fit -- normalize using train stats only.
    (train_seq_n, train_static_n), norm_stats = tsm._normalize(train_data["sequences"], train_data["statics"])

    torch.manual_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_features = train_seq_n.shape[-1]
    model = tsm.GRUResidualModel(n_features=n_features, static_dim=train_static_n.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.SmoothL1Loss()

    train_dataset = tsm.SequenceDataset(train_seq_n, train_static_n, train_data["targets"])
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
        logger.info("[%s] Epoch %d/%d: train_loss=%.4f", variant_name, epoch + 1, n_epochs, train_loss)

    model_path = f"{config.MODELS_DIR}/sequence_model_{variant_name}{MODEL_SUFFIX}.pt"
    scaler_path = f"{config.MODELS_DIR}/sequence_scaler_{variant_name}{MODEL_SUFFIX}.joblib"
    plot_path = f"{config.PLOTS_DIR}/sequence_training_loss_{variant_name}{MODEL_SUFFIX}.png"

    torch.save({
        "state_dict": model.state_dict(),
        "n_features": n_features,
        "static_dim": train_static_n.shape[-1],
        "feature_cols": train_data["feature_cols"],
        "window_size": config.SEQUENCE_WINDOW_SIZE,
        "target_columns": tsm.TARGET_COLUMNS,
        "variant_name": variant_name,
        "n_epochs_trained": n_epochs,
        "trained_via": "rolling_cv_v2_final",
    }, model_path)
    joblib.dump(norm_stats, scaler_path)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(train_losses, label="train_loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Smooth L1 loss")
        ax.set_title(f"Final retraining loss: {variant_name} (fixed {n_epochs} epochs, rolling-CV v2)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
    except Exception as exc:
        logger.warning("[%s] Failed to plot training loss: %s", variant_name, exc)

    return {
        "n_epochs_trained": n_epochs,
        "final_train_loss": train_losses[-1] if train_losses else np.nan,
        "n_train_sequences": len(train_data["sequences"]),
        "model_path": model_path,
        "scaler_path": scaler_path,
    }


def run():
    config.ensure_dirs()

    if not TORCH_AVAILABLE:
        logger.warning("PyTorch is not installed. Skipping final rolling-CV retraining entirely.")
        pd.DataFrame().to_csv(f"{config.OUTPUT_DIR}/rolling_cv_final_training_run_summary.csv", index=False)
        return

    fixed_epoch_counts = _load_fixed_epoch_counts()
    variant_names = [v for v in tsm.FEATURE_SET_VARIANTS if v in fixed_epoch_counts]
    skipped = [v for v in tsm.FEATURE_SET_VARIANTS if v not in fixed_epoch_counts]
    if skipped:
        logger.warning("Skipping variant(s) with no usable fixed_epoch_count: %s", skipped)

    logger.info("=== train_sequence_model_rolling_cv_final starting at %s ===", _now_str())
    logger.info("Variants to retrain (%d): %s", len(variant_names), variant_names)
    logger.info("Fixed epoch counts: %s", {v: fixed_epoch_counts[v] for v in variant_names})

    train_df = pd.read_csv(f"{config.OUTPUT_DIR}/augmented_train_dataset.csv")
    if train_df.empty:
        raise RuntimeError("output/augmented_train_dataset.csv is empty.")
    train_df["current_epoch_utc"] = pd.to_datetime(train_df["current_epoch_utc"], utc=True, format="ISO8601")
    logger.info(
        "Loaded FULL training set: %d rows (augmentation types present: %s).",
        len(train_df),
        sorted(train_df["augmentation_type"].unique()) if "augmentation_type" in train_df.columns else "n/a",
    )

    results = []
    run_start = time.monotonic()

    for i, variant_name in enumerate(variant_names, start=1):
        feature_cols = tsm.FEATURE_SET_VARIANTS[variant_name]
        n_epochs = fixed_epoch_counts[variant_name]

        logger.info(
            "--- [%d/%d] Starting FINAL retraining for variant '%s' at %s (%d feature columns, fixed %d epochs) ---",
            i, len(variant_names), variant_name, _now_str(), len(feature_cols), n_epochs,
        )
        variant_start = time.monotonic()
        status, error_message = "success", ""
        result = {"n_epochs_trained": n_epochs, "final_train_loss": np.nan, "n_train_sequences": np.nan}

        try:
            result = _train_one_variant_final(train_df, feature_cols, variant_name, n_epochs)
        except Exception as exc:
            status = "failed"
            error_message = str(exc)
            logger.error(
                "--- [%d/%d] variant '%s' FAILED after %s: %s ---",
                i, len(variant_names), variant_name, _fmt_elapsed(time.monotonic() - variant_start), exc,
            )
            logger.error("Full traceback:\n%s", traceback.format_exc())
        else:
            logger.info(
                "--- [%d/%d] variant '%s' complete in %s: final_train_loss=%.4f ---",
                i, len(variant_names), variant_name,
                _fmt_elapsed(time.monotonic() - variant_start), result["final_train_loss"],
            )

        total_elapsed = time.monotonic() - run_start
        logger.info("--- [%d/%d] Running total elapsed: %s ---", i, len(variant_names), _fmt_elapsed(total_elapsed))

        results.append({
            "model_variant": variant_name,
            "n_epochs_trained": result["n_epochs_trained"],
            "final_train_loss": result["final_train_loss"],
            "n_train_sequences": result.get("n_train_sequences", np.nan),
            "status": status,
            "elapsed_seconds": round(time.monotonic() - variant_start, 2),
            "error_message": error_message,
        })

    summary_df = pd.DataFrame(results)
    summary_path = f"{config.OUTPUT_DIR}/rolling_cv_final_training_run_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    n_success = int((summary_df["status"] == "success").sum())
    n_failed = int((summary_df["status"] == "failed").sum())
    logger.info(
        "=== train_sequence_model_rolling_cv_final complete. %d/%d succeeded, %d failed. Summary: %s ===",
        n_success, len(variant_names), n_failed, summary_path,
    )
    if n_failed > 0:
        failed_names = summary_df.loc[summary_df["status"] == "failed", "model_variant"].tolist()
        logger.warning("Failed variants (see traceback above and %s for details): %s", summary_path, failed_names)

    print("\n=== FINAL ROLLING-CV RETRAINING SUMMARY ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    run()
