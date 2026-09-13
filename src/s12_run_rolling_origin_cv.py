"""
run_rolling_origin_cv.py

PART C: Rolling-origin (walk-forward) validation for sequence-model
selection, isolated from Part D's final retraining/evaluation.

WHY: a single fixed 2024 validation year (the current val_dataset.csv) can
land on an unrepresentative solar-cycle phase purely by chance. This
script instead evaluates every feature-set variant across 4 different
train/val splits ("folds"), each ending at a different point in the solar
cycle, and uses the AVERAGE behavior across folds for model selection
(Part D) rather than a single point estimate.

DATA SOURCE (per spec): this does NOT build a new dataset. It date-filters
the EXISTING output/augmented_train_dataset.csv, which already contains
every row with current_epoch_utc <= 2023-12-31 (all augmentation types).
    - fold training rows: current_epoch_utc <= fold's train cutoff
      (ANY augmentation_type -- same as normal training)
    - fold validation rows: current_epoch_utc inside the fold's val year
      AND augmentation_type == "original" (never validate against
      synthetically augmented rows)

FOLDS (exactly 4, per spec):
    fold 1: train <= 2018-12-31, val = 2019
    fold 2: train <= 2020-12-31, val = 2021
    fold 3: train <= 2021-12-31, val = 2022
    fold 4: train <= 2022-12-31, val = 2023

For each of the 7 FEATURE_SET_VARIANTS x 4 folds (28 runs total), this
trains a GRU with the EXACT SAME architecture / loss (SmoothL1Loss) /
optimizer (Adam, lr=1e-3) / batch size (64 train / 128 val) /
early-stopping patience (7) / max_epochs (50) / RANDOM_SEED as
train_sequence_model.run() -- copied inline rather than refactored out of
train_sequence_model.py, per the constraint not to modify that file's
existing single-split training path. The ONLY things that vary across
runs are (a) the feature-set (variant's column list) and (b) which rows
are selected for train/val (the fold's cutoffs).

The TEST SET (output/test_dataset.csv) IS NEVER TOUCHED by this script.

Outputs:
    output/rolling_origin_cv_summary.csv
        One row per (variant, fold): model_variant, fold_label, best_epoch,
        best_val_loss, val_n_samples (plus status/error_message for
        transparency on failures). best_epoch is a 1-indexed COUNT of
        epochs trained to reach the best validation loss (i.e. "train for
        this many epochs" -- directly usable as an epoch count, not a
        0-indexed epoch label).

    output/rolling_origin_cv_variant_summary.csv
        One row per variant (7 rows): the Part C.5 aggregation used
        directly by Part D --
            model_variant, fixed_epoch_count (mean best_epoch across the
            variant's successful folds, rounded to nearest int -- this is
            what Part D trains the final model for), mean_best_epoch,
            std_best_val_loss (across folds -- a variant with high std is
            itself a finding: it means that variant's benefit is unstable
            across different solar-cycle phases), mean_best_val_loss,
            n_folds_succeeded.
"""

import logging
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config
import s06_train_sequence_model as tsm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("s12_run_rolling_origin_cv")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# --------------------------------------------------------------------------
# Fold definitions -- EXACTLY these 4, per spec. Do not add/remove/reorder.
# --------------------------------------------------------------------------
FOLDS = [
    {
        "fold_label": "train<=2018_val2019",
        "train_cutoff": "2018-12-31 23:59:59",
        "val_start": "2019-01-01 00:00:00",
        "val_end": "2019-12-31 23:59:59",
    },
    {
        "fold_label": "train<=2020_val2021",
        "train_cutoff": "2020-12-31 23:59:59",
        "val_start": "2021-01-01 00:00:00",
        "val_end": "2021-12-31 23:59:59",
    },
    {
        "fold_label": "train<=2021_val2022",
        "train_cutoff": "2021-12-31 23:59:59",
        "val_start": "2022-01-01 00:00:00",
        "val_end": "2022-12-31 23:59:59",
    },
    {
        "fold_label": "train<=2022_val2023",
        "train_cutoff": "2022-12-31 23:59:59",
        "val_start": "2023-01-01 00:00:00",
        "val_end": "2023-12-31 23:59:59",
    },
]

# Training hyperparameters -- IDENTICAL to train_sequence_model.run().
# Do not change any of these; only feature-set and fold row-selection vary.
MAX_EPOCHS = 50
PATIENCE = 7
TRAIN_BATCH_SIZE = 64
VAL_BATCH_SIZE = 128
LEARNING_RATE = 1e-3


def _now_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _fmt_elapsed(seconds):
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}m{secs:04.1f}s"


def _load_augmented_train_dataset():
    path = f"{config.OUTPUT_DIR}/augmented_train_dataset.csv"
    logger.info("Loading %s ...", path)
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"{path} is empty. Run build_error_attribution_dataset.py -> augment_dataset.py first.")
    df["current_epoch_utc"] = pd.to_datetime(df["current_epoch_utc"], utc=True, format="ISO8601")
    logger.info("Loaded %d rows spanning %s to %s.", len(df), df["current_epoch_utc"].min(), df["current_epoch_utc"].max())
    return df


def _select_fold_rows(full_df, fold):
    """
    Date-filter the already-loaded augmented_train_dataset.csv for one
    fold. Per spec:
        - train: current_epoch_utc <= train_cutoff, ANY augmentation_type
        - val:   current_epoch_utc in [val_start, val_end], AND
                 augmentation_type == "original" only
    """
    train_cutoff = pd.Timestamp(fold["train_cutoff"], tz="UTC")
    val_start = pd.Timestamp(fold["val_start"], tz="UTC")
    val_end = pd.Timestamp(fold["val_end"], tz="UTC")

    fold_train_df = full_df[full_df["current_epoch_utc"] <= train_cutoff].copy()

    val_mask = (full_df["current_epoch_utc"] >= val_start) & (full_df["current_epoch_utc"] <= val_end)
    if "augmentation_type" in full_df.columns:
        val_mask = val_mask & (full_df["augmentation_type"] == "original")
    fold_val_df = full_df[val_mask].copy()

    return fold_train_df, fold_val_df


def _train_one_fold_variant(fold_train_df, fold_val_df, feature_cols, variant_name, fold_label):
    """
    Train one GRU model for one (variant, fold) pair, mirroring
    train_sequence_model.run()'s training loop exactly (architecture,
    loss, optimizer, LR, batch size, early-stopping patience, max_epochs,
    RANDOM_SEED). Returns a dict of result fields; raises on failure so
    the caller's try/except can log + record it.
    """
    train_data = tsm.build_sequences(fold_train_df, config.SEQUENCE_WINDOW_SIZE, feature_cols=feature_cols)
    if train_data is None:
        raise RuntimeError("Could not build any training sequences for this fold (no usable rows).")

    val_data = tsm.build_sequences(fold_val_df, config.SEQUENCE_WINDOW_SIZE, feature_cols=feature_cols)
    if val_data is None:
        raise RuntimeError("Could not build any validation sequences for this fold (no usable rows).")

    missing = [c for c in feature_cols if c not in train_data["feature_cols"]]
    if missing:
        logger.warning(
            "[%s / %s] %d requested feature column(s) not found in fold training data and were dropped: %s",
            variant_name, fold_label, len(missing), missing,
        )

    (train_seq_n, train_static_n, val_seq_n, val_static_n), norm_stats = tsm._normalize(
        train_data["sequences"], train_data["statics"], val_data["sequences"], val_data["statics"]
    )

    torch.manual_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_features = train_seq_n.shape[-1]
    model = tsm.GRUResidualModel(n_features=n_features, static_dim=train_static_n.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.SmoothL1Loss()

    train_dataset = tsm.SequenceDataset(train_seq_n, train_static_n, train_data["targets"])
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

    val_dataset = tsm.SequenceDataset(val_seq_n, val_static_n, val_data["targets"])
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False)

    best_val_loss = float("inf")
    best_epoch_count = None  # 1-indexed count of epochs trained to reach the best val loss
    epochs_without_improvement = 0

    for epoch in range(MAX_EPOCHS):
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

        model.eval()
        val_loss_total, n_val_batches = 0.0, 0
        with torch.no_grad():
            for seq, static, tgt in val_loader:
                seq, static, tgt = seq.to(device), static.to(device), tgt.to(device)
                pred = model(seq, static)
                val_loss_total += loss_fn(pred, tgt).item()
                n_val_batches += 1
        val_loss = val_loss_total / max(n_val_batches, 1)

        logger.info(
            "[%s / %s] Epoch %d: train_loss=%.4f val_loss=%.4f",
            variant_name, fold_label, epoch, train_loss, val_loss,
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch_count = epoch + 1  # 1-indexed: "train for this many epochs"
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            logger.info("[%s / %s] Early stopping at epoch %d.", variant_name, fold_label, epoch)
            break

    if best_epoch_count is None:
        # Val loss never improved even once (shouldn't normally happen since
        # epoch 0 always sets it via best_val_loss starting at inf, but
        # guard anyway).
        best_epoch_count = 1

    return {
        "best_epoch": best_epoch_count,
        "best_val_loss": best_val_loss,
        "val_n_samples": len(val_data["sequences"]),
    }


def run():
    config.ensure_dirs()

    if not TORCH_AVAILABLE:
        logger.warning(
            "PyTorch is not installed. Skipping rolling-origin CV entirely. "
            "Install with `pip install torch` to enable this step."
        )
        pd.DataFrame().to_csv(f"{config.OUTPUT_DIR}/rolling_origin_cv_summary.csv", index=False)
        pd.DataFrame().to_csv(f"{config.OUTPUT_DIR}/rolling_origin_cv_variant_summary.csv", index=False)
        return

    variant_names = list(tsm.FEATURE_SET_VARIANTS.keys())
    logger.info("=== run_rolling_origin_cv starting at %s ===", _now_str())
    logger.info("Variants (%d): %s", len(variant_names), variant_names)
    logger.info("Folds (%d): %s", len(FOLDS), [f["fold_label"] for f in FOLDS])
    logger.info("Total (variant, fold) runs: %d", len(variant_names) * len(FOLDS))

    full_df = _load_augmented_train_dataset()

    # Pre-slice fold rows once (shared across all 7 variants for that fold)
    # rather than re-filtering the full dataframe 28 times.
    fold_rows = {}
    for fold in FOLDS:
        fold_train_df, fold_val_df = _select_fold_rows(full_df, fold)
        fold_rows[fold["fold_label"]] = (fold_train_df, fold_val_df)
        logger.info(
            "Fold '%s': %d train rows (<= %s, any augmentation_type), %d val rows (%s to %s, original only)",
            fold["fold_label"], len(fold_train_df), fold["train_cutoff"],
            len(fold_val_df), fold["val_start"], fold["val_end"],
        )

    results = []
    cv_start = time.monotonic()
    n_total = len(variant_names) * len(FOLDS)
    run_idx = 0

    for variant_name in variant_names:
        feature_cols = tsm.FEATURE_SET_VARIANTS[variant_name]

        for fold in FOLDS:
            run_idx += 1
            fold_label = fold["fold_label"]
            fold_train_df, fold_val_df = fold_rows[fold_label]

            logger.info(
                "--- [%d/%d] Starting variant '%s' / fold '%s' at %s (%d requested feature columns) ---",
                run_idx, n_total, variant_name, fold_label, _now_str(), len(feature_cols),
            )
            run_start = time.monotonic()
            status, error_message = "success", ""
            result = {"best_epoch": np.nan, "best_val_loss": np.nan, "val_n_samples": np.nan}

            try:
                result = _train_one_fold_variant(fold_train_df, fold_val_df, feature_cols, variant_name, fold_label)
            except Exception as exc:
                status = "failed"
                error_message = str(exc)
                logger.error(
                    "--- [%d/%d] variant '%s' / fold '%s' FAILED after %s: %s ---",
                    run_idx, n_total, variant_name, fold_label, _fmt_elapsed(time.monotonic() - run_start), exc,
                )
                logger.error("Full traceback:\n%s", traceback.format_exc())
            else:
                logger.info(
                    "--- [%d/%d] variant '%s' / fold '%s' complete in %s: best_epoch=%s best_val_loss=%s ---",
                    run_idx, n_total, variant_name, fold_label,
                    _fmt_elapsed(time.monotonic() - run_start), result["best_epoch"], result["best_val_loss"],
                )

            total_elapsed = time.monotonic() - cv_start
            logger.info(
                "--- [%d/%d] Running total elapsed: %s ---", run_idx, n_total, _fmt_elapsed(total_elapsed),
            )

            results.append({
                "model_variant": variant_name,
                "fold_label": fold_label,
                "best_epoch": result["best_epoch"],
                "best_val_loss": result["best_val_loss"],
                "val_n_samples": result["val_n_samples"],
                "status": status,
                "error_message": error_message,
            })

    summary_df = pd.DataFrame(results)
    summary_path = f"{config.OUTPUT_DIR}/rolling_origin_cv_summary.csv"
    # Ordered, easy-to-read columns: variant, fold, per spec columns first.
    ordered_cols = ["model_variant", "fold_label", "best_epoch", "best_val_loss", "val_n_samples", "status", "error_message"]
    summary_df = summary_df[ordered_cols]
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved per-(variant,fold) CV summary to %s (%d rows).", summary_path, len(summary_df))

    # -----------------------------------------------------------------
    # Part C.5: per-variant aggregation across folds -> fixed epoch count
    # for Part D, plus stability (std of best_val_loss across folds).
    # -----------------------------------------------------------------
    variant_rows = []
    for variant_name in variant_names:
        sub = summary_df[(summary_df["model_variant"] == variant_name) & (summary_df["status"] == "success")]
        n_succeeded = len(sub)
        if n_succeeded == 0:
            variant_rows.append({
                "model_variant": variant_name,
                "fixed_epoch_count": np.nan,
                "mean_best_epoch": np.nan,
                "std_best_val_loss": np.nan,
                "mean_best_val_loss": np.nan,
                "n_folds_succeeded": 0,
            })
            continue

        mean_best_epoch = sub["best_epoch"].mean()
        variant_rows.append({
            "model_variant": variant_name,
            "fixed_epoch_count": int(round(mean_best_epoch)),
            "mean_best_epoch": mean_best_epoch,
            "std_best_val_loss": sub["best_val_loss"].std(ddof=0) if n_succeeded > 1 else 0.0,
            "mean_best_val_loss": sub["best_val_loss"].mean(),
            "n_folds_succeeded": n_succeeded,
        })

    variant_summary_df = pd.DataFrame(variant_rows)
    variant_summary_path = f"{config.OUTPUT_DIR}/rolling_origin_cv_variant_summary.csv"
    variant_summary_df.to_csv(variant_summary_path, index=False)
    logger.info("Saved per-variant CV aggregation to %s (%d rows).", variant_summary_path, len(variant_summary_df))

    n_success = int((summary_df["status"] == "success").sum())
    n_failed = int((summary_df["status"] == "failed").sum())
    total_elapsed = time.monotonic() - cv_start
    logger.info(
        "=== run_rolling_origin_cv complete at %s. Total elapsed: %s. %d/%d (variant,fold) runs succeeded, %d failed. ===",
        _now_str(), _fmt_elapsed(total_elapsed), n_success, n_total, n_failed,
    )
    if n_failed > 0:
        failed_rows = summary_df[summary_df["status"] == "failed"][["model_variant", "fold_label", "error_message"]]
        logger.warning("Failed (variant, fold) runs (see %s for details):\n%s", summary_path, failed_rows.to_string(index=False))

    print("\n=== ROLLING-ORIGIN CV: PER-VARIANT SUMMARY (feeds Part D) ===")
    print(variant_summary_df.to_string(index=False))
    print("\n=== ROLLING-ORIGIN CV: PER-(VARIANT, FOLD) DETAIL ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    run()
