"""
train_sequence_model_rolling_cv_final_dedup.py

PART D.1 (DEDUP-FIXED): Final retraining of every FEATURE_SET_VARIANTS entry
(all 7) on new-output/augmented_train_dataset.csv, exactly like
train_sequence_model_rolling_cv_final.py, EXCEPT the rows fed into
build_sequences() are filtered in-memory to augmentation_type == "original"
only, eliminating the exact-duplicate-current_epoch_utc rows that
diagnose_duplicate_timestamps.py / diagnose_dedup_confirmation_test.py
identified as the likely root cause of the "original" variant's
worse-than-raw-SGP4 test performance.

REUSED, UNCHANGED, AS-IS (per project convention -- nothing in these files
is modified or reimplemented):
    - train_sequence_model.py: GRUResidualModel, build_sequences,
      _normalize, SequenceDataset, TARGET_COLUMNS, FEATURE_SET_VARIANTS
    - train_sequence_model_rolling_cv_final.py: _load_fixed_epoch_counts,
      _now_str, _fmt_elapsed (imported, not copied)

WHAT'S DIFFERENT FROM train_sequence_model_rolling_cv_final.py:
    1. Training rows are filtered to augmentation_type == "original" before
       build_sequences() is called (in-memory filter only --
       augmented_train_dataset.csv on disk is never modified).
    2. Epoch counts are NOT recomputed -- per the agreed "cheap fix", this
       script reuses the existing fixed_epoch_count values from Part C's
       new-output/rolling_origin_cv_variant_summary.csv unchanged. See the
       CAVEAT note below.
    3. All outputs use the "_rolling_cv_v2_dedup" suffix -- distinct from
       both the v1 ablation artifacts AND the existing (duplicate-timestamp
       -contaminated) "_rolling_cv_v2" artifacts. NEITHER of those is ever
       touched, overwritten, or deleted by this script.

CAVEAT (worth noting in your write-up): the fixed_epoch_count values reused
here were originally selected by Part C's rolling-origin CV, whose FOLD
TRAINING rows also included all augmentation types (only each fold's
validation rows were correctly restricted to "original", per the original
spec). So these epoch counts are "good enough" -- as the single-variant
confirmation test already showed for "original" -- but not perfectly clean,
since they were chosen under the same duplicate-timestamp condition being
removed here. If results look off for a particular variant, re-running Part
C with dedup-filtered FOLD training rows too would be the fully-clean fix.

Outputs (per variant):
    new-output/models/sequence_model_<variant_name>_rolling_cv_v2_dedup.pt
    new-output/models/sequence_scaler_<variant_name>_rolling_cv_v2_dedup.joblib
    new-output/plots/sequence_training_loss_<variant_name>_rolling_cv_v2_dedup.png

Also writes:
    new-output/rolling_cv_final_training_run_summary_dedup.csv
        One row per variant: status, elapsed seconds, epoch count trained,
        final train loss, number of (deduplicated) training sequences, and
        (on failure) the exception message.
"""

import logging
import time
import traceback

import numpy as np
import pandas as pd
import joblib

import config
import s06_train_sequence_model as tsm
import train_sequence_model_rolling_cv_final_predup as tscf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("s16_train_sequence_model_v2_dedup")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

TRAIN_BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# Distinct from both v1 and "_rolling_cv_v2" -- neither existing artifact
# set is ever overwritten by this script.
DEDUP_MODEL_SUFFIX = "_rolling_cv_v2_dedup"


def build_deduplicated_train_df(raw_train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pure in-memory filter: keep only augmentation_type == "original" rows.
    Never writes back to disk. Logs before/after counts and confirms the
    filtered data is (at or near) free of duplicate-timestamp groups.
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

    cluster_sizes = dedup_df.groupby(
        ["object_id", "horizon_hours", "current_epoch_utc"]
    )["current_epoch_utc"].transform("size")
    n_still_dup = int((cluster_sizes > 1).sum())
    pct_still_dup = n_still_dup / n_after * 100.0
    logger.info(
        "Sanity check: %d / %d filtered rows (%.4f%%) are still involved in a duplicate-timestamp "
        "group within their (object_id, horizon_hours) group. This should be at or near 0%%.",
        n_still_dup, n_after, pct_still_dup,
    )
    if pct_still_dup > 1.0:
        logger.warning(
            "Duplicate rate after filtering is higher than expected (%.4f%% > 1%%). This may "
            "indicate genuine duplicate 'original' rows in the source data, not just augmentation "
            "artifacts.", pct_still_dup,
        )

    return dedup_df


def _train_one_variant_final_dedup(dedup_train_df, feature_cols, variant_name, n_epochs):
    """
    Identical training-loop pattern to
    train_sequence_model_rolling_cv_final._train_one_variant_final
    (architecture, SmoothL1Loss, Adam lr=1e-3, batch size 64,
    torch.manual_seed(config.RANDOM_SEED), same device-selection logic, no
    validation split, no early stopping). The only differences: it is fed
    the in-memory-deduplicated dataframe, and it saves to
    "_rolling_cv_v2_dedup"-suffixed paths so nothing existing is touched.
    """
    train_data = tsm.build_sequences(dedup_train_df, config.SEQUENCE_WINDOW_SIZE, feature_cols=feature_cols)
    if train_data is None:
        raise RuntimeError("Could not build any training sequences from the deduplicated training set.")

    missing = [c for c in feature_cols if c not in train_data["feature_cols"]]
    if missing:
        logger.warning(
            "[%s] %d requested feature column(s) not found in training data and were dropped: %s",
            variant_name, len(missing), missing,
        )

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

    model_path = f"{config.MODELS_DIR}/sequence_model_{variant_name}{DEDUP_MODEL_SUFFIX}.pt"
    scaler_path = f"{config.MODELS_DIR}/sequence_scaler_{variant_name}{DEDUP_MODEL_SUFFIX}.joblib"
    plot_path = f"{config.PLOTS_DIR}/sequence_training_loss_{variant_name}{DEDUP_MODEL_SUFFIX}.png"

    torch.save({
        "state_dict": model.state_dict(),
        "n_features": n_features,
        "static_dim": train_static_n.shape[-1],
        "feature_cols": train_data["feature_cols"],
        "window_size": config.SEQUENCE_WINDOW_SIZE,
        "target_columns": tsm.TARGET_COLUMNS,
        "variant_name": variant_name,
        "n_epochs_trained": n_epochs,
        "trained_via": "rolling_cv_v2_final_dedup",
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
        ax.set_title(
            f"Dedup-fixed final retraining loss: {variant_name} "
            f"(fixed {n_epochs} epochs, augmentation_type == 'original' only)"
        )
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
        logger.warning("PyTorch is not installed. Skipping dedup-fixed final retraining entirely.")
        pd.DataFrame().to_csv(f"{config.OUTPUT_DIR}/rolling_cv_final_training_run_summary_dedup.csv", index=False)
        return

    # Reused, unmodified: read the SAME fixed_epoch_count values Part C
    # already produced. Per the agreed "cheap fix", Part C is NOT rerun --
    # see the CAVEAT in this file's module docstring.
    fixed_epoch_counts = tscf._load_fixed_epoch_counts()
    variant_names = [v for v in tsm.FEATURE_SET_VARIANTS if v in fixed_epoch_counts]
    skipped = [v for v in tsm.FEATURE_SET_VARIANTS if v not in fixed_epoch_counts]
    if skipped:
        logger.warning("Skipping variant(s) with no usable fixed_epoch_count: %s", skipped)

    logger.info("=== train_sequence_model_rolling_cv_final_dedup starting at %s ===", tscf._now_str())
    logger.info("Variants to retrain (%d): %s", len(variant_names), variant_names)
    logger.info("Reusing existing fixed epoch counts (Part C NOT rerun): %s",
                {v: fixed_epoch_counts[v] for v in variant_names})

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

    results = []
    run_start = time.monotonic()

    for i, variant_name in enumerate(variant_names, start=1):
        feature_cols = tsm.FEATURE_SET_VARIANTS[variant_name]
        n_epochs = fixed_epoch_counts[variant_name]

        logger.info(
            "--- [%d/%d] Starting DEDUP-FIXED final retraining for variant '%s' at %s "
            "(%d feature columns, fixed %d epochs, %d deduplicated training rows available) ---",
            i, len(variant_names), variant_name, tscf._now_str(), len(feature_cols), n_epochs, len(dedup_train_df),
        )
        variant_start = time.monotonic()
        status, error_message = "success", ""
        result = {"n_epochs_trained": n_epochs, "final_train_loss": np.nan, "n_train_sequences": np.nan}

        try:
            result = _train_one_variant_final_dedup(dedup_train_df, feature_cols, variant_name, n_epochs)
        except Exception as exc:
            status = "failed"
            error_message = str(exc)
            logger.error(
                "--- [%d/%d] variant '%s' FAILED after %s: %s ---",
                i, len(variant_names), variant_name, tscf._fmt_elapsed(time.monotonic() - variant_start), exc,
            )
            logger.error("Full traceback:\n%s", traceback.format_exc())
        else:
            logger.info(
                "--- [%d/%d] variant '%s' complete in %s: final_train_loss=%.4f ---",
                i, len(variant_names), variant_name,
                tscf._fmt_elapsed(time.monotonic() - variant_start), result["final_train_loss"],
            )

        total_elapsed = time.monotonic() - run_start
        logger.info("--- [%d/%d] Running total elapsed: %s ---", i, len(variant_names), tscf._fmt_elapsed(total_elapsed))

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
    summary_path = f"{config.OUTPUT_DIR}/rolling_cv_final_training_run_summary_dedup.csv"
    summary_df.to_csv(summary_path, index=False)

    n_success = int((summary_df["status"] == "success").sum())
    n_failed = int((summary_df["status"] == "failed").sum())
    logger.info(
        "=== train_sequence_model_rolling_cv_final_dedup complete. %d/%d succeeded, %d failed. Summary: %s ===",
        n_success, len(variant_names), n_failed, summary_path,
    )
    if n_failed > 0:
        failed_names = summary_df.loc[summary_df["status"] == "failed", "model_variant"].tolist()
        logger.warning("Failed variants (see traceback above and %s for details): %s", summary_path, failed_names)

    print("\n=== DEDUP-FIXED FINAL ROLLING-CV RETRAINING SUMMARY ===")
    print(summary_df.to_string(index=False))
    print(
        "\nNote: existing new-output/models/sequence_model_<variant>_rolling_cv_v2.pt/.joblib "
        "artifacts (the duplicate-timestamp-contaminated ones) were NOT touched. New artifacts "
        f"use the '{DEDUP_MODEL_SUFFIX}' suffix. Run evaluate_sequence_ablation_v2_dedup.py next "
        "to compare old vs. new on the test set."
    )


if __name__ == "__main__":
    run()
