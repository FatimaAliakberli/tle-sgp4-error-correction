"""
train_missing_dedup_baselines.py

STANDALONE FILL-IN SCRIPT. Trains the "<variant>_rolling_cv_v2_dedup"
baseline model (train_sequence_model_rolling_cv_final_dedup.py's exact
recipe: dedup filter only, NO oversampling of any kind) for whichever
FEATURE_SET_VARIANTS entries do not already have that artifact under
new-output/models/.

WHY THIS EXISTS
---------------
train_and_eval_postwindow_oversample_filtered_all_variants.py's run showed
that "all_features", "plus_space_weather_full_and_cycle_phase", and
"all_features_plus_cycle_phase" have no "<variant>_rolling_cv_v2_dedup"
baseline on disk (train_sequence_model_rolling_cv_final_dedup.py was
apparently only ever run for a subset of variants). Without that baseline,
there is no way to tell whether the flag-filtered post-window oversampling
augmentation helped, hurt, or did nothing for those particular feature
sets -- you can only compare augmented-vs-raw-SGP4, not
augmented-vs-un-augmented. This script fills in exactly that gap.

SAFETY
------
This is the one script in this experiment sequence that DOES write into
new-output/models/ and new-output/plots/ -- intentionally, since these are
meant to become real baseline artifacts consumable by
add_sequence_variant_predictions_v2_dedup (evaluate_sequence_ablation_v2_dedup.py)
and any other script that already looks for "<variant>_rolling_cv_v2_dedup"
there. To stay safe:
    - It NEVER modifies, retrains, or overwrites an existing
      "<variant>_rolling_cv_v2_dedup" artifact. For every FEATURE_SET_VARIANTS
      entry, it first checks whether BOTH the .pt and .joblib files already
      exist; if so, that variant is skipped entirely (logged, not trained).
    - new-output/augmented_train_dataset.csv is loaded READ-ONLY and never
      written back to.
    - Only variants missing the baseline are trained; a failure on one
      variant does not stop the others.

PIPELINE (per missing variant, matching
train_sequence_model_rolling_cv_final_dedup.py's exact recipe)
---------------------------------------------------------------
1. Load new-output/augmented_train_dataset.csv READ-ONLY, filter in-memory
   to augmentation_type == "original" only (shared across all variants).
2. build_sequences() using that variant's FEATURE_SET_VARIANTS entry.
   NO post-window oversampling, NO row-level augmentation -- this is the
   "step 1 only" baseline.
3. _normalize() (train-stats only, no validation split).
4. Train ONE GRUResidualModel (SmoothL1Loss / Adam lr=1e-3 / batch size 64 /
   torch.manual_seed(config.RANDOM_SEED), no early stopping) for that
   variant's fixed epoch count, read from
   new-output/rolling_origin_cv_variant_summary.csv.
5. Save:
     new-output/models/sequence_model_<variant>_rolling_cv_v2_dedup.pt
     new-output/models/sequence_scaler_<variant>_rolling_cv_v2_dedup.joblib
     new-output/plots/sequence_training_loss_<variant>_rolling_cv_v2_dedup.png

REUSED, UNCHANGED:
    - train_sequence_model.py: FEATURE_SET_VARIANTS, GRUResidualModel,
      build_sequences, _normalize, SequenceDataset, TARGET_COLUMNS

After running this, re-run
train_and_eval_postwindow_oversample_filtered_all_variants.py's evaluation
phase (or evaluate_sequence_ablation_v2_dedup.py directly) to get the
now-complete dedup-vs-filtered comparison for every variant.
"""

import logging
import time
import traceback

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")

import config
import s06_train_sequence_model as tsm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("s17_train_missing_dedup_baselines")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

TRAIN_BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# Matches evaluate_sequence_ablation_v2_dedup.py's DEDUP_MODEL_SUFFIX and
# train_sequence_model_rolling_cv_final_dedup.py's artifact naming exactly.
DEDUP_MODEL_SUFFIX = "_rolling_cv_v2_dedup"


def _dedup_artifact_paths(variant_name):
    model_path = f"{config.MODELS_DIR}/sequence_model_{variant_name}{DEDUP_MODEL_SUFFIX}.pt"
    scaler_path = f"{config.MODELS_DIR}/sequence_scaler_{variant_name}{DEDUP_MODEL_SUFFIX}.joblib"
    return model_path, scaler_path


def _load_fixed_epoch_counts():
    """
    Reads fixed_epoch_count for every FEATURE_SET_VARIANTS entry from
    new-output/rolling_origin_cv_variant_summary.csv.
    """
    summary_path = f"{config.OUTPUT_DIR}/rolling_origin_cv_variant_summary.csv"
    summary_df = pd.read_csv(summary_path)
    counts = {}
    for _, row in summary_df.iterrows():
        variant = row.get("model_variant")
        if variant in tsm.FEATURE_SET_VARIANTS and pd.notna(row.get("fixed_epoch_count")):
            counts[variant] = int(row["fixed_epoch_count"])
    return counts


def find_missing_dedup_variants():
    """
    Returns the list of FEATURE_SET_VARIANTS entries that do NOT already
    have both a .pt and .joblib "<variant>_rolling_cv_v2_dedup" artifact
    under config.MODELS_DIR.
    """
    missing = []
    for variant_name in tsm.FEATURE_SET_VARIANTS:
        model_path, scaler_path = _dedup_artifact_paths(variant_name)
        import os
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            logger.info("'%s%s' already exists (%s, %s) -- will NOT retrain.",
                        variant_name, DEDUP_MODEL_SUFFIX, model_path, scaler_path)
        else:
            missing.append(variant_name)
    return missing


def build_deduplicated_train_df(raw_train_df: pd.DataFrame) -> pd.DataFrame:
    """
    In-memory-only filter: keep augmentation_type == "original" rows.
    Never writes back to disk.
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


def _train_one_dedup_baseline(dedup_train_df, variant_name, n_epochs):
    """
    train_sequence_model_rolling_cv_final_dedup.py's exact training-loop
    pattern: dedup filter only, no oversampling. GRUResidualModel /
    SmoothL1Loss / Adam lr=1e-3 / batch size 64 /
    torch.manual_seed(config.RANDOM_SEED) / same device-selection logic,
    fixed epoch count, no validation split, no early stopping.
    """
    display_name = f"{variant_name}{DEDUP_MODEL_SUFFIX}"

    feature_cols = tsm.FEATURE_SET_VARIANTS[variant_name]
    logger.info(
        "[%s] Building training sequences (window_size=%d, %d feature columns)...",
        display_name, config.SEQUENCE_WINDOW_SIZE, len(feature_cols),
    )
    train_data = tsm.build_sequences(dedup_train_df, config.SEQUENCE_WINDOW_SIZE, feature_cols=feature_cols)
    if train_data is None:
        raise RuntimeError(f"[{display_name}] Could not build any training sequences.")
    n_sequences = len(train_data["sequences"])
    logger.info("[%s] Training sequences (dedup-only, no oversampling): %d", display_name, n_sequences)

    (train_seq_n, train_static_n), norm_stats = tsm._normalize(
        train_data["sequences"], train_data["statics"]
    )

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
        logger.info("[%s] Epoch %d/%d: train_loss=%.4f", display_name, epoch + 1, n_epochs, train_loss)

    model_path, scaler_path = _dedup_artifact_paths(variant_name)
    plot_path = f"{config.PLOTS_DIR}/sequence_training_loss_{display_name}.png"

    torch.save({
        "state_dict": model.state_dict(),
        "n_features": n_features,
        "static_dim": train_static_n.shape[-1],
        "feature_cols": train_data["feature_cols"],
        "window_size": config.SEQUENCE_WINDOW_SIZE,
        "target_columns": tsm.TARGET_COLUMNS,
        "variant_name": variant_name,
        "n_epochs_trained": n_epochs,
        "trained_via": "s17_train_missing_dedup_baselines",
    }, model_path)
    joblib.dump(norm_stats, scaler_path)

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(train_losses, label="train_loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Smooth L1 loss")
        ax.set_title(f"Dedup-only baseline: {variant_name} (fixed {n_epochs} epochs, no oversampling)")
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
        "n_train_sequences": n_sequences,
        "final_train_loss": train_losses[-1] if train_losses else np.nan,
    }


def run():
    config.ensure_dirs()

    if not TORCH_AVAILABLE:
        logger.error("PyTorch is not installed. Cannot train any dedup baselines.")
        return

    missing_variants = find_missing_dedup_variants()
    if not missing_variants:
        logger.info("Every FEATURE_SET_VARIANTS entry already has a '%s' artifact. Nothing to do.",
                     DEDUP_MODEL_SUFFIX)
        return

    fixed_epoch_counts = _load_fixed_epoch_counts()
    variant_names = [v for v in missing_variants if v in fixed_epoch_counts]
    skipped_no_epoch_count = [v for v in missing_variants if v not in fixed_epoch_counts]
    if skipped_no_epoch_count:
        logger.warning(
            "Skipping variant(s) missing a usable fixed_epoch_count in rolling_origin_cv_variant_summary.csv: %s",
            skipped_no_epoch_count,
        )
    logger.info("Missing dedup baselines to train (%d): %s", len(variant_names), variant_names)
    if not variant_names:
        logger.info("Nothing left to train after filtering for a usable fixed_epoch_count.")
        return

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

    results = []
    run_start = time.monotonic()
    for i, variant_name in enumerate(variant_names, start=1):
        n_epochs = fixed_epoch_counts[variant_name]
        variant_start = time.monotonic()
        status, error_message = "success", ""
        n_sequences, final_train_loss = np.nan, np.nan

        logger.info("--- [%d/%d] Starting missing dedup baseline '%s' (fixed %d epochs) ---",
                     i, len(variant_names), variant_name, n_epochs)
        try:
            result = _train_one_dedup_baseline(dedup_train_df, variant_name, n_epochs)
            n_sequences = result["n_train_sequences"]
            final_train_loss = result["final_train_loss"]
        except Exception as exc:
            status = "failed"
            error_message = str(exc)
            logger.error("--- [%d/%d] '%s' FAILED: %s ---", i, len(variant_names), variant_name, exc)
            logger.error("Full traceback:\n%s", traceback.format_exc())
        else:
            logger.info("--- [%d/%d] '%s' complete in %.1fs: final_train_loss=%.4f ---",
                         i, len(variant_names), variant_name, time.monotonic() - variant_start, final_train_loss)

        results.append({
            "model_variant": variant_name,
            "n_epochs_trained": n_epochs,
            "n_train_sequences": n_sequences,
            "final_train_loss": final_train_loss,
            "status": status,
            "elapsed_seconds": round(time.monotonic() - variant_start, 2),
            "error_message": error_message,
        })

    results_df = pd.DataFrame(results)
    print("\n=== MISSING DEDUP BASELINES -- TRAINING SUMMARY ===")
    print(results_df.to_string(index=False))
    logger.info(
        "Done in %.1fs total. %d/%d missing baseline(s) trained successfully.",
        time.monotonic() - run_start,
        int((results_df["status"] == "success").sum()) if not results_df.empty else 0,
        len(variant_names),
    )
    logger.info(
        "Re-run evaluate_sequence_ablation_v2_dedup.py or "
        "train_and_eval_postwindow_oversample_filtered_all_variants.py's evaluation phase now "
        "to get the complete dedup-vs-filtered comparison for every variant."
    )


if __name__ == "__main__":
    run()
