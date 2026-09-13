"""
train_and_eval_all_features_plus_cycle_phase_filtered.py

STANDALONE FILL-IN SCRIPT for the ONE remaining gap in the flag-filtered
post-window oversampling ablation: the "all_features_plus_cycle_phase"
variant's oversampled model, which failed with an out-of-memory error when
trained as the 7th variant in a single long-running process
(train_and_eval_postwindow_oversample_filtered_all_variants.py).

WHY THIS SHOULD SUCCEED WHERE THE EARLIER ATTEMPT DIDN'T
----------------------------------------------------------
The earlier failure happened building the BASE (non-oversampled) sequence
array -- shape (341468, 16, 47), ~980 MiB -- which is essentially the same
allocation that train_missing_dedup_baselines.py performed successfully for
this exact variant (46 feature columns, same row count) in a separate,
freshly-started process. That strongly suggests the failure was caused by
memory fragmentation left over from the six other variants already trained
earlier in that same long-running process, not a hard memory ceiling.
Running this variant alone, in its own process, should avoid that. As an
extra safeguard, intermediate large arrays are explicitly deleted and
gc.collect() is called between stages.

WHAT THIS DOES
--------------
1. Loads new-output/augmented_train_dataset.csv READ-ONLY, filters to
   augmentation_type == "original" only (never written back to disk).
2. build_sequences() for "all_features_plus_cycle_phase" (its
   FEATURE_SET_VARIANTS entry already exists in train_sequence_model.py --
   no new variant definition needed here).
3. Flag-filtered oversampling-eligibility + oversampling (identical logic
   to train_and_eval_postwindow_oversample_filtered_all_variants.py).
4. Trains for 5 fixed epochs (this variant's known fixed_epoch_count from
   new-output/rolling_origin_cv_variant_summary.csv), same
   architecture/optimizer/loss/batch-size/seed as every other model in this
   ablation.
5. Saves artifacts to the SAME paths the all-variants script would have
   used, so downstream evaluation scripts (e.g.
   evaluate_dedup_vs_filtered_all_variants.py) pick it up with no changes:
     new-output/test_output/models/sequence_model_all_features_plus_cycle_phase_postwindow_oversample_filtered_test.pt
     new-output/test_output/models/sequence_scaler_all_features_plus_cycle_phase_postwindow_oversample_filtered_test.joblib
     new-output/test_output/plots/sequence_training_loss_all_features_plus_cycle_phase_postwindow_oversample_filtered_test.png
6. Immediately evaluates it against new-output/test_dataset.csv, alongside
   raw SGP4 and this variant's existing dedup-only baseline (already
   trained by train_missing_dedup_baselines.py), and prints/saves a 3-row
   comparison so you have a complete picture without re-running the whole
   ablation.

Does not modify any existing file. Does not touch
new-output/augmented_train_dataset.csv, new-output/models, or
new-output/plots (its dedup-only baseline there is read, never rewritten).

REUSED, UNCHANGED:
    - train_sequence_model.py: FEATURE_SET_VARIANTS, GRUResidualModel,
      build_sequences, _normalize, SequenceDataset, TARGET_COLUMNS
    - evaluate_sequence_ablation_v2_dedup.py: add_sequence_variant_predictions_v2_dedup
    - evaluate_model.py: _metrics
"""

import gc
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
logger = logging.getLogger("s23_train_eval_postwindow_oversample_filtered_cycle_phase_fillin")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

OVERSAMPLE_HIGH_ERROR_QUANTILE = getattr(config, "AUGMENTATION_HIGH_ERROR_QUANTILE", 0.95)
OVERSAMPLE_FACTOR = getattr(config, "AUGMENTATION_OVERSAMPLE_FACTOR", 3)
FLAG_COLUMNS = ["outlier_flag", "truth_offset_flag", "huge_gap_flag"]

VARIANT_NAME = "all_features_plus_cycle_phase"

TEST_OUTPUT_DIR = f"{config.OUTPUT_DIR}/test_output"
TEST_MODELS_DIR = f"{TEST_OUTPUT_DIR}/models"
TEST_PLOTS_DIR = f"{TEST_OUTPUT_DIR}/plots"

TRAIN_BATCH_SIZE = 64
LEARNING_RATE = 1e-3

RAW_COLUMN = "position_error_km"
DEDUP_MODEL_SUFFIX = "_rolling_cv_v2_dedup"
NEW_MODEL_SUFFIX = "_postwindow_oversample_filtered_test"
DEDUP_MODEL_DISPLAY_NAME = f"{VARIANT_NAME}{DEDUP_MODEL_SUFFIX}"
NEW_MODEL_DISPLAY_NAME = f"{VARIANT_NAME}{NEW_MODEL_SUFFIX}"


def _ensure_test_output_dirs():
    os.makedirs(TEST_MODELS_DIR, exist_ok=True)
    os.makedirs(TEST_PLOTS_DIR, exist_ok=True)


def _load_fixed_epoch_count(variant_name=VARIANT_NAME):
    summary_path = f"{config.OUTPUT_DIR}/rolling_origin_cv_variant_summary.csv"
    summary_df = pd.read_csv(summary_path)
    row = summary_df[summary_df["model_variant"] == variant_name]
    if row.empty:
        raise RuntimeError(f"No row with model_variant == '{variant_name}' found in {summary_path}.")
    n_epochs = int(row.iloc[0]["fixed_epoch_count"])
    logger.info("Loaded fixed_epoch_count=%d for variant '%s' from %s.", n_epochs, variant_name, summary_path)
    return n_epochs


def build_deduplicated_train_df(raw_train_df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(raw_train_df)
    logger.info("Full augmented_train_dataset.csv row count (in memory, unmodified on disk): %d", n_before)
    if "augmentation_type" not in raw_train_df.columns:
        raise RuntimeError("'augmentation_type' column not found in augmented_train_dataset.csv.")
    dedup_df = raw_train_df[raw_train_df["augmentation_type"] == "original"].copy()
    n_after = len(dedup_df)
    logger.info(
        "Filtered to augmentation_type == 'original' only: %d -> %d rows (%.2f%% retained). "
        "In-memory pandas filter only -- the CSV on disk is untouched.",
        n_before, n_after, (n_after / n_before * 100.0) if n_before > 0 else float("nan"),
    )
    if n_after == 0:
        raise RuntimeError("No rows remain after filtering to augmentation_type == 'original'.")
    return dedup_df


def _coerce_bool_column(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.map(lambda v: str(v).strip().lower() in ("true", "1", "1.0", "yes") if pd.notna(v) else False)


def compute_window_eligibility(meta, dedup_train_df: pd.DataFrame) -> np.ndarray:
    missing_flag_cols = [c for c in FLAG_COLUMNS if c not in dedup_train_df.columns]
    if missing_flag_cols:
        raise RuntimeError(f"Expected flag column(s) missing from augmented_train_dataset.csv: {missing_flag_cols}")

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
            "%d / %d window(s) could not be matched back to a source row for flag lookup "
            "(treating as eligible by default).", n_unmatched, len(merged),
        )
    for c in FLAG_COLUMNS:
        merged[c] = merged[c].fillna(False).astype(bool)

    ineligible_mask = merged[FLAG_COLUMNS[0]] | merged[FLAG_COLUMNS[1]] | merged[FLAG_COLUMNS[2]]
    eligible_mask = (~ineligible_mask).values

    n_total = len(eligible_mask)
    n_eligible = int(eligible_mask.sum())
    logger.info(
        "Oversampling-eligibility filter: %d / %d windows (%.2f%%) eligible, %d excluded.",
        n_eligible, n_total, (n_eligible / n_total * 100.0) if n_total else float("nan"), n_total - n_eligible,
    )
    return eligible_mask


def apply_postwindow_oversampling_filtered(train_data, eligible_mask: np.ndarray):
    sequences = train_data["sequences"]
    statics = train_data["statics"]
    targets = train_data["targets"]
    meta = train_data["meta"]

    n_before = len(sequences)
    magnitudes = np.linalg.norm(targets, axis=1)
    eligible_magnitudes = magnitudes[eligible_mask]
    if len(eligible_magnitudes) == 0:
        raise RuntimeError("No eligible windows available to compute an oversampling threshold from.")

    threshold = float(np.quantile(eligible_magnitudes, OVERSAMPLE_HIGH_ERROR_QUANTILE))
    high_error_mask = eligible_mask & (magnitudes >= threshold)
    n_high_error = int(high_error_mask.sum())

    logger.info(
        "Post-window oversampling: quantile=%.4f over %d eligible window(s) -> threshold=%.6f km. "
        "%d / %d total windows (%.2f%%) identified as eligible high-error.",
        OVERSAMPLE_HIGH_ERROR_QUANTILE, len(eligible_magnitudes), threshold,
        n_high_error, n_before, (n_high_error / n_before * 100.0) if n_before > 0 else float("nan"),
    )

    n_extra_copies = max(OVERSAMPLE_FACTOR - 1, 0)
    if n_high_error == 0 or n_extra_copies == 0:
        return {
            "sequences": sequences, "statics": statics, "targets": targets,
            "meta": list(meta), "feature_cols": train_data["feature_cols"],
        }

    high_error_indices = np.nonzero(high_error_mask)[0]
    dup_indices = np.tile(high_error_indices, n_extra_copies)

    sequences_out = np.concatenate([sequences, sequences[dup_indices]], axis=0)
    statics_out = np.concatenate([statics, statics[dup_indices]], axis=0)
    targets_out = np.concatenate([targets, targets[dup_indices]], axis=0)
    meta_out = list(meta) + [meta[i] for i in dup_indices]

    n_after = len(sequences_out)
    logger.info(
        "Post-window oversampling applied: %d eligible high-error window(s) duplicated %d extra "
        "time(s) each (OVERSAMPLE_FACTOR=%d). Total training-sequence count: %d -> %d.",
        n_high_error, n_extra_copies, OVERSAMPLE_FACTOR, n_before, n_after,
    )
    return {
        "sequences": sequences_out, "statics": statics_out, "targets": targets_out,
        "meta": meta_out, "feature_cols": train_data["feature_cols"],
    }


def _train_model(oversampled_data, n_epochs):
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
        "trained_via": "postwindow_oversample_filtered_test_isolated_retry",
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
            f"Post-window oversample (flag-filtered) test: {VARIANT_NAME} "
            f"(fixed {n_epochs} epochs, quantile={OVERSAMPLE_HIGH_ERROR_QUANTILE}, factor={OVERSAMPLE_FACTOR})"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
    except Exception as exc:
        logger.warning("Failed to plot training loss: %s", exc)

    return {"model_path": model_path, "scaler_path": scaler_path, "plot_path": plot_path,
            "final_train_loss": train_losses[-1] if train_losses else np.nan}


def add_checkpoint_predictions(df, model_path, scaler_path, display_name):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    norm_stats = joblib.load(scaler_path)

    model = tsm.GRUResidualModel(
        n_features=checkpoint["n_features"], static_dim=checkpoint["static_dim"],
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

    gc.collect()

    n_epochs = _load_fixed_epoch_count(VARIANT_NAME)

    train_path = f"{config.OUTPUT_DIR}/augmented_train_dataset.csv"
    logger.info("Loading (READ-ONLY) training dataset from %s ...", train_path)
    raw_train_df = pd.read_csv(train_path)
    if raw_train_df.empty:
        raise RuntimeError(f"{train_path} is empty.")
    raw_train_df["current_epoch_utc"] = pd.to_datetime(raw_train_df["current_epoch_utc"], utc=True, format="ISO8601")
    dedup_train_df = build_deduplicated_train_df(raw_train_df)
    del raw_train_df
    gc.collect()

    feature_cols = tsm.FEATURE_SET_VARIANTS[VARIANT_NAME]
    logger.info(
        "Building base training sequences for '%s' (window_size=%d, %d feature columns)...",
        VARIANT_NAME, config.SEQUENCE_WINDOW_SIZE, len(feature_cols),
    )
    base_train_data = tsm.build_sequences(dedup_train_df, config.SEQUENCE_WINDOW_SIZE, feature_cols=feature_cols)
    if base_train_data is None:
        raise RuntimeError("Could not build any training sequences.")
    logger.info("Base training sequences (no augmentation): %d", len(base_train_data["sequences"]))

    eligible_mask = compute_window_eligibility(base_train_data["meta"], dedup_train_df)
    del dedup_train_df
    gc.collect()

    oversampled_data = apply_postwindow_oversampling_filtered(base_train_data, eligible_mask)
    del base_train_data, eligible_mask
    gc.collect()

    training_result = None
    try:
        logger.info("Training '%s' for %d fixed epoch(s)...", NEW_MODEL_DISPLAY_NAME, n_epochs)
        training_result = _train_model(oversampled_data, n_epochs)
        logger.info(
            "Training complete. final_train_loss=%.4f. Saved: %s, %s, %s",
            training_result["final_train_loss"], training_result["model_path"],
            training_result["scaler_path"], training_result["plot_path"],
        )
    except Exception:
        logger.exception("Training failed.")
    del oversampled_data
    gc.collect()

    # --- Evaluate: raw SGP4, existing dedup baseline, and this new model ---
    test_path = f"{config.OUTPUT_DIR}/test_dataset.csv"
    test_df = pd.read_csv(test_path)
    if test_df.empty:
        logger.warning("Test dataset is empty. Nothing to evaluate.")
        return
    test_df["current_epoch_utc"] = pd.to_datetime(test_df["current_epoch_utc"], utc=True, format="ISO8601")

    rows = []
    try:
        rows.append({"model": "raw_sgp4", **_metrics(test_df[RAW_COLUMN]),
                     "pct_samples_improved": np.nan, "mean_pct_improvement": np.nan})
    except Exception:
        logger.exception("Failed computing metrics for raw_sgp4.")

    try:
        test_df, ok, dedup_error_col = add_sequence_variant_predictions_v2_dedup(test_df, VARIANT_NAME)
        if ok:
            pct_improved, mean_improve = _improvement_stats(test_df[RAW_COLUMN], test_df[dedup_error_col])
            rows.append({"model": DEDUP_MODEL_DISPLAY_NAME, **_metrics(test_df[dedup_error_col].dropna()),
                         "pct_samples_improved": pct_improved, "mean_pct_improvement": mean_improve})
        else:
            logger.warning("'%s' unavailable.", DEDUP_MODEL_DISPLAY_NAME)
    except Exception:
        logger.exception("Failed evaluating '%s'.", DEDUP_MODEL_DISPLAY_NAME)

    if training_result is not None:
        try:
            test_df, ok, new_error_col = add_checkpoint_predictions(
                test_df, training_result["model_path"], training_result["scaler_path"], NEW_MODEL_DISPLAY_NAME
            )
            if ok:
                pct_improved, mean_improve = _improvement_stats(test_df[RAW_COLUMN], test_df[new_error_col])
                rows.append({"model": NEW_MODEL_DISPLAY_NAME, **_metrics(test_df[new_error_col].dropna()),
                             "pct_samples_improved": pct_improved, "mean_pct_improvement": mean_improve})
            else:
                logger.warning("'%s' unavailable.", NEW_MODEL_DISPLAY_NAME)
        except Exception:
            logger.exception("Failed evaluating '%s'.", NEW_MODEL_DISPLAY_NAME)
    else:
        logger.warning("Skipping evaluation of the new model since training did not complete.")

    summary_df = pd.DataFrame(rows)
    ordered_cols = ["model", "n_samples", "mae_km", "median_ae_km", "rmse_km", "p90_km", "p95_km", "max_km",
                    "pct_samples_improved", "mean_pct_improvement"]
    summary_df = summary_df[[c for c in ordered_cols if c in summary_df.columns]]
    summary_path = f"{TEST_OUTPUT_DIR}/diagnostics_all_features_plus_cycle_phase_filtered_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved comparison table to %s (%d rows).", summary_path, len(summary_df))

    print(f"\n=== {VARIANT_NAME}: DEDUP-ONLY vs. FLAG-FILTERED vs. RAW SGP4 (test set) ===")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    else:
        print("No rows could be evaluated.")

    logger.info(
        "Done. Re-run evaluate_dedup_vs_filtered_all_variants.py to fold this into the full 7-variant table."
    )


if __name__ == "__main__":
    run()
