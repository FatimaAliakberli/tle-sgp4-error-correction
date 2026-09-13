"""
diagnose_dedup_confirmation_test.py

Standalone confirmation script (read-only w.r.t. every existing file and
w.r.t. new-output/augmented_train_dataset.csv and all existing model
artifacts). No pipeline file is modified.

BACKGROUND
----------
diagnose_duplicate_timestamps.py established that essentially all
(object_id, horizon_hours) groups in new-output/augmented_train_dataset.csv
contain rows that share an exact current_epoch_utc (original + its
augmented copies). train_sequence_model.build_sequences() sorts each group
by current_epoch_utc and slides a fixed-size window over consecutive rows
with no augmentation_type filtering, so a training window built from this
file can span far fewer real distinct moments than an equivalent test-time
window (test data is "original" rows only, one row per timestamp).

This script runs ONE controlled confirmation experiment: retrain just the
"original" feature-set variant, for its already-selected fixed epoch count,
on an IN-MEMORY-ONLY deduplicated copy of the training data (augmentation_type
== "original" rows only), and compare its test-set performance directly
against the existing "original_rolling_cv_v2" model (unchanged, not
retrained) and against raw SGP4.

What this script does NOT do:
    - It never writes back to augmented_train_dataset.csv or any other
      existing input file.
    - It never overwrites any existing model artifact -- all outputs use
      the "_dedup_test" suffix, distinct from both v1 and
      "_rolling_cv_v2" paths.
    - It never retrains or touches the existing "original_rolling_cv_v2"
      model; that model is loaded read-only via
      evaluate_sequence_ablation_v2.add_sequence_variant_predictions_v2.

Outputs:
    new-output/models/sequence_model_original_dedup_test.pt
    new-output/models/sequence_scaler_original_dedup_test.joblib
    new-output/plots/sequence_training_loss_original_dedup_test.png
    new-output/diagnostics_dedup_confirmation_summary.csv
"""

import logging

import numpy as np
import pandas as pd
import joblib

import config
import s06_train_sequence_model as tsm
from evaluate_sequence_ablation_v2_predup import add_sequence_variant_predictions_v2
from s05_evaluate_baseline_and_raw_sgp4 import _metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("s14_confirm_dedup_fix")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

VARIANT_NAME = "original"
DEDUP_SUFFIX = "_dedup_test"
V2_SUFFIX = "_rolling_cv_v2"

RAW_COLUMN = "position_error_km"
TRAIN_BATCH_SIZE = 64
LEARNING_RATE = 1e-3


def load_fixed_epoch_count(variant_name: str) -> int:
    path = f"{config.OUTPUT_DIR}/rolling_origin_cv_variant_summary.csv"
    logger.info("Loading fixed_epoch_count for variant '%s' from %s ...", variant_name, path)
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"{path} is empty or missing.")

    row = df[df["model_variant"] == variant_name]
    if row.empty:
        raise RuntimeError(f"No row for model_variant == '{variant_name}' found in {path}.")

    fixed = row.iloc[0]["fixed_epoch_count"]
    if pd.isna(fixed):
        raise RuntimeError(
            f"fixed_epoch_count is NaN for variant '{variant_name}' in {path} "
            f"(all its rolling-CV folds may have failed)."
        )
    return int(fixed)


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
        "Sanity check: %d / %d filtered rows (%.4f%%) are STILL involved in a duplicate-timestamp "
        "group within their (object_id, horizon_hours) group. This should be at or near 0%%.",
        n_still_dup, n_after, pct_still_dup,
    )
    if pct_still_dup > 1.0:
        logger.warning(
            "Duplicate rate after filtering is higher than expected (%.4f%% > 1%%). "
            "This may indicate the raw data itself has genuine duplicate 'original' rows "
            "(e.g. re-ingested TLEs), not just augmentation artifacts.",
            pct_still_dup,
        )

    return dedup_df


def train_dedup_test_model(dedup_train_df: pd.DataFrame, feature_cols, n_epochs: int) -> dict:
    """
    Mirrors train_sequence_model_rolling_cv_final.py's final-training loop
    exactly (architecture, SmoothL1Loss, Adam lr=1e-3, batch size 64,
    torch.manual_seed(config.RANDOM_SEED), same device-selection logic, no
    validation split, no early stopping) -- the ONLY difference is that it
    is fed the in-memory-deduplicated dataframe instead of the full
    (all-augmentation-types) training set, and it saves to distinct
    "_dedup_test"-suffixed paths so nothing existing is touched.
    """
    logger.info(
        "[%s] Building training sequences from DEDUPLICATED data (window_size=%d, %d feature columns)...",
        VARIANT_NAME, config.SEQUENCE_WINDOW_SIZE, len(feature_cols),
    )
    train_data = tsm.build_sequences(dedup_train_df, config.SEQUENCE_WINDOW_SIZE, feature_cols=feature_cols)
    if train_data is None:
        raise RuntimeError("Could not build any training sequences from the deduplicated training set.")

    missing = [c for c in feature_cols if c not in train_data["feature_cols"]]
    if missing:
        logger.warning(
            "[%s] %d requested feature column(s) not found in training data and were dropped: %s",
            VARIANT_NAME, len(missing), missing,
        )
    logger.info(
        "[%s] Deduplicated training sequences built: %d (vs. whatever the full-augmentation "
        "training run built -- see rolling_cv_final_training_run_summary.csv for that count).",
        VARIANT_NAME, len(train_data["sequences"]),
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
        logger.info("[%s-dedup_test] Epoch %d/%d: train_loss=%.4f", VARIANT_NAME, epoch + 1, n_epochs, train_loss)

    model_path = f"{config.MODELS_DIR}/sequence_model_{VARIANT_NAME}{DEDUP_SUFFIX}.pt"
    scaler_path = f"{config.MODELS_DIR}/sequence_scaler_{VARIANT_NAME}{DEDUP_SUFFIX}.joblib"
    plot_path = f"{config.PLOTS_DIR}/sequence_training_loss_{VARIANT_NAME}{DEDUP_SUFFIX}.png"

    torch.save({
        "state_dict": model.state_dict(),
        "n_features": n_features,
        "static_dim": train_static_n.shape[-1],
        "feature_cols": train_data["feature_cols"],
        "window_size": config.SEQUENCE_WINDOW_SIZE,
        "target_columns": tsm.TARGET_COLUMNS,
        "variant_name": VARIANT_NAME,
        "n_epochs_trained": n_epochs,
        "trained_via": "dedup_confirmation_test",
    }, model_path)
    joblib.dump(norm_stats, scaler_path)
    logger.info("[%s-dedup_test] Saved model to %s and scaler to %s", VARIANT_NAME, model_path, scaler_path)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(train_losses, label="train_loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Smooth L1 loss")
        ax.set_title(
            f"Dedup confirmation retrain: {VARIANT_NAME} (fixed {n_epochs} epochs, "
            f"augmentation_type == 'original' only)"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        logger.info("[%s-dedup_test] Saved training-loss plot to %s", VARIANT_NAME, plot_path)
    except Exception as exc:
        logger.warning("[%s-dedup_test] Failed to plot training loss: %s", VARIANT_NAME, exc)

    return {
        "model_path": model_path,
        "scaler_path": scaler_path,
        "n_epochs_trained": n_epochs,
        "final_train_loss": train_losses[-1] if train_losses else np.nan,
        "n_train_sequences": len(train_data["sequences"]),
    }


def add_dedup_test_predictions(test_df: pd.DataFrame) -> tuple:
    """
    Same load-checkpoint / build-sequences / normalize / forward-pass
    pattern as evaluate_sequence_ablation_v2.add_sequence_variant_predictions_v2,
    but pointed at this script's own "_dedup_test" model/scaler paths.
    Eval-mode, no_grad, forward pass only -- no training happens here.
    """
    model_path = f"{config.MODELS_DIR}/sequence_model_{VARIANT_NAME}{DEDUP_SUFFIX}.pt"
    scaler_path = f"{config.MODELS_DIR}/sequence_scaler_{VARIANT_NAME}{DEDUP_SUFFIX}.joblib"

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    norm_stats = joblib.load(scaler_path)

    model = tsm.GRUResidualModel(
        n_features=checkpoint["n_features"],
        static_dim=checkpoint["static_dim"],
        n_targets=len(checkpoint["target_columns"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    data = tsm.build_sequences(test_df, checkpoint["window_size"], feature_cols=checkpoint["feature_cols"])
    if data is None:
        raise RuntimeError("Could not build evaluation sequences for the dedup_test model.")

    seq_n = (data["sequences"] - norm_stats["seq_mean"]) / norm_stats["seq_std"]
    static_n = (data["statics"] - norm_stats["static_mean"]) / norm_stats["static_std"]

    with torch.no_grad():
        preds = model(torch.from_numpy(seq_n.astype(np.float32)), torch.from_numpy(static_n.astype(np.float32)))
    preds = preds.numpy()

    diff = data["targets"] - preds
    seq_errors = np.linalg.norm(diff, axis=1)

    error_col = f"{VARIANT_NAME}{DEDUP_SUFFIX}_corrected_error_km"
    meta_df = pd.DataFrame(data["meta"], columns=["object_id", "horizon_hours", "current_epoch_utc"])
    meta_df[error_col] = seq_errors

    test_df = test_df.merge(meta_df, on=["object_id", "horizon_hours", "current_epoch_utc"], how="left")
    return test_df, error_col


def compute_improvement_stats(raw: pd.Series, corrected: pd.Series) -> tuple:
    """
    pct_samples_improved: fraction (as a %) of rows (with both raw and
    corrected present) where corrected < raw.
    mean_pct_improvement: mean of (raw - corrected) / raw * 100, computed
    only over rows where both values are present and raw != 0.
    Same convention as evaluate_sequence_ablation.py's _build_row.
    """
    both_present = raw.notna() & corrected.notna()
    if both_present.sum() == 0:
        return np.nan, np.nan

    raw_p = raw[both_present]
    corrected_p = corrected[both_present]

    pct_improved = float((corrected_p < raw_p).mean() * 100.0)

    nonzero_raw = both_present & (raw != 0)
    if nonzero_raw.sum() == 0:
        mean_pct_improvement = np.nan
    else:
        raw_nz = raw[nonzero_raw]
        corrected_nz = corrected[nonzero_raw]
        mean_pct_improvement = float(((raw_nz - corrected_nz) / raw_nz * 100.0).mean())

    return pct_improved, mean_pct_improvement


def build_summary_row(model_name: str, errors: pd.Series, raw: pd.Series = None, corrected: pd.Series = None) -> dict:
    metrics = _metrics(errors.dropna() if hasattr(errors, "dropna") else errors)
    row = {"model": model_name}
    row.update(metrics)
    if raw is not None and corrected is not None:
        pct_improved, mean_pct_improvement = compute_improvement_stats(raw, corrected)
        row["pct_samples_improved"] = pct_improved
        row["mean_pct_improvement"] = mean_pct_improvement
    else:
        row["pct_samples_improved"] = np.nan
        row["mean_pct_improvement"] = np.nan
    return row


def run():
    config.ensure_dirs()

    if not TORCH_AVAILABLE:
        logger.error("PyTorch is not installed; this confirmation script requires it. Aborting.")
        pd.DataFrame().to_csv(f"{config.OUTPUT_DIR}/diagnostics_dedup_confirmation_summary.csv", index=False)
        return

    summary_rows = []

    # --- 1-2: load training data read-only, build in-memory dedup copy ---
    train_path = f"{config.OUTPUT_DIR}/augmented_train_dataset.csv"
    logger.info("Loading (READ-ONLY) training dataset from %s ...", train_path)
    raw_train_df = pd.read_csv(train_path)
    if raw_train_df.empty:
        logger.error("%s is empty. Aborting.", train_path)
        pd.DataFrame().to_csv(f"{config.OUTPUT_DIR}/diagnostics_dedup_confirmation_summary.csv", index=False)
        return
    raw_train_df["current_epoch_utc"] = pd.to_datetime(
        raw_train_df["current_epoch_utc"], utc=True, format="ISO8601"
    )

    try:
        dedup_train_df = build_deduplicated_train_df(raw_train_df)
    except Exception:
        logger.exception("Failed to build deduplicated training dataframe. Aborting.")
        pd.DataFrame().to_csv(f"{config.OUTPUT_DIR}/diagnostics_dedup_confirmation_summary.csv", index=False)
        return

    # --- 3: fixed epoch count for the "original" variant ---
    try:
        n_epochs = load_fixed_epoch_count(VARIANT_NAME)
        logger.info("Using fixed_epoch_count=%d for variant '%s' (from Part C rolling-origin CV).", n_epochs, VARIANT_NAME)
    except Exception:
        logger.exception("Failed to load fixed_epoch_count. Aborting.")
        pd.DataFrame().to_csv(f"{config.OUTPUT_DIR}/diagnostics_dedup_confirmation_summary.csv", index=False)
        return

    feature_cols = tsm.FEATURE_SET_VARIANTS[VARIANT_NAME]

    # --- 4-5: train + save the dedup_test model ---
    dedup_test_trained = False
    try:
        train_result = train_dedup_test_model(dedup_train_df, feature_cols, n_epochs)
        dedup_test_trained = True
        logger.info(
            "Dedup-test training complete: %d epochs, final_train_loss=%.4f, %d training sequences.",
            train_result["n_epochs_trained"], train_result["final_train_loss"], train_result["n_train_sequences"],
        )
    except Exception:
        logger.exception("Dedup-test training FAILED. Will still attempt to evaluate raw SGP4 and the "
                          "existing v2 model for partial comparison.")

    # --- load test set once, shared across both evaluations ---
    test_path = f"{config.OUTPUT_DIR}/test_dataset.csv"
    logger.info("Loading test dataset from %s ...", test_path)
    test_df = pd.read_csv(test_path)
    if test_df.empty:
        logger.error("Test dataset is empty. Aborting.")
        pd.DataFrame().to_csv(f"{config.OUTPUT_DIR}/diagnostics_dedup_confirmation_summary.csv", index=False)
        return
    test_df["current_epoch_utc"] = pd.to_datetime(test_df["current_epoch_utc"], utc=True, format="ISO8601")

    # --- raw SGP4 row ---
    if RAW_COLUMN not in test_df.columns:
        logger.error("Column '%s' not found in test dataset; cannot proceed.", RAW_COLUMN)
        pd.DataFrame().to_csv(f"{config.OUTPUT_DIR}/diagnostics_dedup_confirmation_summary.csv", index=False)
        return
    try:
        summary_rows.append(build_summary_row("raw_sgp4", test_df[RAW_COLUMN]))
        logger.info("raw_sgp4 metrics computed.")
    except Exception:
        logger.exception("Failed computing raw_sgp4 metrics.")

    # --- 7: existing v2 "original" model, unchanged, not retrained ---
    v2_error_col = None
    try:
        test_df, ok, v2_error_col = add_sequence_variant_predictions_v2(test_df, VARIANT_NAME)
        if not ok:
            logger.warning(
                "Existing '%s%s' model artifacts unavailable; that row will be absent from the summary.",
                VARIANT_NAME, V2_SUFFIX,
            )
        else:
            summary_rows.append(build_summary_row(
                f"{VARIANT_NAME}{V2_SUFFIX}",
                test_df[v2_error_col],
                raw=test_df[RAW_COLUMN],
                corrected=test_df[v2_error_col],
            ))
            logger.info("Existing '%s%s' model scored on test set.", VARIANT_NAME, V2_SUFFIX)
    except Exception:
        logger.exception("Failed scoring the existing '%s%s' model; that row will be absent from the summary.",
                          VARIANT_NAME, V2_SUFFIX)

    # --- 6: evaluate the new dedup_test model ---
    dedup_error_col = None
    if dedup_test_trained:
        try:
            test_df, dedup_error_col = add_dedup_test_predictions(test_df)
            summary_rows.append(build_summary_row(
                f"{VARIANT_NAME}{DEDUP_SUFFIX}",
                test_df[dedup_error_col],
                raw=test_df[RAW_COLUMN],
                corrected=test_df[dedup_error_col],
            ))
            logger.info("New '%s%s' model scored on test set.", VARIANT_NAME, DEDUP_SUFFIX)
        except Exception:
            logger.exception("Failed scoring the new '%s%s' model; that row will be absent from the summary.",
                              VARIANT_NAME, DEDUP_SUFFIX)
    else:
        logger.warning("Skipping dedup_test evaluation because training failed earlier.")

    # --- 9: save comparison table ---
    summary_cols = [
        "model", "n_samples", "mae_km", "median_ae_km", "rmse_km", "p90_km", "p95_km", "max_km",
        "pct_samples_improved", "mean_pct_improvement",
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df[[c for c in summary_cols if c in summary_df.columns]]
    summary_path = f"{config.OUTPUT_DIR}/diagnostics_dedup_confirmation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved dedup confirmation summary to %s (%d rows).", summary_path, len(summary_df))

    # --- 10: final console verdict ---
    print("\n=== DEDUP CONFIRMATION TEST: raw SGP4 vs existing original_rolling_cv_v2 vs new original_dedup_test ===")
    if summary_df.empty:
        print("No models could be evaluated -- see log warnings/errors above.")
        logger.info("diagnose_dedup_confirmation_test complete (no rows).")
        return
    print(summary_df.to_string(index=False))

    row_by_model = {r["model"]: r for r in summary_rows}
    raw_mae = row_by_model.get("raw_sgp4", {}).get("mae_km", np.nan)
    v2_mae = row_by_model.get(f"{VARIANT_NAME}{V2_SUFFIX}", {}).get("mae_km", np.nan)
    dedup_mae = row_by_model.get(f"{VARIANT_NAME}{DEDUP_SUFFIX}", {}).get("mae_km", np.nan)

    print("\n--- VERDICT ---")
    if np.isnan(dedup_mae):
        print("Could not evaluate the dedup_test model (see errors above) -- no verdict possible.")
    elif np.isnan(v2_mae):
        print(
            f"original_dedup_test MAE = {dedup_mae:.3f} km, but the existing original_rolling_cv_v2 "
            f"model could not be scored for comparison -- no verdict possible on whether dedup fixed "
            f"the regression vs. that specific model."
        )
    else:
        fixed_vs_v2 = dedup_mae < v2_mae
        also_beats_raw = (not np.isnan(raw_mae)) and (dedup_mae <= raw_mae)
        print(
            f"raw_sgp4 MAE:                {raw_mae:.3f} km\n"
            f"original_rolling_cv_v2 MAE:  {v2_mae:.3f} km\n"
            f"original_dedup_test MAE:     {dedup_mae:.3f} km"
        )
        if fixed_vs_v2 and also_beats_raw:
            print(
                "CONFIRMED: removing duplicate-timestamp rows (training on augmentation_type == "
                "'original' only) improved the sequence model's test MAE relative to the "
                "duplicate-timestamp-contaminated original_rolling_cv_v2 model, AND the dedup_test "
                "model now matches or beats raw SGP4. The duplicate-timestamp training rows look "
                "like the root cause of the regression."
            )
        elif fixed_vs_v2 and not also_beats_raw:
            print(
                "PARTIALLY CONFIRMED: removing duplicate-timestamp rows improved the sequence "
                "model's test MAE relative to original_rolling_cv_v2, but the dedup_test model "
                "still does not beat raw SGP4 -- duplicate timestamps were LIKELY A contributor, "
                "but other factors are probably also at play."
            )
        else:
            print(
                "NOT CONFIRMED: the dedup_test model's MAE is not better than "
                "original_rolling_cv_v2's -- removing duplicate-timestamp rows alone did not fix "
                "the regression, so the root cause likely lies elsewhere (or in addition to this)."
            )

    logger.info("diagnose_dedup_confirmation_test complete.")


if __name__ == "__main__":
    run()
