"""
run_sequence_ablation.py

Driver for the sequence-model feature-set ablation study.

Loops over every entry in train_sequence_model.FEATURE_SET_VARIANTS and
trains each one by calling the refactored, variant-parameterized
train_sequence_model.run(variant_name) from Step 2. Architecture, loss,
optimizer, LR, batch size, early stopping, and RANDOM_SEED are unchanged
across variants -- this script only controls WHICH variant runs and WHEN.

Each variant's training is wrapped in its own try/except so that one
variant failing (e.g. a missing feature column blowing up something
downstream, an out-of-memory error, etc.) does not stop the rest of the
ablation from running. Failures are logged with a full traceback and the
script continues to the next variant.

Logging follows the project's existing convention (logging.basicConfig at
INFO level, a named logger per module) and adds explicit per-stage timing
so there are no silent multi-minute gaps in the log while a variant trains
-- each variant logs a start heartbeat, an elapsed-time heartbeat on
completion or failure, and the running-total elapsed time across the
whole ablation so far.

Outputs (per variant, written by train_sequence_model.run()):
    output/models/sequence_model_<variant_name>.pt
    output/models/sequence_scaler_<variant_name>.joblib
    output/plots/sequence_training_loss_<variant_name>.png

Additionally, this script writes:
    output/sequence_ablation_run_summary.csv
        One row per variant: status (success/failed), elapsed seconds,
        number of step feature columns requested, and (on failure) the
        exception message.
"""

import logging
import time
import traceback
from datetime import datetime, timezone

import pandas as pd

import config
import s06_train_sequence_model
from s06_train_sequence_model import FEATURE_SET_VARIANTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("s07_run_sequence_ablation_v1")


def _now_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _fmt_elapsed(seconds):
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}m{secs:04.1f}s"


def run():
    config.ensure_dirs()

    variant_names = list(FEATURE_SET_VARIANTS.keys())
    logger.info("=== run_sequence_ablation starting at %s ===", _now_str())
    logger.info("Variants to run (%d): %s", len(variant_names), variant_names)

    results = []
    ablation_start = time.monotonic()

    for i, variant_name in enumerate(variant_names, start=1):
        n_requested_cols = len(FEATURE_SET_VARIANTS[variant_name])
        logger.info(
            "--- [%d/%d] Starting variant '%s' at %s (%d requested step feature columns) ---",
            i, len(variant_names), variant_name, _now_str(), n_requested_cols,
        )

        variant_start = time.monotonic()
        status = "success"
        error_message = ""

        try:
            train_sequence_model.run(variant_name=variant_name)
        except Exception as exc:
            status = "failed"
            error_message = str(exc)
            logger.error(
                "--- [%d/%d] Variant '%s' FAILED after %s: %s ---",
                i, len(variant_names), variant_name, _fmt_elapsed(time.monotonic() - variant_start), exc,
            )
            logger.error("Full traceback for variant '%s':\n%s", variant_name, traceback.format_exc())
        else:
            elapsed = time.monotonic() - variant_start
            logger.info(
                "--- [%d/%d] Variant '%s' complete in %s ---",
                i, len(variant_names), variant_name, _fmt_elapsed(elapsed),
            )

        variant_elapsed = time.monotonic() - variant_start
        total_elapsed = time.monotonic() - ablation_start
        logger.info(
            "--- [%d/%d] Running total elapsed: %s (variant took %s) ---",
            i, len(variant_names), _fmt_elapsed(total_elapsed), _fmt_elapsed(variant_elapsed),
        )

        results.append({
            "variant_name": variant_name,
            "status": status,
            "n_requested_step_feature_columns": n_requested_cols,
            "elapsed_seconds": round(variant_elapsed, 2),
            "error_message": error_message,
        })

    total_elapsed = time.monotonic() - ablation_start
    summary_df = pd.DataFrame(results)
    summary_path = f"{config.OUTPUT_DIR}/sequence_ablation_run_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    n_success = int((summary_df["status"] == "success").sum())
    n_failed = int((summary_df["status"] == "failed").sum())
    logger.info(
        "=== run_sequence_ablation complete at %s. Total elapsed: %s. %d/%d variants succeeded, %d failed. ===",
        _now_str(), _fmt_elapsed(total_elapsed), n_success, len(variant_names), n_failed,
    )
    if n_failed > 0:
        failed_names = summary_df.loc[summary_df["status"] == "failed", "variant_name"].tolist()
        logger.warning("Failed variants (see traceback above and %s for details): %s", summary_path, failed_names)
    logger.info("Per-variant run summary written to %s", summary_path)

    print("\n=== SEQUENCE ABLATION RUN SUMMARY ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    run()
