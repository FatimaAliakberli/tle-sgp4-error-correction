"""
run_basic_pipeline.py (originally run_all.py)

Runs the "quick path" pipeline end-to-end, in order:
    1. s01_compute_tle_statistics.py
    2. s02_build_error_attribution_dataset.py
    3. s03_augment_and_split_dataset.py
    4. s04_train_baseline_model.py
    5. s06_train_sequence_model.py   (optional -- skipped gracefully if no torch)
    6. s05_evaluate_baseline_and_raw_sgp4.py

This reproduces the baseline (GBM) vs. raw SGP4 result using the original,
single train/val/test split. It does NOT run the full ablation study,
rolling-origin CV, dedup fix, or flag-filtered oversampling stages -- see
the "Full path" table in the repo README for those (they are intentionally
run by hand, one at a time, rather than chained here).

Each stage is wrapped in try/except so a failure in one stage is logged
but does not necessarily prevent later stages from being attempted (though
later stages may themselves report missing inputs if an earlier stage
failed to produce output).
"""

import logging
import time

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_basic_pipeline")

STAGES = [
    ("compute_tle_statistics", "s01_compute_tle_statistics"),
    ("build_error_attribution_dataset", "s02_build_error_attribution_dataset"),
    ("augment_and_split_dataset", "s03_augment_and_split_dataset"),
    ("train_baseline_model", "s04_train_baseline_model"),
    ("train_sequence_model", "s06_train_sequence_model"),
    ("evaluate_baseline_and_raw_sgp4", "s05_evaluate_baseline_and_raw_sgp4"),
]


def run():
    config.ensure_dirs()
    logger.info("Starting full pipeline run.")

    for stage_name, module_name in STAGES:
        logger.info("=" * 70)
        logger.info("STAGE: %s", stage_name)
        logger.info("=" * 70)
        start = time.time()
        try:
            module = __import__(module_name)
            module.run()
        except Exception as exc:
            logger.error("Stage '%s' failed: %s", stage_name, exc, exc_info=True)
            logger.error("Continuing to next stage...")
        else:
            elapsed = time.time() - start
            logger.info("Stage '%s' completed in %.1fs", stage_name, elapsed)

    logger.info("Pipeline run complete. See the outputs/ directory for results.")


if __name__ == "__main__":
    run()
