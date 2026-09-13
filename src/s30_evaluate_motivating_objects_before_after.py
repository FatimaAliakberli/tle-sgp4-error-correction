"""
evaluate_motivating_objects_before_after.py
STANDALONE, EVAL-ONLY, READ-ONLY SCRIPT. 
Automatically identifies the Top-N worst offenders by raw SGP4 p95 error 
and generates before/after metrics and enhanced plots for them.
"""
import logging
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config
import s06_train_sequence_model as tsm
from s05_evaluate_baseline_and_raw_sgp4 import _metrics
from evaluate_sequence_ablation_v1 import add_sequence_variant_predictions
from evaluate_sequence_ablation_v2_predup import add_sequence_variant_predictions_v2, MODEL_SUFFIX as V2_SUFFIX
from s19_evaluate_sequence_ablation_v2_dedup import (
    add_sequence_variant_predictions_v2_dedup,
    DEDUP_MODEL_SUFFIX,
)
from s24_evaluate_dedup_vs_filtered_all_variants import (
    add_checkpoint_predictions,
    TEST_MODELS_DIR,
    FILTERED_MODEL_SUFFIX,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate_motivating_objects")

RAW_COLUMN = "position_error_km"
TOP_N_OBJECTS = 5  # <-- CHANGE THIS to analyze more or fewer objects (e.g., 5 or 10)

OUTPUT_SUMMARY_PATH = f"{config.OUTPUT_DIR}/motivating_objects_before_after_summary.csv"
PLOTS_DIR = config.PLOTS_DIR

def _metrics_for_ids(df, error_col, object_ids):
    if error_col not in df.columns:
        return None
    sub = df[df["object_id"].isin(object_ids)]
    series = sub[error_col].dropna()
    if series.empty:
        return None
    return _metrics(series)

def _add_rows(rows, model, era, object_label, metrics_dict):
    if metrics_dict is None:
        return
    row = {"model": model, "era": era, "object_id": object_label}
    row.update(metrics_dict)
    rows.append(row)

def run():
    test_path = f"{config.OUTPUT_DIR}/test_dataset.csv"
    logger.info("Loading test dataset (read-only) from %s ...", test_path)
    test_df = pd.read_csv(test_path)
    
    if test_df.empty:
        logger.warning("Test dataset is empty. Nothing to evaluate.")
        return
        
    test_df["current_epoch_utc"] = pd.to_datetime(
        test_df["current_epoch_utc"], utc=True, format="ISO8601"
    )
    test_df["object_id"] = test_df["object_id"].astype(str)

    # ------------------------------------------------------------------
    # AUTOMATICALLY IDENTIFY TOP-N WORST OFFENDERS BY RAW SGP4 P95 ERROR
    # ------------------------------------------------------------------
    logger.info("Identifying top %d worst offenders by raw SGP4 p95 error...", TOP_N_OBJECTS)
    raw_p95_per_object = test_df.groupby("object_id")[RAW_COLUMN].quantile(0.95)
    top_n_object_ids = raw_p95_per_object.sort_values(ascending=False).head(TOP_N_OBJECTS).index.tolist()
    
    # Create a clean label mapping for filenames and display (replaces spaces/slashes)
    OBJECT_TARGETS = {
        obj_id.replace(" ", "_").replace("/", "_").replace("-", "_"): obj_id 
        for obj_id in top_n_object_ids
    }
    
    logger.info("Selected objects for analysis: %s", list(OBJECT_TARGETS.values()))

    # Restrict all further work to just these objects for efficiency
    df = test_df[test_df["object_id"].isin(top_n_object_ids)].copy()
    if df.empty:
        logger.error("None of the selected objects were found in the test dataset.")
        return

    rows = []

    # ------------------------------------------------------------------
    # 1. Raw SGP4, per object
    # ------------------------------------------------------------------
    logger.info("Evaluating raw SGP4 (no correction), per object...")
    for clean_label, obj_id in OBJECT_TARGETS.items():
        try:
            m = _metrics_for_ids(df, RAW_COLUMN, [obj_id])
            _add_rows(rows, "raw_sgp4", "raw", clean_label, m)
        except Exception:
            logger.exception("Failed computing raw SGP4 metrics for '%s'.", clean_label)

    # ------------------------------------------------------------------
    # 2. BEFORE: v1 "sequence_<variant>.pt" artifacts
    # ------------------------------------------------------------------
    for variant_name in tsm.FEATURE_SET_VARIANTS:
        display = f"sequence_{variant_name}"
        try:
            df_eval, ok, error_col = add_sequence_variant_predictions(df.copy(), variant_name)
        except Exception:
            logger.exception("Failed evaluating BEFORE (v1) variant '%s'.", display)
            continue
        if not ok:
            logger.warning("BEFORE (v1) checkpoint for '%s' not found/usable. Skipping.", display)
            continue
        for clean_label, obj_id in OBJECT_TARGETS.items():
            try:
                m = _metrics_for_ids(df_eval, error_col, [obj_id])
                if m is None:
                    logger.warning("No overlapping rows for BEFORE (v1) '%s' on object '%s'.", display, clean_label)
                _add_rows(rows, display, "before", clean_label, m)
            except Exception:
                logger.exception("Failed scoring BEFORE (v1) '%s' on object '%s'.", display, clean_label)

    # ------------------------------------------------------------------
    # 3. BEFORE: old, duplicate-timestamp-contaminated "_rolling_cv_v2" artifacts
    # ------------------------------------------------------------------
    for variant_name in tsm.FEATURE_SET_VARIANTS:
        display = f"sequence_{variant_name}{V2_SUFFIX}"
        try:
            df_eval, ok, error_col = add_sequence_variant_predictions_v2(df.copy(), variant_name)
        except Exception:
            logger.exception("Failed evaluating BEFORE (old v2) variant '%s'.", display)
            continue
        if not ok:
            logger.warning("BEFORE (old v2) checkpoint for '%s' not found/usable. Skipping.", display)
            continue
        for clean_label, obj_id in OBJECT_TARGETS.items():
            try:
                m = _metrics_for_ids(df_eval, error_col, [obj_id])
                if m is None:
                    logger.warning("No overlapping rows for BEFORE (old v2) '%s' on object '%s'.", display, clean_label)
                _add_rows(rows, display, "before", clean_label, m)
            except Exception:
                logger.exception("Failed scoring BEFORE (old v2) '%s' on object '%s'.", display, clean_label)

    # ------------------------------------------------------------------
    # 4. AFTER: dedup-only "_rolling_cv_v2_dedup" artifacts
    # ------------------------------------------------------------------
    for variant_name in tsm.FEATURE_SET_VARIANTS:
        display = f"sequence_{variant_name}{DEDUP_MODEL_SUFFIX}"
        try:
            df_eval, ok, error_col = add_sequence_variant_predictions_v2_dedup(df.copy(), variant_name)
        except Exception:
            logger.exception("Failed evaluating AFTER (dedup-only) variant '%s'.", display)
            continue
        if not ok:
            logger.warning("AFTER (dedup-only) checkpoint for '%s' not found/usable. Skipping.", display)
            continue
        for clean_label, obj_id in OBJECT_TARGETS.items():
            try:
                m = _metrics_for_ids(df_eval, error_col, [obj_id])
                if m is None:
                    logger.warning("No overlapping rows for AFTER (dedup-only) '%s' on object '%s'.", display, clean_label)
                _add_rows(rows, display, "after", clean_label, m)
            except Exception:
                logger.exception("Failed scoring AFTER (dedup-only) '%s' on object '%s'.", display, clean_label)

    # ------------------------------------------------------------------
    # 5. AFTER: flag-filtered "_postwindow_oversample_filtered_test" artifacts
    # ------------------------------------------------------------------
    for variant_name in tsm.FEATURE_SET_VARIANTS:
        display = f"sequence_{variant_name}{FILTERED_MODEL_SUFFIX}"
        model_path = f"{TEST_MODELS_DIR}/sequence_model_{variant_name}{FILTERED_MODEL_SUFFIX}.pt"
        scaler_path = f"{TEST_MODELS_DIR}/sequence_scaler_{variant_name}{FILTERED_MODEL_SUFFIX}.joblib"
        if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
            logger.warning(
                "AFTER (flag-filtered) checkpoint for '%s' not found (expected %s, %s). Skipping.",
                display, model_path, scaler_path,
            )
            continue
        try:
            df_eval, ok, error_col = add_checkpoint_predictions(df.copy(), model_path, scaler_path, f"{variant_name}{FILTERED_MODEL_SUFFIX}")
        except Exception:
            logger.exception("Failed evaluating AFTER (flag-filtered) variant '%s'.", display)
            continue
        if not ok:
            logger.warning("AFTER (flag-filtered) checkpoint for '%s' not usable (sequence build failed). Skipping.", display)
            continue
        for clean_label, obj_id in OBJECT_TARGETS.items():
            try:
                m = _metrics_for_ids(df_eval, error_col, [obj_id])
                if m is None:
                    logger.warning("No overlapping rows for AFTER (flag-filtered) '%s' on object '%s'.", display, clean_label)
                _add_rows(rows, display, "after", clean_label, m)
            except Exception:
                logger.exception("Failed scoring AFTER (flag-filtered) '%s' on object '%s'.", display, clean_label)

    # ------------------------------------------------------------------
    # Assemble, save, and report
    # ------------------------------------------------------------------
    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        logger.error("No rows could be evaluated at all. Writing an empty summary.")
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        summary_df.to_csv(OUTPUT_SUMMARY_PATH, index=False)
        return

    ordered_cols = [
        "model", "era", "object_id",
        "n_samples", "mae_km", "median_ae_km", "rmse_km", "p90_km", "p95_km", "max_km",
    ]
    summary_df = summary_df[[c for c in ordered_cols if c in summary_df.columns]]
    era_rank = {"raw": 0, "before": 1, "after": 2}
    summary_df["_era_rank"] = summary_df["era"].map(era_rank).fillna(3)
    summary_df = summary_df.sort_values(["object_id", "_era_rank", "model"]).drop(columns=["_era_rank"])
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    summary_df.to_csv(OUTPUT_SUMMARY_PATH, index=False)
    logger.info("Saved before/after summary to %s (%d rows).", OUTPUT_SUMMARY_PATH, len(summary_df))
    
    print("\n=== MOTIVATING-OBJECT BEFORE/AFTER COMPARISON (test set) ===")
    print(summary_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Per-object verdict and enhanced plotting
    # ------------------------------------------------------------------
    print("\n=== PER-OBJECT VERDICT ===")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    for clean_label, obj_id in OBJECT_TARGETS.items():
        obj_df = summary_df[summary_df["object_id"] == clean_label]
        if obj_df.empty:
            print(f"\n{clean_label}: no data available.")
            continue
            
        raw_rows = obj_df[obj_df["era"] == "raw"]
        before_rows = obj_df[obj_df["era"] == "before"]
        after_rows = obj_df[obj_df["era"] == "after"]
        
        raw_p95 = raw_rows["p95_km"].iloc[0] if len(raw_rows) > 0 else np.nan
        best_before_row = before_rows.loc[before_rows["p95_km"].idxmin()] if len(before_rows) > 0 else None
        best_after_row = after_rows.loc[after_rows["p95_km"].idxmin()] if len(after_rows) > 0 else None
        
        print(f"\n{clean_label}:")
        print(f"  raw SGP4 p95:            {raw_p95:.3f} km" if not np.isnan(raw_p95) else "  raw SGP4 p95:            N/A")
        if best_before_row is not None:
            print(f"  best BEFORE model:       {best_before_row['model']} (p95 = {best_before_row['p95_km']:.3f} km)")
        if best_after_row is not None:
            print(f"  best AFTER model:        {best_after_row['model']} (p95 = {best_after_row['p95_km']:.3f} km)")
            
        if best_after_row is not None and not np.isnan(raw_p95):
            beats_raw = best_after_row["p95_km"] < raw_p95
            print(f"  does ANY after model beat raw SGP4 p95?          {'YES' if beats_raw else 'NO'} "
                  f"({best_after_row['p95_km']:.3f} vs {raw_p95:.3f} km)")
        if best_after_row is not None and best_before_row is not None:
            beats_before = best_after_row["p95_km"] < best_before_row["p95_km"]
            print(f"  does ANY after model beat the best before model? {'YES' if beats_before else 'NO'} "
                  f"({best_after_row['p95_km']:.3f} vs {best_before_row['p95_km']:.3f} km)")

        # ------------------------------------------------------------------
        # Generate enhanced bar chart for this object
        # ------------------------------------------------------------------
        bar_labels, bar_values = [], []
        if len(raw_rows) > 0:
            bar_labels.append("raw_sgp4")
            bar_values.append(raw_rows["p95_km"].iloc[0])
        if len(before_rows) > 0:
            best_before_row = before_rows.loc[before_rows["p95_km"].idxmin()]
            bar_labels.append(f"best_before\n({best_before_row['model']})")
            bar_values.append(best_before_row["p95_km"])
        for _, r in after_rows.iterrows():
            bar_labels.append(r["model"])
            bar_values.append(r["p95_km"])
            
        if not bar_labels:
            logger.warning("Skipping plot for '%s': nothing to plot.", clean_label)
            continue
            
        fig, ax = plt.subplots(figsize=(max(9.0, 0.9 * len(bar_labels)), 6.0))
        colors = []
        for lbl in bar_labels:
            if lbl == "raw_sgp4":
                colors.append("tab:gray")
            elif lbl.startswith("best_before"):
                colors.append("tab:orange")
            else:
                colors.append("tab:blue")
                
        ax.bar(range(len(bar_labels)), bar_values, color=colors)
        ax.set_xticks(range(len(bar_labels)))
        ax.set_xticklabels(bar_labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("p95 position error (km)")
        ax.set_title(f"{clean_label.replace('_', ' ')}: p95 error, raw SGP4 vs. models")
        
        # Add raw SGP4 reference line and checkmarks for improvements
        if not np.isnan(raw_p95):
            ax.axhline(y=raw_p95, color='red', linestyle='dashed', linewidth=2, zorder=10)
            for i, (lbl, val) in enumerate(zip(bar_labels, bar_values)):
                if lbl != "raw_sgp4" and val < raw_p95:
                    ax.text(i, val + (raw_p95 * 0.02), "✓", ha='center', va='bottom', 
                            fontsize=14, color="#2ca02c", fontweight="bold", zorder=11)
                    
        fig.tight_layout()
        plot_path = f"{PLOTS_DIR}/motivating_objects_before_after_{clean_label}_enhanced.png"
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved enhanced plot: %s", plot_path)

    logger.info("evaluate_motivating_objects complete.")

if __name__ == "__main__":
    run()
