"""
generate_ablation_summary_v2.py

PART D.3: Reads output/ablation_summary.csv (v1) and
output/ablation_summary_v2.csv (v2, rolling-origin-CV-selected), and
writes a plain-language output/ablation_summary_v2.txt answering exactly
the three questions from the spec:

    (a) How do the 2 new cycle-phase variants compare to their
        non-cycle-phase counterparts? (v2-internal comparison, since v1
        never had these variants)
    (b) Did rolling-origin model selection change results for the OTHER
        variants (orbital_jump, tracking_cadence, original, all_features)
        even though they didn't get any new features? This isolates how
        much of any v1->v2 change is due to better model selection (a
        different, cross-validated epoch count instead of early-stopping
        against a single arbitrary 2024 val year) vs. the new feature
        itself.
    (c) An explicit before/after (v1 vs v2) comparison against v1's
        numbers for every variant that exists in both files.

This is a standalone, re-runnable summarizer -- it does NOT retrain or
re-evaluate anything, matching the existing generate_ablation_summary.py
convention. It also does not modify ablation_summary.csv, ablation_summary.txt,
or ablation_summary_v2.csv.

Output:
    output/ablation_summary_v2.txt
"""

import logging

import numpy as np
import pandas as pd

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("s20_summarize_ablation_v1_and_v2")

METRIC_FOR_COMPARISON = "mae_km"

# v1 model_variant name -> v2 model_variant name, for the 5 variants that
# exist in BOTH files (the 2 new cycle-phase variants only exist in v2).
COMMON_VARIANTS = {
    "original": ("sequence_original", "sequence_original_rolling_cv_v2"),
    "plus_orbital_jump": ("sequence_plus_orbital_jump", "sequence_plus_orbital_jump_rolling_cv_v2"),
    "plus_space_weather_full": ("sequence_plus_space_weather_full", "sequence_plus_space_weather_full_rolling_cv_v2"),
    "plus_tracking_cadence": ("sequence_plus_tracking_cadence", "sequence_plus_tracking_cadence_rolling_cv_v2"),
    "all_features": ("sequence_all_features", "sequence_all_features_rolling_cv_v2"),
}

# Variants that received NO new features between v1 and v2 -- any v1->v2
# change for these is attributable entirely to rolling-origin CV changing
# the fixed epoch count (better model selection), not to new inputs.
# ("original" is included too, as the zero-added-features reference point.)
NO_NEW_FEATURE_VARIANTS = ["original", "plus_orbital_jump", "plus_tracking_cadence", "all_features"]

# The 2 new Part B cycle-phase variants, each paired with its non-cycle-phase
# counterpart, v2-only (v1 has no cycle-phase columns at all).
CYCLE_PHASE_COMPARISONS = [
    (
        "plus_space_weather_full_and_cycle_phase vs plus_space_weather_full",
        "sequence_plus_space_weather_full_and_cycle_phase_rolling_cv_v2",
        "sequence_plus_space_weather_full_rolling_cv_v2",
    ),
    (
        "all_features_plus_cycle_phase vs all_features",
        "sequence_all_features_plus_cycle_phase_rolling_cv_v2",
        "sequence_all_features_rolling_cv_v2",
    ),
]

RAW_VARIANT = "raw_sgp4"
BASELINE_VARIANT = "baseline_tree"


def _overall_row(df, model_variant):
    sub = df[(df["model_variant"] == model_variant) & (df["group_type"] == "overall")]
    if sub.empty:
        return None
    return sub.iloc[0]


def _pct_change(new_val, old_val):
    """Percent change from old_val to new_val, where POSITIVE means new_val
    is lower/better (an improvement) than old_val."""
    if old_val is None or pd.isna(old_val) or old_val == 0 or pd.isna(new_val):
        return np.nan
    return (old_val - new_val) / old_val * 100.0


def run():
    config.ensure_dirs()

    v1_path = f"{config.OUTPUT_DIR}/ablation_summary.csv"
    v2_path = f"{config.OUTPUT_DIR}/ablation_summary_v2.csv"

    try:
        v1_df = pd.read_csv(v1_path)
    except FileNotFoundError:
        logger.warning("%s not found. Run evaluate_sequence_ablation.py (v1) first.", v1_path)
        return
    try:
        v2_df = pd.read_csv(v2_path)
    except FileNotFoundError:
        logger.warning("%s not found. Run evaluate_sequence_ablation_v2.py (Part D.2) first.", v2_path)
        return

    if v1_df.empty or v2_df.empty:
        logger.warning("One or both of %s / %s is empty. Nothing to summarize.", v1_path, v2_path)
        return

    lines = []
    lines.append("Rolling-Origin CV Ablation Summary (v2) -- Before/After Comparison")
    lines.append("=" * 68)
    lines.append("")
    lines.append(f"Sources: {v1_path} (v1, single fixed-2024-val-year selection)")
    lines.append(f"         {v2_path} (v2, rolling-origin cross-validation selection)")
    lines.append(f"Comparison metric: {METRIC_FOR_COMPARISON} (lower is better)")
    lines.append("All comparisons below use the 'overall' (all-horizons-combined) rows.")
    lines.append("")

    # -----------------------------------------------------------------
    # Sanity check: raw SGP4 and baseline tree should be numerically
    # identical (or very close) between v1 and v2, since neither is
    # retrained and both files evaluate against the same test_dataset.csv.
    # -----------------------------------------------------------------
    lines.append("Sanity check: unchanged models (raw SGP4, baseline tree)")
    lines.append("-" * 58)
    for variant in [RAW_VARIANT, BASELINE_VARIANT]:
        r1 = _overall_row(v1_df, variant)
        r2 = _overall_row(v2_df, variant)
        if r1 is None or r2 is None:
            lines.append(f"  {variant}: not available in both files -- cannot sanity-check.")
            continue
        mae1, mae2 = r1[METRIC_FOR_COMPARISON], r2[METRIC_FOR_COMPARISON]
        diff = abs(mae1 - mae2)
        flag = "" if diff < 1e-6 else "  <-- UNEXPECTED: these should be identical (same unretrained model/data)"
        lines.append(f"  {variant}: v1 mae_km={mae1:.6f}, v2 mae_km={mae2:.6f} (diff={diff:.6f}){flag}")
    lines.append("")

    # -----------------------------------------------------------------
    # (a) The 2 new cycle-phase variants vs. their non-cycle-phase
    # counterparts (v2-internal; these variants don't exist in v1 at all).
    # -----------------------------------------------------------------
    lines.append("(a) Do the new solar-cycle-context features help? (v2-only comparison)")
    lines.append("-" * 72)
    for label, with_cycle_name, without_cycle_name in CYCLE_PHASE_COMPARISONS:
        row_with = _overall_row(v2_df, with_cycle_name)
        row_without = _overall_row(v2_df, without_cycle_name)
        if row_with is None or row_without is None:
            lines.append(f"  {label}: NOT AVAILABLE in {v2_path} -- did Part D.1/D.2 run for both variants?")
            continue
        mae_with = row_with[METRIC_FOR_COMPARISON]
        mae_without = row_without[METRIC_FOR_COMPARISON]
        imp = _pct_change(mae_with, mae_without)
        verdict = "IMPROVES on" if imp > 0 else "does NOT improve on"
        lines.append(
            f"  {label}:\n"
            f"    with cycle-phase:    mae_km={mae_with:.4f} ({with_cycle_name})\n"
            f"    without cycle-phase: mae_km={mae_without:.4f} ({without_cycle_name})\n"
            f"    => Adding solar-cycle context {verdict} the base feature set ({imp:+.1f}% change in MAE)."
        )
        lines.append("")

    # -----------------------------------------------------------------
    # (b) Did rolling-origin CV change results for variants that got NO
    # new features? This isolates "better model selection" from "new
    # feature effect".
    # -----------------------------------------------------------------
    lines.append("(b) Did better model selection alone (no new features) change results?")
    lines.append("-" * 72)
    lines.append(
        "  These variants have the EXACT SAME feature columns in v1 and v2 -- any\n"
        "  change below comes purely from rolling-origin CV choosing a different\n"
        "  (cross-validated) epoch count, instead of early-stopping against a\n"
        "  single, possibly-unrepresentative, fixed 2024 validation year."
    )
    lines.append("")
    no_feature_changes = []
    for key in NO_NEW_FEATURE_VARIANTS:
        v1_name, v2_name = COMMON_VARIANTS[key]
        r1 = _overall_row(v1_df, v1_name)
        r2 = _overall_row(v2_df, v2_name)
        if r1 is None or r2 is None:
            lines.append(f"  {key}: NOT AVAILABLE in both files -- skipping.")
            continue
        mae1, mae2 = r1[METRIC_FOR_COMPARISON], r2[METRIC_FOR_COMPARISON]
        change = _pct_change(mae2, mae1)
        no_feature_changes.append((key, change))
        direction = "improved" if change > 0 else ("got worse" if change < 0 else "unchanged")
        lines.append(
            f"  {key}: v1 mae_km={mae1:.4f} -> v2 mae_km={mae2:.4f} "
            f"({change:+.1f}%, {direction} under rolling-origin CV selection alone)"
        )
    lines.append("")
    valid_changes = [c for _, c in no_feature_changes if not pd.isna(c)]
    if valid_changes:
        mean_change = float(np.mean(valid_changes))
        if mean_change > 1.0:
            lines.append(
                f"  => On average, model selection ALONE improved these {len(valid_changes)} "
                f"no-new-feature variants by {mean_change:+.1f}% MAE. This means a meaningful "
                "share of any v1->v2 improvement elsewhere is likely due to better model "
                "selection, not new features -- interpret feature-driven claims with this in mind."
            )
        elif mean_change < -1.0:
            lines.append(
                f"  => On average, model selection ALONE made these {len(valid_changes)} "
                f"no-new-feature variants {mean_change:+.1f}% worse under rolling-origin CV. "
                "This suggests the CV-selected epoch counts may be more conservative "
                "(e.g. earlier stopping) than the single-fixed-year selection was."
            )
        else:
            lines.append(
                f"  => On average, model selection alone had a small effect on these "
                f"{len(valid_changes)} no-new-feature variants ({mean_change:+.1f}% MAE change). "
                "Most of any v1->v2 improvement seen elsewhere is more likely attributable "
                "to the new features themselves rather than to model selection."
            )
    lines.append("")

    # -----------------------------------------------------------------
    # (c) Explicit before/after (v1 vs v2) for every variant in both files.
    # -----------------------------------------------------------------
    lines.append("(c) Explicit before/after (v1 vs v2) for every variant present in both")
    lines.append("-" * 72)
    for key, (v1_name, v2_name) in COMMON_VARIANTS.items():
        r1 = _overall_row(v1_df, v1_name)
        r2 = _overall_row(v2_df, v2_name)
        if r1 is None or r2 is None:
            lines.append(f"  {key}: NOT AVAILABLE in both files -- skipping.")
            continue
        mae1, mae2 = r1[METRIC_FOR_COMPARISON], r2[METRIC_FOR_COMPARISON]
        change = _pct_change(mae2, mae1)
        lines.append(
            f"  {key}:\n"
            f"    v1 ({v1_name}): mae_km={mae1:.4f}, n_samples={int(r1['n_samples'])}\n"
            f"    v2 ({v2_name}): mae_km={mae2:.4f}, n_samples={int(r2['n_samples'])}\n"
            f"    change: {change:+.1f}% ({'improvement' if change > 0 else 'regression' if change < 0 else 'no change'})"
        )
        lines.append("")

    # Best model overall, v2, for quick reference.
    v2_overall = v2_df[v2_df["group_type"] == "overall"].copy()
    if not v2_overall.empty:
        best_row = v2_overall.loc[v2_overall[METRIC_FOR_COMPARISON].idxmin()]
        lines.append("Best model overall in v2 (all models, by overall MAE)")
        lines.append("-" * 57)
        lines.append(f"  {best_row['model_variant']}: mae_km={best_row[METRIC_FOR_COMPARISON]:.4f}")
        lines.append("")

    lines.append("(Generated by generate_ablation_summary_v2.py from ablation_summary.csv and")
    lines.append(" ablation_summary_v2.csv. Neither v1 file was modified. See")
    lines.append(" output/plots/ablation_comparison_v2.png and")
    lines.append(" output/plots/ablation_comparison_sequence_only_v2.png for the corresponding charts.)")

    out_path = f"{config.OUTPUT_DIR}/ablation_summary_v2.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("Wrote %s", out_path)
    print("\n".join(lines))


if __name__ == "__main__":
    run()
