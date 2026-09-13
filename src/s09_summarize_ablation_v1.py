"""
generate_ablation_summary.py

Reads output/ablation_summary.csv (produced by evaluate_sequence_ablation.py)
and writes a short, plain-language output/ablation_summary.txt, in a style
similar to the existing error_attribution_summary.txt, answering:

    - Which single feature group produced the largest improvement over the
      original sequence model ("sequence_original")?
    - Does "sequence_all_features" beat every individual single-group
      addition, or does combining groups underperform some single addition?
    - How does each model variant compare to raw SGP4 and to the baseline
      tree model?

This is a standalone, re-runnable summarizer: it does NOT retrain or
re-evaluate anything. It only reads the CSV that evaluate_sequence_ablation.py
already wrote and reports on it, so it can be run any time after that script
has produced output/ablation_summary.csv (including as a separate step,
re-run as many times as you like, e.g. after re-running the ablation with
different data).

Output:
    output/ablation_summary.txt
"""

import logging

import numpy as np
import pandas as pd

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("s09_summarize_ablation_v1")

# Model-variant names, matching evaluate_sequence_ablation.py's
# VARIANT_DISPLAY_NAMES / FULL_MODEL_ORDER.
RAW_VARIANT = "raw_sgp4"
BASELINE_VARIANT = "baseline_tree"
SEQ_ORIGINAL_VARIANT = "sequence_original"
SEQ_SINGLE_ADD_VARIANTS = {
    "sequence_plus_orbital_jump": "orbital-element-jump features",
    "sequence_plus_space_weather_full": "space-weather lag/rolling/flag features",
    "sequence_plus_tracking_cadence": "tracking-cadence/density features",
}
SEQ_ALL_VARIANT = "sequence_all_features"

# Primary metric used for all "which is better" comparisons in this report.
METRIC_FOR_COMPARISON = "mae_km"


def _overall_row(df, model_variant):
    """Return the single 'overall' (all-horizons-combined) row for a model
    variant, or None if it isn't present in the summary CSV."""
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

    summary_path = f"{config.OUTPUT_DIR}/ablation_summary.csv"
    try:
        df = pd.read_csv(summary_path)
    except FileNotFoundError:
        logger.warning(
            "%s not found. Run evaluate_sequence_ablation.py first to produce it.", summary_path
        )
        return

    if df.empty:
        logger.warning("%s is empty. Nothing to summarize.", summary_path)
        return

    lines = []
    lines.append("Sequence Model Feature-Set Ablation Summary")
    lines.append("=" * 44)
    lines.append("")
    lines.append(f"Source: {summary_path}")
    lines.append(f"Comparison metric: {METRIC_FOR_COMPARISON} (lower is better)")
    lines.append("")

    raw_row = _overall_row(df, RAW_VARIANT)
    baseline_row = _overall_row(df, BASELINE_VARIANT)
    orig_row = _overall_row(df, SEQ_ORIGINAL_VARIANT)
    all_row = _overall_row(df, SEQ_ALL_VARIANT)

    # -----------------------------------------------------------------
    # Section 1: overall comparison of every variant vs raw SGP4 and vs
    # the baseline tree model, using the "overall" (all-horizons-combined)
    # rows.
    # -----------------------------------------------------------------
    lines.append("Overall comparison (test set, all horizons combined)")
    lines.append("-" * 53)
    if raw_row is None:
        lines.append("  Raw SGP4 row not found in ablation_summary.csv -- cannot compute comparisons.")
    else:
        raw_mae = raw_row[METRIC_FOR_COMPARISON]
        baseline_mae = baseline_row[METRIC_FOR_COMPARISON] if baseline_row is not None else None

        lines.append(
            f"  Raw SGP4 (no correction): mae_km={raw_mae:.4f}, n_samples={int(raw_row['n_samples'])}"
        )
        if baseline_row is not None:
            imp_vs_raw = _pct_change(baseline_mae, raw_mae)
            lines.append(
                f"  Baseline tree model:      mae_km={baseline_mae:.4f} ({imp_vs_raw:+.1f}% vs raw SGP4)"
            )
        else:
            lines.append("  Baseline tree model:      NOT AVAILABLE in ablation_summary.csv")
        lines.append("")

        seq_variant_order = [SEQ_ORIGINAL_VARIANT] + list(SEQ_SINGLE_ADD_VARIANTS.keys()) + [SEQ_ALL_VARIANT]
        for variant in seq_variant_order:
            row = _overall_row(df, variant)
            if row is None:
                lines.append(f"  {variant}: NOT AVAILABLE in ablation_summary.csv")
                continue
            mae = row[METRIC_FOR_COMPARISON]
            imp_vs_raw = _pct_change(mae, raw_mae)
            line = f"  {variant}: mae_km={mae:.4f} ({imp_vs_raw:+.1f}% vs raw SGP4"
            if baseline_row is not None:
                imp_vs_baseline = _pct_change(mae, baseline_mae)
                line += f", {imp_vs_baseline:+.1f}% vs baseline tree model"
            line += ")"
            lines.append(line)
        lines.append("")

    # -----------------------------------------------------------------
    # Section 2: which single feature group helped the sequence model
    # most, relative to "sequence_original"?
    # -----------------------------------------------------------------
    lines.append("Which single feature group helped the sequence model most?")
    lines.append("-" * 60)
    if orig_row is None:
        lines.append(f"  '{SEQ_ORIGINAL_VARIANT}' row not found -- cannot compute single-group improvements.")
    else:
        orig_mae = orig_row[METRIC_FOR_COMPARISON]
        lines.append(f"  Baseline for this comparison: '{SEQ_ORIGINAL_VARIANT}' mae_km={orig_mae:.4f}")

        single_add_results = []
        for variant, description in SEQ_SINGLE_ADD_VARIANTS.items():
            row = _overall_row(df, variant)
            if row is None:
                lines.append(f"  + {description} ({variant}): NOT AVAILABLE")
                continue
            mae = row[METRIC_FOR_COMPARISON]
            imp = _pct_change(mae, orig_mae)
            single_add_results.append((variant, description, mae, imp))
            lines.append(
                f"  + {description} ({variant}): mae_km={mae:.4f} ({imp:+.1f}% vs '{SEQ_ORIGINAL_VARIANT}')"
            )

        valid_results = [r for r in single_add_results if not pd.isna(r[3])]
        lines.append("")
        if valid_results:
            best = max(valid_results, key=lambda r: r[3])  # largest positive % improvement
            if best[3] > 0:
                lines.append(
                    f"  => Largest single-group improvement: {best[1]} ({best[0]}), "
                    f"{best[3]:+.1f}% lower MAE than the original feature set."
                )
            else:
                lines.append(
                    "  => No single feature group improved on the original feature set "
                    "(every individual addition increased MAE). Least-harmful addition: "
                    f"{best[1]} ({best[0]}, {best[3]:+.1f}%)."
                )
        else:
            lines.append("  => Could not determine the best single feature group (missing data).")
        lines.append("")

    # -----------------------------------------------------------------
    # Section 3: does combining every feature group ("all_features") beat
    # every individual single-group addition, or does combining them
    # underperform some individual addition?
    # -----------------------------------------------------------------
    lines.append("Does combining all feature groups beat every individual addition?")
    lines.append("-" * 68)
    if all_row is None:
        lines.append(f"  '{SEQ_ALL_VARIANT}' row not found -- cannot answer this question.")
    else:
        all_mae = all_row[METRIC_FOR_COMPARISON]
        lines.append(f"  '{SEQ_ALL_VARIANT}' mae_km={all_mae:.4f}")

        comparisons = []
        for variant, description in SEQ_SINGLE_ADD_VARIANTS.items():
            row = _overall_row(df, variant)
            if row is None:
                continue
            mae = row[METRIC_FOR_COMPARISON]
            comparisons.append((variant, description, mae, all_mae < mae))

        if not comparisons:
            lines.append("  No individual single-group variants available to compare against.")
        else:
            for variant, description, mae, beats in comparisons:
                verdict = "beats" if beats else "does NOT beat"
                lines.append(
                    f"  '{SEQ_ALL_VARIANT}' {verdict} '{variant}' ({all_mae:.4f} vs {mae:.4f} km MAE)"
                )
            lines.append("")

            beats_all = all(c[3] for c in comparisons)
            if beats_all:
                lines.append(
                    f"  => '{SEQ_ALL_VARIANT}' beat every individual single-group addition: "
                    "the feature groups' benefits appear additive (or at least non-conflicting)."
                )
            else:
                worse_than = [c for c in comparisons if not c[3]]
                worse_names = ", ".join(f"{d} ({v})" for v, d, _, _ in worse_than)
                lines.append(
                    f"  => '{SEQ_ALL_VARIANT}' underperformed at least one individual addition "
                    f"({worse_names}). This suggests some feature groups may interact "
                    "negatively (redundant or conflicting signal) rather than combining cleanly, "
                    "and the single best-performing addition may be preferable to using all of them."
                )
        lines.append("")

    lines.append("(Generated by generate_ablation_summary.py from output/ablation_summary.csv.")
    lines.append(" All comparisons above use the 'overall' (all-horizons-combined) mae_km values;")
    lines.append(" see ablation_summary.csv for the full by-horizon breakdown, and")
    lines.append(" output/plots/ablation_comparison.png / ablation_comparison_sequence_only.png")
    lines.append(" for the corresponding charts.)")

    out_path = f"{config.OUTPUT_DIR}/ablation_summary.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("Wrote %s", out_path)
    print("\n".join(lines))


if __name__ == "__main__":
    run()
