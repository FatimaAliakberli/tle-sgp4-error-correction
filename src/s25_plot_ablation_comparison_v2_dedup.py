"""
plot_enhanced_ablation.py
Standalone script to generate enhanced ablation comparison plots with:
1. Maximally distinct colors per model
2. Dashed red reference line at raw SGP4 level per horizon
3. Green arrows pointing UP from below to highlight bars that beat raw SGP4
4. Legend moved to the bottom to avoid overlapping with right-side labels
"""
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("s25_plot_ablation_comparison_v2_dedup")

# Reconstruct the model orders from your evaluation script
from s06_train_sequence_model import FEATURE_SET_VARIANTS
from evaluate_sequence_ablation_v2_predup import VARIANT_DISPLAY_NAMES_V2

DEDUP_MODEL_SUFFIX = "_rolling_cv_v2_dedup"
VARIANT_DISPLAY_NAMES_DEDUP = {
    variant: f"sequence_{variant}{DEDUP_MODEL_SUFFIX}" for variant in FEATURE_SET_VARIANTS
}

OLD_SEQUENCE_ORDER = list(VARIANT_DISPLAY_NAMES_V2.values())
DEDUP_SEQUENCE_ORDER = list(VARIANT_DISPLAY_NAMES_DEDUP.values())
FULL_MODEL_ORDER = ["raw_sgp4", "baseline_tree"] + OLD_SEQUENCE_ORDER + DEDUP_SEQUENCE_ORDER


def get_maximally_distinct_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        sat = 0.85 if i % 2 == 0 else 0.65
        val = 0.90 if i % 3 != 1 else 0.70
        colors.append(mcolors.hsv_to_rgb([hue, sat, val]))
    return np.array(colors)


def plot_grouped_bar_mae_with_arrows(df, model_order, title, save_path, baseline_model="raw_sgp4"):
    # Increased height slightly to accommodate the bottom legend comfortably
    fig, ax = plt.subplots(figsize=(18, 10))

    plot_df = df[df["group_type"] == "by_horizon"].copy()
    plot_df["model_variant"] = pd.Categorical(
        plot_df["model_variant"], categories=model_order, ordered=True
    )
    plot_df = plot_df.dropna(subset=["mae_km"])

    horizons = sorted(plot_df["horizon_hours"].unique())
    n_models = len(model_order)
    x_positions = np.arange(len(horizons))

    distinct_colors = get_maximally_distinct_colors(n_models)
    color_map = {model: distinct_colors[i] for i, model in enumerate(model_order)}
    color_map["raw_sgp4"] = np.array([0.15, 0.15, 0.15, 1.0])
    color_map["baseline_tree"] = np.array([0.55, 0.55, 0.55, 1.0])

    group_width = 0.80
    bar_width = group_width / n_models
    offset_start = -group_width / 2 + bar_width / 2

    for i, model in enumerate(model_order):
        model_data = plot_df[plot_df["model_variant"] == model]
        if model_data.empty:
            continue

        y_vals = model_data["mae_km"].values
        x_vals = x_positions + offset_start + i * bar_width

        ax.bar(
            x_vals, y_vals, bar_width,
            label=model,
            color=color_map[model],
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )

    for h_idx, horizon in enumerate(horizons):
        baseline_rows = plot_df[
            (plot_df["horizon_hours"] == horizon)
            & (plot_df["model_variant"] == baseline_model)
        ]
        if baseline_rows.empty:
            continue
        y_ref = baseline_rows["mae_km"].values[0]

        ax.hlines(
            y=y_ref,
            xmin=h_idx - group_width / 2 - 0.02,
            xmax=h_idx + group_width / 2 + 0.02,
            colors="red",
            linestyles="dashed",
            linewidth=2.2,
            zorder=10,
        )
        ax.text(
            h_idx + group_width / 2 + 0.03, y_ref,
            f" Raw SGP4 ({y_ref:.3f} km)",
            color="darkred", va="center", ha="left",
            fontsize=9, fontweight="bold", zorder=11,
        )

        for model in model_order:
            if model == baseline_model:
                continue
            model_data = plot_df[
                (plot_df["model_variant"] == model)
                & (plot_df["horizon_hours"] == horizon)
            ]
            if model_data.empty:
                continue
            mae_val = model_data["mae_km"].values[0]
            if mae_val < y_ref:
                model_idx = model_order.index(model)
                bar_x = x_positions[h_idx] + offset_start + model_idx * bar_width

                arrow_start_y = max(mae_val - 0.025, 0.005)
                ax.annotate(
                    "",
                    xy=(bar_x, mae_val + 0.002),
                    xytext=(bar_x, arrow_start_y - 0.018),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="#2ca02c",
                        lw=2.2,
                        mutation_scale=15,
                    ),
                    zorder=12,
                )
                ax.text(
                    bar_x, mae_val + 0.006,
                    "✓",
                    ha="center", va="bottom",
                    fontsize=11, color="#2ca02c", fontweight="bold",
                    zorder=12,
                )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Propagation Horizon (hours)", fontsize=12)
    ax.set_ylabel("Mean Absolute Error (km)", fontsize=12)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{int(h)}h" for h in horizons], fontsize=12, fontweight="bold")

    ax.grid(axis="y", alpha=0.25, linestyle="--", zorder=0)
    ax.set_axisbelow(True)

    y_max = plot_df["mae_km"].max()
    ax.set_ylim(0, y_max * 1.08)

    # --- LEGEND MOVED TO BOTTOM ---
    ax.legend(
        title="Model Variant",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.35),  # Pushes it below the x-axis
        frameon=False,
        fontsize=7,
        ncol=4,  # 4 columns to keep it compact
    )

    # bbox_inches='tight' in savefig will automatically expand the canvas 
    # to include the legend placed outside the axes.
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved enhanced plot to {save_path}")


def main():
    csv_path = f"{config.OUTPUT_DIR}/ablation_summary_v2_dedup.csv"
    try:
        summary_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        logger.error(f"Could not find {csv_path}. Run evaluate_sequence_ablation_v2_dedup.py first.")
        return

    if summary_df.empty:
        logger.warning("Summary CSV is empty. Nothing to plot.")
        return

    logger.info("Generating enhanced FULL comparison plot (with arrows)...")
    plot_grouped_bar_mae_with_arrows(
        summary_df, FULL_MODEL_ORDER,
        "MAE by horizon: raw SGP4 vs baseline vs OLD (contaminated) vs NEW (dedup-fixed) sequence variants",
        f"{config.PLOTS_DIR}/ablation_comparison_v2_dedup_enhanced.png",
        baseline_model="raw_sgp4",
    )

    logger.info("Generating enhanced SEQUENCE-ONLY comparison plot (with arrows)...")
    plot_grouped_bar_mae_with_arrows(
        summary_df, OLD_SEQUENCE_ORDER + DEDUP_SEQUENCE_ORDER,
        "MAE by horizon: sequence-model variants only -- OLD (contaminated) vs NEW (dedup-fixed)",
        f"{config.PLOTS_DIR}/ablation_comparison_sequence_only_v2_dedup_enhanced.png",
        baseline_model="raw_sgp4",
    )

    logger.info("Plot generation complete!")


if __name__ == "__main__":
    main()
