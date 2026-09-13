"""
plot_enhanced_motivating_objects.py
Standalone script to generate enhanced p95 error plots for the two 
motivating objects (CZ-4C DEB and FREGAT DEB). 
Reads from the existing CSV output and adds:
1. Maximally distinct colors per model
2. Dashed red reference line at raw SGP4 p95 level
3. Green arrows pointing RIGHT from left to highlight models that beat raw SGP4
4. Horizontal bar chart layout to avoid label overlap
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
logger = logging.getLogger("s31_plot_motivating_objects_before_after")

OBJECT_LABELS = ["CZ-4C_DEB", "FREGAT_DEB"]
METRIC_COL = "p95_km"

def get_maximally_distinct_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        sat = 0.85 if i % 2 == 0 else 0.65
        val = 0.90 if i % 3 != 1 else 0.70
        colors.append(mcolors.hsv_to_rgb([hue, sat, val]))
    return np.array(colors)


def plot_motivating_object_with_arrows(df, object_label, save_path):
    obj_df = df[df["object_id"] == object_label].copy()
    if obj_df.empty:
        logger.warning("No data found for %s. Skipping plot.", object_label)
        return

    # Ensure raw_sgp4 is first, then sort the rest
    raw_row = obj_df[obj_df["model"] == "raw_sgp4"]
    other_rows = obj_df[obj_df["model"] != "raw_sgp4"].sort_values(by=["era", "model"])
    plot_df = pd.concat([raw_row, other_rows]).reset_index(drop=True)

    models = plot_df["model"].tolist()
    p95_vals = plot_df[METRIC_COL].values
    raw_p95 = raw_row[METRIC_COL].values[0] if not raw_row.empty else np.nan

    n_models = len(models)
    
    # Horizontal bar chart: more space for labels
    fig_height = max(8, 0.45 * n_models)  # Adjust height based on number of models
    fig, ax = plt.subplots(figsize=(12, fig_height))

    # Colour assignment
    distinct_colors = get_maximally_distinct_colors(n_models)
    color_map = {model: distinct_colors[i] for i, model in enumerate(models)}
    if "raw_sgp4" in color_map:
        color_map["raw_sgp4"] = np.array([0.15, 0.15, 0.15, 1.0])

    # Horizontal bars (y positions are 0, 1, 2, ... n_models-1)
    y_positions = np.arange(n_models)
    bar_height = 0.75

    for i, model in enumerate(models):
        ax.barh(
            y_positions[i], p95_vals[i], bar_height,
            label=model,
            color=color_map[model],
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )

    # Reference line + arrows for models that beat raw SGP4
    if not np.isnan(raw_p95):
        # Dashed red vertical reference line
        ax.axvline(
            x=raw_p95,
            color="red",
            linestyle="dashed",
            linewidth=2.2,
            zorder=10,
        )
        # Label at the top of the line
        ax.text(
            raw_p95 * 1.005, n_models - 0.5,
            f"Raw SGP4 ({raw_p95:.2f} km)",
            color="darkred", va="top", ha="left",
            fontsize=10, fontweight="bold", zorder=11,
        )

        # Find every model whose bar is to the LEFT of the line (better)
        for i, model in enumerate(models):
            if model == "raw_sgp4":
                continue
            if p95_vals[i] < raw_p95:
                # Green arrow pointing RIGHT from bar end toward the line
                ax.annotate(
                    "",
                    xy=(p95_vals[i] + 0.05 * raw_p95, y_positions[i]),  # arrow tip
                    xytext=(p95_vals[i] * 0.95, y_positions[i]),        # arrow tail
                    arrowprops=dict(
                        arrowstyle="->",
                        color="#2ca02c",
                        lw=2.5,
                        mutation_scale=18,
                    ),
                    zorder=12,
                )
                # Checkmark above the bar
                ax.text(
                    p95_vals[i] * 0.98, y_positions[i] + 0.35,
                    "✓",
                    ha="right", va="bottom",
                    fontsize=12, color="#2ca02c", fontweight="bold",
                    zorder=12,
                )

    # Formatting
    ax.set_title(f"{object_label.replace('_', ' ')}: p95 error, raw SGP4 vs. models", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("p95 position error (km)", fontsize=12)
    
    # Set y-axis labels (model names) - now horizontal so no overlap
    ax.set_yticks(y_positions)
    ax.set_yticklabels(models, fontsize=9)
    
    # Invert y-axis so raw_sgp4 is at the top
    ax.invert_yaxis()
    
    ax.grid(axis="x", alpha=0.25, linestyle="--", zorder=0)
    ax.set_axisbelow(True)

    x_max = p95_vals.max()
    ax.set_xlim(0, x_max * 1.15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved enhanced plot for {object_label} to {save_path}")


def main():
    csv_path = f"{config.OUTPUT_DIR}/motivating_objects_before_after_summary.csv"
    try:
        summary_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        logger.error(f"Could not find {csv_path}. Run evaluate_motivating_objects_before_after.py first.")
        return

    if summary_df.empty:
        logger.warning("Summary CSV is empty. Nothing to plot.")
        return

    for label in OBJECT_LABELS:
        plot_motivating_object_with_arrows(
            summary_df, 
            label,
            f"{config.PLOTS_DIR}/motivating_objects_before_after_{label}_enhanced.png"
        )

    logger.info("Plot generation complete!")

if __name__ == "__main__":
    main()
