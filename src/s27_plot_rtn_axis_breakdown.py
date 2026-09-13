"""
plot_rtn_axis_breakdown.py
Generates publication-ready visualizations highlighting the subtle 
differences between model variants by plotting PERCENTAGE CHANGE 
relative to the raw_sgp4 baseline, with explicit data labels.
"""
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("plot_rtn_axis")

# --- Configuration ---
# NOTE: previously hardcoded to "new-output/..." -- now routed through
# config.OUTPUT_DIR / config.PLOTS_DIR like every other script in the
# pipeline, so all CSVs and plots land under one unified outputs/ tree
# instead of being split across output/ and new-output/.
CSV_PATH = os.path.join(config.OUTPUT_DIR, "rtn_axis_breakdown_summary.csv")
OUTPUT_PREFIX = os.path.join(config.PLOTS_DIR, "rtn_axis_breakdown_")
RAW_MODEL = "raw_sgp4"

# Metrics we want to visualize
METRICS_TO_PLOT = [
    ("mae_km", "MAE", "mae_pct_change"),
    ("median_ae_km", "Median AE", "median_ae_pct_change"),
    ("p95_km", "95th Percentile Error", "p95_pct_change"),
]

def clean_model_name(name):
    """Shorten model names for better x-axis readability."""
    name = name.replace("_rolling_cv_v2_dedup", " (Dedup)")
    name = name.replace("_postwindow_oversample_filtered_test", " (Filtered)")
    name = name.replace("plus_", "+")
    name = name.replace("all_features", "All Feats")
    name = name.replace("space_weather_full_and_cycle_phase", "SW + Cycle")
    name = name.replace("space_weather_full", "Space Weather")
    name = name.replace("orbital_jump", "Orb. Jump")
    name = name.replace("tracking_cadence", "Track. Cadence")
    return name

def main():
    config.ensure_dirs()
    logger.info(f"Loading data from {CSV_PATH}...")
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        logger.error(f"File {CSV_PATH} not found. Please ensure it is in the current directory.")
        return

    # 1. Extract Raw SGP4 baseline for each axis
    raw_sgp4 = df[df['model'] == RAW_MODEL].set_index('axis')[['mae_km', 'median_ae_km', 'p95_km']]
    
    # 2. Filter out raw_sgp4 for the comparison dataframe
    comp_df = df[df['model'] != RAW_MODEL].copy()
    
    # 3. Calculate Percentage Change relative to Raw SGP4 for each axis
    for metric, _, pct_col in METRICS_TO_PLOT:
        comp_df[pct_col] = comp_df.apply(
            lambda row: ((row[metric] - raw_sgp4.loc[row['axis'], metric]) / raw_sgp4.loc[row['axis'], metric]) * 100,
            axis=1
        )

    # 4. Clean up model names for the plot
    comp_df['model_clean'] = comp_df['model'].apply(clean_model_name)
    
    # Define a logical order for the x-axis (Dedup models first, then Filtered)
    model_order = comp_df['model_clean'].unique()

    # Set publication-quality styling
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    palette = {'radial': '#2166AC', 'along_track': '#D6604D', 'cross_track': '#1A9850'}

    # =========================================================================
    # PLOT 1: Grouped Bar Chart with Percentage Change & Data Labels
    # =========================================================================
    for metric, metric_name, pct_col in METRICS_TO_PLOT:
        logger.info(f"Generating {metric_name} percentage change plot...")
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        sns.barplot(
            data=comp_df,
            x='model_clean',
            y=pct_col,
            hue='axis',
            palette=palette,
            order=model_order,
            ax=ax,
            alpha=0.9
        )
        
        # Add a clear baseline at 0%
        ax.axhline(0, color='black', linewidth=1.5, linestyle='--', zorder=10)
        
        # Formatting
        ax.set_ylabel(f'% Change vs. Raw SGP4', fontsize=12, fontweight='bold')
        ax.set_xlabel('Model Variant', fontsize=12, fontweight='bold')
        ax.set_title(f'Relative {metric_name} Performance by RTN Axis', fontsize=14, fontweight='bold', pad=15)
        plt.xticks(rotation=30, ha='right', fontsize=10)
        ax.legend(title='RTN Axis', title_fontsize=11, fontsize=10, loc='upper right')
        
        # Add explicit data labels on top of each bar (Crucial for small differences!)
        for container in ax.containers:
            # Format: show 1 decimal place and a % sign. 
            # We use padding to push the text slightly above the bar.
            ax.bar_label(container, fmt='%.1f%%', fontsize=9, fontweight='bold', padding=3)
        
        plt.tight_layout()
        output_path = f"{OUTPUT_PREFIX}{metric.split('_')[0]}_pct_change.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
        plt.close()

    # =========================================================================
    # PLOT 2: Dot Plot (Forest Plot style) - Often better for tiny differences
    # =========================================================================
    logger.info("Generating Dot Plot (Forest Plot style) for MAE...")
    metric, metric_name, pct_col = METRICS_TO_PLOT[0] # Focus on MAE for the dot plot
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use stripplot or scatter to create a clean dot plot
    sns.stripplot(
        data=comp_df,
        x=pct_col,
        y='model_clean',
        hue='axis',
        palette=palette,
        order=model_order,
        size=8,
        ax=ax,
        dodge=True,
        alpha=0.8
    )
    
    ax.axvline(0, color='black', linewidth=1.5, linestyle='--')
    ax.set_xlabel(f'% Change in {metric_name} vs. Raw SGP4', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model Variant', fontsize=12, fontweight='bold')
    ax.set_title(f'Dot Plot: Relative {metric_name} Performance', fontsize=14, fontweight='bold', pad=15)
    ax.legend(title='RTN Axis', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add text annotations next to the dots
    for i, row in comp_df.iterrows():
        # Find the exact x-position seaborn gave this dot (approximate based on dodge)
        # A simpler approach is to just write the value on the plot
        pass # (Dot plots are self-explanatory with the grid, but we can add labels if desired)

    plt.tight_layout()
    output_path = f"{OUTPUT_PREFIX}mae_dotplot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()

    logger.info("Plotting complete!")

if __name__ == "__main__":
    main()
