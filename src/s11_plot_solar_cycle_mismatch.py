"""
plot_solar_cycle_zoomed.py
Creates a clean, zoomed-in visualization of the solar cycle mismatch,
focusing on the actual data range (2010-2026). 
Loads the space weather CSV directly using the correct column names.
"""
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("s11_plot_solar_cycle_mismatch")

# Boundary dates (fallback to your known dates if not in config)
try:
    TRAIN_END = pd.Timestamp(config.TRAIN_END_DATE, tz="UTC")
    VAL_START = pd.Timestamp(config.VALIDATION_START_DATE, tz="UTC")
    VAL_END = pd.Timestamp(config.VALIDATION_END_DATE, tz="UTC")
    TEST_START = pd.Timestamp(config.TEST_START_DATE, tz="UTC")
    TEST_END = pd.Timestamp(config.TEST_END_DATE, tz="UTC")
except AttributeError:
    TRAIN_END = pd.Timestamp("2023-12-31", tz="UTC")
    VAL_START = pd.Timestamp("2024-01-01", tz="UTC")
    VAL_END = pd.Timestamp("2024-12-31", tz="UTC")
    TEST_START = pd.Timestamp("2025-01-01", tz="UTC")
    TEST_END = pd.Timestamp("2026-08-01", tz="UTC")

F107_THRESHOLD = getattr(config, "HIGH_FLUX_F107_THRESHOLD", 150)
KP_THRESHOLD = getattr(config, "GEOMAGNETIC_STORM_KP_THRESHOLD", 5)


def plot_solar_cycle_zoomed(csv_path, save_path):
    logger.info("Loading space weather data from %s...", csv_path)
    
    # FIX: Use 'timestamp_utc' instead of 'date' to match your CSV structure
    df = pd.read_csv(csv_path, parse_dates=["timestamp_utc"])
    
    # Ensure timezone-aware for comparison with boundaries
    if df["timestamp_utc"].dt.tz is None:
        df["timestamp_utc"] = df["timestamp_utc"].dt.tz_localize("UTC")
    
    # Filter to 2010 onwards for cleaner visualization
    df_plot = df[df["timestamp_utc"] >= "2010-01-01"].copy()
    
    if df_plot.empty:
        logger.error("No data found after 2010. Check CSV format.")
        return

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, 
                                    gridspec_kw={"height_ratios": [1, 1]})
    
    # --- Panel 1: F10.7 ---
    ax1.plot(df_plot["timestamp_utc"], df_plot["f107"], color="#E8710A", linewidth=0.6, 
             alpha=0.8, label="F10.7")
    ax1.axhline(y=F107_THRESHOLD, color="red", linestyle="--", linewidth=1.5, 
                label=f"High-flux threshold ({F107_THRESHOLD} sfu)")
    ax1.set_ylabel("F10.7 (sfu)", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 400)  # Zoom in to show detail (exclude extreme outliers)
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    
    # --- Panel 2: Kp ---
    ax2.plot(df_plot["timestamp_utc"], df_plot["kp"], color="#1F77B4", linewidth=0.6, 
             alpha=0.8, label="Kp")
    ax2.axhline(y=KP_THRESHOLD, color="red", linestyle="--", linewidth=1.5, 
                label=f"Storm threshold ({KP_THRESHOLD})")
    ax2.set_ylabel("Kp index", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 9)
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    
    # --- Add train/val/test boundary shading ---
    # Train region
    ax1.axvspan(df_plot["timestamp_utc"].min(), TRAIN_END, alpha=0.15, color="green", label="Train")
    ax2.axvspan(df_plot["timestamp_utc"].min(), TRAIN_END, alpha=0.15, color="green")
    
    # Val region
    ax1.axvspan(VAL_START, VAL_END, alpha=0.15, color="orange", label="Val")
    ax2.axvspan(VAL_START, VAL_END, alpha=0.15, color="orange")
    
    # Test region
    ax1.axvspan(TEST_START, TEST_END, alpha=0.15, color="red", label="Test")
    ax2.axvspan(TEST_START, TEST_END, alpha=0.15, color="red")
    
    # Add vertical boundary lines for clarity
    for ax in [ax1, ax2]:
        ax.axvline(x=TRAIN_END, color="black", linestyle="-", linewidth=1.2, alpha=0.7)
        ax.axvline(x=TEST_START, color="black", linestyle="-", linewidth=1.2, alpha=0.7)
    
    # --- X-axis formatting ---
    ax2.set_xlabel("Date", fontsize=12, fontweight="bold")
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 7]))
    
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
    
    # --- Title and layout ---
    fig.suptitle("Solar Cycle 25: Space Weather Distribution Across Train/Val/Test Splits\n"
                 "Zoomed view (2010–2026)", 
                 fontsize=13, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved zoomed solar cycle plot to %s", save_path)


def main():
    # Use the path from your log
    csv_path = "space_weather/space_weather.csv"
    save_path = f"{config.PLOTS_DIR}/solar_cycle_timeseries_zoomed.png"
    
    try:
        plot_solar_cycle_zoomed(csv_path, save_path)
    except FileNotFoundError:
        logger.error(f"Could not find {csv_path}. Please check the path.")
    except Exception:
        logger.exception("Failed to generate zoomed solar cycle plot.")


if __name__ == "__main__":
    main()
