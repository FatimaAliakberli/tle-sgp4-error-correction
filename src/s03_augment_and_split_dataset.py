"""
augment_dataset.py

1. Physics-based feature augmentation (applied to the WHOLE dataset):
   - orbital decay rate, perigee decay rate
   - eccentricity change
   - time since last geomagnetic storm / high F10.7 event
   (rolling means of B*/mean-motion, space-weather lags/rolling means, and
   storm/high-flux flags are already computed upstream in
   build_error_attribution_dataset.py / space_weather_utils.py)

2. Time-based train/validation/test split.

3. Training-only data augmentation (train split only):
   - small Gaussian noise on numerical orbital + space-weather features
   - simulated missing-TLE updates (inflate time-since-last-tle features)
   - oversampling of high-error samples

Outputs:
    output/augmented_train_dataset.csv
    output/val_dataset.csv
    output/test_dataset.csv
"""

import logging

import numpy as np
import pandas as pd

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("s03_augment_and_split_dataset")

np.random.seed(config.RANDOM_SEED)


def add_physics_augmented_features(df):
    """Add dataset-wide physics-based features that require looking across
    an object's own time-ordered history (still causal: only uses data at
    or before each row's current_epoch_utc)."""
    if df.empty:
        return df

    df = df.sort_values(["object_id", "current_epoch_utc"]).reset_index(drop=True)
    out_parts = []

    for object_id, g in df.groupby("object_id", sort=False):
        g = g.sort_values("current_epoch_utc").copy()

        # Orbital decay rate: d(semi_major_axis)/dt using consecutive current-epoch samples
        t_hours = (g["current_epoch_utc"] - g["current_epoch_utc"].iloc[0]).dt.total_seconds() / 3600.0
        sma = g["semi_major_axis_km"].values
        perigee = g["perigee_altitude_km"].values
        ecc = g["eccentricity"].values

        decay_rate = np.full(len(g), np.nan)
        perigee_decay_rate = np.full(len(g), np.nan)
        ecc_change = np.full(len(g), np.nan)

        t_vals = t_hours.values
        for i in range(1, len(g)):
            dt = t_vals[i] - t_vals[i - 1]
            if dt > 1e-6:
                decay_rate[i] = (sma[i] - sma[i - 1]) / dt
                perigee_decay_rate[i] = (perigee[i] - perigee[i - 1]) / dt
                ecc_change[i] = ecc[i] - ecc[i - 1]

        g["orbital_decay_rate_km_per_hr"] = decay_rate
        g["perigee_decay_rate_km_per_hr"] = perigee_decay_rate
        g["eccentricity_change"] = ecc_change

        # Time since last geomagnetic storm / high-flux event, using the
        # current-epoch space weather flags already present in the dataset.
        storm_flag_col = "sw_current_geomagnetic_storm_flag"
        flux_flag_col = "sw_current_high_flux_flag"

        last_storm_time = None
        last_flux_time = None
        time_since_storm = np.full(len(g), np.nan)
        time_since_flux = np.full(len(g), np.nan)

        epochs = g["current_epoch_utc"].values
        storm_flags = g[storm_flag_col].values if storm_flag_col in g.columns else np.zeros(len(g))
        flux_flags = g[flux_flag_col].values if flux_flag_col in g.columns else np.zeros(len(g))

        for i in range(len(g)):
            cur_t = pd.Timestamp(epochs[i])
            if last_storm_time is not None:
                time_since_storm[i] = (cur_t - last_storm_time).total_seconds() / 3600.0
            if last_flux_time is not None:
                time_since_flux[i] = (cur_t - last_flux_time).total_seconds() / 3600.0
            if storm_flags[i]:
                last_storm_time = cur_t
            if flux_flags[i]:
                last_flux_time = cur_t

        g["hours_since_last_storm"] = time_since_storm
        g["hours_since_last_high_flux"] = time_since_flux

        out_parts.append(g)

    result = pd.concat(out_parts, axis=0).sort_index()
    return result


def time_based_split(df):
    """Split by current_epoch_utc according to config's date boundaries."""
    if df.empty:
        return df.copy(), df.copy(), df.copy()

    epoch = pd.to_datetime(df["current_epoch_utc"], utc=True)

    train_end = pd.Timestamp(config.TRAIN_END_DATE, tz="UTC")
    val_start = pd.Timestamp(config.VALIDATION_START_DATE, tz="UTC")
    val_end = pd.Timestamp(config.VALIDATION_END_DATE, tz="UTC")
    test_start = pd.Timestamp(config.TEST_START_DATE, tz="UTC")
    test_end = pd.Timestamp(config.TEST_END_DATE, tz="UTC")

    train_df = df[epoch <= train_end].copy()
    val_df = df[(epoch >= val_start) & (epoch <= val_end)].copy()
    test_df = df[(epoch >= test_start) & (epoch <= test_end)].copy()

    return train_df, val_df, test_df


NUMERIC_NOISE_COLUMNS_PREFIXES = (
    "inclination_deg", "raan_deg", "eccentricity", "arg_perigee_deg", "mean_anomaly_deg",
    "mean_motion_rev_per_day", "semi_major_axis_km", "perigee_altitude_km", "apogee_altitude_km",
    "altitude_km", "bstar", "sw_current_", "sw_target_", "sw_mid_", "density_proxy",
)


def _numeric_noise_columns(df):
    cols = []
    for c in df.columns:
        if df[c].dtype.kind not in "fi":
            continue
        if c.startswith(NUMERIC_NOISE_COLUMNS_PREFIXES) or any(c.startswith(p) for p in NUMERIC_NOISE_COLUMNS_PREFIXES):
            cols.append(c)
    return cols


def apply_training_only_augmentation(train_df):
    """Apply augmentation strategies that are ONLY valid for the training set."""
    if not config.ALLOW_DATA_AUGMENTATION or train_df.empty:
        return train_df

    df = train_df.copy()
    rng = np.random.default_rng(config.RANDOM_SEED)

    # 1. Small Gaussian noise on numerical orbital + space-weather features.
    #    Noise is added only to INPUT features; residual labels are left
    #    untouched (kept "consistent" by using very small noise magnitude,
    #    as permitted by the spec when full physical re-derivation is
    #    impractical).
    noisy_df = df.copy()
    noise_cols = _numeric_noise_columns(df)
    for col in noise_cols:
        vals = noisy_df[col].values.astype(float)
        finite_mask = np.isfinite(vals)
        if finite_mask.sum() == 0:
            continue
        std = np.nanstd(vals[finite_mask])
        if std == 0 or np.isnan(std):
            continue
        noise = rng.normal(0.0, config.AUGMENTATION_NOISE_STD * std, size=vals.shape)
        vals[finite_mask] = vals[finite_mask] + noise[finite_mask]
        noisy_df[col] = vals
    noisy_df["augmentation_type"] = "gaussian_noise"

    # 2. Simulated missing TLE updates: inflate time-since-last-tle features
    #    for a random subset of rows (as if an update had been skipped).
    missing_df = df.copy()
    missing_mask = rng.random(len(missing_df)) < config.AUGMENTATION_MISSING_TLE_PROB
    for col in ["hours_since_last_tle", "minutes_since_last_tle"]:
        if col in missing_df.columns:
            multiplier = 1.0 + rng.uniform(0.5, 2.0, size=len(missing_df))
            missing_df.loc[missing_mask, col] = (
                missing_df.loc[missing_mask, col].astype(float) * multiplier[missing_mask]
            )
    for col in ["number_of_tles_last_24h", "number_of_tles_last_72h"]:
        if col in missing_df.columns:
            missing_df.loc[missing_mask, col] = (
                missing_df.loc[missing_mask, col].astype(float) - 1
            ).clip(lower=0)
    missing_df["augmentation_type"] = "simulated_missing_tle"
    missing_df = missing_df[missing_mask]

    df["augmentation_type"] = "original"
    augmented = pd.concat([df, noisy_df, missing_df], axis=0, ignore_index=True)

    # 3. Oversample high-error samples (based on |along_track_residual_km|).
    if config.AUGMENTATION_OVERSAMPLE_HIGH_ERROR and "along_track_residual_km" in augmented.columns:
        abs_resid = augmented["along_track_residual_km"].abs()
        finite = abs_resid.replace([np.inf, -np.inf], np.nan).dropna()
        if len(finite) > 0:
            thresh = finite.quantile(config.AUGMENTATION_HIGH_ERROR_QUANTILE)
            high_error_rows = augmented[abs_resid >= thresh]
            if len(high_error_rows) > 0:
                extra = pd.concat(
                    [high_error_rows] * (config.AUGMENTATION_OVERSAMPLE_FACTOR - 1),
                    axis=0, ignore_index=True,
                ) if config.AUGMENTATION_OVERSAMPLE_FACTOR > 1 else pd.DataFrame()
                if not extra.empty:
                    extra["augmentation_type"] = "oversampled_high_error"
                    augmented = pd.concat([augmented, extra], axis=0, ignore_index=True)

    # Optional label clipping for training stability (documented, off by default)
    if config.TRAIN_LABEL_CLIP_KM is not None:
        for col in config.RESIDUAL_TARGETS:
            if col in augmented.columns:
                augmented[col] = augmented[col].clip(-config.TRAIN_LABEL_CLIP_KM, config.TRAIN_LABEL_CLIP_KM)

    return augmented


def run():
    config.ensure_dirs()

    dataset_path = f"{config.OUTPUT_DIR}/error_attribution_dataset.csv"
    logger.info("Loading %s", dataset_path)
    df = pd.read_csv(dataset_path)

    if df.empty:
        logger.warning("Error attribution dataset is empty; writing empty splits.")
        for name in ["augmented_train_dataset", "val_dataset", "test_dataset"]:
            pd.DataFrame().to_csv(f"{config.OUTPUT_DIR}/{name}.csv", index=False)
        return

    for col in ["current_epoch_utc", "target_epoch_utc", "truth_tle_epoch_utc"]:
        df[col] = pd.to_datetime(df[col], utc=True, format="ISO8601")

    logger.info("Adding physics-augmented features (dataset-wide)...")
    df = add_physics_augmented_features(df)

    logger.info("Splitting into train/val/test by time...")
    train_df, val_df, test_df = time_based_split(df)
    logger.info("Split sizes -- train: %d, val: %d, test: %d", len(train_df), len(val_df), len(test_df))

    logger.info("Applying training-only augmentation...")
    train_aug_df = apply_training_only_augmentation(train_df)
    logger.info("Augmented train size: %d (from %d)", len(train_aug_df), len(train_df))

    if "augmentation_type" not in val_df.columns:
        val_df = val_df.copy()
        val_df["augmentation_type"] = "original"
    if "augmentation_type" not in test_df.columns:
        test_df = test_df.copy()
        test_df["augmentation_type"] = "original"

    train_aug_df.to_csv(f"{config.OUTPUT_DIR}/augmented_train_dataset.csv", index=False)
    val_df.to_csv(f"{config.OUTPUT_DIR}/val_dataset.csv", index=False)
    test_df.to_csv(f"{config.OUTPUT_DIR}/test_dataset.csv", index=False)

    logger.info("augment_dataset complete.")


if __name__ == "__main__":
    run()
