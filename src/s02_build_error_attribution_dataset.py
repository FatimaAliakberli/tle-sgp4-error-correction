"""
build_error_attribution_dataset.py

Builds the core per-sample error-attribution dataset used to train the
residual-correction models.

For each object, for each TLE at current_epoch t0, for each horizon in
config.HORIZONS_HOURS:
    1. Find an approximate "truth" TLE within MAX_TRUTH_OFFSET_HOURS of
       target_time = t0 + horizon.
    2. Propagate the CURRENT TLE to target_time -> SGP4 prediction.
    3. Propagate the TRUTH TLE to target_time -> approximate truth.
    4. Compute the position residual (labels), in both Cartesian and RTN.
    5. Attach orbital / drag / space-weather / tracking-cadence /
       physics-augmented features computed ONLY from information available
       at t0 (no look-ahead into the truth TLE).

Outputs:
    output/error_attribution_dataset.csv (or .parquet)
    output/error_attribution_summary.txt
"""

import logging
import time

import numpy as np
import pandas as pd

import config
import tle_utils
import feature_utils
import space_weather_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("s02_build_error_attribution_dataset")

# How often (in TLEs) to log progress inside a single large object, and how
# often (in objects) to log progress across the whole dataset. Tune these
# down if you want chattier logs, up if they're too noisy.
LOG_EVERY_N_TLES = 1000
LOG_EVERY_N_OBJECTS = 1


def build_object_samples(object_name, sats, ts, sw_df):
    """Build all samples (across all horizons) for one object."""
    epochs = [tle_utils.epoch_datetime(s) for s in sats]
    rows = []
    n_sats = len(sats)

    # Pre-extract raw orbital elements for every TLE (cheap, no propagation)
    raw_elements = [feature_utils.orbital_elements_from_sat(s) for s in sats]

    for i, cur in enumerate(sats):
        if n_sats >= LOG_EVERY_N_TLES and i > 0 and i % LOG_EVERY_N_TLES == 0:
            logger.info("  ... %s: processed %d/%d TLEs (%d samples so far)",
                        object_name, i, n_sats, len(rows))

        cur_epoch = epochs[i]

        # Only prior TLEs are usable for cadence / change features.
        prior_epochs = epochs[:i]
        prior_bstars = [raw_elements[k]["bstar"] for k in range(i)]
        prior_mean_motions = [raw_elements[k]["mean_motion_rev_per_day"] for k in range(i)]
        prior_semi_major_axes = [raw_elements[k]["semi_major_axis_km"] for k in range(i)]
        prior_perigee_altitudes = [raw_elements[k]["perigee_altitude_km"] for k in range(i)]

        cadence_feats = feature_utils.compute_tracking_cadence_features(
            cur_epoch, prior_epochs, prior_bstars, prior_mean_motions,
            prior_semi_major_axes, prior_perigee_altitudes,
        )

        cur_elements = raw_elements[i]

        for horizon_h in config.HORIZONS_HOURS:
            target_time = cur_epoch + pd.Timedelta(hours=horizon_h)

            # find nearest truth TLE within offset (search forward from i)
            best_j, best_offset = None, None
            for j in range(i + 1, len(sats)):
                offset_h = abs((epochs[j] - target_time).total_seconds()) / 3600.0
                if offset_h <= config.MAX_TRUTH_OFFSET_HOURS:
                    if best_offset is None or offset_h < best_offset:
                        best_offset, best_j = offset_h, j
                if epochs[j] > target_time + pd.Timedelta(hours=config.MAX_TRUTH_OFFSET_HOURS):
                    break

            if best_j is None:
                continue

            truth_sat = sats[best_j]

            try:
                sgp4_pos, sgp4_vel = feature_utils.propagate_state(cur, ts, target_time)
                truth_pos, _truth_vel = feature_utils.propagate_state(truth_sat, ts, target_time)
            except Exception as exc:
                logger.debug("Propagation failed for %s at %s (h=%s): %s", object_name, cur_epoch, horizon_h, exc)
                continue

            delta = truth_pos - sgp4_pos
            position_error_km = float(np.linalg.norm(delta))

            radial, along, cross, rtn_valid = feature_utils.project_residual_to_rtn(sgp4_pos, sgp4_vel, truth_pos)

            row = {
                "object_id": object_name,
                "current_epoch_utc": cur_epoch,
                "target_epoch_utc": target_time,
                "horizon_hours": horizon_h,

                # labels
                "position_error_km": position_error_km,
                "velocity_error_km_s": np.nan,  # truth velocity is itself an SGP4 estimate; kept nan by default
                "radial_residual_km": radial,
                "along_track_residual_km": along,
                "cross_track_residual_km": cross,
                "delta_x_km": float(delta[0]),
                "delta_y_km": float(delta[1]),
                "delta_z_km": float(delta[2]),
                "rtn_valid": rtn_valid,

                # prediction info
                "sgp4_pos_x_km": float(sgp4_pos[0]),
                "sgp4_pos_y_km": float(sgp4_pos[1]),
                "sgp4_pos_z_km": float(sgp4_pos[2]),
                "truth_pos_x_km": float(truth_pos[0]),
                "truth_pos_y_km": float(truth_pos[1]),
                "truth_pos_z_km": float(truth_pos[2]),
                "truth_tle_epoch_utc": epochs[best_j],
                "truth_offset_hours": best_offset,

                # data-quality flags
                "outlier_flag": False,   # filled after computing dataset-wide threshold
                "huge_gap_flag": bool(cadence_feats.get("hours_since_last_tle", 0) is not None and
                                       (np.nan_to_num(cadence_feats.get("hours_since_last_tle", 0), nan=0.0) >
                                        config.MAX_SEQUENCE_GAP_HOURS)),
                "truth_offset_flag": bool(best_offset > config.MAX_TRUTH_OFFSET_HOURS * 0.5),
            }

            # orbital / drag features (from current TLE only)
            row.update(cur_elements)

            # tracking cadence features
            row.update(cadence_feats)

            # space weather timestamps are resolved in a single batched call
            # after the loop (see build_all_samples) for performance --
            # per-row space-weather lookups do not scale to large datasets.
            midpoint = cur_epoch + (target_time - cur_epoch) / 2
            row["_sw_midpoint"] = midpoint

            rows.append(row)

    return rows


def run():
    config.ensure_dirs()
    ts = tle_utils.get_timescale()
    pipeline_start = time.time()

    logger.info("Loading TLE dataset from %s ...", config.TLE_DATASET_DIR)
    t0 = time.time()
    satellites = tle_utils.load_all_tle_history(config.TLE_DATASET_DIR)
    total_tles = sum(len(v) for v in satellites.values())
    logger.info("Loaded %d objects, %d total TLEs, in %.1fs.",
                len(satellites), total_tles, time.time() - t0)

    satellites, dropped_objects = tle_utils.filter_high_gap_objects(
        satellites,
        max_mean_interval_hours=config.MAX_OBJECT_MEAN_INTERVAL_HOURS,
        min_tle_count=config.MIN_OBJECT_TLE_COUNT,
        min_interval_hours=config.MIN_INTERVAL_HOURS,
    )
    if dropped_objects:
        logger.info("Proceeding with %d objects after tracking-cadence filter.", len(satellites))

    logger.info("Loading space weather data...")
    t0 = time.time()
    sw_df = space_weather_utils.load_space_weather()
    logger.info("Loaded %d space-weather rows in %.1fs.", len(sw_df), time.time() - t0)

    all_rows = []
    n_objects_ok, n_objects_failed = 0, 0
    n_objects_total = len(satellites)

    logger.info("Building error-attribution samples for %d objects...", n_objects_total)
    for obj_idx, (object_name, sats) in enumerate(satellites.items(), start=1):
        t_obj = time.time()
        try:
            if len(sats) < 2:
                logger.info("[%d/%d] %s: skipped (fewer than 2 TLEs).",
                            obj_idx, n_objects_total, object_name)
                continue

            logger.info("[%d/%d] %s: starting (%d TLEs)...",
                        obj_idx, n_objects_total, object_name, len(sats))
            rows = build_object_samples(object_name, sats, ts, sw_df)
            all_rows.extend(rows)
            n_objects_ok += 1

            elapsed_total = time.time() - pipeline_start
            logger.info(
                "[%d/%d] %s: done in %.1fs -- %d samples generated "
                "(running total: %d samples, %d objects, %.1fs elapsed).",
                obj_idx, n_objects_total, object_name, time.time() - t_obj,
                len(rows), len(all_rows), n_objects_ok, elapsed_total,
            )
        except Exception as exc:
            logger.warning("[%d/%d] %s: FAILED after %.1fs: %s",
                            obj_idx, n_objects_total, object_name, time.time() - t_obj, exc)
            n_objects_failed += 1
            continue

    logger.info("Finished building samples for all objects in %.1fs total. "
                "%d objects succeeded, %d failed, %d raw samples before feature attachment.",
                time.time() - pipeline_start, n_objects_ok, n_objects_failed, len(all_rows))

    df = pd.DataFrame(all_rows)

    if df.empty:
        logger.warning("No samples were generated. Check TLE_dataset contents.")
    else:
        # Batched space-weather feature computation (much faster than doing
        # this per-row: interpolation is vectorized over the whole column).
        logger.info("Computing space-weather features for %d samples (batched)...", len(df))
        t0 = time.time()
        sw_cur = space_weather_utils.get_space_weather_features(sw_df, df["current_epoch_utc"]).add_prefix("sw_current_")
        sw_tgt = space_weather_utils.get_space_weather_features(sw_df, df["target_epoch_utc"]).add_prefix("sw_target_")
        sw_mid = space_weather_utils.get_space_weather_features(sw_df, df["_sw_midpoint"]).add_prefix("sw_mid_")
        sw_cur.index = df.index
        sw_tgt.index = df.index
        sw_mid.index = df.index
        df = pd.concat([df.drop(columns=["_sw_midpoint"]), sw_cur, sw_tgt, sw_mid], axis=1)
        logger.info("Space-weather features attached in %.1fs.", time.time() - t0)

        # density proxy (physics-based augmentation), based on current epoch conditions
        df["density_proxy"] = [
            feature_utils.compute_density_proxy(f107, ap, perigee)
            for f107, ap, perigee in zip(df["sw_current_f107"], df["sw_current_ap"], df["perigee_altitude_km"])
        ]

        # outlier flag: position error beyond 99th percentile per horizon
        for horizon_h in config.HORIZONS_HOURS:
            mask = df["horizon_hours"] == horizon_h
            if mask.sum() > 0:
                thresh = df.loc[mask, "position_error_km"].quantile(0.99)
                df.loc[mask, "outlier_flag"] = df.loc[mask, "position_error_km"] > thresh

        # replace non-finite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Save dataset -- try parquet first, always also provide CSV fallback.
    logger.info("Saving dataset (%d rows) to disk...", len(df))
    t0 = time.time()
    csv_path = f"{config.OUTPUT_DIR}/error_attribution_dataset.csv"
    parquet_path = f"{config.OUTPUT_DIR}/error_attribution_dataset.parquet"
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception as exc:
        logger.info("Parquet save skipped (%s). CSV is available at %s.", exc, csv_path)
    logger.info("Dataset saved in %.1fs.", time.time() - t0)

    # Human readable summary
    with open(f"{config.OUTPUT_DIR}/error_attribution_summary.txt", "w") as f:
        f.write("Error Attribution Dataset Summary\n")
        f.write("==================================\n\n")
        f.write(f"Objects processed successfully: {n_objects_ok}\n")
        f.write(f"Objects failed: {n_objects_failed}\n")
        f.write(f"Objects dropped by tracking-cadence filter (mean_interval_hours > "
                f"{config.MAX_OBJECT_MEAN_INTERVAL_HOURS}): {len(dropped_objects)}\n")
        if dropped_objects:
            for name, info in sorted(dropped_objects.items(), key=lambda kv: -kv[1]["mean_interval_hours"]):
                f.write(f"  - {name}: n_tles={info['n_tles']}, "
                        f"mean_interval_hours={info['mean_interval_hours']:.1f} ({info['reason']})\n")
        f.write(f"Total samples: {len(df)}\n\n")
        if not df.empty:
            f.write("Samples per horizon:\n")
            f.write(df["horizon_hours"].value_counts().sort_index().to_string())
            f.write("\n\nPosition error (km) summary statistics:\n")
            f.write(df.groupby("horizon_hours")["position_error_km"].describe().to_string())
            f.write("\n\nRTN validity rate:\n")
            f.write(str(df["rtn_valid"].mean()))
            f.write("\n\nColumns:\n")
            f.write(", ".join(df.columns))

    logger.info("Wrote %d samples to %s (total pipeline time: %.1fs).",
                len(df), csv_path, time.time() - pipeline_start)


if __name__ == "__main__":
    run()
