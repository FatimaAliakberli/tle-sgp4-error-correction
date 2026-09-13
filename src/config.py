config.py
# """
# config.py

# Central configuration for the TLE residual-correction pipeline.
# Edit values here rather than hard-coding them inside the scripts.
# """

# import os

# # --------------------------------------------------------------------------
# # Paths
# # --------------------------------------------------------------------------
# TLE_DATASET_DIR = "TLE_dataset"
# OUTPUT_DIR = "output"
# PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
# MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
# OBJECT_DIAGNOSTICS_PLOTS_DIR = os.path.join(PLOTS_DIR, "object_diagnostics")
# SPACE_WEATHER_CSV = os.path.join("data", "space_weather", "space_weather.csv")

# # --------------------------------------------------------------------------
# # Horizons for fixed-horizon SGP4 baseline / residual prediction (hours)
# # --------------------------------------------------------------------------
# HORIZONS_HOURS = [12, 24, 36]

# # Kept for backward compatibility with the requested "0.5, 1, 1.5 day" framing
# # (0.5 day = 12h, 1 day = 24h, 1.5 day = 36h) -- HORIZONS_HOURS is authoritative.
# HORIZONS_DAYS = [0.5, 1.0, 1.5]

# # --------------------------------------------------------------------------
# # Time ranges used for stratified statistics
# # --------------------------------------------------------------------------
# TIME_RANGES = {
#     "2010-2020": ("2010-01-01 00:00:00", "2020-12-31 23:59:59"),
#     "2021-2024": ("2021-01-01 00:00:00", "2024-12-31 23:59:59"),
#     "2025": ("2025-01-01 00:00:00", "2025-12-31 23:59:59"),
#     "2026_Jan1_Aug1": ("2026-01-01 00:00:00", "2026-08-01 23:59:59"),
# }

# # --------------------------------------------------------------------------
# # Data-quality thresholds
# # --------------------------------------------------------------------------
# MIN_INTERVAL_HOURS = 0.01          # drop update intervals shorter than this
# MAX_TRUTH_OFFSET_HOURS = 6.0       # max distance between target time and truth TLE epoch

# # Object-level tracking-cadence filter.
# #
# # Some objects (typically debris fragments) are tracked so sparsely that
# # SGP4 diverges by thousands of km between updates. These objects have a
# # mean gap between consecutive TLEs far above the rest of the population
# # (see output/stats_summary.csv -- e.g. IUE DEB at ~268h, COSMOS 1818
# # COOLANT at ~263h, METEOR 2-8 DEB at ~114h, OPS 0856 DEB at ~49h, vs. a
# # population mean of ~34h and 75th percentile of ~26h). Their residual
# # labels are heavy-tailed enough (max errors of 10,000-19,000 km) that they
# # dominate squared-error training loss and degrade predictions for
# # everything else.
# #
# # Objects whose OVERALL mean inter-TLE interval exceeds this threshold are
# # dropped entirely (from statistics, dataset building, training, and
# # evaluation) before any downstream processing happens. Set to None to
# # disable filtering and keep the previous behavior.
# MAX_OBJECT_MEAN_INTERVAL_HOURS = 48.0

# # Optional secondary floor: also drop objects with too few TLEs to compute
# # a meaningful cadence statistic (kept separate from the "fewer than 2
# # TLEs" skip that already exists downstream, in case you want a stricter
# # minimum here). Set to None to disable.
# MIN_OBJECT_TLE_COUNT = None

# # --------------------------------------------------------------------------
# # Train / validation / test split (time based, NOT random)
# # --------------------------------------------------------------------------
# TRAIN_END_DATE = "2023-12-31 23:59:59"
# VALIDATION_START_DATE = "2024-01-01 00:00:00"
# VALIDATION_END_DATE = "2024-12-31 23:59:59"
# TEST_START_DATE = "2025-01-01 00:00:00"
# TEST_END_DATE = "2026-08-01 23:59:59"

# # --------------------------------------------------------------------------
# # Sequence model settings
# # --------------------------------------------------------------------------
# SEQUENCE_WINDOW_SIZE = 16
# MAX_SEQUENCE_GAP_HOURS = 72.0

# # --------------------------------------------------------------------------
# # Reproducibility
# # --------------------------------------------------------------------------
# RANDOM_SEED = 42

# # --------------------------------------------------------------------------
# # Augmentation
# # --------------------------------------------------------------------------
# ALLOW_DATA_AUGMENTATION = True
# AUGMENTATION_NOISE_STD = 0.001          # relative gaussian noise on numeric features
# AUGMENTATION_DROPOUT_PROB = 0.1         # probability of dropping an intermediate seq step
# AUGMENTATION_MISSING_TLE_PROB = 0.1     # probability of simulating a missing TLE update
# AUGMENTATION_OVERSAMPLE_HIGH_ERROR = True
# AUGMENTATION_HIGH_ERROR_QUANTILE = 0.9  # samples above this quantile of |residual| get oversampled
# AUGMENTATION_OVERSAMPLE_FACTOR = 2      # how many extra copies of high-error rows to add

# # --------------------------------------------------------------------------
# # Space weather feature flags
# # --------------------------------------------------------------------------
# GEOMAGNETIC_STORM_KP_THRESHOLD = 5.0
# HIGH_FLUX_F107_THRESHOLD = 150.0

# # Daily sunspot number (SN) threshold above which we flag "high solar
# # activity". SN ~100 corresponds roughly to moderately high solar-cycle
# # activity (comparable in spirit to F10.7 ~150 sfu, hence a similar-style
# # flag alongside HIGH_FLUX_F107_THRESHOLD / GEOMAGNETIC_STORM_KP_THRESHOLD).
# HIGH_SN_THRESHOLD = 100.0

# # Try to download space weather data if the local CSV is missing. This
# # requires internet access and will silently fail back to a template/zeros
# # if it does not work.
# ATTEMPT_SPACE_WEATHER_DOWNLOAD = False

# # --------------------------------------------------------------------------
# # Density proxy scaling (see feature_utils.compute_density_proxy)
# # --------------------------------------------------------------------------
# DENSITY_PROXY_SCALE_HEIGHT_KM = 50.0
# DENSITY_PROXY_F107_REF = 150.0
# DENSITY_PROXY_AP_REF = 15.0

# # --------------------------------------------------------------------------
# # Modeling
# # --------------------------------------------------------------------------
# RESIDUAL_TARGETS = ["radial_residual_km", "along_track_residual_km", "cross_track_residual_km"]
# PRIMARY_TARGET = "along_track_residual_km"

# # Optional clipping of extreme residual labels for TRAINING only (not applied
# # to validation/test evaluation). Set to None to disable.
# TRAIN_LABEL_CLIP_KM = None  # e.g. 50.0

# # --------------------------------------------------------------------------
# # Per-object error diagnostics (object_error_diagnostics.py)
# # --------------------------------------------------------------------------
# # Catastrophic-sample selection: for each object individually, samples with
# # position_error_km at or above this quantile of THAT OBJECT's own error
# # distribution are treated as "catastrophic". This is object-relative (not
# # a single dataset-wide cutoff) so that objects with a lower or higher
# # overall error floor are each judged against their own baseline.
# OBJECT_DIAG_CATASTROPHIC_QUANTILE = 0.99

# # An object needs at least this many samples before we trust a 99th-
# # percentile cutoff computed from its own data; below this we fall back to
# # the dataset-wide `outlier_flag` column (99th percentile per horizon,
# # already computed in build_error_attribution_dataset.py) instead.
# OBJECT_DIAG_MIN_SAMPLES_FOR_OWN_THRESHOLD = 20

# # --- Heuristic classification thresholds (all independently checked; a
# # sample may match zero, one, or several categories) ---

# # "possible_tracking_gap": there was an unusually large gap in TLE updates
# # right before the current TLE used for this prediction.
# OBJECT_DIAG_TRACKING_GAP_HOURS = 36.0

# # "possible_maneuver_or_breakup": an orbital-element jump over the prior
# # 24h or 72h window that is large enough to suggest a maneuver, breakup,
# # or otherwise non-drag-consistent event rather than ordinary secular decay.
# OBJECT_DIAG_BSTAR_CHANGE_THRESHOLD = 5e-4
# OBJECT_DIAG_MEAN_MOTION_CHANGE_THRESHOLD_REV_PER_DAY = 0.001
# OBJECT_DIAG_SEMI_MAJOR_AXIS_CHANGE_THRESHOLD_KM = 1.0
# OBJECT_DIAG_PERIGEE_CHANGE_THRESHOLD_KM = 5.0

# # "possible_space_weather_driven": geomagnetic storm, high F10.7 flux, or
# # high sunspot number flagged at the sample's current or target epoch.
# # (Uses the existing geomagnetic_storm_flag / high_flux_flag / high_sn_flag
# # columns computed by space_weather_utils.get_space_weather_features.)


# def ensure_dirs():
#     """Create all output directories if they do not already exist."""
#     for d in [OUTPUT_DIR, PLOTS_DIR, MODELS_DIR, OBJECT_DIAGNOSTICS_PLOTS_DIR,
#               os.path.dirname(SPACE_WEATHER_CSV) or "."]:
#         os.makedirs(d, exist_ok=True)


"""
config.py

Central configuration for the TLE residual-correction pipeline.
Edit values here rather than hard-coding them inside the scripts.
"""

import os

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
TLE_DATASET_DIR = "data/TLE_dataset"
OUTPUT_DIR = "outputs"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
OBJECT_DIAGNOSTICS_PLOTS_DIR = os.path.join(PLOTS_DIR, "object_diagnostics")
SPACE_WEATHER_CSV = os.path.join("data", "space_weather", "space_weather.csv")

# --------------------------------------------------------------------------
# Horizons for fixed-horizon SGP4 baseline / residual prediction (hours)
# --------------------------------------------------------------------------
HORIZONS_HOURS = [12, 24, 36]

# Kept for backward compatibility with the requested "0.5, 1, 1.5 day" framing
# (0.5 day = 12h, 1 day = 24h, 1.5 day = 36h) -- HORIZONS_HOURS is authoritative.
HORIZONS_DAYS = [0.5, 1.0, 1.5]

# --------------------------------------------------------------------------
# Time ranges used for stratified statistics
# --------------------------------------------------------------------------
TIME_RANGES = {
    "2010-2020": ("2010-01-01 00:00:00", "2020-12-31 23:59:59"),
    "2021-2024": ("2021-01-01 00:00:00", "2024-12-31 23:59:59"),
    "2025": ("2025-01-01 00:00:00", "2025-12-31 23:59:59"),
    "2026_Jan1_Aug1": ("2026-01-01 00:00:00", "2026-08-01 23:59:59"),
}

# --------------------------------------------------------------------------
# Data-quality thresholds
# --------------------------------------------------------------------------
MIN_INTERVAL_HOURS = 0.01          # drop update intervals shorter than this
MAX_TRUTH_OFFSET_HOURS = 6.0       # max distance between target time and truth TLE epoch

# Object-level tracking-cadence filter.
#
# Some objects (typically debris fragments) are tracked so sparsely that
# SGP4 diverges by thousands of km between updates. These objects have a
# mean gap between consecutive TLEs far above the rest of the population
# (see output/stats_summary.csv -- e.g. IUE DEB at ~268h, COSMOS 1818
# COOLANT at ~263h, METEOR 2-8 DEB at ~114h, OPS 0856 DEB at ~49h, vs. a
# population mean of ~34h and 75th percentile of ~26h). Their residual
# labels are heavy-tailed enough (max errors of 10,000-19,000 km) that they
# dominate squared-error training loss and degrade predictions for
# everything else.
#
# Objects whose OVERALL mean inter-TLE interval exceeds this threshold are
# dropped entirely (from statistics, dataset building, training, and
# evaluation) before any downstream processing happens. Set to None to
# disable filtering and keep the previous behavior.
MAX_OBJECT_MEAN_INTERVAL_HOURS = 48.0

# Optional secondary floor: also drop objects with too few TLEs to compute
# a meaningful cadence statistic (kept separate from the "fewer than 2
# TLEs" skip that already exists downstream, in case you want a stricter
# minimum here). Set to None to disable.
MIN_OBJECT_TLE_COUNT = None

# --------------------------------------------------------------------------
# Train / validation / test split (time based, NOT random)
# --------------------------------------------------------------------------
TRAIN_END_DATE = "2023-12-31 23:59:59"
VALIDATION_START_DATE = "2024-01-01 00:00:00"
VALIDATION_END_DATE = "2024-12-31 23:59:59"
TEST_START_DATE = "2025-01-01 00:00:00"
TEST_END_DATE = "2026-08-01 23:59:59"

# --------------------------------------------------------------------------
# Sequence model settings
# --------------------------------------------------------------------------
SEQUENCE_WINDOW_SIZE = 16
MAX_SEQUENCE_GAP_HOURS = 72.0

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Augmentation
# --------------------------------------------------------------------------
ALLOW_DATA_AUGMENTATION = True
AUGMENTATION_NOISE_STD = 0.001          # relative gaussian noise on numeric features
AUGMENTATION_DROPOUT_PROB = 0.1         # probability of dropping an intermediate seq step
AUGMENTATION_MISSING_TLE_PROB = 0.1     # probability of simulating a missing TLE update
AUGMENTATION_OVERSAMPLE_HIGH_ERROR = True
AUGMENTATION_HIGH_ERROR_QUANTILE = 0.9  # samples above this quantile of |residual| get oversampled
AUGMENTATION_OVERSAMPLE_FACTOR = 2      # how many extra copies of high-error rows to add

# --------------------------------------------------------------------------
# Space weather feature flags
# --------------------------------------------------------------------------
GEOMAGNETIC_STORM_KP_THRESHOLD = 5.0
HIGH_FLUX_F107_THRESHOLD = 150.0

# Daily sunspot number (SN) threshold above which we flag "high solar
# activity". SN ~100 corresponds roughly to moderately high solar-cycle
# activity (comparable in spirit to F10.7 ~150 sfu, hence a similar-style
# flag alongside HIGH_FLUX_F107_THRESHOLD / GEOMAGNETIC_STORM_KP_THRESHOLD).
HIGH_SN_THRESHOLD = 100.0

# Try to download space weather data if the local CSV is missing. This
# requires internet access and will silently fail back to a template/zeros
# if it does not work.
ATTEMPT_SPACE_WEATHER_DOWNLOAD = False

# --------------------------------------------------------------------------
# Solar-cycle reference anchors (used only by the optional 0-1
# cycle-phase-position feature in space_weather_utils.compute_cycle_phase_position).
#
# These are published/consensus solar minimum and maximum dates for Solar
# Cycles 24 and 25 (SIDC/NOAA SWPC), deliberately NOT inferred from our own
# limited TLE/space-weather data span -- the whole point of this feature is
# to give the model calendar-based cycle context that doesn't depend on
# whichever slice of the cycle our own dataset happens to cover.
#
# Each entry is (date_str, phase_value). phase_value increases monotonically
# across entries: 0.0 at a minimum, 0.5 at the following maximum, 1.0 at the
# next minimum (which doubles as phase 0.0 of the next cycle), 1.5 at that
# cycle's maximum, etc. space_weather_utils.compute_cycle_phase_position()
# interpolates against this monotonically-increasing scale and then takes
# the result mod 1.0 to produce the final "0 = minimum, ~0.5 = maximum,
# ~1/0 = next minimum" feature.
#
# Cycle 25's maximum date below is provisional (SWPC's most recent published
# estimate as of this writing) since Cycle 25's own following minimum
# hasn't happened yet -- update this list if a more authoritative date is
# published later. Timestamps outside this anchor range (e.g. TLEs from
# the 1960s-2000s, long before Cycle 24) are linearly EXTRAPOLATED from the
# nearest real segment's slope by compute_cycle_phase_position(), not
# clamped or NaN-filled.
SOLAR_CYCLE_ANCHORS = [
    ("2008-12-01", 0.0),   # Cycle 24 minimum
    ("2014-04-01", 0.5),   # Cycle 24 maximum
    ("2019-12-01", 1.0),   # Cycle 24/25 minimum (== Cycle 25 phase 0.0)
    ("2024-07-01", 1.5),   # Cycle 25 maximum (provisional)
]

# --------------------------------------------------------------------------
# Density proxy scaling (see feature_utils.compute_density_proxy)
# --------------------------------------------------------------------------
DENSITY_PROXY_SCALE_HEIGHT_KM = 50.0
DENSITY_PROXY_F107_REF = 150.0
DENSITY_PROXY_AP_REF = 15.0

# --------------------------------------------------------------------------
# Modeling
# --------------------------------------------------------------------------
RESIDUAL_TARGETS = ["radial_residual_km", "along_track_residual_km", "cross_track_residual_km"]
PRIMARY_TARGET = "along_track_residual_km"

# Optional clipping of extreme residual labels for TRAINING only (not applied
# to validation/test evaluation). Set to None to disable.
TRAIN_LABEL_CLIP_KM = None  # e.g. 50.0

# --------------------------------------------------------------------------
# Per-object error diagnostics (object_error_diagnostics.py)
# --------------------------------------------------------------------------
# Catastrophic-sample selection: for each object individually, samples with
# position_error_km at or above this quantile of THAT OBJECT's own error
# distribution are treated as "catastrophic". This is object-relative (not
# a single dataset-wide cutoff) so that objects with a lower or higher
# overall error floor are each judged against their own baseline.
OBJECT_DIAG_CATASTROPHIC_QUANTILE = 0.99

# An object needs at least this many samples before we trust a 99th-
# percentile cutoff computed from its own data; below this we fall back to
# the dataset-wide `outlier_flag` column (99th percentile per horizon,
# already computed in build_error_attribution_dataset.py) instead.
OBJECT_DIAG_MIN_SAMPLES_FOR_OWN_THRESHOLD = 20

# --- Heuristic classification thresholds (all independently checked; a
# sample may match zero, one, or several categories) ---

# "possible_tracking_gap": there was an unusually large gap in TLE updates
# right before the current TLE used for this prediction.
OBJECT_DIAG_TRACKING_GAP_HOURS = 36.0

# "possible_maneuver_or_breakup": an orbital-element jump over the prior
# 24h or 72h window that is large enough to suggest a maneuver, breakup,
# or otherwise non-drag-consistent event rather than ordinary secular decay.
OBJECT_DIAG_BSTAR_CHANGE_THRESHOLD = 5e-4
OBJECT_DIAG_MEAN_MOTION_CHANGE_THRESHOLD_REV_PER_DAY = 0.001
OBJECT_DIAG_SEMI_MAJOR_AXIS_CHANGE_THRESHOLD_KM = 1.0
OBJECT_DIAG_PERIGEE_CHANGE_THRESHOLD_KM = 5.0

# "possible_space_weather_driven": geomagnetic storm, high F10.7 flux, or
# high sunspot number flagged at the sample's current or target epoch.
# (Uses the existing geomagnetic_storm_flag / high_flux_flag / high_sn_flag
# columns computed by space_weather_utils.get_space_weather_features.)


def ensure_dirs():
    """Create all output directories if they do not already exist."""
    for d in [OUTPUT_DIR, PLOTS_DIR, MODELS_DIR, OBJECT_DIAGNOSTICS_PLOTS_DIR,
              os.path.dirname(SPACE_WEATHER_CSV) or "."]:
        os.makedirs(d, exist_ok=True)
