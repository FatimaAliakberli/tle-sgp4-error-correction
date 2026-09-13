"""
space_weather_utils.py

Loading, templating, and feature engineering for space-weather data
(F10.7, Kp, Ap, and now also SN and daily Ap) used as inputs to the
atmospheric-drag related features.

Column meanings (standard GFZ Potsdam Kp/ap/Ap/SN/F10.7 naming):
    f107      - F10.7 cm solar radio flux (daily, sfu)
    kp        - planetary Kp index (3-hourly, quasi-logarithmic 0-9 scale)
    ap        - planetary ap index (3-hourly, linear-scale equivalent of kp)
    sn        - international sunspot number (daily)
    daily_ap  - daily Ap index (linear-scale DAILY average geomagnetic
                activity; distinct from the 3-hourly `ap` column above)

Backward compatibility: older space_weather.csv files may only have the
original 4 columns (timestamp_utc, f107, kp, ap). `sn` and `daily_ap` are
treated as fully optional -- if absent, they are zero-filled rather than
causing a crash, exactly like the existing empty-dataframe fallback.
"""

import os
import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger("space_weather_utils")
logging.basicConfig(level=logging.INFO)

# Columns that MUST be present for the CSV to be considered usable at all.
EXPECTED_COLUMNS = ["timestamp_utc", "f107", "kp", "ap"]

# Additional columns we now support. These are optional for backward
# compatibility with older space_weather.csv files.
OPTIONAL_COLUMNS = ["sn", "daily_ap"]

# Full column set used once a dataframe has been normalized by
# load_space_weather() (required + optional, optional zero-filled if absent).
ALL_COLUMNS = EXPECTED_COLUMNS + OPTIONAL_COLUMNS

# Base space-weather quantities that get lag/rolling-mean feature treatment
# in get_space_weather_features().
FEATURE_BASE_COLUMNS = ["f107", "kp", "ap", "sn", "daily_ap"]


def _write_template_csv(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    template = pd.DataFrame({
        "timestamp_utc": pd.date_range("2010-01-01", periods=3, freq="D", tz="UTC"),
        "f107": [120.0, 120.0, 120.0],
        "kp": [2.0, 2.0, 2.0],
        "ap": [7.0, 7.0, 7.0],
        "sn": [30.0, 30.0, 30.0],
        "daily_ap": [7.0, 7.0, 7.0],
    })
    template.to_csv(path, index=False)


def _try_download_space_weather(path):
    """
    Best-effort optional download of historical space weather data.
    This is intentionally conservative: if anything goes wrong (no
    internet, changed API, parsing failure) we fail gracefully and let
    the caller fall back to a template / zero-filled dataframe.
    """
    if not config.ATTEMPT_SPACE_WEATHER_DOWNLOAD:
        return False
    try:
        import urllib.request
        url = "https://services.swpc.noaa.gov/json/f107_cm_flux.json"
        with urllib.request.urlopen(url, timeout=10) as resp:
            raw = resp.read()
        import json
        data = json.loads(raw)
        df = pd.DataFrame(data)
        if "time_tag" not in df.columns or "flux" not in df.columns:
            return False
        df["timestamp_utc"] = pd.to_datetime(df["time_tag"], utc=True)
        df["f107"] = df["flux"].astype(float)
        df["kp"] = np.nan
        df["ap"] = np.nan
        df["sn"] = np.nan
        df["daily_ap"] = np.nan
        df = df[ALL_COLUMNS]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Downloaded space weather data to %s", path)
        return True
    except Exception as exc:
        logger.warning("Space weather download failed, falling back: %s", exc)
        return False


def load_space_weather(path=None):
    """
    Load the local space-weather CSV. If missing, attempt an optional
    download; if that also fails (or is disabled), create a template file,
    print/log a warning, and return an empty dataframe with the expected
    columns.

    Backward compatible: if the CSV only has the original 4 columns
    (timestamp_utc, f107, kp, ap), `sn` and `daily_ap` are added as
    zero-filled columns rather than causing a failure.

    Returns
    -------
    pandas.DataFrame with columns = ALL_COLUMNS ('sn'/'daily_ap' zero-filled
    if not present in the source CSV), 'timestamp_utc' parsed as UTC
    datetime, sorted ascending.
    """
    if path is None:
        path = config.SPACE_WEATHER_CSV

    if not os.path.exists(path):
        downloaded = _try_download_space_weather(path)
        if not downloaded:
            logger.warning(
                "Space weather CSV not found at %s. Creating a template file. "
                "Fill it in with real F10.7/Kp/Ap/SN/daily-Ap data for best "
                "results. Proceeding with zero-filled / template space-weather "
                "features.",
                path,
            )
            _write_template_csv(path)

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logger.warning("Failed to read space weather CSV %s: %s", path, exc)
        return pd.DataFrame(columns=ALL_COLUMNS)

    missing_required = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_required:
        logger.warning("Space weather CSV missing required columns %s. Returning empty frame.",
                        missing_required)
        return pd.DataFrame(columns=ALL_COLUMNS)

    missing_optional = [c for c in OPTIONAL_COLUMNS if c not in df.columns]
    if missing_optional:
        logger.info(
            "Space weather CSV at %s is missing optional column(s) %s "
            "(older 4-column format). Zero-filling for backward compatibility.",
            path, missing_optional,
        )
        for col in missing_optional:
            df[col] = 0.0

    try:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    except Exception as exc:
        logger.warning("Could not parse timestamp_utc column: %s", exc)
        return pd.DataFrame(columns=ALL_COLUMNS)

    for col in OPTIONAL_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df = df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc").reset_index(drop=True)
    df = df[ALL_COLUMNS]
    return df


def compute_f107_smoothed_13mo(space_weather_df, column="f107"):
    """
    Standard NOAA SWPC / SIDC 13-month smoothed running mean, as used for
    the smoothed sunspot number and (in the same style) smoothed F10.7:

        Rs[m] = (0.5*R[m-6] + R[m-5] + ... + R[m+5] + 0.5*R[m+6]) / 12

    i.e. a centered window of 13 monthly values, with the two edge months
    weighted at 0.5 and the remaining 11 full-weighted, divided by 12 (the
    sum of the weights: 0.5 + 11*1 + 0.5 = 12).

    Steps:
      1. Aggregate the raw (daily/irregular-cadence) `column` series in
         `space_weather_df` into a MONTHLY MEAN series.
      2. Linearly interpolate across any internal calendar months that had
         zero raw observations (data gaps), so the running-mean window
         never has to skip over an internal hole -- only the true start/end
         of the series is handled by the edge fallback below.
      3. Apply the centered 13-month weighted running mean above to every
         month that has a full +/-6-month window available.
      4. EDGE FALLBACK: for months too close to the start or end of the
         available space-weather history to have a full +/-6-month window
         (e.g. the first/last 6 calendar months of coverage), use a
         one-sided TRAILING simple mean over however many months are
         actually available (up to 13), ending at that month, instead of
         producing NaN or raising. This trades a small amount of
         smoothing-formula purity at the two edges for a pipeline that
         never crashes / never emits NaN space-weather features -- the
         same "degrade gracefully" philosophy as load_space_weather()'s
         template/zero-fill fallback elsewhere in this module.

    Returns
    -------
    pandas.Series of the smoothed value, indexed by month-start Timestamp
    (UTC). Empty Series if `space_weather_df` is empty or missing `column`.
    """
    if space_weather_df is None or space_weather_df.empty or column not in space_weather_df.columns:
        return pd.Series(dtype=float)

    monthly = (
        space_weather_df.set_index("timestamp_utc")[column]
        .astype(float)
        .resample("MS")
        .mean()
    )
    # Interpolate internal data-gap months (NaN from resample().mean() on a
    # month with zero raw observations). Only interior gaps are filled this
    # way; True leading/trailing NaNs (before the first / after the last
    # real observation) can't occur here since `monthly`'s index range is
    # itself derived from the data's own min/max timestamp.
    monthly = monthly.interpolate(method="linear", limit_direction="both")

    n_months = len(monthly)
    values = monthly.values
    smoothed = np.empty(n_months)

    for i in range(n_months):
        lo, hi = i - 6, i + 6
        if lo >= 0 and hi < n_months:
            window = values[lo:hi + 1]  # 13 monthly values, centered on i
            smoothed[i] = (0.5 * window[0] + window[1:-1].sum() + 0.5 * window[-1]) / 12.0
        else:
            # Edge fallback: one-sided trailing simple mean, up to 13
            # months, ending at month i.
            trail_lo = max(0, i - 12)
            smoothed[i] = values[trail_lo:i + 1].mean()

    return pd.Series(smoothed, index=monthly.index)


def compute_cycle_phase_position(timestamps):
    """
    0-1 "position within the solar cycle" feature: 0.0 at a solar minimum,
    ~0.5 at the following solar maximum, and back toward 1.0 (== 0.0 of the
    next cycle) at the next minimum. This is a small, deliberately simple
    lookup/interpolation against config.SOLAR_CYCLE_ANCHORS (published
    Cycle 24/25 minimum/maximum dates), NOT anything inferred from our own
    limited data span -- see the comment on SOLAR_CYCLE_ANCHORS for why.

    Implementation: SOLAR_CYCLE_ANCHORS gives a monotonically increasing
    "cycle progress" scale (0.0, 0.5, 1.0, 1.5, ...) at known calendar
    dates. We linearly interpolate against that scale for timestamps
    within the anchor range, and EXTRAPOLATE linearly (using the nearest
    real segment's slope) for timestamps before the first anchor or after
    the last -- e.g. TLEs from the 1960s-2000s (long before Cycle 24), or
    dates past Cycle 25's provisional maximum where no next-minimum date is
    known yet. The raw progress value is then taken mod 1.0 to fold it back
    into the final 0-1 "position within the current cycle" feature.

    Fully vectorized (np.interp), safe to call on large timestamp arrays.

    Parameters
    ----------
    timestamps: iterable of datetime-like (UTC)

    Returns
    -------
    numpy.ndarray of floats in [0, 1), one per input timestamp.
    """
    anchors = sorted(config.SOLAR_CYCLE_ANCHORS, key=lambda a: a[0])
    anchor_dates = pd.to_datetime([d for d, _ in anchors], utc=True)
    anchor_phase = np.array([p for _, p in anchors], dtype=np.float64)

    left_slope = (anchor_phase[1] - anchor_phase[0]) / (anchor_dates[1] - anchor_dates[0]).total_seconds()
    right_slope = (anchor_phase[-1] - anchor_phase[-2]) / (anchor_dates[-1] - anchor_dates[-2]).total_seconds()

    # Extend the anchor arrays with one synthetic point 50 years before the
    # first anchor and 50 years after the last, using each end's real
    # segment slope, so a single vectorized np.interp() call extrapolates
    # linearly instead of clamping flat at the boundary (np.interp's
    # default behavior, which would freeze the phase forever past the last
    # known anchor -- not what we want).
    far_past = anchor_dates[0] - pd.Timedelta(days=365.25 * 50)
    far_future = anchor_dates[-1] + pd.Timedelta(days=365.25 * 50)
    ext_dates = pd.to_datetime(
        [far_past] + list(anchor_dates) + [far_future], utc=True
    )
    ext_phase = np.concatenate([
        [anchor_phase[0] - left_slope * (anchor_dates[0] - far_past).total_seconds()],
        anchor_phase,
        [anchor_phase[-1] + right_slope * (far_future - anchor_dates[-1]).total_seconds()],
    ])

    # IMPORTANT: convert both the anchor scale and the query timestamps to
    # integer time via the SAME pandas construction path (pd.to_datetime on
    # a list/Series, then .astype("int64")). pandas >= 2.x can store
    # datetime64 at ns/us/ms/s resolution depending on how the array was
    # built (e.g. a scalar Timestamp's `.value` is always nanoseconds, but
    # pd.to_datetime() on a list defaults to microsecond resolution in this
    # pandas version) -- mixing those two conversion styles silently
    # produces a 1000x-off timescale. Using the identical path for both
    # arrays here guarantees they share the same resolution.
    ext_ts = ext_dates.astype("int64").astype(np.float64)
    query_ts = pd.to_datetime(pd.Series(list(timestamps)), utc=True).astype("int64").values.astype(np.float64)

    raw_phase = np.interp(query_ts, ext_ts, ext_phase)
    return np.mod(raw_phase, 1.0)


def _interpolate_series(df, timestamps, column):
    """Linearly interpolate `column` of `df` (indexed by timestamp_utc) at `timestamps`."""
    if df.empty:
        return np.zeros(len(timestamps))

    src_t = df["timestamp_utc"].astype("int64").values.astype(np.float64)  # ns since epoch
    src_v = df[column].astype(float).values

    query_t = pd.to_datetime(pd.Series(timestamps), utc=True).astype("int64").values.astype(np.float64)

    # np.interp requires src_t sorted ascending (it is, from load_space_weather)
    return np.interp(query_t, src_t, src_v, left=src_v[0] if len(src_v) else 0.0,
                      right=src_v[-1] if len(src_v) else 0.0)


def get_space_weather_features(space_weather_df, timestamps):
    """
    Compute space-weather features for a list/array of UTC timestamps.

    Parameters
    ----------
    space_weather_df: DataFrame as returned by load_space_weather()
    timestamps: iterable of datetime-like (UTC)

    Returns
    -------
    DataFrame with one row per input timestamp and the following columns
    (for each base quantity in FEATURE_BASE_COLUMNS = f107, kp, ap, sn,
    daily_ap):
        <base>, <base>_lag_6h, <base>_lag_12h, <base>_lag_24h,
        <base>_rolling_mean_24h, <base>_rolling_mean_72h,
    plus:
        geomagnetic_storm_flag, high_flux_flag, high_sn_flag

    Fully backward compatible: if space_weather_df is empty (e.g. an old
    4-column CSV that failed to load, or no file at all), every sn/daily_ap
    derived column is zero-filled rather than raising, matching the
    existing zero-fill behavior for f107/kp/ap.
    """
    timestamps = pd.to_datetime(pd.Series(list(timestamps)), utc=True)
    n = len(timestamps)

    # cycle_phase_position is derived purely from calendar date against
    # published Solar Cycle 24/25 reference dates (config.SOLAR_CYCLE_ANCHORS)
    # -- it does NOT depend on space_weather_df at all, so it's computed
    # once here and included in both the empty-fallback and normal return
    # paths below, exactly like every other column in this function.
    cycle_phase_position = compute_cycle_phase_position(timestamps)

    if space_weather_df is None or space_weather_df.empty:
        # Zero-filled fallback -- pipeline must not crash.
        cols = {}
        for base_col in FEATURE_BASE_COLUMNS:
            cols[base_col] = np.zeros(n)
            for lag_h in [6, 12, 24]:
                cols[f"{base_col}_lag_{lag_h}h"] = np.zeros(n)
            for window_h in [24, 72]:
                cols[f"{base_col}_rolling_mean_{window_h}h"] = np.zeros(n)
        cols["geomagnetic_storm_flag"] = np.zeros(n, dtype=int)
        cols["high_flux_flag"] = np.zeros(n, dtype=int)
        cols["high_sn_flag"] = np.zeros(n, dtype=int)
        # f107_smoothed_13mo has no real f107 series to smooth here, so it
        # zero-fills like every other column in this fallback branch.
        cols["f107_smoothed_13mo"] = np.zeros(n)
        cols["cycle_phase_position"] = cycle_phase_position
        return pd.DataFrame(cols)

    out = {}
    for base_col in FEATURE_BASE_COLUMNS:
        out[base_col] = _interpolate_series(space_weather_df, timestamps, base_col)
        for lag_h in [6, 12, 24]:
            lagged_ts = timestamps - pd.to_timedelta(lag_h, unit="h")
            out[f"{base_col}_lag_{lag_h}h"] = _interpolate_series(space_weather_df, lagged_ts, base_col)

    # Rolling means: approximate by averaging several interpolated lag samples
    # over the requested window (robust even with irregular source cadence).
    for base_col in FEATURE_BASE_COLUMNS:
        for window_h, n_samples in [(24, 5), (72, 9)]:
            offsets = np.linspace(0, window_h, n_samples)
            acc = np.zeros(n)
            for off in offsets:
                shifted_ts = timestamps - pd.to_timedelta(off, unit="h")
                acc += _interpolate_series(space_weather_df, shifted_ts, base_col)
            out[f"{base_col}_rolling_mean_{window_h}h"] = acc / n_samples

    out["geomagnetic_storm_flag"] = (out["kp"] >= config.GEOMAGNETIC_STORM_KP_THRESHOLD).astype(int)
    out["high_flux_flag"] = (out["f107"] >= config.HIGH_FLUX_F107_THRESHOLD).astype(int)
    out["high_sn_flag"] = (out["sn"] >= config.HIGH_SN_THRESHOLD).astype(int)

    # 13-month smoothed F10.7 index: computed once on the monthly-aggregated
    # series (see compute_f107_smoothed_13mo for the standard NOAA/ISES
    # formula and edge-fallback behavior), then interpolated to the
    # requested (much higher cadence) timestamps just like the base columns
    # above. Wrapped in a tiny synthetic DataFrame so it can reuse
    # _interpolate_series unchanged.
    monthly_smoothed = compute_f107_smoothed_13mo(space_weather_df, column="f107")
    if monthly_smoothed.empty:
        out["f107_smoothed_13mo"] = np.zeros(n)
    else:
        smoothed_df = pd.DataFrame({
            "timestamp_utc": monthly_smoothed.index,
            "f107_smoothed_13mo": monthly_smoothed.values,
        })
        out["f107_smoothed_13mo"] = _interpolate_series(smoothed_df, timestamps, "f107_smoothed_13mo")

    out["cycle_phase_position"] = cycle_phase_position

    return pd.DataFrame(out)
