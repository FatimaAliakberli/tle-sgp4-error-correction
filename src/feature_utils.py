"""
feature_utils.py

Orbital, drag-related, tracking-cadence, and physics-based feature
extraction from EarthSatellite (SGP4) objects.
"""

import math
import logging

import numpy as np

import config

logger = logging.getLogger("feature_utils")
logging.basicConfig(level=logging.INFO)

EARTH_RADIUS_KM = 6378.137
MU_EARTH_KM3_S2 = 398600.4418  # standard gravitational parameter


def orbital_elements_from_sat(sat):
    """
    Extract raw TLE / SGP4 orbital elements from an EarthSatellite.

    Returns a dict of primitive orbital features (does not require
    propagation, purely reads the underlying `sat.model` (a Satrec)).
    """
    m = sat.model
    features = {}

    try:
        features["satnum"] = int(m.satnum)
    except Exception:
        features["satnum"] = np.nan

    features["bstar"] = float(getattr(m, "bstar", np.nan))
    features["ndot"] = float(getattr(m, "ndot", np.nan))       # rad / min^2
    features["nddot"] = float(getattr(m, "nddot", np.nan))     # rad / min^3
    features["eccentricity"] = float(getattr(m, "ecco", np.nan))
    features["inclination_deg"] = math.degrees(float(getattr(m, "inclo", np.nan)))
    features["raan_deg"] = math.degrees(float(getattr(m, "nodeo", np.nan)))
    features["arg_perigee_deg"] = math.degrees(float(getattr(m, "argpo", np.nan)))
    features["mean_anomaly_deg"] = math.degrees(float(getattr(m, "mo", np.nan)))

    # mean motion: sgp4 stores no_kozai in rad/min
    mean_motion_rad_per_min = float(getattr(m, "no_kozai", getattr(m, "no", np.nan)))
    features["mean_motion_rad_per_min"] = mean_motion_rad_per_min
    features["mean_motion_rev_per_day"] = mean_motion_rad_per_min * 1440.0 / (2.0 * math.pi)
    features["mean_motion_rad_per_sec"] = mean_motion_rad_per_min / 60.0

    n_rev_per_day = features["mean_motion_rev_per_day"]
    if n_rev_per_day and n_rev_per_day > 0:
        period_min = 1440.0 / n_rev_per_day
        n_rad_per_sec = features["mean_motion_rad_per_sec"]
        semi_major_axis_km = (MU_EARTH_KM3_S2 / (n_rad_per_sec ** 2)) ** (1.0 / 3.0)
    else:
        period_min = np.nan
        semi_major_axis_km = np.nan

    features["orbital_period_min"] = period_min
    features["semi_major_axis_km"] = semi_major_axis_km

    ecc = features["eccentricity"]
    if not np.isnan(semi_major_axis_km) and not np.isnan(ecc):
        perigee_radius_km = semi_major_axis_km * (1.0 - ecc)
        apogee_radius_km = semi_major_axis_km * (1.0 + ecc)
        features["perigee_altitude_km"] = perigee_radius_km - EARTH_RADIUS_KM
        features["apogee_altitude_km"] = apogee_radius_km - EARTH_RADIUS_KM
        features["altitude_km"] = (features["perigee_altitude_km"] + features["apogee_altitude_km"]) / 2.0
    else:
        features["perigee_altitude_km"] = np.nan
        features["apogee_altitude_km"] = np.nan
        features["altitude_km"] = np.nan

    return features


def compute_density_proxy(f107, ap, perigee_altitude_km):
    """
    Very simple, non-physical proxy for atmospheric density that scales
    with solar flux and geomagnetic activity and decays exponentially with
    perigee altitude. This is NOT a substitute for NRLMSISE-00 / JB2008 --
    it only needs to correlate loosely with drag magnitude.
    """
    try:
        f107_scaled = f107 / config.DENSITY_PROXY_F107_REF
        ap_scaled = ap / config.DENSITY_PROXY_AP_REF
        scale_height = config.DENSITY_PROXY_SCALE_HEIGHT_KM
        alt = perigee_altitude_km if perigee_altitude_km is not None and not np.isnan(perigee_altitude_km) else 400.0
        proxy = f107_scaled * max(ap_scaled, 0.1) * math.exp(-alt / (scale_height * 8.0))
        return float(proxy)
    except Exception:
        return np.nan


def try_real_density(epoch_datetime, lat_deg, lon_deg, alt_km):
    """
    Optional real atmospheric density via pymsis, if installed.
    Returns np.nan if pymsis is unavailable or the call fails.
    """
    try:
        import pymsis
        import numpy as _np
        result = pymsis.calculate(
            [epoch_datetime], [lon_deg], [lat_deg], [alt_km],
        )
        # pymsis.calculate returns array with total mass density at index 0
        return float(result[0, 0])
    except Exception:
        return np.nan


def propagate_state(sat, ts, target_datetime):
    """
    Propagate `sat` (EarthSatellite) to `target_datetime` (timezone-aware
    UTC datetime) and return (position_km: np.array(3,), velocity_km_s: np.array(3,)).

    Raises on SGP4 propagation errors (caller should catch).
    """
    t = ts.from_datetime(target_datetime)
    geocentric = sat.at(t)
    pos = np.array(geocentric.position.km, dtype=float)
    vel = np.array(geocentric.velocity.km_per_s, dtype=float)
    return pos, vel


def rtn_unit_vectors(position_km, velocity_km_s):
    """
    Build the Radial / Along-track (Transverse) / Cross-track (Normal)
    right-handed unit-vector basis from a position and velocity vector.
    """
    r = np.asarray(position_km, dtype=float)
    v = np.asarray(velocity_km_s, dtype=float)

    r_norm = np.linalg.norm(r)
    if r_norm < 1e-6:
        raise ValueError("Degenerate position vector for RTN basis.")
    radial_unit = r / r_norm

    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)
    if h_norm < 1e-9:
        raise ValueError("Degenerate angular momentum vector for RTN basis.")
    cross_track_unit = h / h_norm

    along_track_unit = np.cross(cross_track_unit, radial_unit)

    return radial_unit, along_track_unit, cross_track_unit


def project_residual_to_rtn(sgp4_position_km, sgp4_velocity_km_s, truth_position_km):
    """
    Project the position residual (truth - sgp4_prediction) into the RTN
    frame defined by the SGP4-predicted position/velocity.

    Returns
    -------
    (radial_km, along_track_km, cross_track_km, rtn_valid: bool)
    """
    try:
        radial_unit, along_track_unit, cross_track_unit = rtn_unit_vectors(
            sgp4_position_km, sgp4_velocity_km_s
        )
        delta = np.asarray(truth_position_km, dtype=float) - np.asarray(sgp4_position_km, dtype=float)
        radial = float(np.dot(delta, radial_unit))
        along = float(np.dot(delta, along_track_unit))
        cross = float(np.dot(delta, cross_track_unit))
        return radial, along, cross, True
    except Exception as exc:
        logger.debug("RTN projection failed, falling back to Cartesian: %s", exc)
        return np.nan, np.nan, np.nan, False


def compute_tracking_cadence_features(epoch, prior_epochs, prior_bstars, prior_mean_motions,
                                       prior_semi_major_axes, prior_perigee_altitudes):
    """
    Compute tracking-cadence and short-term change features using only
    TLEs strictly BEFORE `epoch` (no look-ahead).

    Parameters
    ----------
    epoch: pandas.Timestamp (UTC) of the current TLE
    prior_epochs: sorted array-like of pandas.Timestamp, all < epoch
    prior_bstars, prior_mean_motions, prior_semi_major_axes, prior_perigee_altitudes:
        arrays aligned with prior_epochs

    Returns
    -------
    dict of cadence / change features
    """
    import pandas as pd

    feats = {}

    if len(prior_epochs) == 0:
        feats["minutes_since_last_tle"] = np.nan
        feats["hours_since_last_tle"] = np.nan
    else:
        last_epoch = prior_epochs[-1]
        delta = (epoch - last_epoch)
        hours = delta.total_seconds() / 3600.0
        feats["hours_since_last_tle"] = hours
        feats["minutes_since_last_tle"] = hours * 60.0

    for window_h, suffix in [(24, "24h"), (72, "72h")]:
        cutoff = epoch - pd.Timedelta(hours=window_h)
        mask = [(pe >= cutoff) and (pe < epoch) for pe in prior_epochs]
        idxs = [i for i, m in enumerate(mask) if m]

        feats[f"number_of_tles_last_{suffix}"] = len(idxs)

        if len(idxs) >= 2:
            window_epochs = [prior_epochs[i] for i in idxs]
            diffs_h = np.diff([e.timestamp() for e in window_epochs]) / 3600.0
            feats[f"mean_update_interval_last_{suffix}"] = float(np.mean(diffs_h)) if len(diffs_h) else np.nan
        else:
            feats[f"mean_update_interval_last_{suffix}"] = np.nan

        def _change(values):
            if len(idxs) == 0:
                return np.nan
            first_idx = idxs[0]
            last_val = values[-1] if len(values) else np.nan
            first_val = values[first_idx] if first_idx < len(values) else np.nan
            try:
                return float(last_val) - float(first_val)
            except Exception:
                return np.nan

        # "last value before epoch" is prior_*[- 1] (most recent prior TLE)
        feats[f"bstar_change_last_{suffix}"] = _change(prior_bstars)
        feats[f"mean_motion_change_last_{suffix}"] = _change(prior_mean_motions)
        feats[f"semi_major_axis_change_last_{suffix}"] = _change(prior_semi_major_axes)
        feats[f"perigee_change_last_{suffix}"] = _change(prior_perigee_altitudes)

    return feats
