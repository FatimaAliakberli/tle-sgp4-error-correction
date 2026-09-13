"""
tle_utils.py

Parsing and cleaning of TLE/GP history files (2-line or 3-line format).

Public functions:
    load_tle_history(filepath)
    load_all_tle_history(dataset_dir)
    clean_satellite_history(satellites)
"""

import os
import glob
import logging

from skyfield.api import EarthSatellite, load

logger = logging.getLogger("tle_utils")
logging.basicConfig(level=logging.INFO)

# A single shared skyfield timescale object (loading it repeatedly is slow).
# builtin=True avoids any network access by using the leap-second/delta-T
# tables bundled with skyfield, which is important for offline/sandboxed
# environments and keeps the pipeline reproducible without internet access.
_TS = load.timescale(builtin=True)


def get_timescale():
    """Return the shared skyfield timescale object."""
    return _TS


def _read_lines_with_fallback_encoding(filepath):
    """Read a text file trying utf-8 first, then latin-1."""
    for encoding in ("utf-8", "latin-1"):
        try:
            with open(filepath, "r", encoding=encoding) as f:
                return f.readlines()
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as exc:
            logger.warning("Could not read %s: %s", filepath, exc)
            return []
    logger.warning("Failed to decode %s with utf-8 or latin-1.", filepath)
    return []


def _is_tle_line1(line):
    return line.startswith("1 ") and len(line) >= 60


def _is_tle_line2(line):
    return line.startswith("2 ") and len(line) >= 60


def _parse_records_from_lines(lines):
    """
    Walk through raw lines and group them into (name, line1, line2) records,
    handling both 2-line and 3-line formats, blank lines, and comments.
    """
    records = []
    cleaned = []
    for raw in lines:
        line = raw.rstrip("\n").rstrip("\r")
        if not line.strip():
            continue
        if line.strip().startswith("#"):
            continue
        cleaned.append(line)

    i = 0
    n = len(cleaned)
    while i < n:
        line = cleaned[i]
        if _is_tle_line1(line):
            # 2-line format starting here
            if i + 1 < n and _is_tle_line2(cleaned[i + 1]):
                records.append((None, line, cleaned[i + 1]))
                i += 2
                continue
            else:
                # malformed, skip this line
                i += 1
                continue
        else:
            # Possibly a name line followed by 2 TLE lines (3-line format)
            if i + 2 < n and _is_tle_line1(cleaned[i + 1]) and _is_tle_line2(cleaned[i + 2]):
                name = line.strip()
                records.append((name, cleaned[i + 1], cleaned[i + 2]))
                i += 3
                continue
            else:
                # Not a recognizable record start; skip forward.
                i += 1
                continue
    return records


def load_tle_history(filepath):
    """
    Load a single TLE history file (2-line or 3-line format).

    Returns
    -------
    dict: {object_name: [EarthSatellite, ...]} (unsorted, may contain
    duplicates; cleanup happens in clean_satellite_history).

    A single file can technically contain multiple named objects if it mixes
    3-line records for different satellites; each distinct name (or the
    filename-derived fallback name for pure 2-line files) becomes its own key.
    """
    result = {}
    lines = _read_lines_with_fallback_encoding(filepath)
    if not lines:
        return result

    records = _parse_records_from_lines(lines)
    if not records:
        logger.warning("No valid TLE records found in %s", filepath)
        return result

    fallback_name = os.path.splitext(os.path.basename(filepath))[0]

    ts = get_timescale()
    for name, l1, l2 in records:
        try:
            sat = EarthSatellite(l1, l2, name if name else fallback_name, ts)
        except Exception as exc:
            logger.warning("Skipping malformed TLE in %s: %s", filepath, exc)
            continue

        object_key = name if name else fallback_name
        # Prefer a stable identifier: NORAD catalog number if available.
        try:
            object_key = f"{object_key}__{sat.model.satnum}"
        except Exception:
            pass

        result.setdefault(object_key, []).append(sat)

    return result


def load_all_tle_history(dataset_dir):
    """
    Load and merge all TLE history files inside dataset_dir (*.txt).

    Returns
    -------
    dict: {object_name: [EarthSatellite, ...]}
    """
    all_satellites = {}
    filepaths = sorted(glob.glob(os.path.join(dataset_dir, "*.txt")))

    if not filepaths:
        logger.warning("No .txt files found in %s", dataset_dir)
        return all_satellites

    for filepath in filepaths:
        try:
            file_result = load_tle_history(filepath)
        except Exception as exc:
            logger.warning("Failed to process file %s: %s", filepath, exc)
            continue

        for object_key, sats in file_result.items():
            all_satellites.setdefault(object_key, []).extend(sats)

    cleaned = clean_satellite_history(all_satellites)
    return cleaned


def clean_satellite_history(satellites):
    """
    Sort each object's TLE list by epoch and remove duplicate epochs
    (keeping the last occurrence, since later files/lines are assumed to be
    at least as authoritative).

    Parameters
    ----------
    satellites: dict {object_name: [EarthSatellite, ...]}

    Returns
    -------
    dict {object_name: [EarthSatellite, ...]} sorted, deduplicated
    """
    cleaned = {}
    for object_key, sats in satellites.items():
        if not sats:
            continue

        # Sort by epoch (Julian date), stable sort preserves original
        # relative order for true ties before dedup.
        try:
            sats_sorted = sorted(sats, key=lambda s: s.model.jdsatepoch + s.model.jdsatepochF)
        except Exception as exc:
            logger.warning("Could not sort TLEs for %s: %s", object_key, exc)
            continue

        # Deduplicate by epoch, keeping the LAST occurrence for each epoch.
        epoch_to_sat = {}
        for sat in sats_sorted:
            try:
                epoch_key = round(sat.model.jdsatepoch + sat.model.jdsatepochF, 8)
            except Exception:
                continue
            epoch_to_sat[epoch_key] = sat  # later entries overwrite earlier ones

        deduped_sorted = [epoch_to_sat[k] for k in sorted(epoch_to_sat.keys())]

        if len(deduped_sorted) == 0:
            continue

        cleaned[object_key] = deduped_sorted

    return cleaned


def epoch_datetime(sat):
    """Return the timezone-aware UTC datetime of a satellite's TLE epoch."""
    return sat.epoch.utc_datetime()


def mean_update_interval_hours(sats, min_interval_hours=0.0):
    """
    Mean gap (in hours) between consecutive TLE epochs for one object's
    (already sorted, deduplicated) satellite list. Mirrors the interval
    calculation used in statistics_analysis.py so filtering and reporting
    stay consistent. Returns np.nan if fewer than 2 usable intervals exist.
    """
    import numpy as np

    if len(sats) < 2:
        return float("nan")

    epochs = [epoch_datetime(s) for s in sats]
    diffs_h = np.diff([e.timestamp() for e in epochs]) / 3600.0
    diffs_h = diffs_h[diffs_h >= min_interval_hours]
    if len(diffs_h) == 0:
        return float("nan")
    return float(np.mean(diffs_h))


def filter_high_gap_objects(satellites, max_mean_interval_hours=None, min_tle_count=None,
                             min_interval_hours=0.0):
    """
    Drop objects whose overall mean inter-TLE interval exceeds
    `max_mean_interval_hours` (sparse/poorly-tracked objects), and/or whose
    total TLE count is below `min_tle_count`. Pass None for either
    threshold to skip that check.

    Parameters
    ----------
    satellites: dict {object_name: [EarthSatellite, ...]} (sorted, as
        returned by load_all_tle_history / clean_satellite_history)
    max_mean_interval_hours: float or None
    min_tle_count: int or None
    min_interval_hours: float, passed through to mean_update_interval_hours

    Returns
    -------
    (kept: dict, dropped: dict) where `dropped` maps
    object_name -> {"n_tles": int, "mean_interval_hours": float, "reason": str}
    """
    if max_mean_interval_hours is None and min_tle_count is None:
        return satellites, {}

    kept = {}
    dropped = {}

    for object_name, sats in satellites.items():
        n_tles = len(sats)
        mean_interval = mean_update_interval_hours(sats, min_interval_hours=min_interval_hours)

        reasons = []
        if min_tle_count is not None and n_tles < min_tle_count:
            reasons.append(f"n_tles={n_tles} < min_tle_count={min_tle_count}")
        if max_mean_interval_hours is not None and mean_interval == mean_interval and \
                mean_interval > max_mean_interval_hours:
            reasons.append(
                f"mean_interval_hours={mean_interval:.1f} > "
                f"max_allowed={max_mean_interval_hours}"
            )

        if reasons:
            dropped[object_name] = {
                "n_tles": n_tles,
                "mean_interval_hours": mean_interval,
                "reason": "; ".join(reasons),
            }
        else:
            kept[object_name] = sats

    if dropped:
        logger.info(
            "Tracking-cadence filter: dropping %d/%d objects (max_mean_interval_hours=%s, "
            "min_tle_count=%s): %s",
            len(dropped), len(satellites), max_mean_interval_hours, min_tle_count,
            ", ".join(sorted(dropped.keys())),
        )

    return kept, dropped
