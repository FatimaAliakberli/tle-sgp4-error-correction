"""
convert_gfz_space_weather.py

Standalone converter: GFZ Potsdam Kp/ap/Ap/SN/F10.7 wide-format file ->
our long-format space_weather.csv (timestamp_utc, f107, kp, ap, sn,
daily_ap).

GFZ format (one row per day, whitespace-separated, fields 0-indexed):
    [0:3]   YYYY MM DD
    [3:5]   days days_m
    [5]     BSR
    [6]     day_in_BSR
    [7:15]  Kp_1..Kp_8      (8 x 3-hourly Kp)
    [15:23] ap_1..ap_8      (8 x 3-hourly ap)
    [23]    Ap              (daily, linear-scale geomagnetic index)
    [24]    SN              (daily sunspot number)
    [25]    F10.7obs
    [26]    F10.7adj
    [27]    D               (data source flag)

Each day expands into 8 rows (one per 3-hour Kp/ap bin). f107, sn, and
daily_ap are daily quantities, so the same value is repeated across all 8
rows for that day -- get_space_weather_features() interpolates/lags them
like any other time series, so the repetition doesn't need special
handling downstream.

Usage:
    python s00_convert_gfz_space_weather.py raw_gfz_file.txt
    python s00_convert_gfz_space_weather.py raw_gfz_file.txt custom_output_path.csv

If the output path is omitted, it defaults to config.SPACE_WEATHER_CSV
(data/space_weather/space_weather.csv), which is exactly where every other
script in this pipeline (space_weather_utils.py via config.py) expects to
find it -- so the common case is just supplying your raw GFZ file.
"""

import os
import sys
import pandas as pd

import config


def convert(input_path, output_path):
    rows = []
    n_skipped = 0
    with open(input_path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 28:
                n_skipped += 1
                continue  # skip malformed/header lines

            try:
                year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                kp_vals = [float(x) for x in parts[7:15]]
                ap_vals = [float(x) for x in parts[15:23]]
                daily_ap = float(parts[23])
                sn = float(parts[24])
                f107_obs = float(parts[25])
            except (ValueError, IndexError):
                n_skipped += 1
                continue

            base = pd.Timestamp(year=year, month=month, day=day, tz="UTC")
            for i in range(8):
                rows.append({
                    "timestamp_utc": base + pd.Timedelta(hours=3 * i),
                    "f107": f107_obs,
                    "kp": kp_vals[i],
                    "ap": ap_vals[i],
                    "sn": sn,
                    "daily_ap": daily_ap,
                })

    if not rows:
        print(f"No valid rows parsed from {input_path} ({n_skipped} lines skipped). "
              f"Nothing written to {output_path}.")
        return

    df = pd.DataFrame(rows).sort_values("timestamp_utc")
    df = df[["timestamp_utc", "f107", "kp", "ap", "sn", "daily_ap"]]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows ({df['timestamp_utc'].min()} to {df['timestamp_utc'].max()}) "
          f"to {output_path} ({n_skipped} malformed/header lines skipped).")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python s00_convert_gfz_space_weather.py raw_gfz_file.txt [output_path.csv]")
        sys.exit(1)
    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) == 3 else config.SPACE_WEATHER_CSV
    convert(in_path, out_path)
