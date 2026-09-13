# Hierarchical Deep Learning for SGP4 Propagation-Error Correction

Code accompanying the paper **"Hierarchical Deep Learning for Space Debris
Detection Using Augmented Hybrid TLE-Grounded Datasets"**, submitted to the
77th International Astronautical Congress (IAC 2026), Antalya, Türkiye.

> Fatima Alakbarli, Hamida Jafarova — Department of Computer Science, ADA
> University, Baku, Azerbaijan.

---

## What this project does

Two-Line Elements (TLEs) plus the SGP4 propagator are the only routinely
available way to track most of the ~40,000+ catalogued objects in low Earth
orbit, but SGP4's accuracy degrades unpredictably, especially during
geomagnetic storms and for objects with orbital-element discontinuities
(manoeuvres, breakups). Standard practice filters out objects with sparse
tracking, but in our dataset of 36 debris objects we found something that
filter misses: objects with perfectly normal tracking cadence that still
occasionally rack up single-update position errors of 100+ km. We call
these **catastrophic errors** (an object's own 99th-percentile error or
worse), and they're the whole reason this project exists.

This repository builds a pipeline that:

1. **Constructs residual-error labels directly from TLE data** — for every
   TLE, it propagates forward with SGP4 to the next TLE's epoch and takes
   the discrepancy (in radial/along-track/cross-track components) as the
   "ground truth" error to learn, since independent ephemerides don't exist
   for uncooperative debris.
2. **Trains two correction models**: a `HistGradientBoostingRegressor`
   baseline and a GRU sequence model, both predicting that residual from
   TLE history plus an extended space-weather feature set (F10.7, Kp, ap,
   sunspot number, daily Ap, plus a derived solar-cycle-phase feature).
3. **Runs a systematic 7-feature-set × 2-augmentation-strategy ablation**
   (with rolling-origin cross-validation) to work out what actually helps.
4. **Attributes each catastrophic error** to a likely physical cause
   (tracking gap / possible manoeuvre-or-breakup / space-weather activity /
   unexplained) via a heuristic per-object diagnostic layer, so failures
   are interpretable instead of just averaged over.

**Headline result:** a small, targeted "solar-cycle phase" feature beats
raw SGP4's 0.409 km test MAE more reliably than aggressive data
augmentation does — and it does so without any oversampling at all. The
gains are concentrated in tail/outlier reduction (p95 and max error), not
in uniformly correcting every sample: across every configuration we tried,
only ~47–52% of individual samples actually improve. See the paper for the
full ablation and the reasoning behind that trade-off.

| | Raw SGP4 | Baseline (GBM) | Sequence, dedup-only (best) | Sequence, flag-filtered (best) |
|---|---|---|---|---|
| MAE (km) | 0.409 | 1.919 | **0.405** | **0.406** |
| p95 (km) | 1.358 | 17.206 | 1.338 | 1.323 |
| Max error (km) | 127.16 | 124.31 | 126.9 | 124.5 |

(Full breakdown, all 14 ablation configurations, and per-object diagnostics
are in the paper's Results section.)

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── data/
│   ├── TLE_dataset/          <- put your raw TLE .txt files here (one per object, or one combined file — see below)
│   └── space_weather/        <- s00_convert_gfz_space_weather.py writes space_weather.csv here
├── outputs/                  <- everything the pipeline generates (CSVs, plots, models). Empty at checkout.
│   ├── plots/
│   ├── models/
│   └── test_output/          <- outputs of the "standalone experiment" scripts, kept separate so they never overwrite the main run
└── src/                      <- all pipeline code (flat directory — see "Why flat?" below)
```

**Why is `src/` flat instead of split into subfolders per stage?** Almost
every script imports helper functions from one or more earlier scripts
(e.g. the final ablation evaluator imports the baseline metric function,
the sequence-model builder, *and* the previous ablation script's
row-summary helper). Splitting into stage subfolders would mean either
breaking those imports or turning this into a proper installable package.
Given the goal is "clone it and run it," a flat, numbered directory is the
more robust choice — run order and grouping are communicated through the
file **names** (`sNN_...`) and the table below instead of through folders.

---

## Setup

```bash
pip install -r requirements.txt
```

`torch` is only required for the sequence-model stages; everything up to
and including the GBM baseline runs on scikit-learn alone.

## What you need to provide

This repo ships without raw data (TLE files and derived outputs are
excluded via `.gitignore` — they're either large, user-specific, or
regenerable). To reproduce anything you need:

1. **A TLE dataset** in `data/TLE_dataset/`: 2-line or 3-line TLE/GP
   history files. `tle_utils.py` parses both formats and sorts/dedupes by
   epoch per object automatically, so you don't need to pre-clean anything.
2. **A raw space-weather file**, in the standard GFZ Potsdam Kp/ap/Ap/SN/
   F10.7 wide format (one row per day — an example, `raw_space_weather.txt`,
   is included in `src/`). Convert it with:

   ```bash
   cd src
   python s00_convert_gfz_space_weather.py raw_space_weather.txt
   ```

   This writes `data/space_weather/space_weather.csv`, the exact path
   `config.py` expects. Pass a second argument to write elsewhere if you
   want, but the default should just work.

All thresholds, paths, date ranges, and hyperparameters live in
`src/config.py` — that's the one file to check/edit before a run, rather
than hunting for hard-coded values inside individual scripts.

---

## Running it

### Quick path — baseline result only

Reproduces the raw-SGP4-vs-baseline-GBM comparison (the first two columns
of the results table above), using a single time-based train/val/test
split (train ≤ 2023, val = 2024, test = 2025–2026):

```bash
cd src
python run_basic_pipeline.py
```

This chains, in order: `s01_compute_tle_statistics.py` →
`s02_build_error_attribution_dataset.py` → `s03_augment_and_split_dataset.py`
→ `s04_train_baseline_model.py` → `s06_train_sequence_model.py` (a single
GRU run, skipped gracefully if `torch` isn't installed) →
`s05_evaluate_baseline_and_raw_sgp4.py`.

### Full path — everything in the paper

The full pipeline trains 14 sequence-model configurations for the main
ablation alone, and is deliberately **not** wired into one runner: several
stages are genuine research detours (a failed experiment, a diagnosed and
fixed bug) that are worth inspecting on their own rather than blindly
chaining through. Run the stages below in order, from inside `src/`.

| # | Script | What it does | Key outputs |
|---|--------|---------------|--------------|
| 00 | `s00_convert_gfz_space_weather.py` | Converts your raw GFZ file to `space_weather.csv` | `data/space_weather/space_weather.csv` |
| 01 | `s01_compute_tle_statistics.py` | TLE update-interval statistics per object / time range; drives the 48h tracking-cadence filter | `outputs/stats_summary.csv` |
| 02 | `s02_build_error_attribution_dataset.py` | Core dataset build: parses TLEs, propagates SGP4, constructs RTN residual labels | `outputs/error_attribution_dataset.csv` |
| 03 | `s03_augment_and_split_dataset.py` | Physics-augmented features, time-based train/val/test split, row-level augmentation (Gaussian noise, simulated missing TLE, oversampling) — the original recipe used only by the baseline model | `outputs/augmented_train_dataset.csv`, `val_dataset.csv`, `test_dataset.csv` |
| 04 | `s04_train_baseline_model.py` | Trains the `HistGradientBoostingRegressor` baseline (one per RTN component) | `outputs/models/baseline_*.joblib` |
| 05 | `s05_evaluate_baseline_and_raw_sgp4.py` | Evaluates raw SGP4 + baseline on the test set | `outputs/evaluation_summary.csv` |
| 06 | `s06_train_sequence_model.py` | Core GRU module — defines the 7 feature-set variants and `build_sequences()`, reused by nearly every later script. Also runnable standalone. | `outputs/models/sequence_*` |
| 07 | `s07_run_sequence_ablation_v1.py` | **v1 ablation**: trains all 7 feature-set variants once, on a single split, *before* the dedup fix (stage 13) | `outputs/models/*_v1` |
| 08 | `evaluate_sequence_ablation_v1.py` | Evaluates the v1 ablation on the test set | `outputs/ablation_summary.csv` |
| 09 | `s09_summarize_ablation_v1.py` | Plain-language summary of the v1 ablation | `outputs/ablation_summary.txt` |
| 10 | `s10_check_solar_cycle_mismatch.py` | Diagnostic quantifying the solar-cycle train/test distribution shift — this motivates the cycle-phase feature and rolling-origin CV | `outputs/solar_cycle_mismatch_summary.csv` |
| 11 | `s11_plot_solar_cycle_mismatch.py` | Produces the paper's solar-cycle timeseries figure | `outputs/plots/solar_cycle_timeseries_zoomed.png` |
| 12 | `s12_run_rolling_origin_cv.py` | 4-fold rolling-origin (walk-forward) CV over the 7 variants, to pick a fixed epoch count per variant robust to solar-cycle phase | `outputs/rolling_origin_cv_variant_summary.csv` |
| 13 | `s13_diagnose_duplicate_timestamps.py` | Diagnostic that **discovers** a duplicate-timestamp bug in sequence-window construction under row-level augmentation | diagnostic report only |
| 14 | `s14_confirm_dedup_fix.py` | Confirms the fix (filtering to `augmentation_type == "original"` before windowing) resolves it | diagnostic report only |
| 15 | `train_sequence_model_rolling_cv_final_predup.py` | **Legacy / pre-fix** retraining of all 7 variants, using the buggy windows. Kept only because the paper's per-horizon table, top-object figures, and space-weather-correlation analysis explicitly use this model for continuity (flagged in the paper as such). **Not** used for the headline ablation numbers. | `outputs/models/*_rolling_cv_v2` |
| 16 | `s16_train_sequence_model_v2_dedup.py` | **Validated / dedup-fixed** retraining — rows are filtered to `augmentation_type == "original"` before windowing. This is the baseline used for every ablation number reported in the paper. | `outputs/models/*_rolling_cv_v2_dedup` |
| 17 | `s17_train_missing_dedup_baselines.py` | Fill-in: trains any `_rolling_cv_v2_dedup` variant that failed/was skipped in stage 16 | `outputs/models/*_rolling_cv_v2_dedup` (remaining variants) |
| 18 | `evaluate_sequence_ablation_v2_predup.py` | Evaluates the legacy pre-dedup rolling-CV models (stage 15) | `outputs/ablation_summary_v2_predup.csv` |
| 19 | `s19_evaluate_sequence_ablation_v2_dedup.py` | Evaluates the validated dedup-fixed rolling-CV models — produces the "sequence, dedup-only" column above | `outputs/ablation_summary_v2_dedup.csv` |
| 20 | `s20_summarize_ablation_v1_and_v2.py` | Combines v1 and v2 summaries into one plain-language report | `outputs/ablation_summary_v1_and_v2.txt` |
| 21 | `s21_train_eval_postwindow_oversample_naive.py` | **Failed experiment, kept for the narrative.** Naive post-window oversampling (top-5% windows by raw residual magnitude, duplicated) on the `original` variant only — every metric gets worse. This is the negative result discussed in the paper. | `outputs/test_output/diagnostics_postwindow_oversample_summary.csv` |
| 22 | `s22_train_eval_postwindow_oversample_filtered_all_variants.py` | **Flag-filtered** post-window oversampling (candidates restricted to windows whose source row has no quality-flag set), across all 7 variants that already have a fixed epoch count | `outputs/test_output/models/*_postwindow_oversample_filtered_test`, comparison CSV |
| 23 | `s23_train_eval_postwindow_oversample_filtered_cycle_phase_fillin.py` | Fill-in for the one variant (`all_features_plus_cycle_phase`) not yet covered by stage 22 | `outputs/test_output/models/all_features_plus_cycle_phase_postwindow_oversample_filtered_test` |
| 24 | `s24_evaluate_dedup_vs_filtered_all_variants.py` | Eval-only: combines every dedup-only + flag-filtered model on disk into the final side-by-side comparison — this is the ablation table in the paper | `outputs/test_output/diagnostics_dedup_vs_filtered_all_variants_summary.csv` |
| 25 | `s25_plot_ablation_comparison_v2_dedup.py` | Produces the per-horizon MAE comparison figure | `outputs/plots/ablation_comparison_v2_dedup_enhanced.png` |
| 26 | `s26_evaluate_rtn_axis_breakdown.py` | Per-axis (radial/along-track/cross-track) MAE %-change vs. raw SGP4, for all 14 configurations | `outputs/rtn_axis_breakdown_summary.csv` |
| 27 | `s27_plot_rtn_axis_breakdown.py` | Plots the RTN axis breakdown figure | `outputs/plots/rtn_axis_breakdown_*.png` |
| 28 | `s28_object_error_diagnostics.py` | Per-object diagnostic classification layer: tags each object's catastrophic samples as possible tracking-gap / maneuver-or-breakup / space-weather-driven / unexplained | `outputs/object_diagnostics_summary.csv`, per-object plots |
| 29 | `s29_diagnose_outliers.py` | Supporting diagnostic on the label-quality flags that motivate the flag-filtered oversampling design | diagnostic report only |
| 30 | `s30_evaluate_motivating_objects_before_after.py` | Before/after (legacy vs. dedup-only vs. flag-filtered) analysis for the two motivating objects, CZ-4C DEB and FREGAT DEB | `outputs/motivating_objects_before_after_summary.csv` |
| 31 | `s31_plot_motivating_objects_before_after.py` | Produces the CZ-4C / FREGAT p95-error before/after figures | `outputs/plots/motivating_objects_before_after_*.png` |

A handful of scripts (`s01`, `s10`, `s12`, `s13`) are read-only diagnostics
— they print/log a report instead of writing a headline artifact — but
still run them in order, since later stages assume the data-quality
picture they establish.

---

## Known caveats (worth keeping in mind while reading results)

These are called out explicitly in the paper's Limitations section, and
matter for anyone extending this code rather than just reading about it:

- **The GBM baseline and the sequence models are not trained on the same
  augmentation recipe** (baseline: original row-level augmentation;
  sequence models: dedup-only, optionally + flag-filtered post-window
  oversampling). This confounds architecture with training-data treatment
  in the baseline-vs-sequence-model comparison — retraining the baseline
  on the dedup-only condition is future work.
- **All ablation numbers come from single training runs with a fixed
  seed.** The differences between configurations (0.001–0.008 km around a
  ~0.41 km baseline) are small enough that multi-seed replication would be
  needed to say which are statistically meaningful, not just numerically
  different.
- **The per-horizon / per-object / space-weather-correlation diagnostics**
  in the paper use the legacy *pre-dedup* model (stage 15 above), not the
  validated dedup-fixed one, and are explicitly flagged there as
  illustrative of the diagnostic methodology rather than current numbers.
- **These are tail-error correctors, not universal improvers.** Only
  ~47–52% of individual test samples improve in any given configuration;
  the reported MAE/p95 gains come from shrinking the worst outliers, which
  matters for the paper's motivating failure mode (catastrophic single-
  update errors) but means uniform deployment across every object and
  horizon isn't the right takeaway.



