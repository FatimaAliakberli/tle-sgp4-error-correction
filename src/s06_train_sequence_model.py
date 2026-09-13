"""
train_sequence_model.py

Optional GRU-based sequence residual model that uses an object's previous
TLE-derived states (up to SEQUENCE_WINDOW_SIZE steps) to predict the SGP4
residual (RTN) at the current sample's horizon.

Sequences are built per (object_id, horizon_hours) group, ordered by
current_epoch_utc, using only rows STRICTLY BEFORE the sample being
predicted (no look-ahead). Each step in the sequence carries the step
feature columns for the selected variant (see FEATURE_SET_VARIANTS below)
plus the previous sample's along-track residual.

If PyTorch is not installed, this script logs a clear warning and exits
without raising, so the rest of the pipeline (run_all.py) can continue.

--------------------------------------------------------------------------
FEATURE-SET ABLATION (variant staging)
--------------------------------------------------------------------------
This script trains the SAME GRU architecture / loss / optimizer / LR /
batch size / early-stopping / RANDOM_SEED for every variant. The ONLY
thing that changes between variants is which columns feed each sequence
step (STEP_FEATURE_COLUMNS -> FEATURE_SET_VARIANTS[variant_name]).

Variants are defined additively on top of "original":
    - "plus_orbital_jump", "plus_space_weather_full", and
      "plus_tracking_cadence" each add ONE feature group to "original"
      INDEPENDENTLY of one another (they do not stack with each other).
    - "all_features" is the only cumulative variant: original + all three
      groups combined (deduplicated, order-preserving).

This lets a later evaluation step isolate the marginal effect of each
feature group on held-out error, while "all_features" shows whether the
groups' benefits (if any) are additive, redundant, or interacting.

Outputs (per variant, if torch available):
    output/models/sequence_model_<variant_name>.pt
    output/models/sequence_scaler_<variant_name>.joblib
    output/plots/sequence_training_loss_<variant_name>.png
"""

import argparse
import logging

import numpy as np
import pandas as pd
import joblib

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("s06_train_sequence_model")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# --------------------------------------------------------------------------
# Base ("original") per-step feature columns -- unchanged from the current
# production sequence model. Kept as its own name (rather than only living
# inside FEATURE_SET_VARIANTS) so existing callers that import
# STEP_FEATURE_COLUMNS directly (e.g. evaluate_model.py's default path)
# keep working unmodified.
# --------------------------------------------------------------------------
STEP_FEATURE_COLUMNS = [
    "inclination_deg", "raan_deg", "eccentricity", "arg_perigee_deg", "mean_anomaly_deg",
    "mean_motion_rev_per_day", "semi_major_axis_km", "perigee_altitude_km", "apogee_altitude_km",
    "bstar", "ndot", "nddot",
    "hours_since_last_tle",
    "sw_current_f107", "sw_current_kp", "sw_current_ap",
    "sw_current_sn", "sw_current_daily_ap",
]
# NOTE: sw_current_sn / sw_current_daily_ap require an error_attribution_dataset.csv
# built after the space_weather_utils.py extension that adds sn/daily_ap. If an
# older dataset is loaded that lacks these columns, build_sequences() below already
# filters the requested feature list down to whatever's actually present in `df`,
# so this degrades gracefully rather than raising -- it just trains with fewer
# step features.

# --------------------------------------------------------------------------
# Feature group add-ons for the ablation. Each group is a self-contained
# list of columns; grouping them by name (rather than just inlining into
# FEATURE_SET_VARIANTS) makes it easy to see exactly what each ablation arm
# contributes and to reuse the groups in "all_features".
# --------------------------------------------------------------------------

# Orbital-element-jump features (11 columns): short-window decay rates and
# 24h/72h change-in-orbital-element features. These are the signals most
# likely to help the model recognize maneuvers, breakups, or other
# non-secular-decay events that a purely "current state" feature set can't
# see. orbital_decay_rate_km_per_hr / perigee_decay_rate_km_per_hr /
# eccentricity_change come from augment_dataset.add_physics_augmented_features;
# the rest come from the upstream error_attribution_dataset.csv.
ORBITAL_JUMP_FEATURE_COLUMNS = [
    "orbital_decay_rate_km_per_hr",
    "perigee_decay_rate_km_per_hr",
    "eccentricity_change",
    "bstar_change_last_24h",
    "mean_motion_change_last_24h",
    "semi_major_axis_change_last_24h",
    "perigee_change_last_24h",
    "bstar_change_last_72h",
    "mean_motion_change_last_72h",
    "semi_major_axis_change_last_72h",
    "perigee_change_last_72h",
]

# Space-weather lag/rolling/flag features (11 columns), added on top of the
# 5 "current instantaneous" space-weather features already in "original"
# (sw_current_f107/kp/ap/sn/daily_ap). Deliberately restricted to
# sw_current_* (never sw_target_* or sw_mid_*): those describe conditions
# at or after the prediction target time and would leak future information
# into a per-step input feature.
SPACE_WEATHER_FULL_FEATURE_COLUMNS = [
    "sw_current_f107_lag_6h",
    "sw_current_f107_lag_12h",
    "sw_current_f107_lag_24h",
    "sw_current_kp_lag_6h",
    "sw_current_kp_lag_12h",
    "sw_current_kp_lag_24h",
    "sw_current_f107_rolling_mean_24h",
    "sw_current_kp_rolling_mean_24h",
    "sw_current_geomagnetic_storm_flag",
    "sw_current_high_flux_flag",
    "sw_current_high_sn_flag",
]

# Tracking-cadence / density features (4 columns): how densely/sparsely an
# object has been tracked recently. Complements hours_since_last_tle
# (already in "original") with counts and mean spacing over two windows.
TRACKING_CADENCE_FEATURE_COLUMNS = [
    "number_of_tles_last_24h",
    "mean_update_interval_last_24h",
    "number_of_tles_last_72h",
    "mean_update_interval_last_72h",
]

# Solar-cycle context features (Part A): the standard NOAA/ISES 13-month
# smoothed F10.7 index, plus the optional 0-1 cycle-phase-position feature
# derived from published Solar Cycle 24/25 reference dates. Both are
# computed once in space_weather_utils.get_space_weather_features() and
# therefore already flow through build_error_attribution_dataset.py's
# existing sw_current_/sw_target_/sw_mid_ prefixing with no extra plumbing.
# Restricted to sw_current_* only, for the same no-look-ahead reasoning as
# SPACE_WEATHER_FULL_FEATURE_COLUMNS above (sw_target_*/sw_mid_* describe
# conditions at or after the prediction target time).
#
# This group is deliberately kept separate from SPACE_WEATHER_FULL_FEATURE_COLUMNS
# (rather than folded into it) so "plus_space_weather_full" and
# "all_features" stay EXACTLY as they were for the existing 5 variants --
# the two new variants below are the only place cycle-phase context is
# added, so its marginal effect can be isolated cleanly.
CYCLE_PHASE_FEATURE_COLUMNS = [
    "sw_current_f107_smoothed_13mo",
    "sw_current_cycle_phase_position",
]


def _dedup_preserve_order(cols):
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


FEATURE_SET_VARIANTS = {
    # Current production feature set. Baseline for the ablation.
    "original": list(STEP_FEATURE_COLUMNS),

    # original + orbital-element-jump features, in isolation.
    "plus_orbital_jump": _dedup_preserve_order(
        STEP_FEATURE_COLUMNS + ORBITAL_JUMP_FEATURE_COLUMNS
    ),

    # original + full space-weather (lag/rolling/flag) features, in isolation.
    "plus_space_weather_full": _dedup_preserve_order(
        STEP_FEATURE_COLUMNS + SPACE_WEATHER_FULL_FEATURE_COLUMNS
    ),

    # original + tracking-cadence/density features, in isolation.
    "plus_tracking_cadence": _dedup_preserve_order(
        STEP_FEATURE_COLUMNS + TRACKING_CADENCE_FEATURE_COLUMNS
    ),

    # original + ALL of the above, combined. The only cumulative variant.
    "all_features": _dedup_preserve_order(
        STEP_FEATURE_COLUMNS
        + ORBITAL_JUMP_FEATURE_COLUMNS
        + SPACE_WEATHER_FULL_FEATURE_COLUMNS
        + TRACKING_CADENCE_FEATURE_COLUMNS
    ),

    # -----------------------------------------------------------------
    # NEW (Part B): isolate the marginal effect of solar-cycle context on
    # top of the existing space-weather feature set(s). Everything above
    # this point is untouched from the original 5-variant ablation.
    # -----------------------------------------------------------------

    # plus_space_weather_full's column list + the new cycle-phase columns.
    # Compare directly against "plus_space_weather_full" to isolate the
    # marginal effect of adding 13-month-smoothed F10.7 / cycle-phase
    # context on top of the existing lag/rolling/flag space-weather group.
    "plus_space_weather_full_and_cycle_phase": _dedup_preserve_order(
        STEP_FEATURE_COLUMNS
        + SPACE_WEATHER_FULL_FEATURE_COLUMNS
        + CYCLE_PHASE_FEATURE_COLUMNS
    ),

    # all_features' column list + the same new cycle-phase columns. Compare
    # directly against "all_features" to isolate the marginal effect of
    # cycle-phase context when every other feature group is already present.
    "all_features_plus_cycle_phase": _dedup_preserve_order(
        STEP_FEATURE_COLUMNS
        + ORBITAL_JUMP_FEATURE_COLUMNS
        + SPACE_WEATHER_FULL_FEATURE_COLUMNS
        + TRACKING_CADENCE_FEATURE_COLUMNS
        + CYCLE_PHASE_FEATURE_COLUMNS
    ),
}


TARGET_COLUMNS = ["radial_residual_km", "along_track_residual_km", "cross_track_residual_km"]


def build_sequences(df, window_size, feature_cols=None):
    """
    Build (sequence, static_features, targets) tuples per sample.

    For each (object_id, horizon_hours) group sorted by current_epoch_utc,
    sample index k (k >= 1) uses rows [max(0, k-window_size) : k) as its
    input sequence (previous residuals included), and predicts targets at
    row k.

    Parameters
    ----------
    df : pd.DataFrame
    window_size : int
    feature_cols : list[str] or None
        Which per-step feature columns to use. Defaults to
        STEP_FEATURE_COLUMNS (the "original" variant) for backward
        compatibility with existing callers (e.g. evaluate_model.py's
        default single-model evaluation path). Columns not present in
        `df` are silently dropped so this degrades gracefully rather than
        raising.
    """
    if feature_cols is None:
        feature_cols = STEP_FEATURE_COLUMNS

    sequences, statics, targets, meta = [], [], [], []

    feature_cols = [c for c in feature_cols if c in df.columns]
    n_features = len(feature_cols) + 1  # +1 for previous residual (along-track)

    for (object_id, horizon_h), g in df.groupby(["object_id", "horizon_hours"], sort=False):
        g = g.sort_values("current_epoch_utc").reset_index(drop=True)
        if len(g) < 2:
            continue

        step_feats = g[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
        prev_resid = g["along_track_residual_km"].shift(1).fillna(0.0).values

        step_matrix = np.concatenate([step_feats, prev_resid.reshape(-1, 1)], axis=1)

        for k in range(1, len(g)):
            start = max(0, k - window_size)
            seq = step_matrix[start:k]
            if len(seq) < window_size:
                pad = np.zeros((window_size - len(seq), n_features))
                seq = np.concatenate([pad, seq], axis=0)

            row = g.iloc[k]
            target_vals = row[TARGET_COLUMNS].apply(pd.to_numeric, errors="coerce").values
            if np.any(np.isnan(target_vals)):
                continue

            static = np.array([horizon_h], dtype=float)

            sequences.append(seq)
            statics.append(static)
            targets.append(target_vals.astype(float))
            meta.append((object_id, horizon_h, row["current_epoch_utc"]))

    if len(sequences) == 0:
        return None

    return {
        "sequences": np.stack(sequences).astype(np.float32),
        "statics": np.stack(statics).astype(np.float32),
        "targets": np.stack(targets).astype(np.float32),
        "meta": meta,
        "feature_cols": feature_cols,
    }


if TORCH_AVAILABLE:
    class SequenceDataset(Dataset):
        def __init__(self, seq, static, target):
            self.seq = torch.from_numpy(seq)
            self.static = torch.from_numpy(static)
            self.target = torch.from_numpy(target)

        def __len__(self):
            return len(self.seq)

        def __getitem__(self, idx):
            return self.seq[idx], self.static[idx], self.target[idx]

    class GRUResidualModel(nn.Module):
        def __init__(self, n_features, static_dim, hidden_size=64, n_targets=3):
            super().__init__()
            self.gru = nn.GRU(input_size=n_features, hidden_size=hidden_size, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(hidden_size + static_dim, 64),
                nn.ReLU(),
                nn.Linear(64, n_targets),
            )

        def forward(self, seq, static):
            _, h_n = self.gru(seq)
            h_last = h_n[-1]  # (batch, hidden_size)
            combined = torch.cat([h_last, static], dim=1)
            return self.head(combined)


def _normalize(train_seq, train_static, other_seq=None, other_static=None):
    seq_mean = train_seq.mean(axis=(0, 1), keepdims=True)
    seq_std = train_seq.std(axis=(0, 1), keepdims=True) + 1e-8
    static_mean = train_static.mean(axis=0, keepdims=True)
    static_std = train_static.std(axis=0, keepdims=True) + 1e-8

    train_seq_n = (train_seq - seq_mean) / seq_std
    train_static_n = (train_static - static_mean) / static_std

    result = [train_seq_n, train_static_n]
    if other_seq is not None:
        result.append((other_seq - seq_mean) / seq_std)
        result.append((other_static - static_mean) / static_std)

    stats = {"seq_mean": seq_mean, "seq_std": seq_std, "static_mean": static_mean, "static_std": static_std}
    return result, stats


def run(variant_name="original"):
    """
    Train ONE named feature-set variant end to end (data loading, sequence
    construction, normalization, GRU training with early stopping, and
    artifact saving), using variant-specific output paths.

    Architecture / loss / optimizer / LR / batch size / early-stopping /
    RANDOM_SEED are IDENTICAL across all variants -- only the per-step
    input feature columns differ, via FEATURE_SET_VARIANTS[variant_name].
    """
    config.ensure_dirs()

    if variant_name not in FEATURE_SET_VARIANTS:
        raise ValueError(
            f"Unknown variant '{variant_name}'. Valid options: {sorted(FEATURE_SET_VARIANTS)}"
        )
    feature_cols = FEATURE_SET_VARIANTS[variant_name]

    if not TORCH_AVAILABLE:
        logger.warning(
            "PyTorch is not installed. Skipping sequence model training. "
            "Install with `pip install torch` to enable this optional step; "
            "the rest of the pipeline is unaffected."
        )
        return

    logger.info("=== train_sequence_model: variant '%s' ===", variant_name)
    logger.info("[%s] Requested %d step feature columns.", variant_name, len(feature_cols))

    train_df = pd.read_csv(f"{config.OUTPUT_DIR}/augmented_train_dataset.csv")
    val_df = pd.read_csv(f"{config.OUTPUT_DIR}/val_dataset.csv")
    if not train_df.empty:
        train_df["current_epoch_utc"] = pd.to_datetime(train_df["current_epoch_utc"], utc=True, format="ISO8601")
    if not val_df.empty:
        val_df["current_epoch_utc"] = pd.to_datetime(val_df["current_epoch_utc"], utc=True, format="ISO8601")

    # Sequence construction uses only "original" (non-augmented) rows to
    # avoid duplicated/noisy rows breaking the temporal ordering assumption.
    if "augmentation_type" in train_df.columns:
        train_df = train_df[train_df["augmentation_type"] == "original"]

    if train_df.empty:
        logger.warning("[%s] No training data available for sequence model. Skipping.", variant_name)
        return

    logger.info("[%s] Building training sequences (window_size=%d)...", variant_name, config.SEQUENCE_WINDOW_SIZE)
    train_data = build_sequences(train_df, config.SEQUENCE_WINDOW_SIZE, feature_cols=feature_cols)
    val_data = (
        build_sequences(val_df, config.SEQUENCE_WINDOW_SIZE, feature_cols=feature_cols)
        if not val_df.empty else None
    )

    if train_data is None:
        logger.warning("[%s] Could not build any training sequences. Skipping.", variant_name)
        return

    missing = [c for c in feature_cols if c not in train_data["feature_cols"]]
    if missing:
        logger.warning(
            "[%s] %d requested feature column(s) not found in training data and were dropped: %s",
            variant_name, len(missing), missing,
        )
    logger.info(
        "[%s] Using %d step feature columns (+1 previous-residual channel). Train sequences: %d",
        variant_name, len(train_data["feature_cols"]), len(train_data["sequences"]),
    )

    if val_data is not None:
        (train_seq_n, train_static_n, val_seq_n, val_static_n), norm_stats = _normalize(
            train_data["sequences"], train_data["statics"], val_data["sequences"], val_data["statics"]
        )
        logger.info("[%s] Validation sequences: %d", variant_name, len(val_data["sequences"]))
    else:
        (train_seq_n, train_static_n), norm_stats = _normalize(train_data["sequences"], train_data["statics"])
        val_seq_n, val_static_n = None, None

    torch.manual_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_features = train_seq_n.shape[-1]
    model = GRUResidualModel(n_features=n_features, static_dim=train_static_n.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.SmoothL1Loss()

    train_dataset = SequenceDataset(train_seq_n, train_static_n, train_data["targets"])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    if val_seq_n is not None:
        val_dataset = SequenceDataset(val_seq_n, val_static_n, val_data["targets"])
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    else:
        val_loader = None

    max_epochs = 50
    patience = 7
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    train_losses, val_losses = [], []
    best_state = None

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for seq, static, tgt in train_loader:
            seq, static, tgt = seq.to(device), static.to(device), tgt.to(device)
            optimizer.zero_grad()
            pred = model(seq, static)
            loss = loss_fn(pred, tgt)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(train_loss)

        if val_loader is not None:
            model.eval()
            val_loss_total, n_val_batches = 0.0, 0
            with torch.no_grad():
                for seq, static, tgt in val_loader:
                    seq, static, tgt = seq.to(device), static.to(device), tgt.to(device)
                    pred = model(seq, static)
                    val_loss_total += loss_fn(pred, tgt).item()
                    n_val_batches += 1
            val_loss = val_loss_total / max(n_val_batches, 1)
            val_losses.append(val_loss)

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                epochs_without_improvement += 1

            logger.info("[%s] Epoch %d: train_loss=%.4f val_loss=%.4f", variant_name, epoch, train_loss, val_loss)

            if epochs_without_improvement >= patience:
                logger.info("[%s] Early stopping at epoch %d.", variant_name, epoch)
                break
        else:
            logger.info("[%s] Epoch %d: train_loss=%.4f", variant_name, epoch, train_loss)
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model_path = f"{config.MODELS_DIR}/sequence_model_{variant_name}.pt"
    scaler_path = f"{config.MODELS_DIR}/sequence_scaler_{variant_name}.joblib"
    plot_path = f"{config.PLOTS_DIR}/sequence_training_loss_{variant_name}.png"

    torch.save({
        "state_dict": model.state_dict(),
        "n_features": n_features,
        "static_dim": train_static_n.shape[-1],
        "feature_cols": train_data["feature_cols"],
        "window_size": config.SEQUENCE_WINDOW_SIZE,
        "target_columns": TARGET_COLUMNS,
        "variant_name": variant_name,
    }, model_path)

    joblib.dump(norm_stats, scaler_path)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(train_losses, label="train_loss")
        if val_losses:
            ax.plot(val_losses, label="val_loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Smooth L1 loss")
        ax.set_title(f"Sequence model training loss ({variant_name})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
    except Exception as exc:
        logger.warning("[%s] Failed to plot training loss: %s", variant_name, exc)

    logger.info(
        "[%s] train_sequence_model complete. Best val loss: %s. Saved: %s, %s, %s",
        variant_name, best_val_loss if val_loader else "n/a", model_path, scaler_path, plot_path,
    )


def _parse_args():
    parser = argparse.ArgumentParser(description="Train a single named sequence-model feature-set variant.")
    parser.add_argument(
        "--variant",
        default="original",
        choices=sorted(FEATURE_SET_VARIANTS),
        help="Which entry of FEATURE_SET_VARIANTS to train (default: original).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(variant_name=args.variant)
