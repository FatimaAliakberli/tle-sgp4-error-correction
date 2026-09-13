"""
train_baseline_model.py

Trains gradient-boosted residual-correction models using scikit-learn.

Design choice (documented per spec):
    We train ONE model PER RESIDUAL COMPONENT (radial / along-track /
    cross-track), each using `horizon_hours` as an input feature rather
    than training 9 separate horizon-specific models. This is the
    "simpler, robust" option: it lets each model share statistical
    strength across horizons (12h/24h/36h error patterns are correlated),
    keeps the number of trained artifacts small (3 instead of 9), and
    still lets the model learn horizon-dependent behavior because
    horizon_hours is available as a plain numeric feature to the trees.

Model: sklearn.ensemble.HistGradientBoostingRegressor (native NaN support,
fast, robust to unscaled/heterogeneous features -- ideal for this tabular
orbital-feature problem).

Outputs:
    output/models/baseline_model.joblib      (dict of 3 fitted models)
    output/models/scaler.joblib              (feature imputer/encoder state)
    output/feature_importance.csv
    output/plots/baseline_feature_importance.png
"""

import logging
import json

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("s04_train_baseline_model")

# Columns that must NEVER be used as model inputs (identifiers, labels, or
# anything derived from the future truth TLE).
LEAKY_OR_ID_COLUMNS = {
    "object_id", "current_epoch_utc", "target_epoch_utc",
    "position_error_km", "velocity_error_km_s",
    "radial_residual_km", "along_track_residual_km", "cross_track_residual_km",
    "delta_x_km", "delta_y_km", "delta_z_km", "rtn_valid",
    "truth_pos_x_km", "truth_pos_y_km", "truth_pos_z_km",
    "truth_tle_epoch_utc", "truth_offset_hours",
    "outlier_flag", "huge_gap_flag", "truth_offset_flag",
    "augmentation_type",
}


def build_feature_matrix(df, object_id_categories=None):
    """
    Build the model input matrix X (numeric only, object_id label-encoded).

    Returns
    -------
    X: pd.DataFrame, feature_columns: list[str], object_id_categories: pd.Index
    """
    df = df.copy()

    if object_id_categories is None:
        object_id_categories = pd.Index(sorted(df["object_id"].astype(str).unique()))

    object_id_map = {cat: i for i, cat in enumerate(object_id_categories)}
    df["object_id_encoded"] = df["object_id"].astype(str).map(object_id_map).fillna(-1).astype(int)

    feature_columns = [
        c for c in df.columns
        if c not in LEAKY_OR_ID_COLUMNS and c != "object_id_encoded"
        and df[c].dtype.kind in "fibc"
    ]
    feature_columns = sorted(set(feature_columns))
    feature_columns = ["object_id_encoded"] + feature_columns

    X = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    return X, feature_columns, object_id_categories


def train_one_target(X_train, y_train, X_val, y_val, target_name):
    model = HistGradientBoostingRegressor(
        random_state=config.RANDOM_SEED,
        max_iter=300,
        early_stopping=True,
        validation_fraction=None if X_val is None else 0.1,
        l2_regularization=0.1,
    )

    if X_val is not None and len(X_val) > 0:
        # HistGradientBoostingRegressor supports an explicit validation set
        # via early_stopping + monitor is not directly exposed; simplest
        # robust approach is to concatenate and let internal validation_fraction
        # handle it, OR fit then manually check. We fit on train only and
        # report val score separately for transparency.
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_mae = float(np.mean(np.abs(val_pred - y_val)))
        logger.info("[%s] validation MAE: %.4f km", target_name, val_mae)
    else:
        model.fit(X_train, y_train)
        val_mae = None

    return model, val_mae


def run():
    config.ensure_dirs()

    train_df = pd.read_csv(f"{config.OUTPUT_DIR}/augmented_train_dataset.csv")
    val_df = pd.read_csv(f"{config.OUTPUT_DIR}/val_dataset.csv")

    if train_df.empty:
        logger.warning("Training dataset is empty. Skipping baseline model training.")
        return

    X_train, feature_columns, object_id_categories = build_feature_matrix(train_df)
    X_val, _, _ = build_feature_matrix(val_df, object_id_categories=object_id_categories) if not val_df.empty else (None, None, None)

    models = {}
    val_maes = {}
    importances = []

    for target in config.RESIDUAL_TARGETS:
        y_train = pd.to_numeric(train_df[target], errors="coerce")
        valid_mask = y_train.notna() & X_train.notna().any(axis=1)
        Xt = X_train[valid_mask]
        yt = y_train[valid_mask]

        if len(Xt) == 0:
            logger.warning("No valid training samples for target %s. Skipping.", target)
            continue

        if X_val is not None and not val_df.empty:
            y_val = pd.to_numeric(val_df[target], errors="coerce")
            val_valid = y_val.notna() & X_val.notna().any(axis=1)
            Xv, yv = X_val[val_valid], y_val[val_valid]
        else:
            Xv, yv = None, None

        model, val_mae = train_one_target(Xt, yt, Xv, yv, target)
        models[target] = model
        val_maes[target] = val_mae

        try:
            result = permutation_importance(
                model, Xt.sample(min(2000, len(Xt)), random_state=config.RANDOM_SEED),
                yt.loc[Xt.sample(min(2000, len(Xt)), random_state=config.RANDOM_SEED).index],
                n_repeats=3, random_state=config.RANDOM_SEED, n_jobs=1,
            )
            for col, imp_mean, imp_std in zip(feature_columns, result.importances_mean, result.importances_std):
                importances.append({
                    "target": target, "feature": col,
                    "importance_mean": imp_mean, "importance_std": imp_std,
                })
        except Exception as exc:
            logger.warning("Permutation importance failed for %s: %s", target, exc)

    joblib.dump(models, f"{config.MODELS_DIR}/baseline_model.joblib")
    joblib.dump({
        "feature_columns": feature_columns,
        "object_id_categories": list(object_id_categories),
    }, f"{config.MODELS_DIR}/scaler.joblib")

    importance_df = pd.DataFrame(importances)
    importance_df.to_csv(f"{config.OUTPUT_DIR}/feature_importance.csv", index=False)

    with open(f"{config.MODELS_DIR}/baseline_model_metadata.json", "w") as f:
        json.dump({"val_mae_km": val_maes}, f, indent=2)

    # Feature importance plot for the primary target
    if not importance_df.empty:
        try:
            primary = importance_df[importance_df["target"] == config.PRIMARY_TARGET]
            primary = primary.sort_values("importance_mean", ascending=False).head(20)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(primary["feature"][::-1], primary["importance_mean"][::-1])
            ax.set_title(f"Top feature importances ({config.PRIMARY_TARGET})")
            ax.set_xlabel("Permutation importance")
            fig.tight_layout()
            fig.savefig(f"{config.PLOTS_DIR}/baseline_feature_importance.png", dpi=120)
            plt.close(fig)
        except Exception as exc:
            logger.warning("Failed to plot feature importance: %s", exc)

    logger.info("train_baseline_model complete. Validation MAE per target: %s", val_maes)


if __name__ == "__main__":
    run()
