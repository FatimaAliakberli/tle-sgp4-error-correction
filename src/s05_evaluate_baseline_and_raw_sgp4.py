"""
evaluate_model.py

Evaluates on the TEST SET ONLY:
    1. Raw SGP4 baseline (position_error_km, already in the dataset)
    2. SGP4 + baseline (HistGradientBoosting) residual model
    3. SGP4 + sequence (GRU) residual model, if trained

Because RTN is an orthonormal basis, the magnitude of the residual
correction error (||actual_RTN - predicted_RTN||) equals the true 3D
Cartesian position error after correction -- no need to reconstruct
Cartesian vectors.

Outputs:
    output/evaluation_summary.csv
    output/plots/horizon_error_comparison.png
    output/plots/object_error_comparison.png
    output/plots/residual_before_after.png
"""

import logging

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from s04_train_baseline_model import build_feature_matrix, LEAKY_OR_ID_COLUMNS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("s05_evaluate_baseline_and_raw_sgp4")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _metrics(errors_km):
    errors_km = np.asarray(errors_km, dtype=float)
    errors_km = errors_km[np.isfinite(errors_km)]
    if len(errors_km) == 0:
        return {k: np.nan for k in
                ["n_samples", "mae_km", "median_ae_km", "rmse_km", "p90_km", "p95_km", "max_km"]}
    return {
        "n_samples": int(len(errors_km)),
        "mae_km": float(np.mean(errors_km)),
        "median_ae_km": float(np.median(errors_km)),
        "rmse_km": float(np.sqrt(np.mean(errors_km ** 2))),
        "p90_km": float(np.percentile(errors_km, 90)),
        "p95_km": float(np.percentile(errors_km, 95)),
        "max_km": float(np.max(errors_km)),
    }


def add_baseline_predictions(df):
    """Adds columns: baseline_pred_radial/along/cross_km, baseline_corrected_error_km"""
    try:
        models = joblib.load(f"{config.MODELS_DIR}/baseline_model.joblib")
        scaler_meta = joblib.load(f"{config.MODELS_DIR}/scaler.joblib")
    except FileNotFoundError:
        logger.warning("Baseline model artifacts not found. Skipping baseline evaluation.")
        return df, False

    feature_columns = scaler_meta["feature_columns"]
    object_id_categories = pd.Index(scaler_meta["object_id_categories"])

    X, _, _ = build_feature_matrix(df, object_id_categories=object_id_categories)
    X = X.reindex(columns=feature_columns, fill_value=np.nan)

    preds = {}
    for target, model in models.items():
        try:
            preds[target] = model.predict(X)
        except Exception as exc:
            logger.warning("Prediction failed for %s: %s", target, exc)
            preds[target] = np.full(len(X), np.nan)

    df = df.copy()
    for target in config.RESIDUAL_TARGETS:
        col = f"baseline_pred_{target}"
        df[col] = preds.get(target, np.full(len(df), np.nan))

    actual = df[config.RESIDUAL_TARGETS].apply(pd.to_numeric, errors="coerce").values
    predicted = df[[f"baseline_pred_{t}" for t in config.RESIDUAL_TARGETS]].values
    diff = actual - predicted
    df["baseline_corrected_error_km"] = np.linalg.norm(diff, axis=1)

    return df, True


def add_sequence_predictions(df):
    """Adds column: sequence_corrected_error_km, if the sequence model exists."""
    if not TORCH_AVAILABLE:
        return df, False

    import torch.nn as nn
    try:
        checkpoint = torch.load(f"{config.MODELS_DIR}/sequence_model.pt", map_location="cpu", weights_only=False)
        norm_stats = joblib.load(f"{config.MODELS_DIR}/sequence_scaler.joblib")
    except FileNotFoundError:
        logger.info("Sequence model artifacts not found. Skipping sequence evaluation.")
        return df, False

    from s06_train_sequence_model import GRUResidualModel, build_sequences

    model = GRUResidualModel(
        n_features=checkpoint["n_features"],
        static_dim=checkpoint["static_dim"],
        n_targets=len(checkpoint["target_columns"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    data = build_sequences(df, checkpoint["window_size"])
    if data is None:
        logger.warning("Could not build sequences for evaluation set. Skipping sequence evaluation.")
        return df, False

    seq_n = (data["sequences"] - norm_stats["seq_mean"]) / norm_stats["seq_std"]
    static_n = (data["statics"] - norm_stats["static_mean"]) / norm_stats["static_std"]

    with torch.no_grad():
        preds = model(torch.from_numpy(seq_n.astype(np.float32)), torch.from_numpy(static_n.astype(np.float32)))
    preds = preds.numpy()

    diff = data["targets"] - preds
    seq_errors = np.linalg.norm(diff, axis=1)

    meta_df = pd.DataFrame(data["meta"], columns=["object_id", "horizon_hours", "current_epoch_utc"])
    meta_df["sequence_corrected_error_km"] = seq_errors

    df = df.merge(meta_df, on=["object_id", "horizon_hours", "current_epoch_utc"], how="left")
    return df, True


def summarize(df, group_cols, label):
    rows = []
    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["group_type"] = label

        raw = _metrics(g["position_error_km"])
        row.update({f"sgp4_{k}": v for k, v in raw.items()})

        if "baseline_corrected_error_km" in g.columns:
            corrected = _metrics(g["baseline_corrected_error_km"])
            row.update({f"baseline_{k}": v for k, v in corrected.items()})
            improved = (g["baseline_corrected_error_km"] < g["position_error_km"]).mean() * 100
            pct_improve = ((g["position_error_km"] - g["baseline_corrected_error_km"]) /
                           g["position_error_km"].replace(0, np.nan)).mean() * 100
            row["baseline_pct_samples_improved"] = improved
            row["baseline_mean_pct_improvement"] = pct_improve

        if "sequence_corrected_error_km" in g.columns and g["sequence_corrected_error_km"].notna().any():
            seq_corrected = _metrics(g["sequence_corrected_error_km"].dropna())
            row.update({f"sequence_{k}": v for k, v in seq_corrected.items()})

        rows.append(row)
    return rows


def run():
    config.ensure_dirs()

    test_df = pd.read_csv(f"{config.OUTPUT_DIR}/test_dataset.csv")
    if test_df.empty:
        logger.warning("Test dataset is empty. Nothing to evaluate.")
        pd.DataFrame().to_csv(f"{config.OUTPUT_DIR}/evaluation_summary.csv", index=False)
        return
    test_df["current_epoch_utc"] = pd.to_datetime(test_df["current_epoch_utc"], utc=True, format="ISO8601")

    test_df, has_baseline = add_baseline_predictions(test_df)
    test_df, has_sequence = add_sequence_predictions(test_df)

    all_rows = []
    all_rows.extend(summarize(test_df, ["horizon_hours"], "by_horizon"))
    all_rows.extend(summarize(test_df, ["object_id"], "by_object"))

    for range_name, (start_str, end_str) in config.TIME_RANGES.items():
        start = pd.Timestamp(start_str, tz="UTC")
        end = pd.Timestamp(end_str, tz="UTC")
        sub = test_df[(test_df["current_epoch_utc"] >= start) & (test_df["current_epoch_utc"] <= end)]
        if len(sub) > 0:
            sub = sub.copy()
            sub["time_range"] = range_name
            all_rows.extend(summarize(sub, ["time_range"], "by_time_range"))

    # low vs high error objects (median SGP4 error per object)
    obj_median_error = test_df.groupby("object_id")["position_error_km"].median()
    if len(obj_median_error) > 1:
        threshold = obj_median_error.median()
        test_df["object_error_tier"] = test_df["object_id"].map(
            lambda o: "high_error" if obj_median_error.get(o, 0) >= threshold else "low_error"
        )
        all_rows.extend(summarize(test_df, ["object_error_tier"], "by_error_tier"))

    all_rows.extend(summarize(test_df, [], "overall") if False else [])
    # overall summary (no groupby key)
    overall_row = {"group_type": "overall"}
    overall_row.update({f"sgp4_{k}": v for k, v in _metrics(test_df["position_error_km"]).items()})
    if "baseline_corrected_error_km" in test_df.columns:
        overall_row.update({f"baseline_{k}": v for k, v in _metrics(test_df["baseline_corrected_error_km"]).items()})
        overall_row["baseline_pct_samples_improved"] = (
            test_df["baseline_corrected_error_km"] < test_df["position_error_km"]
        ).mean() * 100
    if "sequence_corrected_error_km" in test_df.columns and test_df["sequence_corrected_error_km"].notna().any():
        overall_row.update({f"sequence_{k}": v for k, v in _metrics(test_df["sequence_corrected_error_km"].dropna()).items()})
    all_rows.append(overall_row)

    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv(f"{config.OUTPUT_DIR}/evaluation_summary.csv", index=False)

    # --- plots ---
    try:
        fig, ax = plt.subplots(figsize=(7, 4))
        horizons = sorted(test_df["horizon_hours"].unique())
        sgp4_means = [test_df.loc[test_df["horizon_hours"] == h, "position_error_km"].mean() for h in horizons]
        ax.plot(horizons, sgp4_means, marker="o", label="Raw SGP4")
        if has_baseline:
            baseline_means = [test_df.loc[test_df["horizon_hours"] == h, "baseline_corrected_error_km"].mean() for h in horizons]
            ax.plot(horizons, baseline_means, marker="o", label="SGP4 + baseline model")
        if has_sequence and "sequence_corrected_error_km" in test_df.columns:
            seq_means = [test_df.loc[test_df["horizon_hours"] == h, "sequence_corrected_error_km"].mean() for h in horizons]
            ax.plot(horizons, seq_means, marker="o", label="SGP4 + sequence model")
        ax.set_xlabel("Horizon (hours)")
        ax.set_ylabel("Mean position error (km)")
        ax.set_title("Error by horizon: raw SGP4 vs residual-corrected")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{config.PLOTS_DIR}/horizon_error_comparison.png", dpi=120)
        plt.close(fig)
    except Exception as exc:
        logger.warning("Failed horizon comparison plot: %s", exc)

    try:
        obj_stats = test_df.groupby("object_id")["position_error_km"].mean().sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(obj_stats.index[::-1], obj_stats.values[::-1])
        ax.set_xlabel("Mean SGP4 position error (km)")
        ax.set_title("Top 15 objects by mean SGP4 error (test set)")
        fig.tight_layout()
        fig.savefig(f"{config.PLOTS_DIR}/object_error_comparison.png", dpi=120)
        plt.close(fig)
    except Exception as exc:
        logger.warning("Failed object comparison plot: %s", exc)

    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(test_df["position_error_km"].dropna(), bins=50, alpha=0.7)
        axes[0].set_title("Before correction (raw SGP4)")
        axes[0].set_xlabel("Position error (km)")
        if has_baseline:
            axes[1].hist(test_df["baseline_corrected_error_km"].dropna(), bins=50, alpha=0.7, color="orange")
        axes[1].set_title("After baseline correction")
        axes[1].set_xlabel("Position error (km)")
        fig.tight_layout()
        fig.savefig(f"{config.PLOTS_DIR}/residual_analysis.png", dpi=120)
        fig.savefig(f"{config.PLOTS_DIR}/residual_before_after.png", dpi=120)
        plt.close(fig)
    except Exception as exc:
        logger.warning("Failed residual before/after plot: %s", exc)

    # Console summary
    print("\n=== EVALUATION SUMMARY (test set) ===")
    print(summary_df[summary_df["group_type"] == "overall"].to_string(index=False))
    print("\nBy horizon:")
    print(summary_df[summary_df["group_type"] == "by_horizon"].to_string(index=False))

    logger.info("evaluate_model complete.")


if __name__ == "__main__":
    run()
