"""
diagnose_duplicate_timestamps.py

Standalone diagnostic script (read-only with respect to every existing
pipeline file; no model training or loading; pure pandas).

Hypothesis under test
----------------------
When ALL augmentation_type rows (original + gaussian_noise +
simulated_missing_tle + oversampled_high_error) are used to build sequence
training windows -- as the rolling-CV pipeline intentionally does -- many
objects likely end up with 2-4 rows sharing the EXACT SAME
current_epoch_utc within a given (object_id, horizon_hours) group, because
augmentation perturbs feature VALUES (or duplicates whole rows) without
changing the timestamp. train_sequence_model.build_sequences() sorts each
(object_id, horizon_hours) group by current_epoch_utc and slides a
fixed-size window over consecutive rows, with no augmentation_type
filtering. If several rows share a timestamp, a single training window can
contain multiple "time steps" that are really the same real-world moment
repeated with slightly different feature values -- something that never
happens at test time (test data is "original" rows only, one row per
timestamp).

This script quantifies how often that actually happens and shows concrete
example windows, without importing or exercising any model code.

Outputs:
    new-output/diagnostics_duplicate_timestamps_summary.csv
    new-output/diagnostics_duplicate_timestamps_examples.csv
"""

import logging

import numpy as np
import pandas as pd

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("s13_diagnose_duplicate_timestamps")

TRAIN_PATH_NAME = "augmented_train_dataset.csv"
N_EXAMPLE_GROUPS = 20
CLUSTER_SIZE_BUCKET_LABELS = ["1", "2", "3", "4+"]


def _bucket_cluster_size(size: int) -> str:
    if size <= 1:
        return "1"
    if size == 2:
        return "2"
    if size == 3:
        return "3"
    return "4+"


def load_train_dataset() -> pd.DataFrame:
    train_path = f"{config.OUTPUT_DIR}/{TRAIN_PATH_NAME}"
    logger.info("Loading augmented training dataset from %s ...", train_path)
    df = pd.read_csv(train_path)
    if df.empty:
        return df
    df["current_epoch_utc"] = pd.to_datetime(
        df["current_epoch_utc"], utc=True, format="ISO8601"
    )
    return df


def report_overall_composition(df: pd.DataFrame) -> None:
    total = len(df)
    logger.info("Overall training dataset composition (%d rows total):", total)
    if "augmentation_type" not in df.columns:
        logger.warning("Column 'augmentation_type' not found; skipping composition report.")
        return
    counts = df["augmentation_type"].value_counts(dropna=False)
    pct = (counts / total * 100.0).round(3)
    comp_df = pd.DataFrame({"row_count": counts, "pct_of_total": pct})
    print("\n=== Overall training-row composition by augmentation_type ===")
    print(comp_df.to_string())
    for aug_type, row in comp_df.iterrows():
        logger.info(
            "  %-25s %8d rows (%.3f%%)", str(aug_type), int(row["row_count"]), row["pct_of_total"]
        )


def compute_timestamp_cluster_sizes(df: pd.DataFrame) -> pd.Series:
    """
    For every row, returns the number of rows (including itself) that share
    the exact same current_epoch_utc within that row's own
    (object_id, horizon_hours) group. Index-aligned with df.
    """
    cluster_sizes = df.groupby(
        ["object_id", "horizon_hours", "current_epoch_utc"]
    )["current_epoch_utc"].transform("size")
    return cluster_sizes


def compute_per_group_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (object_id, horizon_hours) group: number of rows involved in a
    duplicate-timestamp cluster (ts_cluster_size > 1) and the largest
    cluster size seen in that group. One row per group.
    """
    grouped = df.groupby(["object_id", "horizon_hours"]).agg(
        n_rows=("ts_cluster_size", "size"),
        n_duplicate_rows=("ts_cluster_size", lambda s: int((s > 1).sum())),
        max_cluster_size=("ts_cluster_size", "max"),
    ).reset_index()
    return grouped


def compute_aggregate_stats(df: pd.DataFrame, per_group: pd.DataFrame) -> dict:
    total_rows = len(df)
    rows_with_dup = int((df["ts_cluster_size"] > 1).sum())
    pct_rows_with_dup = (rows_with_dup / total_rows * 100.0) if total_rows > 0 else np.nan

    # Distinct (object_id, horizon_hours, current_epoch_utc) clusters, sized.
    cluster_table = df.groupby(
        ["object_id", "horizon_hours", "current_epoch_utc"]
    ).size().reset_index(name="cluster_size")
    cluster_table["bucket"] = cluster_table["cluster_size"].apply(_bucket_cluster_size)

    dist_rows = []
    for bucket in CLUSTER_SIZE_BUCKET_LABELS:
        bucket_clusters = cluster_table[cluster_table["bucket"] == bucket]
        n_clusters = len(bucket_clusters)
        n_rows_in_bucket = int(bucket_clusters["cluster_size"].sum())
        dist_rows.append({
            "cluster_size_bucket": bucket,
            "num_timestamp_clusters": n_clusters,
            "num_rows_in_bucket": n_rows_in_bucket,
            "pct_of_all_rows": (n_rows_in_bucket / total_rows * 100.0) if total_rows > 0 else np.nan,
        })
    distribution_df = pd.DataFrame(dist_rows)

    # Breakdown by augmentation_type, restricted to rows inside a
    # duplicate-timestamp cluster (ts_cluster_size > 1).
    dup_rows_df = df[df["ts_cluster_size"] > 1]
    if "augmentation_type" in df.columns and len(dup_rows_df) > 0:
        aug_counts = dup_rows_df["augmentation_type"].value_counts(dropna=False)
        aug_pct = (aug_counts / len(dup_rows_df) * 100.0).round(3)
        aug_breakdown_df = pd.DataFrame({
            "augmentation_type": aug_counts.index,
            "count_in_duplicate_clusters": aug_counts.values,
            "pct_of_duplicate_cluster_rows": aug_pct.values,
        })
    else:
        aug_breakdown_df = pd.DataFrame(
            columns=["augmentation_type", "count_in_duplicate_clusters", "pct_of_duplicate_cluster_rows"]
        )

    n_groups_total = len(per_group)
    n_groups_with_dup = int((per_group["n_duplicate_rows"] > 0).sum())

    return {
        "total_rows": total_rows,
        "rows_with_duplicate_timestamp": rows_with_dup,
        "pct_rows_with_duplicate_timestamp": pct_rows_with_dup,
        "n_object_horizon_groups": n_groups_total,
        "n_object_horizon_groups_with_duplicates": n_groups_with_dup,
        "pct_object_horizon_groups_with_duplicates": (
            n_groups_with_dup / n_groups_total * 100.0 if n_groups_total > 0 else np.nan
        ),
        "distribution_df": distribution_df,
        "augmentation_breakdown_df": aug_breakdown_df,
    }


def find_best_window(sub_df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Given all rows for one (object_id, horizon_hours) group (already sorted
    by current_epoch_utc, index reset), return the window_size-row
    consecutive slice that contains the most duplicate-timestamp rows --
    i.e. the clearest example of what build_sequences() would hand the
    model for this group. Assumes len(sub_df) >= window_size.
    """
    n = len(sub_df)
    sizes = sub_df["ts_cluster_size"].to_numpy()
    best_start = 0
    best_dup_count = -1
    for start in range(0, n - window_size + 1):
        dup_count = int(np.sum(sizes[start:start + window_size] > 1))
        if dup_count > best_dup_count:
            best_dup_count = dup_count
            best_start = start
    return sub_df.iloc[best_start:best_start + window_size]


def build_example_windows(df: pd.DataFrame, per_group: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Select up to N_EXAMPLE_GROUPS (object_id, horizon_hours) groups with the
    most duplicate-timestamp rows (and at least window_size total rows, so a
    full sliding window can actually be formed the way build_sequences()
    would form one), and for each, extract the single clearest example
    window.
    """
    candidates = per_group[
        (per_group["n_duplicate_rows"] > 0) & (per_group["n_rows"] >= window_size)
    ].sort_values(
        ["n_duplicate_rows", "max_cluster_size"], ascending=False
    ).head(N_EXAMPLE_GROUPS)

    if candidates.empty:
        logger.warning(
            "No (object_id, horizon_hours) group has both a duplicate timestamp AND at least "
            "SEQUENCE_WINDOW_SIZE=%d total rows, so no example windows can be built.",
            window_size,
        )
        return pd.DataFrame(
            columns=["object_id", "horizon_hours", "window_index", "step_index_in_window",
                     "current_epoch_utc", "augmentation_type"]
        )

    example_rows = []
    df_sorted = df.sort_values(["object_id", "horizon_hours", "current_epoch_utc"])
    grouped = df_sorted.groupby(["object_id", "horizon_hours"])

    for window_index, (_, cand_row) in enumerate(candidates.iterrows()):
        object_id = cand_row["object_id"]
        horizon_hours = cand_row["horizon_hours"]
        try:
            sub_df = grouped.get_group((object_id, horizon_hours)).reset_index(drop=True)
        except KeyError:
            logger.warning(
                "Could not re-fetch group (object_id=%s, horizon_hours=%s); skipping.",
                object_id, horizon_hours,
            )
            continue

        window_df = find_best_window(sub_df, window_size)
        for step_index, (_, row) in enumerate(window_df.iterrows()):
            example_rows.append({
                "object_id": object_id,
                "horizon_hours": horizon_hours,
                "window_index": window_index,
                "step_index_in_window": step_index,
                "current_epoch_utc": row["current_epoch_utc"],
                "augmentation_type": row.get("augmentation_type", np.nan),
            })

        n_dup_in_window = int((window_df["ts_cluster_size"] > 1).sum())
        logger.info(
            "Example window %d/%d: object_id=%s horizon_hours=%s -- %d/%d steps in this window "
            "share a timestamp with another step (group max cluster size: %d, group total dup "
            "rows: %d)",
            window_index + 1, len(candidates), object_id, horizon_hours,
            n_dup_in_window, window_size, cand_row["max_cluster_size"], cand_row["n_duplicate_rows"],
        )

    return pd.DataFrame(example_rows)


def save_aggregate_summary(agg_stats: dict, summary_path: str) -> None:
    scalar_df = pd.DataFrame([{
        "metric": "total_rows",
        "value": agg_stats["total_rows"],
    }, {
        "metric": "rows_with_duplicate_timestamp",
        "value": agg_stats["rows_with_duplicate_timestamp"],
    }, {
        "metric": "pct_rows_with_duplicate_timestamp",
        "value": agg_stats["pct_rows_with_duplicate_timestamp"],
    }, {
        "metric": "n_object_horizon_groups",
        "value": agg_stats["n_object_horizon_groups"],
    }, {
        "metric": "n_object_horizon_groups_with_duplicates",
        "value": agg_stats["n_object_horizon_groups_with_duplicates"],
    }, {
        "metric": "pct_object_horizon_groups_with_duplicates",
        "value": agg_stats["pct_object_horizon_groups_with_duplicates"],
    }])

    with open(summary_path, "w", newline="") as f:
        f.write("# section: overall_scalar_stats\n")
        scalar_df.to_csv(f, index=False)
        f.write("\n# section: timestamp_cluster_size_distribution\n")
        agg_stats["distribution_df"].to_csv(f, index=False)
        f.write("\n# section: augmentation_type_breakdown_of_duplicate_rows\n")
        agg_stats["augmentation_breakdown_df"].to_csv(f, index=False)


def run():
    config.ensure_dirs()

    df = load_train_dataset()
    if df.empty:
        logger.warning("Augmented training dataset is empty. Nothing to diagnose.")
        pd.DataFrame().to_csv(f"{config.OUTPUT_DIR}/diagnostics_duplicate_timestamps_summary.csv", index=False)
        pd.DataFrame().to_csv(f"{config.OUTPUT_DIR}/diagnostics_duplicate_timestamps_examples.csv", index=False)
        return

    required_cols = {"object_id", "horizon_hours", "current_epoch_utc"}
    missing = required_cols - set(df.columns)
    if missing:
        logger.error("Training dataset is missing required columns %s. Aborting.", missing)
        return

    # --- point 2: overall composition ---
    report_overall_composition(df)

    # --- point 3: per-(object_id, horizon_hours) duplicate-timestamp stats ---
    logger.info("Computing per-row timestamp cluster sizes within each (object_id, horizon_hours) group...")
    df["ts_cluster_size"] = compute_timestamp_cluster_sizes(df)

    logger.info("Aggregating duplicate-timestamp stats per (object_id, horizon_hours) group...")
    per_group = compute_per_group_stats(df)
    logger.info(
        "Computed stats for %d (object_id, horizon_hours) groups; %d of them contain at least "
        "one duplicate timestamp.",
        len(per_group), int((per_group["n_duplicate_rows"] > 0).sum()),
    )

    # --- point 4: aggregate across all groups ---
    logger.info("Computing dataset-wide aggregate duplicate-timestamp statistics...")
    agg_stats = compute_aggregate_stats(df, per_group)

    print("\n=== Timestamp-cluster-size distribution (distinct (object_id, horizon_hours, "
          "current_epoch_utc) clusters) ===")
    print(agg_stats["distribution_df"].to_string(index=False))

    print("\n=== Augmentation-type breakdown of rows inside a duplicate-timestamp cluster ===")
    if agg_stats["augmentation_breakdown_df"].empty:
        print("(no duplicate-timestamp rows found)")
    else:
        print(agg_stats["augmentation_breakdown_df"].to_string(index=False))

    # --- point 5: simulate build_sequences() windows for ~20 example groups ---
    window_size = config.SEQUENCE_WINDOW_SIZE
    logger.info(
        "Building up to %d example sliding windows (window size=%d, matching "
        "config.SEQUENCE_WINDOW_SIZE) from the groups with the most duplicate-timestamp rows...",
        N_EXAMPLE_GROUPS, window_size,
    )
    examples_df = build_example_windows(df, per_group, window_size)

    # --- point 6: save outputs ---
    summary_path = f"{config.OUTPUT_DIR}/diagnostics_duplicate_timestamps_summary.csv"
    save_aggregate_summary(agg_stats, summary_path)
    logger.info("Saved aggregate duplicate-timestamp summary to %s", summary_path)

    examples_path = f"{config.OUTPUT_DIR}/diagnostics_duplicate_timestamps_examples.csv"
    examples_df.to_csv(examples_path, index=False)
    logger.info(
        "Saved %d example window rows (from %d distinct windows) to %s",
        len(examples_df), examples_df["window_index"].nunique() if not examples_df.empty else 0,
        examples_path,
    )

    # --- point 8: final console verdict ---
    pct_affected = agg_stats["pct_rows_with_duplicate_timestamp"]
    print("\n=== FINAL SUMMARY ===")
    print(
        f"{agg_stats['rows_with_duplicate_timestamp']:,} / {agg_stats['total_rows']:,} "
        f"training rows ({pct_affected:.2f}%) share an exact current_epoch_utc with at least "
        f"one other row within their own (object_id, horizon_hours) group."
    )
    print(
        f"{agg_stats['n_object_horizon_groups_with_duplicates']:,} / "
        f"{agg_stats['n_object_horizon_groups']:,} (object_id, horizon_hours) groups "
        f"({agg_stats['pct_object_horizon_groups_with_duplicates']:.2f}%) contain at least one "
        f"duplicate-timestamp cluster."
    )

    if np.isnan(pct_affected):
        verdict = "Could not compute a verdict (no rows to evaluate)."
    elif pct_affected >= 15.0:
        verdict = (
            f"VERDICT: {pct_affected:.1f}% of training rows share a timestamp with 1+ other "
            f"rows -- this looks like a MEANINGFUL contributor to training-data quality issues. "
            f"A large share of sliding windows built by build_sequences() likely contain "
            f"repeated/near-repeated time steps that never occur at test time."
        )
    elif pct_affected >= 5.0:
        verdict = (
            f"VERDICT: {pct_affected:.1f}% of training rows share a timestamp with 1+ other "
            f"rows -- this is a MODERATE effect, worth checking against per-variant validation "
            f"performance rather than dismissing outright."
        )
    else:
        verdict = (
            f"VERDICT: only {pct_affected:.1f}% of training rows share a timestamp with 1+ other "
            f"rows -- this looks like a MINOR effect on its own, though it may still matter for "
            f"specific heavily-augmented objects (see the example windows above)."
        )
    print(verdict)

    logger.info("diagnose_duplicate_timestamps complete.")


if __name__ == "__main__":
    run()
