"""
F_0.5 scoring utilities.

Used for:
  - sweeping thresholds during model training (pick the cutoff that
    maximizes macro F_0.5 on your held-out validation split)
  - self-scoring before you ever touch the real leaderboard

Only depends on pandas + stdlib.
"""

import pandas as pd


def f0_5_score(predicted_ids, true_ids):
    """Score ONE Source-1 entity. predicted_ids/true_ids: iterables of IDs."""
    predicted_ids = set(predicted_ids)
    true_ids = set(true_ids)

    if not predicted_ids and not true_ids:
        return 1.0  # correctly predicted singleton
    if not predicted_ids or not true_ids:
        return 0.0  # missed everything, or false-positived a singleton

    tp = len(predicted_ids & true_ids)
    precision = tp / len(predicted_ids)
    recall = tp / len(true_ids)

    if precision == 0 and recall == 0:
        return 0.0

    beta = 0.5
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


def _split_ids(cell):
    if pd.isna(cell) or cell == "":
        return []
    return cell.split(",")


def macro_f0_5(predictions_df, ground_truth_df):
    """
    predictions_df / ground_truth_df: columns [source1_entity_id, matched_entity_ids]
    (comma-separated string, empty for singletons). Returns the macro-averaged
    F_0.5 across every entity present in ground_truth_df.
    """
    merged = predictions_df.merge(
        ground_truth_df, on="source1_entity_id", suffixes=("_pred", "_true"), how="right"
    )
    scores = []
    for _, row in merged.iterrows():
        pred = _split_ids(row.get("matched_entity_ids_pred"))
        true = _split_ids(row.get("matched_entity_ids_true"))
        scores.append(f0_5_score(pred, true))
    return sum(scores) / len(scores) if scores else 0.0


def macro_f0_5_by_group(predictions_df, ground_truth_df, group_col, group_lookup):
    """
    Same as macro_f0_5 but also breaks the score out by a grouping key
    (e.g. country) so you can check e.g. whether performance holds up
    on entities you're less confident about.

    group_lookup: dict {source1_entity_id: group_value}
    Returns (overall_score, {group_value: score}).
    """
    merged = predictions_df.merge(
        ground_truth_df, on="source1_entity_id", suffixes=("_pred", "_true"), how="right"
    )
    per_row_scores = []
    group_scores = {}
    for _, row in merged.iterrows():
        pred = _split_ids(row.get("matched_entity_ids_pred"))
        true = _split_ids(row.get("matched_entity_ids_true"))
        s = f0_5_score(pred, true)
        per_row_scores.append(s)
        g = group_lookup.get(row["source1_entity_id"], "unknown")
        group_scores.setdefault(g, []).append(s)

    overall = sum(per_row_scores) / len(per_row_scores) if per_row_scores else 0.0
    group_avg = {g: sum(v) / len(v) for g, v in group_scores.items()}
    return overall, group_avg
