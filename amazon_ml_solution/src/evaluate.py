"""
evaluate.py — Exact local F_0.5 scorer (macro-averaged, entity-level).

The formula MUST match the competition specification:
  F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
  Macro-averaged across ALL Source 1 entities.
  Singletons: correct empty-for-empty prediction = 1.0 per entity.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-ENTITY F_0.5
# ─────────────────────────────────────────────────────────────────────────────

def entity_f05(predicted: set, ground_truth: set) -> float:
    """
    Compute F_0.5 for a single Source 1 entity.

    Both inputs are sets of S2/S3 IDs.
    Returns 1.0 for correct empty predictions (true singletons).
    Returns 0.0 if either is empty while the other is not.
    """
    pred = set(predicted) if predicted else set()
    true = set(ground_truth) if ground_truth else set()

    # Both empty → singleton correctly predicted
    if not pred and not true:
        return 1.0

    tp = len(pred & true)

    precision = tp / len(pred) if pred else 0.0
    recall    = tp / len(true) if true else 0.0

    if precision == 0.0 or recall == 0.0:
        return 0.0

    return (1.25 * precision * recall) / (0.25 * precision + recall)


# ─────────────────────────────────────────────────────────────────────────────
# MACRO F_0.5 OVER ALL S1 ENTITIES
# ─────────────────────────────────────────────────────────────────────────────

def macro_f05(
    predictions: dict[str, set],
    ground_truth: dict[str, set],
) -> dict:
    """
    Compute macro-averaged F_0.5 over every S1 entity in ground_truth.

    Args:
        predictions:  {s1_id: set_of_predicted_ids}
        ground_truth: {s1_id: set_of_true_ids}

    Returns:
        dict with keys:
            macro_f05, mean_precision, mean_recall,
            singleton_f05, multi_match_f05,
            n_entities, n_singletons, n_multi
    """
    per_entity = []
    precisions = []
    recalls    = []
    singleton_scores  = []
    multi_scores      = []

    for s1_id, true_ids in ground_truth.items():
        pred_ids = predictions.get(s1_id, set())
        f = entity_f05(pred_ids, true_ids)
        per_entity.append(f)

        tp = len(pred_ids & true_ids)
        prec = tp / len(pred_ids) if pred_ids else (1.0 if not true_ids else 0.0)
        rec  = tp / len(true_ids) if true_ids else (1.0 if not pred_ids else 0.0)
        precisions.append(prec)
        recalls.append(rec)

        if not true_ids:
            singleton_scores.append(f)
        else:
            multi_scores.append(f)

    result = {
        "macro_f05":       float(np.mean(per_entity)) if per_entity else 0.0,
        "mean_precision":  float(np.mean(precisions))  if precisions else 0.0,
        "mean_recall":     float(np.mean(recalls))     if recalls    else 0.0,
        "singleton_f05":   float(np.mean(singleton_scores)) if singleton_scores else None,
        "multi_match_f05": float(np.mean(multi_scores))     if multi_scores     else None,
        "n_entities":      len(per_entity),
        "n_singletons":    len(singleton_scores),
        "n_multi":         len(multi_scores),
    }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD SWEEP
# ─────────────────────────────────────────────────────────────────────────────

def threshold_sweep(
    scores_df: pd.DataFrame,
    ground_truth: dict[str, set],
    lo: float = 0.05,
    hi: float = 0.95,
    step: float = 0.01,
) -> pd.DataFrame:
    """
    Sweep decision thresholds and compute macro F_0.5 at each.

    Args:
        scores_df: DataFrame with columns [source1_entity_id, candidate_entity_id, score]
        ground_truth: {s1_id: set_of_true_ids}
        lo, hi, step: threshold grid

    Returns:
        DataFrame sorted by macro_f05 descending with columns:
            threshold, macro_f05, mean_precision, mean_recall, n_predictions
    """
    thresholds = np.arange(lo, hi + step / 2, step)
    rows = []

    for t in thresholds:
        preds: dict[str, set] = {}
        # Initialize all S1 entities with empty prediction
        for s1_id in ground_truth:
            preds[s1_id] = set()

        above = scores_df[scores_df["score"] >= t]
        for _, row in above.iterrows():
            s1 = row["source1_entity_id"]
            if s1 in preds:
                preds[s1].add(row["candidate_entity_id"])

        metrics = macro_f05(preds, ground_truth)
        metrics["threshold"] = round(float(t), 4)
        metrics["n_predictions"] = int(above.shape[0])
        rows.append(metrics)

    result = pd.DataFrame(rows).sort_values("macro_f05", ascending=False)
    return result


def find_best_threshold(
    scores_df: pd.DataFrame,
    ground_truth: dict[str, set],
    lo: float = 0.05,
    hi: float = 0.95,
    step: float = 0.01,
    verbose: bool = True,
) -> tuple[float, dict]:
    """Return (best_threshold, metrics_at_best_threshold)."""
    sweep = threshold_sweep(scores_df, ground_truth, lo, hi, step)
    best_row = sweep.iloc[0]
    best_t   = float(best_row["threshold"])
    best_metrics = best_row.to_dict()

    if verbose:
        log.info(
            "Best threshold: %.4f  |  macro F0.5=%.4f  |  prec=%.4f  |  rec=%.4f",
            best_t,
            best_metrics["macro_f05"],
            best_metrics["mean_precision"],
            best_metrics["mean_recall"],
        )
    return best_t, best_metrics


# ─────────────────────────────────────────────────────────────────────────────
# BLOCKING RECALL MEASUREMENT
# ─────────────────────────────────────────────────────────────────────────────

def blocking_recall(
    candidates: dict[str, set],
    ground_truth: dict[str, set],
) -> dict:
    """
    Measure how many true match pairs are contained in the candidate set.

    Args:
        candidates:   {s1_id: set_of_candidate_ids}
        ground_truth: {s1_id: set_of_true_ids}

    Returns:
        dict with total_true_pairs, covered_pairs, recall, missed_pairs
    """
    total = 0
    covered = 0
    missed = []

    for s1_id, true_ids in ground_truth.items():
        if not true_ids:
            continue
        cands = candidates.get(s1_id, set())
        for tid in true_ids:
            total += 1
            if tid in cands:
                covered += 1
            else:
                missed.append((s1_id, tid))

    recall = covered / total if total > 0 else 1.0
    return {
        "total_true_pairs": total,
        "covered_pairs":    covered,
        "blocking_recall":  recall,
        "missed_pairs":     len(missed),
        "missed_examples":  missed[:20],   # show first 20 for debugging
    }


if __name__ == "__main__":
    # Quick sanity check
    gt   = {"S1-001": {"S2-010", "S3-020"}, "S1-002": set()}
    pred = {"S1-001": {"S2-010"},            "S1-002": set()}
    m = macro_f05(pred, gt)
    print("Sanity check:", m)
    # S1-001: precision=1.0, recall=0.5 → F0.5 = (1.25*1.0*0.5)/(0.25*1.0+0.5) = 0.625/0.75 = 0.833
    # S1-002: both empty → 1.0
    # macro = (0.833 + 1.0) / 2 = 0.917
    assert abs(m["macro_f05"] - 0.9166666) < 1e-4, f"Got {m['macro_f05']}"
    print("Sanity check PASSED.")
