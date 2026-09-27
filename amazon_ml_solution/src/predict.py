"""
predict.py — Inference pipeline for the Amazon ML Challenge.

Takes a trained LightGBM model + candidate pairs + features
and produces:
  - matching_results.tsv  (leaderboard submission)
  - candidate_pairs.tsv   (blocking artifact)

Applies per-entity threshold and singleton protection.
Includes all consistency checks before writing output.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import OUTPUT_DIR, THRESHOLD
from evaluate import macro_f05
from features import get_feature_columns
from io_utils import write_candidate_pairs, write_matching_results

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SCORE CANDIDATES
# ─────────────────────────────────────────────────────────────────────────────

def score_candidates(
    model,
    feat_df: pd.DataFrame,
    feat_cols: list[str],
    batch_size: int = 500_000,
) -> pd.DataFrame:
    """
    Score all candidate pairs with the trained model.
    Returns feat_df with an added 'score' column.
    """
    log.info("Scoring %d candidate pairs ...", len(feat_df))
    X = feat_df[feat_cols].values

    scores = np.empty(len(feat_df), dtype=np.float32)
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        batch_proba = model.predict_proba(X[start:end])[:, 1]
        scores[start:end] = batch_proba
        log.info("  ... scored %d / %d", end, len(X))

    feat_df = feat_df.copy()
    feat_df["score"] = scores
    return feat_df


# ─────────────────────────────────────────────────────────────────────────────
# APPLY THRESHOLD
# ─────────────────────────────────────────────────────────────────────────────

def apply_threshold(
    scores_df: pd.DataFrame,
    all_s1_ids: list[str],
    threshold: float,
    country_map: Optional[dict[str, str]] = None,
    country_thresholds: Optional[dict[str, float]] = None,
    fallback_threshold: Optional[float] = None,
) -> dict[str, set[str]]:
    """
    Convert pair scores into per-S1-entity match predictions.

    Args:
        scores_df:           DataFrame with [source1_entity_id, candidate_entity_id, score]
        all_s1_ids:          All S1 entity IDs (to include singletons with empty predictions)
        threshold:           Global decision threshold
        country_map:         {s1_id: country_norm} for per-country threshold lookup
        country_thresholds:  {country: threshold} overrides
        fallback_threshold:  For unseen countries (e.g., France). Defaults to global threshold.

    Returns:
        {s1_id: set_of_matched_ids}
    """
    # Initialize all S1 with empty prediction (singletons by default)
    predictions: dict[str, set[str]] = {s1: set() for s1 in all_s1_ids}

    for _, row in scores_df.iterrows():
        s1_id   = row["source1_entity_id"]
        cand_id = row["candidate_entity_id"]
        score   = row["score"]

        # Determine effective threshold for this entity
        if country_map and country_thresholds:
            country = country_map.get(s1_id, "")
            t = country_thresholds.get(
                country,
                fallback_threshold if fallback_threshold is not None else threshold
            )
        else:
            t = threshold

        if score >= t:
            predictions[s1_id].add(cand_id)

    n_matched   = sum(1 for v in predictions.values() if v)
    n_singleton = sum(1 for v in predictions.values() if not v)
    log.info(
        "Threshold=%.4f → %d entities with matches, %d predicted singletons.",
        threshold, n_matched, n_singleton
    )
    return predictions


# ─────────────────────────────────────────────────────────────────────────────
# CONSISTENCY CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def validate_predictions(
    predictions: dict[str, set[str]],
    candidates: dict[str, set[str]],
    all_s1_ids: list[str],
    valid_pool_ids: set[str],
) -> list[str]:
    """
    Run all consistency checks. Returns list of error messages (empty = pass).
    """
    errors = []

    # Check every S1 entity has a row
    missing = set(all_s1_ids) - set(predictions.keys())
    if missing:
        errors.append(f"Missing S1 entities: {len(missing)} (e.g. {list(missing)[:3]})")

    # Check every prediction ID is S2/S3 (not S1)
    for s1_id, matched in predictions.items():
        self_matches = {m for m in matched if m.startswith("S1-")}
        if self_matches:
            errors.append(f"S1 self-matches for {s1_id}: {self_matches}")

        # Check IDs exist in test pool
        unknown = matched - valid_pool_ids
        if unknown:
            errors.append(f"Unknown pool IDs for {s1_id}: {unknown}")

        # Check no duplicates (sets guarantee this, but double-check)
        if len(matched) != len(set(matched)):
            errors.append(f"Duplicate matched IDs for {s1_id}")

        # Check matches are subset of candidates
        cands = candidates.get(s1_id, set())
        not_in_cands = matched - cands
        if not_in_cands:
            errors.append(f"Matched IDs not in candidates for {s1_id}: {not_in_cands}")

    return errors


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PREDICTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_prediction(
    model,
    feat_df: pd.DataFrame,
    feat_cols: list[str],
    candidates: dict[str, set[str]],
    all_s1_ids: list[str],
    valid_pool_ids: set[str],
    threshold: float,
    country_map: Optional[dict[str, str]] = None,
    country_thresholds: Optional[dict[str, float]] = None,
    output_dir: Path = OUTPUT_DIR,
    submission_name: str = "submission_01",
    ground_truth: Optional[dict[str, set[str]]] = None,
) -> dict[str, set[str]]:
    """
    Full prediction pipeline:
      1. Score all candidate pairs
      2. Apply threshold → match predictions
      3. Consistency check
      4. Write output files
      5. Optionally compute local F_0.5 if ground_truth provided

    Returns: predictions dict
    """
    # ── Score ──────────────────────────────────────────────────────────────
    scored_df = score_candidates(model, feat_df, feat_cols)

    scores_df = scored_df[["source1_entity_id", "candidate_entity_id", "score"]]

    # ── Threshold ──────────────────────────────────────────────────────────
    predictions = apply_threshold(
        scores_df, all_s1_ids, threshold,
        country_map=country_map,
        country_thresholds=country_thresholds,
        fallback_threshold=THRESHOLD.get("unseen_country_fallback") or threshold,
    )

    # ── Consistency checks ─────────────────────────────────────────────────
    errors = validate_predictions(predictions, candidates, all_s1_ids, valid_pool_ids)
    if errors:
        log.error("PREDICTION CONSISTENCY ERRORS:")
        for e in errors:
            log.error("  %s", e)
        log.warning("Attempting auto-fix ...")
        # Auto-fix: remove bad IDs
        for s1_id in list(predictions.keys()):
            predictions[s1_id] = {
                m for m in predictions[s1_id]
                if not m.startswith("S1-")
                and m in valid_pool_ids
                and m in candidates.get(s1_id, set())
            }
    else:
        log.info("All consistency checks PASSED.")

    # ── Write output files ─────────────────────────────────────────────────
    out_dir = output_dir / submission_name
    out_dir.mkdir(parents=True, exist_ok=True)

    write_matching_results(predictions, out_dir / "matching_results.tsv")
    write_candidate_pairs(candidates, out_dir / "candidate_pairs.tsv")

    # Also write to top-level output/ for the validator script
    write_matching_results(predictions, output_dir / "matching_results.tsv")
    write_candidate_pairs(candidates, output_dir / "candidate_pairs.tsv")

    # ── Local F_0.5 (if ground truth available) ────────────────────────────
    if ground_truth:
        metrics = macro_f05(predictions, ground_truth)
        log.info(
            "LOCAL F_0.5 = %.4f  (precision=%.4f  recall=%.4f  "
            "singleton_f05=%.4f  multi_match_f05=%.4f)",
            metrics["macro_f05"],
            metrics["mean_precision"],
            metrics["mean_recall"],
            metrics.get("singleton_f05") or -1,
            metrics.get("multi_match_f05") or -1,
        )
        import json
        with open(out_dir / "local_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    log.info("Submission written to %s", out_dir)
    return predictions
