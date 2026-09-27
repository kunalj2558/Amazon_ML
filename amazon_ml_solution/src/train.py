"""
train.py — LightGBM pairwise match classifier training.

Implements the training protocol from 04_FEATURE_MODEL_TRAINING_PLAN.md:
  - Entity-level train/val/test split (no pair-level leakage)
  - Hard negative mining from blocking candidates
  - Early stopping on validation loss
  - Model selection by validation macro F_0.5 (not AUC)
  - All artifacts saved with experiment metadata
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import EXPERIMENTS_DIR, LGBM, MODELS_DIR, RANDOM_SEED, SPLIT
from evaluate import find_best_threshold, macro_f05
from features import get_feature_columns

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / VAL / TEST SPLIT (entity-level)
# ─────────────────────────────────────────────────────────────────────────────

def entity_level_split(
    s1_ids: list[str],
    ground_truth: dict[str, set[str]],
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    seed: int   = RANDOM_SEED,
) -> tuple[set[str], set[str], set[str]]:
    """
    Split Source 1 entity IDs into train / val / test sets.
    Split is at entity level so no pair from the same S1 entity appears
    in both train and validation.
    """
    all_ids = list(s1_ids)
    np.random.seed(seed)
    np.random.shuffle(all_ids)

    n = len(all_ids)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train_ids = set(all_ids[:n_train])
    val_ids   = set(all_ids[n_train:n_train + n_val])
    test_ids  = set(all_ids[n_train + n_val:])

    log.info("Entity split: train=%d  val=%d  test=%d", len(train_ids), len(val_ids), len(test_ids))
    return train_ids, val_ids, test_ids


# ─────────────────────────────────────────────────────────────────────────────
# HARD NEGATIVE SAMPLING
# ─────────────────────────────────────────────────────────────────────────────

def build_training_pairs(
    feat_df: pd.DataFrame,
    ground_truth: dict[str, set[str]],
    train_s1_ids: set[str],
    neg_ratio: int = SPLIT["neg_ratio"],
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Build (positive + hard_negative) training DataFrame from feature matrix.

    Positives: pairs where candidate_entity_id ∈ ground_truth[source1_entity_id]
    Hard negatives: candidates (from blocking) that are NOT true matches.
    Sampled at neg_ratio hard negatives per positive.
    """
    # Filter to train S1 entities only
    train_df = feat_df[feat_df["source1_entity_id"].isin(train_s1_ids)].copy()
    log.info("Training pairs before sampling: %d", len(train_df))

    positives = train_df[train_df["label"] == 1]
    negatives = train_df[train_df["label"] == 0]

    n_pos = len(positives)
    n_neg_target = n_pos * neg_ratio

    log.info("Positives: %d | Negatives available: %d | Target negatives: %d",
             n_pos, len(negatives), n_neg_target)

    if len(negatives) > n_neg_target:
        negatives = negatives.sample(n=n_neg_target, random_state=seed)

    combined = pd.concat([positives, negatives], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    log.info("Training dataset: %d pairs (pos=%d, neg=%d)",
             len(combined), n_pos, len(negatives))
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# LIGHTGBM TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_lgbm(
    feat_df: pd.DataFrame,
    ground_truth: dict[str, set[str]],
    train_s1_ids: set[str],
    val_s1_ids:   set[str],
    cfg: dict = LGBM,
    neg_ratio: int = SPLIT["neg_ratio"],
    experiment_name: str = "exp",
) -> tuple[lgb.LGBMClassifier, list[str], float, dict]:
    """
    Train a LightGBM pairwise classifier and return the model + metadata.

    Returns:
        model:       trained LGBMClassifier
        feat_cols:   list of feature column names used
        best_thresh: F_0.5-optimized decision threshold on validation
        metrics:     dict of validation metrics
    """
    feat_cols = get_feature_columns(feat_df)
    log.info("Training with %d features.", len(feat_cols))

    # ── Build training and validation DataFrames ───────────────────────────
    train_pairs = build_training_pairs(feat_df, ground_truth, train_s1_ids, neg_ratio)
    val_df      = feat_df[feat_df["source1_entity_id"].isin(val_s1_ids)].copy()

    X_train = train_pairs[feat_cols].values
    y_train = train_pairs["label"].values

    X_val   = val_df[feat_cols].values
    y_val   = val_df["label"].values

    log.info("X_train: %s  X_val: %s", X_train.shape, X_val.shape)

    # ── Train ──────────────────────────────────────────────────────────────
    model = lgb.LGBMClassifier(**cfg)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(stopping_rounds=cfg.get("early_stopping_rounds", 100), verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    log.info("Best iteration: %d", model.best_iteration_)

    # ── Score validation set ───────────────────────────────────────────────
    val_df = val_df.copy()
    val_df["score"] = model.predict_proba(X_val)[:, 1]

    # Build val ground truth for only val S1 entities
    val_gt = {s1: gt for s1, gt in ground_truth.items() if s1 in val_s1_ids}

    # Build scores_df for threshold search
    scores_df = val_df[["source1_entity_id", "candidate_entity_id", "score"]].copy()
    scores_df.columns = ["source1_entity_id", "candidate_entity_id", "score"]

    best_thresh, val_metrics = find_best_threshold(
        scores_df, val_gt, lo=0.05, hi=0.95, step=0.01, verbose=True
    )

    log.info("Validation F_0.5 @ threshold=%.4f: %.4f", best_thresh, val_metrics["macro_f05"])

    # ── Save artifacts ─────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODELS_DIR / f"lgbm_{experiment_name}_{ts}.pkl"
    meta_path  = EXPERIMENTS_DIR / f"experiment_{experiment_name}_{ts}.json"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    meta = {
        "experiment": experiment_name,
        "timestamp": ts,
        "n_features": len(feat_cols),
        "feature_cols": feat_cols,
        "best_iteration": int(model.best_iteration_),
        "best_threshold": best_thresh,
        "validation_metrics": val_metrics,
        "lgbm_config": cfg,
        "neg_ratio": neg_ratio,
        "train_pairs": len(train_pairs),
        "n_positives": int((train_pairs["label"] == 1).sum()),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log.info("Model saved → %s", model_path)
    log.info("Experiment saved → %s", meta_path)

    return model, feat_cols, best_thresh, val_metrics


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE REPORT
# ─────────────────────────────────────────────────────────────────────────────

def feature_importance_report(model: lgb.LGBMClassifier, feat_cols: list[str]) -> pd.DataFrame:
    """Return feature importance DataFrame sorted by gain."""
    importances = model.feature_importances_
    report = pd.DataFrame({
        "feature": feat_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    log.info("Top 10 features:\n%s", report.head(10).to_string())
    return report


# ─────────────────────────────────────────────────────────────────────────────
# LOAD SAVED MODEL
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str | Path) -> lgb.LGBMClassifier:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    log.info("Loaded model from %s", model_path)
    return model
