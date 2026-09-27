"""
run_pipeline.py — Master end-to-end pipeline runner for the Amazon ML Challenge 2026.

Usage:
    python run_pipeline.py --mode train        # preprocess + block + feats + train + validate
    python run_pipeline.py --mode predict      # load model + predict on test set
    python run_pipeline.py --mode full         # train + predict (all steps)
    python run_pipeline.py --mode eda          # EDA report only

All intermediate artifacts are saved to avoid re-running expensive steps.
Set SKIP_PREPROCESS=1, SKIP_BLOCKING=1, SKIP_FEATURES=1 env vars to skip steps.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import (
    BLOCKING, DATA_PROCESSED, DATA_RAW_TEST, DATA_RAW_TRAIN,
    EXPERIMENTS_DIR, FEATURES, LGBM, MODELS_DIR, OUTPUT_DIR,
    RANDOM_SEED, SPLIT, THRESHOLD,
)
from evaluate import blocking_recall, find_best_threshold, macro_f05
from features import build_feature_matrix, get_feature_columns
from io_utils import (
    ground_truth_to_dict, load_parquet, read_ground_truth,
    read_source, save_json, save_parquet, write_candidate_pairs,
    write_matching_results,
)
from preprocess import run_preprocessing

log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(EXPERIMENTS_DIR / "pipeline.log", encoding="utf-8"),
    ]
)


# ─────────────────────────────────────────────────────────────────────────────
# EDA REPORT
# ─────────────────────────────────────────────────────────────────────────────

def run_eda() -> None:
    """Quick EDA report saved to experiments/eda_report.json."""
    log.info("=" * 60)
    log.info("PHASE 1: EDA")
    log.info("=" * 60)

    s1 = read_source(DATA_RAW_TRAIN / "train_source1.tsv", "train_s1")
    s2 = read_source(DATA_RAW_TRAIN / "train_source2.tsv", "train_s2")
    s3 = read_source(DATA_RAW_TRAIN / "train_source3.tsv", "train_s3")
    gt = read_ground_truth(DATA_RAW_TRAIN / "train_ground_truth.tsv")
    gt_dict = ground_truth_to_dict(gt)

    # Match cardinality
    match_counts = {s1_id: len(v) for s1_id, v in gt_dict.items()}
    counts_series = pd.Series(match_counts)

    n_singletons = (counts_series == 0).sum()
    n_total = len(counts_series)

    report = {
        "train_s1_count": len(s1),
        "train_s2_count": len(s2),
        "train_s3_count": len(s3),
        "ground_truth_pairs": int(counts_series.sum()),
        "s1_with_matches": int((counts_series > 0).sum()),
        "s1_singletons": int(n_singletons),
        "singleton_rate": float(n_singletons / n_total),
        "max_matches_per_s1": int(counts_series.max()),
        "mean_matches_per_s1": float(counts_series.mean()),
        "country_distribution_s1": s1["country"].value_counts().to_dict(),
        "country_distribution_s2": s2["country"].value_counts().to_dict(),
        "null_rates_s1": s1.isnull().mean().to_dict(),
        "null_rates_s2": s2.isnull().mean().to_dict(),
        "match_count_distribution": counts_series.value_counts().sort_index().to_dict(),
    }

    save_json(report, EXPERIMENTS_DIR / "eda_report.json")
    log.info("EDA report saved. Singleton rate=%.2f%%", report["singleton_rate"] * 100)
    log.info("Train: S1=%d  S2=%d  S3=%d  GT_pairs=%d",
             report["train_s1_count"], report["train_s2_count"],
             report["train_s3_count"], report["ground_truth_pairs"])
    return report


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_train_pipeline() -> tuple:
    """
    Full training pipeline:
      1. Preprocessing
      2. Blocking
      3. Feature engineering
      4. Model training + validation
    Returns (model, feat_cols, best_threshold)
    """
    t_start = time.time()

    # ── Step 1: Preprocessing ─────────────────────────────────────────────
    log.info("=" * 60)
    log.info("PHASE 2: PREPROCESSING")
    log.info("=" * 60)
    paths = run_preprocessing(DATA_RAW_TRAIN, DATA_RAW_TEST, DATA_PROCESSED)

    train_s1 = load_parquet(paths["train_s1"])
    train_s2 = load_parquet(paths["train_s2"])
    train_s3 = load_parquet(paths["train_s3"])

    # ── Step 2: Blocking ──────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("PHASE 3: BLOCKING")
    log.info("=" * 60)

    blocking_meta_path = DATA_PROCESSED / "train_blocking_meta.parquet"
    candidates_path    = DATA_PROCESSED / "train_candidates.pkl"

    gt = read_ground_truth(DATA_RAW_TRAIN / "train_ground_truth.tsv")
    gt_dict = ground_truth_to_dict(gt)

    if blocking_meta_path.exists() and candidates_path.exists():
        log.info("Loading cached blocking results ...")
        blocking_meta = load_parquet(blocking_meta_path)
        with open(candidates_path, "rb") as f:
            candidates = pickle.load(f)
    else:
        from blocking import run_blocking
        candidates, blocking_meta = run_blocking(
            train_s1, train_s2, train_s3, BLOCKING, ground_truth=gt_dict
        )
        save_parquet(blocking_meta, blocking_meta_path)
        with open(candidates_path, "wb") as f:
            pickle.dump(candidates, f)

    # Blocking recall measurement
    recall_stats = blocking_recall(candidates, gt_dict)
    log.info("Blocking recall: %.4f  (%d/%d pairs)",
             recall_stats["blocking_recall"],
             recall_stats["covered_pairs"],
             recall_stats["total_true_pairs"])
    save_json(recall_stats, EXPERIMENTS_DIR / "blocking_metrics.json")

    # ── Step 3: Feature Engineering ───────────────────────────────────────
    log.info("=" * 60)
    log.info("PHASE 4: FEATURE ENGINEERING")
    log.info("=" * 60)

    feat_path = DATA_PROCESSED / "train_features.parquet"

    if feat_path.exists():
        log.info("Loading cached features ...")
        feat_df = load_parquet(feat_path)
    else:
        pool_df = pd.concat([train_s2, train_s3], ignore_index=True)
        # Only compute features for candidate pairs (blocking output)
        feat_df = build_feature_matrix(
            blocking_meta,
            train_s1,
            pool_df,
            label_map=gt_dict,
        )
        save_parquet(feat_df, feat_path)

    log.info("Feature matrix: %d rows × %d cols", len(feat_df), len(feat_df.columns))

    # ── Step 4: Train / Val Split ─────────────────────────────────────────
    log.info("=" * 60)
    log.info("PHASE 5: MODEL TRAINING")
    log.info("=" * 60)

    from train import entity_level_split, feature_importance_report, train_lgbm

    all_s1_ids = train_s1["entity_id"].tolist()
    train_ids, val_ids, test_ids = entity_level_split(
        all_s1_ids, gt_dict,
        train_frac=SPLIT["train_frac"],
        val_frac=SPLIT["val_frac"],
        seed=SPLIT["seed"],
    )

    model, feat_cols, best_thresh, val_metrics = train_lgbm(
        feat_df, gt_dict, train_ids, val_ids,
        cfg=LGBM, neg_ratio=SPLIT["neg_ratio"],
        experiment_name="baseline"
    )

    # Feature importance
    importance_df = feature_importance_report(model, feat_cols)
    importance_df.to_csv(EXPERIMENTS_DIR / "feature_importance.csv", index=False)

    elapsed = time.time() - t_start
    log.info("Training pipeline done in %.1f seconds.", elapsed)
    log.info("Validation F_0.5=%.4f at threshold=%.4f", val_metrics["macro_f05"], best_thresh)

    return model, feat_cols, best_thresh, candidates


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION PIPELINE (Test Set)
# ─────────────────────────────────────────────────────────────────────────────

def run_predict_pipeline(
    model,
    feat_cols: list[str],
    threshold: float,
    submission_name: str = "submission_01",
) -> None:
    """
    Run inference on the test set and produce final submission files.
    """
    log.info("=" * 60)
    log.info("PHASE 6: TEST SET PREDICTION")
    log.info("=" * 60)

    # ── Load preprocessed test data ────────────────────────────────────────
    paths = run_preprocessing(DATA_RAW_TRAIN, DATA_RAW_TEST, DATA_PROCESSED)
    test_s1 = load_parquet(paths["test_s1"])
    test_s2 = load_parquet(paths["test_s2"])
    test_s3 = load_parquet(paths["test_s3"])
    pool_df = pd.concat([test_s2, test_s3], ignore_index=True)

    all_s1_ids    = test_s1["entity_id"].tolist()
    valid_pool_ids = set(pool_df["entity_id"].tolist())

    # ── Blocking on test set ───────────────────────────────────────────────
    test_blocking_meta_path = DATA_PROCESSED / "test_blocking_meta.parquet"
    test_candidates_path    = DATA_PROCESSED / "test_candidates.pkl"

    if test_blocking_meta_path.exists() and test_candidates_path.exists():
        log.info("Loading cached test blocking results ...")
        test_blocking_meta = load_parquet(test_blocking_meta_path)
        with open(test_candidates_path, "rb") as f:
            candidates = pickle.load(f)
    else:
        from blocking import run_blocking
        candidates, test_blocking_meta = run_blocking(
            test_s1, test_s2, test_s3, BLOCKING, ground_truth=None
        )
        save_parquet(test_blocking_meta, test_blocking_meta_path)
        with open(test_candidates_path, "wb") as f:
            pickle.dump(candidates, f)

    # ── Features on test candidates ────────────────────────────────────────
    test_feat_path = DATA_PROCESSED / "test_features.parquet"
    if test_feat_path.exists():
        log.info("Loading cached test features ...")
        feat_df = load_parquet(test_feat_path)
    else:
        feat_df = build_feature_matrix(test_blocking_meta, test_s1, pool_df, label_map=None)
        save_parquet(feat_df, test_feat_path)

    # ── Predict ────────────────────────────────────────────────────────────
    from predict import run_prediction

    # Build country map for per-country thresholds
    country_map = dict(zip(test_s1["entity_id"], test_s1["country_norm"]))

    predictions = run_prediction(
        model, feat_df, feat_cols,
        candidates=candidates,
        all_s1_ids=all_s1_ids,
        valid_pool_ids=valid_pool_ids,
        threshold=threshold,
        country_map=country_map,
        country_thresholds=THRESHOLD.get("country_overrides", {}),
        output_dir=OUTPUT_DIR,
        submission_name=submission_name,
        ground_truth=None,
    )

    log.info("Test prediction complete. Check output/matching_results.tsv")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 Pipeline")
    parser.add_argument("--mode", choices=["eda", "train", "predict", "full"], default="full",
                        help="Pipeline mode")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to saved model for predict mode")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Decision threshold (overrides tuned threshold)")
    parser.add_argument("--submission-name", type=str, default="submission_01",
                        help="Submission folder name")
    args = parser.parse_args()

    log.info("Amazon ML Challenge 2026 — Starting pipeline (mode=%s)", args.mode)

    if args.mode in ["eda"]:
        run_eda()

    elif args.mode in ["train", "full"]:
        run_eda()
        model, feat_cols, best_thresh, _ = run_train_pipeline()

        if args.threshold:
            best_thresh = args.threshold
            log.info("Using override threshold: %.4f", best_thresh)

        if args.mode == "full":
            run_predict_pipeline(model, feat_cols, best_thresh, args.submission_name)

    elif args.mode == "predict":
        if not args.model_path:
            # Find latest model
            model_files = sorted(MODELS_DIR.glob("*.pkl"))
            if not model_files:
                log.error("No model found. Run with --mode train first.")
                sys.exit(1)
            model_path = model_files[-1]
        else:
            model_path = Path(args.model_path)

        from train import load_model
        model = load_model(model_path)

        # Load experiment metadata to get feat_cols and threshold
        meta_files = sorted(EXPERIMENTS_DIR.glob("experiment_*.json"))
        if meta_files:
            with open(meta_files[-1]) as f:
                meta = json.load(f)
            feat_cols   = meta["feature_cols"]
            best_thresh = args.threshold or meta["best_threshold"]
        else:
            log.error("No experiment metadata found.")
            sys.exit(1)

        run_predict_pipeline(model, feat_cols, best_thresh, args.submission_name)

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
