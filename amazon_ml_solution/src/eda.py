"""
eda.py — Standalone EDA script. Run before the main pipeline.
Produces experiments/eda_report.json and prints key statistics.

Usage:
    python src/eda.py
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_RAW_TRAIN, EXPERIMENTS_DIR
from io_utils import ground_truth_to_dict, read_ground_truth, read_source

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def run_eda() -> dict:
    log.info("=" * 60)
    log.info("EDA — Loading training data ...")
    log.info("=" * 60)

    s1 = read_source(DATA_RAW_TRAIN / "train_source1.tsv", "S1")
    s2 = read_source(DATA_RAW_TRAIN / "train_source2.tsv", "S2")
    s3 = read_source(DATA_RAW_TRAIN / "train_source3.tsv", "S3")
    gt = read_ground_truth(DATA_RAW_TRAIN / "train_ground_truth.tsv")
    gt_dict = ground_truth_to_dict(gt)

    # ── Schema checks ──────────────────────────────────────────────────────
    log.info("\n--- SCHEMA CHECKS ---")
    for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
        log.info("%s: %d rows | nulls: %s", name, len(df),
                 df[["business_name","business_address","country"]].isnull().sum().to_dict())
        dup_ids = df["entity_id"].duplicated().sum()
        if dup_ids:
            log.warning("  %s has %d duplicate entity_ids!", name, dup_ids)

    # ── Ground truth analysis ──────────────────────────────────────────────
    log.info("\n--- GROUND TRUTH ANALYSIS ---")
    match_counts = {s1_id: len(ids) for s1_id, ids in gt_dict.items()}
    n = len(match_counts)
    n_singletons = sum(1 for v in match_counts.values() if v == 0)
    n_with_matches = n - n_singletons

    log.info("Total S1 entities in GT: %d", n)
    log.info("Singletons (no match): %d (%.1f%%)", n_singletons, 100*n_singletons/n)
    log.info("With at least 1 match: %d (%.1f%%)", n_with_matches, 100*n_with_matches/n)
    log.info("Max matches for single S1: %d", max(match_counts.values()))
    log.info("Avg matches (non-singleton): %.2f",
             sum(v for v in match_counts.values() if v > 0) / max(n_with_matches, 1))

    # Distribution
    dist = Counter(match_counts.values())
    log.info("Match count distribution:")
    for k in sorted(dist.keys())[:10]:
        log.info("  %d matches: %d S1 entities", k, dist[k])

    # S2 vs S3 matches
    n_s2_only = sum(1 for ids in gt_dict.values()
                    if ids and all(i.startswith("S2-") for i in ids))
    n_s3_only = sum(1 for ids in gt_dict.values()
                    if ids and all(i.startswith("S3-") for i in ids))
    n_both    = sum(1 for ids in gt_dict.values()
                    if ids and any(i.startswith("S2-") for i in ids)
                    and any(i.startswith("S3-") for i in ids))
    log.info("S2-only matches: %d | S3-only: %d | Both S2+S3: %d",
             n_s2_only, n_s3_only, n_both)

    # ── Country distribution ───────────────────────────────────────────────
    log.info("\n--- COUNTRY DISTRIBUTION ---")
    for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
        log.info("%s countries: %s", name, df["country"].value_counts().to_dict())

    # ── Name / Address length stats ────────────────────────────────────────
    log.info("\n--- TEXT LENGTH STATS ---")
    for name, df in [("S1", s1), ("S2", s2)]:
        name_lens = df["business_name"].str.len().describe()
        addr_lens = df["business_address"].str.len().describe()
        log.info("%s name len: mean=%.1f, max=%.1f", name,
                 name_lens["mean"], name_lens["max"])
        log.info("%s addr len: mean=%.1f, max=%.1f", name,
                 addr_lens["mean"], addr_lens["max"])

    # ── Noise examples ─────────────────────────────────────────────────────
    log.info("\n--- SAMPLE MATCHED PAIRS (first 5) ---")
    shown = 0
    for s1_id, match_ids in gt_dict.items():
        if not match_ids:
            continue
        s1_row = s1[s1["entity_id"] == s1_id]
        if s1_row.empty:
            continue
        s1_name = s1_row.iloc[0]["business_name"]
        s1_addr = s1_row.iloc[0]["business_address"]
        for mid in list(match_ids)[:2]:
            src = s2 if mid.startswith("S2-") else s3
            m_row = src[src["entity_id"] == mid]
            if m_row.empty:
                continue
            m_name = m_row.iloc[0]["business_name"]
            m_addr = m_row.iloc[0]["business_address"]
            log.info("  S1[%s]: '%s' @ '%s'", s1_id, s1_name[:60], s1_addr[:60])
            log.info("  →[%s]: '%s' @ '%s'", mid, m_name[:60], m_addr[:60])
            log.info("")
        shown += 1
        if shown >= 5:
            break

    # ── Save report ────────────────────────────────────────────────────────
    report = {
        "s1_count": len(s1),
        "s2_count": len(s2),
        "s3_count": len(s3),
        "gt_s1_count": n,
        "singleton_count": n_singletons,
        "singleton_rate": round(n_singletons / n, 4),
        "with_matches_count": n_with_matches,
        "max_matches": max(match_counts.values()),
        "s2_only": n_s2_only,
        "s3_only": n_s3_only,
        "both_s2_s3": n_both,
        "country_s1": s1["country"].value_counts().to_dict(),
        "country_s2": s2["country"].value_counts().to_dict(),
        "country_s3": s3["country"].value_counts().to_dict(),
        "match_distribution": {str(k): v for k, v in sorted(dist.items())},
    }
    out_path = EXPERIMENTS_DIR / "eda_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info("\nEDA report saved → %s", out_path)
    return report


if __name__ == "__main__":
    run_eda()
