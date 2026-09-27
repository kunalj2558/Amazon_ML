"""
io_utils.py — Safe, validated TSV I/O for the Amazon ML Challenge pipeline.

All files in this challenge are tab-separated. Reading without sep='\t'
produces a single corrupt column. Every read goes through this module.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# COLUMN DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]
GT_COLS     = ["source1_entity_id", "matched_entity_ids"]
MATCH_COLS  = ["source1_entity_id", "matched_entity_ids"]
CAND_COLS   = ["source1_entity_id", "candidate_entity_ids"]


# ─────────────────────────────────────────────────────────────────────────────
# READ HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def read_source(path: str | Path, label: str = "") -> pd.DataFrame:
    """Read a source TSV file (S1/S2/S3). Always uses sep='\\t'."""
    path = Path(path)
    log.info("Reading %s %s", label, path)
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    # Validate columns
    missing = [c for c in SOURCE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}. Got {list(df.columns)}")
    # Strip whitespace from entity_id
    df["entity_id"] = df["entity_id"].str.strip()
    # Fill genuinely missing text with empty string
    for col in ["business_name", "business_address", "country"]:
        df[col] = df[col].fillna("").str.strip()
    log.info("  → %d rows", len(df))
    return df


def read_ground_truth(path: str | Path) -> pd.DataFrame:
    """Read train_ground_truth.tsv. Returns df with matched_entity_ids as str."""
    path = Path(path)
    log.info("Reading ground truth %s", path)
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = [c for c in GT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    df["source1_entity_id"]  = df["source1_entity_id"].str.strip()
    df["matched_entity_ids"] = df["matched_entity_ids"].fillna("").str.strip()
    log.info("  → %d rows", len(df))
    return df


def ground_truth_to_dict(gt_df: pd.DataFrame) -> dict[str, set[str]]:
    """Convert ground-truth DataFrame into {s1_id: set_of_matched_ids}.
    Vectorized implementation — handles 2M+ rows efficiently.
    """
    def _parse_ids(s: str) -> set[str]:
        s = s.strip()
        return set(s.split(",")) if s else set()

    result = dict(zip(
        gt_df["source1_entity_id"],
        gt_df["matched_entity_ids"].apply(_parse_ids),
    ))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# WRITE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def write_matching_results(predictions: dict[str, set[str]], path: str | Path) -> None:
    """Write matching_results.tsv in the exact required format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for s1_id, matched in sorted(predictions.items()):
        rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(sorted(matched)) if matched else "",
        })
    df = pd.DataFrame(rows, columns=MATCH_COLS)
    df.to_csv(path, sep="\t", index=False)
    log.info("Wrote matching_results.tsv → %s  (%d rows)", path, len(df))


def write_candidate_pairs(candidates: dict[str, set[str]], path: str | Path) -> None:
    """Write candidate_pairs.tsv in the exact required format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for s1_id, cands in sorted(candidates.items()):
        rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(sorted(cands)) if cands else "",
        })
    df = pd.DataFrame(rows, columns=CAND_COLS)
    df.to_csv(path, sep="\t", index=False)
    log.info("Wrote candidate_pairs.tsv → %s  (%d rows)", path, len(df))


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# PARQUET HELPERS (fast internal I/O)
# ─────────────────────────────────────────────────────────────────────────────

def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log.info("Saved parquet → %s  (%d rows)", path, len(df))


def load_parquet(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    log.info("Loaded parquet ← %s  (%d rows)", path, len(df))
    return df
