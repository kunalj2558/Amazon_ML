"""
features.py — Pairwise similarity feature computation for candidate pairs.

Covers all feature groups from 04_FEATURE_MODEL_TRAINING_PLAN.md:
  A. Name features (16)
  B. Address features (13)
  C. Phonetic features (3)
  D. Retrieval features (7)
  E. Metadata features (7)
  F. Interaction features (6)

Total: ~52 features per pair.

Design rules:
  - Every feature has defined behavior for missing values (filled with MISSING_FILL=-1).
  - Calculations are deterministic and label-free.
  - Vectorized where possible; pair loop only for string comparisons.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from rapidfuzz import distance as rfdist
from rapidfuzz import fuzz as rffuzz

from config import FEATURES

log = logging.getLogger(__name__)

MISSING = FEATURES["missing_fill"]


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def safe_jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets. Returns 0 if both empty."""
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def token_set(text: str) -> set[str]:
    if not text:
        return set()
    return set(text.split())


def char_ngram_set(text: str, n: int) -> set[str]:
    if not text or len(text) < n:
        return set()
    padded = f" {text} "
    return {padded[i:i+n] for i in range(len(padded) - n + 1)}


def safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# GROUP A: NAME FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def compute_name_features(row_s1: pd.Series, row_s23: pd.Series) -> dict:
    n1 = str(row_s1.get("name_clean", "") or "")
    n2 = str(row_s23.get("name_clean", "") or "")
    n1_raw = str(row_s1.get("business_name", "") or "")
    n2_raw = str(row_s23.get("business_name", "") or "")

    t1 = token_set(n1)
    t2 = token_set(n2)

    c3_1 = char_ngram_set(n1, 3)
    c3_2 = char_ngram_set(n2, 3)
    c4_1 = char_ngram_set(n1, 4)
    c4_2 = char_ngram_set(n2, 4)

    feats = {}

    # Exact matches
    feats["name_exact_raw"]   = int(n1_raw.lower() == n2_raw.lower())
    feats["name_exact_clean"] = int(n1 == n2)

    # Jaccard
    feats["name_jaccard_word"] = safe_jaccard(t1, t2)
    feats["name_jaccard_char3"] = safe_jaccard(c3_1, c3_2)
    feats["name_jaccard_char4"] = safe_jaccard(c4_1, c4_2)

    # Edit distance features (rapidfuzz — normalized 0..1)
    if n1 and n2:
        feats["name_levenshtein_norm"] = 1.0 - rfdist.Levenshtein.normalized_distance(n1, n2)
        feats["name_jaro_winkler"]     = rffuzz.WRatio(n1, n2) / 100.0
        feats["name_ratio"]            = rffuzz.ratio(n1, n2) / 100.0
        feats["name_partial_ratio"]    = rffuzz.partial_ratio(n1, n2) / 100.0
        feats["name_token_sort_ratio"] = rffuzz.token_sort_ratio(n1, n2) / 100.0
        feats["name_token_set_ratio"]  = rffuzz.token_set_ratio(n1, n2) / 100.0
    else:
        for k in ["name_levenshtein_norm", "name_jaro_winkler", "name_ratio",
                  "name_partial_ratio", "name_token_sort_ratio", "name_token_set_ratio"]:
            feats[k] = MISSING

    # Token overlap
    feats["name_common_token_count"] = len(t1 & t2)
    feats["name_common_token_ratio"] = safe_div(len(t1 & t2), max(len(t1), len(t2), 1))

    # Length features
    feats["name_length_ratio"] = safe_div(min(len(n1), len(n2)), max(len(n1), len(n2), 1))
    feats["name_len_s1"]  = len(n1)
    feats["name_len_s23"] = len(n2)

    # First token match
    tok1 = n1.split()[0] if n1.split() else ""
    tok2 = n2.split()[0] if n2.split() else ""
    feats["name_first_token_match"] = int(tok1 == tok2 and tok1 != "")

    # Subset check (one name is a sub-multiset of the other)
    feats["name_is_subset"] = int(t1.issubset(t2) or t2.issubset(t1))

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# GROUP B: ADDRESS FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def compute_address_features(row_s1: pd.Series, row_s23: pd.Series) -> dict:
    a1 = str(row_s1.get("address_clean", "") or "")
    a2 = str(row_s23.get("address_clean", "") or "")

    t1 = token_set(a1)
    t2 = token_set(a2)
    c3_1 = char_ngram_set(a1, 3)
    c3_2 = char_ngram_set(a2, 3)

    p1 = str(row_s1.get("postal_code", "") or "").strip()
    p2 = str(row_s23.get("postal_code", "") or "").strip()

    nums1 = set(row_s1.get("address_numbers", []) or [])
    nums2 = set(row_s23.get("address_numbers", []) or [])

    feats = {}

    feats["address_exact"] = int(a1 == a2 and a1 != "")

    # Jaccard
    feats["address_jaccard_word"] = safe_jaccard(t1, t2)
    feats["address_jaccard_char3"] = safe_jaccard(c3_1, c3_2)

    # Edit distance
    if a1 and a2:
        feats["address_levenshtein_norm"] = 1.0 - rfdist.Levenshtein.normalized_distance(a1, a2)
        feats["address_token_sort_ratio"] = rffuzz.token_sort_ratio(a1, a2) / 100.0
        feats["address_token_set_ratio"]  = rffuzz.token_set_ratio(a1, a2) / 100.0
    else:
        for k in ["address_levenshtein_norm", "address_token_sort_ratio", "address_token_set_ratio"]:
            feats[k] = MISSING

    # Token overlap
    feats["address_common_token_count"] = len(t1 & t2)
    feats["address_common_token_ratio"] = safe_div(len(t1 & t2), max(len(t1), len(t2), 1))
    feats["address_length_ratio"]       = safe_div(min(len(a1), len(a2)), max(len(a1), len(a2), 1))

    # Postal code features
    feats["postal_exact"]         = int(bool(p1) and bool(p2) and p1 == p2)
    feats["postal_prefix3_match"] = int(bool(p1) and bool(p2) and p1[:3] == p2[:3] and len(p1) >= 3)
    feats["both_have_postal"]     = int(bool(p1) and bool(p2))
    feats["one_has_postal"]       = int(bool(p1) != bool(p2))

    # Numeric token overlap (house numbers, etc.)
    feats["address_number_overlap"] = safe_jaccard(nums1, nums2)
    # House number exact: first numeric token match
    first_num1 = sorted(nums1)[0] if nums1 else ""
    first_num2 = sorted(nums2)[0] if nums2 else ""
    feats["house_number_exact"] = int(bool(first_num1) and first_num1 == first_num2)

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# GROUP C: PHONETIC FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def compute_phonetic_features(row_s1: pd.Series, row_s23: pd.Series) -> dict:
    feats = {}
    pm1  = str(row_s1.get("name_dbl_meta_primary", "") or "")
    pm2  = str(row_s23.get("name_dbl_meta_primary", "") or "")
    ps1  = str(row_s1.get("name_dbl_meta_secondary", "") or "")
    ps2  = str(row_s23.get("name_dbl_meta_secondary", "") or "")

    feats["phonetic_primary_match"]   = int(bool(pm1) and pm1 == pm2)
    feats["phonetic_secondary_match"] = int(bool(ps1) and ps1 == ps2)
    feats["phonetic_any_match"]       = int(
        feats["phonetic_primary_match"] or feats["phonetic_secondary_match"]
    )
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# GROUP D: RETRIEVAL FEATURES (from blocking metadata)
# ─────────────────────────────────────────────────────────────────────────────

def compute_retrieval_features(meta_row: pd.Series) -> dict:
    feats = {}
    feats["passes_hit"]       = int(meta_row.get("passes_hit", 0) or 0)
    feats["B_char_score"]     = float(meta_row.get("B_char_tfidf_score", MISSING) or MISSING)
    feats["B_char_rank"]      = float(meta_row.get("B_char_tfidf_rank", MISSING) or MISSING)
    feats["C_token_score"]    = float(meta_row.get("C_token_tfidf_score", MISSING) or MISSING)
    feats["C_token_rank"]     = float(meta_row.get("C_token_tfidf_rank", MISSING) or MISSING)

    # Best rank across passes
    b_rank = feats["B_char_rank"]
    c_rank = feats["C_token_rank"]
    valid_ranks = [r for r in [b_rank, c_rank] if r > 0]
    feats["best_retrieval_rank"]  = min(valid_ranks) if valid_ranks else MISSING
    # Best score across passes
    b_score = feats["B_char_score"]
    c_score = feats["C_token_score"]
    valid_scores = [s for s in [b_score, c_score] if s > 0]
    feats["best_retrieval_score"] = max(valid_scores) if valid_scores else MISSING

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# GROUP E: METADATA FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def compute_metadata_features(row_s1: pd.Series, row_s23: pd.Series) -> dict:
    feats = {}
    c1 = str(row_s1.get("country_norm", "") or "")
    c2 = str(row_s23.get("country_norm", "") or "")
    feats["country_equal"] = int(c1 == c2 and c1 != "")

    src_id = str(row_s23.get("entity_id", ""))
    feats["source_is_s2"] = int(src_id.startswith("S2-"))
    feats["source_is_s3"] = int(src_id.startswith("S3-"))

    feats["name_missing_s1"]  = int(not row_s1.get("name_clean", ""))
    feats["name_missing_s23"] = int(not row_s23.get("name_clean", ""))
    feats["addr_missing_s1"]  = int(not row_s1.get("address_clean", ""))
    feats["addr_missing_s23"] = int(not row_s23.get("address_clean", ""))

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# GROUP F: INTERACTION FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def compute_interaction_features(name_feats: dict, addr_feats: dict, phon_feats: dict) -> dict:
    feats = {}

    nj   = name_feats.get("name_jaccard_word", 0.0)
    pe   = addr_feats.get("postal_exact", 0)
    ntsr = name_feats.get("name_token_set_ratio", 0.0)
    aj   = addr_feats.get("address_jaccard_word", 0.0)
    pa   = phon_feats.get("phonetic_any_match", 0)
    nts  = name_feats.get("name_token_sort_ratio", 0.0)

    feats["high_name_and_exact_postal"] = int(nj > 0.7 and pe == 1)
    feats["high_name_and_high_address"] = int(nj > 0.6 and aj > 0.6)
    feats["high_name_low_address"]      = int(nj > 0.6 and aj < 0.3)
    feats["low_name_high_address"]      = int(nj < 0.3 and aj > 0.6)
    feats["phonetic_and_fuzzy_agree"]   = int(pa == 1 and ntsr > 0.7)
    feats["exact_name_disagree_addr"]   = int(
        name_feats.get("name_exact_clean", 0) == 1
        and addr_feats.get("address_jaccard_word", 1.0) < 0.2
    )

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# FULL PAIR FEATURE VECTOR
# ─────────────────────────────────────────────────────────────────────────────

def compute_pair_features(
    row_s1: pd.Series,
    row_s23: pd.Series,
    meta_row: Optional[pd.Series] = None,
) -> dict:
    """Compute all features for a single (S1, S2/S3) candidate pair."""
    nf = compute_name_features(row_s1, row_s23)
    af = compute_address_features(row_s1, row_s23)
    pf = compute_phonetic_features(row_s1, row_s23)
    rf = compute_retrieval_features(meta_row if meta_row is not None else pd.Series({}))
    mf = compute_metadata_features(row_s1, row_s23)
    inf= compute_interaction_features(nf, af, pf)

    return {**nf, **af, **pf, **rf, **mf, **inf}


# ─────────────────────────────────────────────────────────────────────────────
# BATCH FEATURE COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    meta_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    label_map: Optional[dict[str, set[str]]] = None,
) -> pd.DataFrame:
    """
    Compute features for all candidate pairs in meta_df.

    Args:
        meta_df:   DataFrame with (source1_entity_id, candidate_entity_id, ...)
        s1_df:     Preprocessed Source 1 DataFrame
        pool_df:   Preprocessed Source 2+3 DataFrame
        label_map: {s1_id: set_of_true_match_ids} for training labels (optional)

    Returns:
        DataFrame with all features + optional 'label' column
    """
    log.info("Building feature matrix for %d candidate pairs ...", len(meta_df))

    # Index for fast row lookups
    s1_idx   = s1_df.set_index("entity_id")
    pool_idx = pool_df.set_index("entity_id")

    feature_rows = []
    missing_s1   = 0
    missing_s23  = 0

    for i, meta_row in meta_df.iterrows():
        s1_id   = meta_row["source1_entity_id"]
        cand_id = meta_row["candidate_entity_id"]

        if s1_id not in s1_idx.index:
            missing_s1 += 1
            continue
        if cand_id not in pool_idx.index:
            missing_s23 += 1
            continue

        row_s1  = s1_idx.loc[s1_id]
        row_s23 = pool_idx.loc[cand_id]
        feats   = compute_pair_features(row_s1, row_s23, meta_row)

        feats["source1_entity_id"]   = s1_id
        feats["candidate_entity_id"] = cand_id

        if label_map is not None:
            feats["label"] = int(cand_id in label_map.get(s1_id, set()))

        feature_rows.append(feats)

        if (i + 1) % 100_000 == 0:
            log.info("  ... processed %d / %d pairs", i + 1, len(meta_df))

    if missing_s1 or missing_s23:
        log.warning("Skipped %d missing S1 and %d missing pool IDs", missing_s1, missing_s23)

    result = pd.DataFrame(feature_rows)
    log.info("Feature matrix: %d rows × %d columns", len(result), len(result.columns))
    return result


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature column names (exclude IDs and label)."""
    exclude = {"source1_entity_id", "candidate_entity_id", "label",
               "pass_names", "passes_hit"}
    return [c for c in df.columns if c not in exclude and not c.endswith("_names")]
