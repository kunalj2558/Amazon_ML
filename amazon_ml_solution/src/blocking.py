"""
blocking.py — Multi-pass candidate generation for Business Entity Resolution.

Implements 7 complementary blocking passes (A–G) as specified in
03_BLOCKING_CANDIDATE_GENERATION_PLAN.md. The union of all passes is the
candidate set written to candidate_pairs.tsv.

Design rules:
  - Never silently truncate candidates without diagnosing the key.
  - Measure incremental recall per pass on training data.
  - Never generate S1→S1 pairs.
  - Store retrieval metadata (rank, score, passes_hit) for use as features.
"""

from __future__ import annotations

import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import BLOCKING, EXPERIMENTS_DIR

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Batched sparse cosine retrieval
# ─────────────────────────────────────────────────────────────────────────────

def _sparse_top_k(
    query_matrix: csr_matrix,
    index_matrix: csr_matrix,
    index_ids: list[str],
    top_k: int,
    batch_size: int = 5000,
) -> list[list[tuple[str, float]]]:
    """
    For each query row, find top-K index rows by cosine similarity.
    Returns list of lists: [ [(id, score), ...], ... ]
    Processes in batches to control memory.
    """
    results = []
    n_queries = query_matrix.shape[0]
    index_ids_arr = np.array(index_ids)

    for start in range(0, n_queries, batch_size):
        end   = min(start + batch_size, n_queries)
        batch = query_matrix[start:end]
        # (batch_size × n_index) dense similarity matrix
        sims  = (batch @ index_matrix.T).toarray()  # type: ignore
        # For each row, get top_k indices
        for row in sims:
            top_idx  = np.argpartition(row, -min(top_k, len(row)))[-top_k:]
            top_idx  = top_idx[np.argsort(row[top_idx])[::-1]]
            results.append([(index_ids_arr[i], float(row[i])) for i in top_idx if row[i] > 0])

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PASS A — Exact normalized name match
# ─────────────────────────────────────────────────────────────────────────────

def pass_a_exact_name(
    s1_df: pd.DataFrame,
    pool_df: pd.DataFrame,
) -> dict[str, set[str]]:
    """
    Inverted index on name_clean. Perfect for exact-match variants.
    High precision, cheap.
    """
    log.info("[Pass A] Exact name matching ...")
    # Build: name_clean → list of pool IDs
    name_to_ids: dict[str, list[str]] = defaultdict(list)
    for _, row in pool_df.iterrows():
        if row["name_clean"]:
            name_to_ids[row["name_clean"]].append(row["entity_id"])

    candidates: dict[str, set[str]] = defaultdict(set)
    for _, row in s1_df.iterrows():
        s1_id = row["entity_id"]
        key   = row["name_clean"]
        if key and key in name_to_ids:
            candidates[s1_id].update(name_to_ids[key])

    total = sum(len(v) for v in candidates.values())
    log.info("[Pass A] Generated %d candidate pairs.", total)
    return dict(candidates)


# ─────────────────────────────────────────────────────────────────────────────
# PASS B — Character TF-IDF Retrieval on names
# ─────────────────────────────────────────────────────────────────────────────

def pass_b_char_tfidf(
    s1_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    cfg: dict,
) -> tuple[dict[str, set[str]], dict[str, dict[str, float]], dict[str, dict[str, int]]]:
    """
    Character n-gram TF-IDF retrieval on name_clean.
    Returns:
        candidates dict
        score_map: {s1_id: {cand_id: score}}
        rank_map:  {s1_id: {cand_id: rank}}
    """
    log.info("[Pass B] Char TF-IDF retrieval (top_k=%d) ...", cfg["top_k"])
    vectorizer = TfidfVectorizer(
        analyzer   = cfg["analyzer"],
        ngram_range= cfg["ngram_range"],
        sublinear_tf= cfg["sublinear_tf"],
        min_df     = cfg["min_df"],
    )
    pool_texts = pool_df["name_clean"].fillna("").tolist()
    pool_ids   = pool_df["entity_id"].tolist()

    index_mat = vectorizer.fit_transform(pool_texts)
    query_mat = vectorizer.transform(s1_df["name_clean"].fillna("").tolist())
    s1_ids    = s1_df["entity_id"].tolist()

    top_k_results = _sparse_top_k(
        query_mat, index_mat, pool_ids, cfg["top_k"],
        batch_size=BLOCKING["batch_size"]
    )

    candidates: dict[str, set[str]] = defaultdict(set)
    score_map:  dict[str, dict[str, float]] = defaultdict(dict)
    rank_map:   dict[str, dict[str, int]]   = defaultdict(dict)

    for s1_id, results in zip(s1_ids, top_k_results):
        for rank, (cand_id, score) in enumerate(results, 1):
            candidates[s1_id].add(cand_id)
            score_map[s1_id][cand_id] = score
            rank_map[s1_id][cand_id]  = rank

    total = sum(len(v) for v in candidates.values())
    log.info("[Pass B] Generated %d candidate pairs.", total)
    return dict(candidates), dict(score_map), dict(rank_map)


# ─────────────────────────────────────────────────────────────────────────────
# PASS C — Token TF-IDF Retrieval (word-level)
# ─────────────────────────────────────────────────────────────────────────────

def pass_c_token_tfidf(
    s1_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    cfg: dict,
) -> tuple[dict[str, set[str]], dict[str, dict[str, float]], dict[str, dict[str, int]]]:
    """
    Word n-gram TF-IDF on name_clean. Catches token-level overlap/reordering.
    Complement to char TF-IDF.
    """
    log.info("[Pass C] Token TF-IDF retrieval (top_k=%d) ...", cfg["top_k"])
    vectorizer = TfidfVectorizer(
        analyzer   = cfg["analyzer"],
        ngram_range= cfg["ngram_range"],
        sublinear_tf= cfg["sublinear_tf"],
        min_df     = cfg["min_df"],
    )
    pool_texts = pool_df["name_clean"].fillna("").tolist()
    pool_ids   = pool_df["entity_id"].tolist()

    index_mat = vectorizer.fit_transform(pool_texts)
    query_mat = vectorizer.transform(s1_df["name_clean"].fillna("").tolist())
    s1_ids    = s1_df["entity_id"].tolist()

    top_k_results = _sparse_top_k(
        query_mat, index_mat, pool_ids, cfg["top_k"],
        batch_size=BLOCKING["batch_size"]
    )

    candidates: dict[str, set[str]] = defaultdict(set)
    score_map:  dict[str, dict[str, float]] = defaultdict(dict)
    rank_map:   dict[str, dict[str, int]]   = defaultdict(dict)

    for s1_id, results in zip(s1_ids, top_k_results):
        for rank, (cand_id, score) in enumerate(results, 1):
            candidates[s1_id].add(cand_id)
            score_map[s1_id][cand_id] = score
            rank_map[s1_id][cand_id]  = rank

    total = sum(len(v) for v in candidates.values())
    log.info("[Pass C] Generated %d candidate pairs.", total)
    return dict(candidates), dict(score_map), dict(rank_map)


# ─────────────────────────────────────────────────────────────────────────────
# PASS D — Postal / Numeric address blocking
# ─────────────────────────────────────────────────────────────────────────────

def pass_d_postal(
    s1_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    cfg: dict,
) -> dict[str, set[str]]:
    """
    Block on (country, postal_code). Very high precision candidates.
    Also optionally combines with first name token for narrower keys.
    """
    log.info("[Pass D] Postal code blocking ...")
    # Build index: key → [pool_ids]
    postal_index: dict[str, list[str]] = defaultdict(list)
    for _, row in pool_df.iterrows():
        pc = row["postal_code"].strip()
        if not pc:
            continue
        country = row["country_norm"]
        key     = f"{country}|{pc}"
        postal_index[key].append(row["entity_id"])

        # Narrower key: country + postal + first name token
        if cfg.get("use_first_name_token") and row["name_tokens"]:
            first_tok = row["name_tokens"][0] if row["name_tokens"] else ""
            narrow_key = f"{country}|{pc}|{first_tok}"
            postal_index[narrow_key].append(row["entity_id"])

    candidates: dict[str, set[str]] = defaultdict(set)
    for _, row in s1_df.iterrows():
        s1_id   = row["entity_id"]
        pc      = row["postal_code"].strip()
        country = row["country_norm"]
        if not pc:
            continue
        key = f"{country}|{pc}"
        if key in postal_index:
            cands = postal_index[key]
            # Skip if block is too large (common postal code)
            if len(cands) <= BLOCKING["max_candidates_per_s1"]:
                candidates[s1_id].update(cands)

        if cfg.get("use_first_name_token") and row["name_tokens"]:
            first_tok  = row["name_tokens"][0] if row["name_tokens"] else ""
            narrow_key = f"{country}|{pc}|{first_tok}"
            if narrow_key in postal_index:
                candidates[s1_id].update(postal_index[narrow_key])

    total = sum(len(v) for v in candidates.values())
    log.info("[Pass D] Generated %d candidate pairs.", total)
    return dict(candidates)


# ─────────────────────────────────────────────────────────────────────────────
# PASS E — Phonetic blocking (Double Metaphone)
# ─────────────────────────────────────────────────────────────────────────────

def pass_e_phonetic(
    s1_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    cfg: dict,
) -> dict[str, set[str]]:
    """
    Block on country + Double Metaphone primary code of business name.
    Handles transliterations and phonetic spelling variants.
    """
    log.info("[Pass E] Phonetic blocking ...")
    phonetic_index: dict[str, list[str]] = defaultdict(list)
    for _, row in pool_df.iterrows():
        code    = row["name_dbl_meta_primary"]
        country = row["country_norm"]
        if not code:
            continue
        key = f"{country}|{code}"
        phonetic_index[key].append(row["entity_id"])

    max_block = cfg.get("max_phonetic_block_size", 500)
    candidates: dict[str, set[str]] = defaultdict(set)
    for _, row in s1_df.iterrows():
        s1_id   = row["entity_id"]
        code    = row["name_dbl_meta_primary"]
        country = row["country_norm"]
        if not code:
            continue
        key = f"{country}|{code}"
        if key in phonetic_index:
            block = phonetic_index[key]
            if len(block) <= max_block:
                candidates[s1_id].update(block)

    total = sum(len(v) for v in candidates.values())
    log.info("[Pass E] Generated %d candidate pairs.", total)
    return dict(candidates)


# ─────────────────────────────────────────────────────────────────────────────
# PASS F — MinHash LSH on character n-grams
# ─────────────────────────────────────────────────────────────────────────────

def pass_f_minhash(
    s1_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    cfg: dict,
) -> dict[str, set[str]]:
    """
    MinHash LSH on character trigrams of (name + address).
    Catches word-order variants, compound names, partial strings.
    """
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        log.warning("[Pass F] datasketch not installed. Skipping MinHash LSH.")
        return {}

    log.info("[Pass F] MinHash LSH (num_perm=%d, threshold=%.2f) ...",
             cfg["num_perm"], cfg["threshold"])
    n = cfg["ngram_size"]

    def make_minhash(text: str) -> MinHash:
        m = MinHash(num_perm=cfg["num_perm"])
        chars = {text[i:i+n] for i in range(max(0, len(text) - n + 1))}
        for c in chars:
            m.update(c.encode("utf-8"))
        return m

    lsh = MinHashLSH(threshold=cfg["threshold"], num_perm=cfg["num_perm"])

    # Build LSH index with pool records
    pool_hashes = {}
    for _, row in pool_df.iterrows():
        pid  = row["entity_id"]
        text = (row["name_clean"] + " " + row["address_clean"]).strip()
        mh   = make_minhash(text)
        pool_hashes[pid] = mh
        try:
            lsh.insert(pid, mh)
        except ValueError:
            pass  # duplicate key — skip

    candidates: dict[str, set[str]] = defaultdict(set)
    for _, row in s1_df.iterrows():
        s1_id = row["entity_id"]
        text  = (row["name_clean"] + " " + row["address_clean"]).strip()
        mh    = make_minhash(text)
        try:
            result = lsh.query(mh)
            candidates[s1_id].update(result)
        except Exception:
            pass

    total = sum(len(v) for v in candidates.values())
    log.info("[Pass F] Generated %d candidate pairs.", total)
    return dict(candidates)


# ─────────────────────────────────────────────────────────────────────────────
# PASS G — Sorted-token prefix blocking
# ─────────────────────────────────────────────────────────────────────────────

def pass_g_prefix(
    s1_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    cfg: dict,
) -> dict[str, set[str]]:
    """
    Block on country + first N chars of alphabetically sorted name tokens.
    Catches word transpositions cheaply.
    """
    log.info("[Pass G] Sorted-token prefix blocking (prefix_len=%d) ...", cfg["prefix_len"])
    plen = cfg["prefix_len"]

    prefix_index: dict[str, list[str]] = defaultdict(list)
    for _, row in pool_df.iterrows():
        prefix  = row["name_prefix4"][:plen]
        country = row["country_norm"]
        if not prefix:
            continue
        key = f"{country}|{prefix}"
        prefix_index[key].append(row["entity_id"])

    candidates: dict[str, set[str]] = defaultdict(set)
    for _, row in s1_df.iterrows():
        s1_id   = row["entity_id"]
        prefix  = row["name_prefix4"][:plen]
        country = row["country_norm"]
        if not prefix:
            continue
        key = f"{country}|{prefix}"
        if key in prefix_index:
            block = prefix_index[key]
            # Only use narrow prefix blocks to avoid explosion
            if len(block) <= BLOCKING["max_candidates_per_s1"]:
                candidates[s1_id].update(block)

    total = sum(len(v) for v in candidates.values())
    log.info("[Pass G] Generated %d candidate pairs.", total)
    return dict(candidates)


# ─────────────────────────────────────────────────────────────────────────────
# UNION + METADATA ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

def union_candidates(
    s1_ids: list[str],
    pass_results: list[tuple[str, dict[str, set[str]]]],
    score_maps:   dict[str, dict[str, dict[str, float]]] | None = None,
    rank_maps:    dict[str, dict[str, dict[str, int]]]   | None = None,
    max_cap: int = 500,
) -> tuple[dict[str, set[str]], pd.DataFrame]:
    """
    Union all pass results into a single candidate dict.
    Also builds a metadata DataFrame for each (s1_id, cand_id) pair.

    Returns:
        candidates: {s1_id: set_of_candidate_ids}
        meta_df:    DataFrame with retrieval metadata per pair
    """
    log.info("Unioning %d blocking passes ...", len(pass_results))

    # How many passes hit each pair
    pass_count: dict[tuple[str, str], int]   = defaultdict(int)
    passes_hit: dict[tuple[str, str], list]  = defaultdict(list)

    candidates: dict[str, set[str]] = {s1: set() for s1 in s1_ids}

    for pass_name, pass_cands in pass_results:
        for s1_id, cand_ids in pass_cands.items():
            for cid in cand_ids:
                candidates[s1_id].add(cid)
                key = (s1_id, cid)
                pass_count[key] += 1
                passes_hit[key].append(pass_name)

    # Hard cap: if any S1 has too many candidates, diagnose and trim
    capped = 0
    for s1_id in s1_ids:
        if len(candidates[s1_id]) > max_cap:
            log.warning(
                "S1 %s has %d candidates → capping at %d. "
                "Consider diagnosing this blocking key.",
                s1_id, len(candidates[s1_id]), max_cap
            )
            # Keep candidates with highest pass count first
            ranked = sorted(
                candidates[s1_id],
                key=lambda cid: pass_count.get((s1_id, cid), 0),
                reverse=True,
            )
            candidates[s1_id] = set(ranked[:max_cap])
            capped += 1

    if capped:
        log.warning("Capped %d S1 entities.", capped)

    # Build metadata rows
    meta_rows = []
    for s1_id, cand_ids in candidates.items():
        for cid in cand_ids:
            key = (s1_id, cid)
            row = {
                "source1_entity_id":   s1_id,
                "candidate_entity_id": cid,
                "passes_hit":          len(passes_hit.get(key, [])),
                "pass_names":          ",".join(passes_hit.get(key, [])),
            }
            if score_maps:
                for pass_name, smap in score_maps.items():
                    row[f"{pass_name}_score"] = smap.get(s1_id, {}).get(cid, -1.0)
            if rank_maps:
                for pass_name, rmap in rank_maps.items():
                    row[f"{pass_name}_rank"] = rmap.get(s1_id, {}).get(cid, -1)
            meta_rows.append(row)

    meta_df = pd.DataFrame(meta_rows)
    log.info(
        "Union complete: %d total candidate pairs across %d S1 entities.",
        len(meta_df), len(candidates)
    )
    return candidates, meta_df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BLOCKING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_blocking(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    cfg: dict,
    ground_truth: dict[str, set[str]] | None = None,
) -> tuple[dict[str, set[str]], pd.DataFrame]:
    """
    Run all blocking passes (A–G) for S1 vs S2 and S1 vs S3.

    Args:
        s1_df: preprocessed Source 1 DataFrame
        s2_df: preprocessed Source 2 DataFrame
        s3_df: preprocessed Source 3 DataFrame
        cfg:   BLOCKING config dict
        ground_truth: if provided, blocking recall is measured after each pass

    Returns:
        candidates: {s1_id: set_of_candidate_ids}
        meta_df:    retrieval metadata DataFrame
    """
    s1_ids = s1_df["entity_id"].tolist()
    pool_df = pd.concat([s2_df, s3_df], ignore_index=True)

    log.info("Starting multi-pass blocking: %d S1 vs %d pool records",
             len(s1_df), len(pool_df))

    # Running union for recall measurement
    running_union: dict[str, set[str]] = {s1: set() for s1 in s1_ids}

    def _measure_recall(label: str, new_cands: dict[str, set[str]]) -> None:
        if ground_truth is None:
            return
        for s1, cids in new_cands.items():
            running_union[s1].update(cids)
        from evaluate import blocking_recall as _br
        stats = _br(running_union, ground_truth)
        log.info(
            "[Recall after %s] %.4f  (%d/%d pairs covered, %d missed)",
            label, stats["blocking_recall"],
            stats["covered_pairs"], stats["total_true_pairs"],
            stats["missed_pairs"]
        )

    # ── Pass A — exact name ────────────────────────────────────────────────
    cands_a = pass_a_exact_name(s1_df, pool_df)
    _measure_recall("Pass A", cands_a)

    # ── Pass B — char TF-IDF ───────────────────────────────────────────────
    cands_b, scores_b, ranks_b = pass_b_char_tfidf(s1_df, pool_df, cfg["char_tfidf"])
    _measure_recall("Pass B", cands_b)

    # ── Pass C — token TF-IDF ─────────────────────────────────────────────
    cands_c, scores_c, ranks_c = pass_c_token_tfidf(s1_df, pool_df, cfg["token_tfidf"])
    _measure_recall("Pass C", cands_c)

    # ── Pass D — postal ────────────────────────────────────────────────────
    cands_d = pass_d_postal(s1_df, pool_df, cfg["postal"])
    _measure_recall("Pass D", cands_d)

    # ── Pass E — phonetic ──────────────────────────────────────────────────
    cands_e = pass_e_phonetic(s1_df, pool_df, cfg["phonetic"])
    _measure_recall("Pass E", cands_e)

    # ── Pass F — MinHash ───────────────────────────────────────────────────
    cands_f = pass_f_minhash(s1_df, pool_df, cfg["minhash"])
    _measure_recall("Pass F", cands_f)

    # ── Pass G — sorted prefix ─────────────────────────────────────────────
    cands_g = pass_g_prefix(s1_df, pool_df, cfg["prefix"])
    _measure_recall("Pass G", cands_g)

    # ── Union ──────────────────────────────────────────────────────────────
    all_passes = [
        ("A_exact",      cands_a),
        ("B_char_tfidf", cands_b),
        ("C_token_tfidf",cands_c),
        ("D_postal",     cands_d),
        ("E_phonetic",   cands_e),
        ("F_minhash",    cands_f),
        ("G_prefix",     cands_g),
    ]
    score_maps = {"B_char_tfidf": scores_b, "C_token_tfidf": scores_c}
    rank_maps  = {"B_char_tfidf": ranks_b,  "C_token_tfidf": ranks_c}

    candidates, meta_df = union_candidates(
        s1_ids, all_passes, score_maps, rank_maps,
        max_cap=cfg["max_candidates_per_s1"]
    )

    # Final recall measurement
    if ground_truth:
        from evaluate import blocking_recall as _br
        final_stats = _br(candidates, ground_truth)
        log.info("FINAL BLOCKING RECALL: %.4f", final_stats["blocking_recall"])
        # Save missed pairs for analysis
        missed_df = pd.DataFrame(final_stats["missed_examples"], columns=["s1_id", "true_id"])
        missed_df.to_csv(EXPERIMENTS_DIR / "missed_true_pairs.csv", index=False)

    return candidates, meta_df


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("Blocking module loaded successfully. Run via run_pipeline.py.")
