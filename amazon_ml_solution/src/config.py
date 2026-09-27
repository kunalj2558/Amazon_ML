"""
config.py — Central configuration for the Amazon ML Challenge 2026 solution.
All tunable values live here. Never scatter magic numbers across source files.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_TRAIN = Path(r"D:\Deksto-1\Desktop\Hackthons\Projects\Amazon_ML\6ab10eb3b23ba_student_resource\student_resource\dataset\train")
DATA_RAW_TEST  = Path(r"D:\Deksto-1\Desktop\Hackthons\Projects\Amazon_ML\6ab10eb3b23ba_student_resource\student_resource\dataset\test")
UTILS_DIR      = Path(r"D:\Deksto-1\Desktop\Hackthons\Projects\Amazon_ML\6ab10eb3b23ba_student_resource\student_resource\utils")

DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR     = ROOT / "models"
EXPERIMENTS_DIR= ROOT / "experiments"
OUTPUT_DIR     = ROOT / "output"
SUBMISSIONS_DIR= ROOT / "submissions"

# Create directories
for d in [DATA_PROCESSED, MODELS_DIR, EXPERIMENTS_DIR, OUTPUT_DIR, SUBMISSIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# RANDOM SEEDS  (determinism)
# ─────────────────────────────────────────────────────────────────────────────

RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

PREPROCESSING = {
    # Character n-gram sizes to store
    "char_ngram_sizes": [2, 3, 4],
    # Legal-abbreviation expansion dictionary (lowercase → full form)
    "name_abbrev_map": {
        "corp": "corporation",
        "corp.": "corporation",
        "pvt": "private",
        "pvt.": "private",
        "ltd": "limited",
        "ltd.": "limited",
        "inc": "incorporated",
        "inc.": "incorporated",
        "co": "company",
        "co.": "company",
        "llc": "limited liability company",
        "llp": "limited liability partnership",
        "intl": "international",
        "int'l": "international",
        "mfg": "manufacturing",
        "svcs": "services",
        "assoc": "associates",
        "assocs": "associates",
        "bros": "brothers",
        "hldgs": "holdings",
        "grp": "group",
        "mgmt": "management",
        "svc": "service",
        "natl": "national",
        "indl": "industrial",
        "ind": "industries",
        "tech": "technology",
        "sys": "systems",
        "soln": "solutions",
        "solns": "solutions",
        "ent": "enterprises",
        "entp": "enterprises",
        "dept": "department",
    },
    # Address abbreviation expansion
    "addr_abbrev_map": {
        "rd": "road",
        "rd.": "road",
        "st": "street",
        "st.": "street",
        "ave": "avenue",
        "ave.": "avenue",
        "blvd": "boulevard",
        "blvd.": "boulevard",
        "dr": "drive",
        "dr.": "drive",
        "ln": "lane",
        "ln.": "lane",
        "hwy": "highway",
        "hwy.": "highway",
        "fl": "floor",
        "flr": "floor",
        "apt": "apartment",
        "ste": "suite",
        "bldg": "building",
        "no.": "number",
        "no": "number",
        "nagar": "nagar",  # Keep Indian terms
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# BLOCKING
# ─────────────────────────────────────────────────────────────────────────────

BLOCKING = {
    # Pass B — char TF-IDF retrieval
    "char_tfidf": {
        "analyzer": "char_wb",
        "ngram_range": (2, 5),
        "sublinear_tf": True,
        "min_df": 1,
        "top_k": 50,         # top candidates per S1 from this pass
    },
    # Pass C — token TF-IDF / BM25 retrieval
    "token_tfidf": {
        "analyzer": "word",
        "ngram_range": (1, 2),
        "sublinear_tf": True,
        "min_df": 1,
        "top_k": 50,
    },
    # Pass D — postal / numeric
    "postal": {
        "use_country_prefix": True,
        "use_first_name_token": True,
    },
    # Pass E — phonetic
    "phonetic": {
        "use_double_metaphone": True,
        "max_phonetic_block_size": 500,   # discard blocks larger than this
    },
    # Pass F — MinHash LSH
    "minhash": {
        "num_perm": 128,
        "threshold": 0.35,
        "ngram_size": 3,
    },
    # Pass G — sorted-token prefix
    "prefix": {
        "prefix_len": 4,
    },
    # Global
    "max_candidates_per_s1": 500,          # hard cap; diagnose before applying
    "batch_size": 10_000,                  # S1 batch size for matrix multiplication
}

# ─────────────────────────────────────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────────────────────────────────────

FEATURES = {
    "use_phonetic": True,
    "use_embeddings": False,      # set True after sparse pipeline is stable
    "use_minhash_score": True,
    "missing_fill": -1.0,         # fill value for undefined similarities
}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL — LightGBM
# ─────────────────────────────────────────────────────────────────────────────

LGBM = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.03,
    "num_leaves": 127,
    "min_child_samples": 30,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "n_estimators": 2000,
    "early_stopping_rounds": 100,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
    "class_weight": "balanced",
}

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING SPLIT
# ─────────────────────────────────────────────────────────────────────────────

SPLIT = {
    "train_frac": 0.70,
    "val_frac":   0.15,
    "test_frac":  0.15,
    "seed": RANDOM_SEED,
    # negative sampling ratio  (hard negatives per positive)
    "neg_ratio": 5,
}

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLD = {
    "search_lo": 0.05,
    "search_hi": 0.95,
    "search_step": 0.01,
    "global_default": 0.5,      # fallback before search is run
    # per-country overrides — filled after validation
    "country_overrides": {},
    # unseen-country fallback (France in test)
    "unseen_country_fallback": None,   # None = use global_default
}
