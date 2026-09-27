"""
preprocess.py — Deterministic text normalization for business names and addresses.

Design principles (from the planning docs):
  - Language-agnostic: must handle US, India, AND France (unseen in training).
  - Do NOT blindly ASCII-strip accented characters (French: é, ñ, ü).
  - Keep both raw and normalized representations.
  - Deterministic: running twice produces identical output.
  - Vectorized: uses pandas string ops, not Python loops over rows.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Optional

import pandas as pd

from config import DATA_PROCESSED, PREPROCESSING, RANDOM_SEED
from io_utils import load_parquet, read_source, save_parquet

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# COMPILE REGEX PATTERNS ONCE
# ─────────────────────────────────────────────────────────────────────────────

# Punctuation to replace with space (preserve alphanumerics + spaces + hyphens)
_PUNCT_RE    = re.compile(r"[^\w\s\-]", re.UNICODE)
# Multiple whitespace → single space
_SPACE_RE    = re.compile(r"\s+")
# Extract all numeric tokens (house numbers, PINs, ZIP codes)
_NUMS_RE     = re.compile(r"\b\d+\b")
# 5–6 digit postal code (covers Indian PIN and US ZIP)
_POSTAL_RE   = re.compile(r"\b(\d{5,6})\b")
# Ampersand variants
_AMP_RE      = re.compile(r"\s*&\s*", re.UNICODE)


# ─────────────────────────────────────────────────────────────────────────────
# ABBREVIATION MAPS (from config)
# ─────────────────────────────────────────────────────────────────────────────

NAME_ABBREV  = PREPROCESSING["name_abbrev_map"]
ADDR_ABBREV  = PREPROCESSING["addr_abbrev_map"]


# ─────────────────────────────────────────────────────────────────────────────
# CORE NORMALIZATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _unicode_normalize(text: str) -> str:
    """NFC normalize: compose canonical equivalents. Preserve accented chars."""
    return unicodedata.normalize("NFC", text)


def _fold_accents(text: str) -> str:
    """Create an accent-folded variant (ASCII approximation).
    Used as a secondary representation — NOT a replacement for the primary.
    Useful for French: 'Société' → 'Societe'
    """
    return unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("ascii")


def _expand_abbreviations(tokens: list[str], abbrev_map: dict[str, str]) -> list[str]:
    """Expand tokens that exactly match an abbreviation entry."""
    return [abbrev_map.get(tok, tok) for tok in tokens]


def normalize_name(text: str) -> str:
    """
    Normalize a business name string.

    Returns the cleaned, lowercase, abbreviation-expanded string.
    All information-destroying steps (token deletion) are avoided unless
    validated. Preserves French accented characters in primary form.
    """
    if not text or not text.strip():
        return ""

    s = _unicode_normalize(text)
    # Normalize ampersand variants → ' and '
    s = _AMP_RE.sub(" and ", s)
    # Lowercase
    s = s.lower()
    # Replace punctuation (except hyphens) with space
    s = _PUNCT_RE.sub(" ", s)
    # Collapse whitespace
    s = _SPACE_RE.sub(" ", s).strip()
    # Tokenize and expand abbreviations
    tokens = s.split()
    tokens = _expand_abbreviations(tokens, NAME_ABBREV)
    # Re-join
    return " ".join(tokens)


def normalize_address(text: str) -> str:
    """
    Normalize a business address string.

    Conservative: only expands unambiguous abbreviations.
    Preserves numeric tokens — they are the strongest address signal.
    """
    if not text or not text.strip():
        return ""

    s = _unicode_normalize(text)
    s = _AMP_RE.sub(" and ", s)
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _SPACE_RE.sub(" ", s).strip()
    tokens = s.split()
    tokens = _expand_abbreviations(tokens, ADDR_ABBREV)
    return " ".join(tokens)


def extract_postal_code(text: str) -> str:
    """Extract the first 5–6 digit postal / PIN / ZIP code from address text."""
    if not text:
        return ""
    m = _POSTAL_RE.search(text)
    return m.group(1) if m else ""


def extract_address_numbers(text: str) -> list[str]:
    """Return all numeric tokens found in address (house numbers, PIN, etc.)."""
    if not text:
        return []
    return _NUMS_RE.findall(text)


def make_char_ngrams(text: str, n: int) -> set[str]:
    """Return set of character n-grams (padded with spaces) for a text string."""
    if len(text) < n:
        return set()
    padded = f" {text} "
    return {padded[i:i+n] for i in range(len(padded) - n + 1)}


def make_sorted_tokens(text: str) -> str:
    """Return alphabetically sorted token string — catches word transpositions."""
    return " ".join(sorted(text.split()))


# ─────────────────────────────────────────────────────────────────────────────
# PHONETIC ENCODING
# ─────────────────────────────────────────────────────────────────────────────

def _get_double_metaphone(text: str) -> tuple[str, str]:
    """Return (primary, secondary) Double Metaphone codes for the first word."""
    try:
        from phonetics import dmetaphone
        first_word = text.split()[0] if text.split() else ""
        if not first_word:
            return ("", "")
        result = dmetaphone(first_word)
        return (result[0] or "", result[1] or "")
    except Exception:
        return ("", "")


def _get_soundex(text: str) -> str:
    try:
        import jellyfish
        first_word = text.split()[0] if text.split() else ""
        return jellyfish.soundex(first_word) if first_word else ""
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# FULL RECORD PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all normalizations to a source dataframe.
    Adds derived columns alongside the originals (raw fields are preserved).

    Input columns expected:  entity_id, business_name, business_address, country
    Added columns:
        name_clean, name_folded, name_tokens (list), name_sorted,
        name_char2 (set), name_char3 (set), name_char4 (set),
        name_prefix4, name_dbl_meta_primary, name_dbl_meta_secondary, name_soundex,
        address_clean, address_folded, address_tokens (list), address_sorted,
        address_char3 (set), address_numbers (list), postal_code,
        country_norm
    """
    log.info("Preprocessing %d records ...", len(df))

    # ── Names ──────────────────────────────────────────────────────────────
    df["name_clean"]  = df["business_name"].apply(normalize_name)
    df["name_folded"] = df["name_clean"].apply(_fold_accents)

    df["name_tokens"] = df["name_clean"].str.split()
    df["name_sorted"] = df["name_clean"].apply(make_sorted_tokens)
    df["name_prefix4"] = df["name_sorted"].str[:4]

    for n in [2, 3, 4]:
        col = f"name_char{n}"
        df[col] = df["name_clean"].apply(lambda t: make_char_ngrams(t, n))

    # Phonetic (first-word based)
    phon = df["name_clean"].apply(_get_double_metaphone)
    df["name_dbl_meta_primary"]   = phon.apply(lambda x: x[0])
    df["name_dbl_meta_secondary"] = phon.apply(lambda x: x[1])
    df["name_soundex"] = df["name_clean"].apply(_get_soundex)

    # ── Addresses ──────────────────────────────────────────────────────────
    df["address_clean"]  = df["business_address"].apply(normalize_address)
    df["address_folded"] = df["address_clean"].apply(_fold_accents)

    df["address_tokens"] = df["address_clean"].str.split()
    df["address_sorted"] = df["address_clean"].apply(make_sorted_tokens)

    df["address_char3"]  = df["address_clean"].apply(lambda t: make_char_ngrams(t, 3))
    df["address_numbers"] = df["business_address"].apply(extract_address_numbers)
    df["postal_code"]    = df["business_address"].apply(extract_postal_code)

    # ── Country ────────────────────────────────────────────────────────────
    df["country_norm"] = df["country"].str.lower().str.strip()

    # ── Combined text for embedding retrieval (Phase 3 Pass H) ────────────
    df["embedding_text"] = (
        df["name_clean"] + " [SEP] " + df["address_clean"]
    )

    log.info("Preprocessing done. Total columns: %d", len(df.columns))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_preprocessing(
    data_train_dir: Path,
    data_test_dir: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """
    Preprocess all 6 source files and write Parquet artifacts.

    Returns: dict of label → parquet path
    """
    from io_utils import read_source

    files = {
        "train_s1": (data_train_dir / "train_source1.tsv", "S1"),
        "train_s2": (data_train_dir / "train_source2.tsv", "S2"),
        "train_s3": (data_train_dir / "train_source3.tsv", "S3"),
        "test_s1":  (data_test_dir  / "test_source1.tsv",  "S1"),
        "test_s2":  (data_test_dir  / "test_source2.tsv",  "S2"),
        "test_s3":  (data_test_dir  / "test_source3.tsv",  "S3"),
    }

    output_paths = {}
    for label, (tsv_path, src) in files.items():
        out_path = output_dir / f"{label}_processed.parquet"
        if out_path.exists():
            log.info("Skipping %s (already exists)", out_path.name)
            output_paths[label] = out_path
            continue
        df = read_source(tsv_path, label=f"[{src}] {label}")
        df = preprocess_dataframe(df)
        save_parquet(df, out_path)
        output_paths[label] = out_path

    log.info("All preprocessing complete.")
    return output_paths


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from config import DATA_RAW_TRAIN, DATA_RAW_TEST, DATA_PROCESSED
    paths = run_preprocessing(DATA_RAW_TRAIN, DATA_RAW_TEST, DATA_PROCESSED)
    log.info("Output paths: %s", paths)
