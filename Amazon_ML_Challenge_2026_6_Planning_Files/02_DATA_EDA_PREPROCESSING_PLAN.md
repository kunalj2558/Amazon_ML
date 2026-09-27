# Data, EDA and Preprocessing Implementation Plan

## 1. Objective

Build a deterministic preprocessing layer that exposes multiple representations of each business record without destroying information needed for matching.

The challenge explicitly identifies name noise, address noise, missing components, transliteration, abbreviations, punctuation differences, word-order changes and typos. France is present only in test, so normalization must not depend on the training-country list.

## 2. Data contract

Expected tables:

```text
train_source1.tsv
train_source2.tsv
train_source3.tsv
train_ground_truth.tsv

test_source1.tsv
test_source2.tsv
test_source3.tsv
```

Required fields:

```text
entity_id
business_name
business_address
country
```

Ground truth:

```text
source1_entity_id
matched_entity_ids
```

Never assume the file's physical order is meaningful.

## 3. EDA checklist

Create `01_eda.ipynb` or an equivalent reproducible script.

### 3.1 Schema checks

For every file:

- column names;
- dtypes;
- row count;
- unique ID count;
- duplicate IDs;
- null/empty rates;
- whitespace-only values;
- unusually long strings;
- malformed IDs;
- source inferred from ID prefix.

Assertions:

```python
assert set(source1.entity_id.str[:3]) == {"S1-"}  # after inspecting actual data
assert source2.entity_id.str.startswith("S2-").all()
assert source3.entity_id.str.startswith("S3-").all()
```

Do not hard-code the exact country set.

### 3.2 Ground-truth analysis

Calculate:

- singleton percentage;
- distribution of match count per S1;
- percentage of S1 with only S2 matches;
- only S3 matches;
- both S2 and S3 matches;
- maximum number of matches per S1;
- match cardinality by country;
- match cardinality by source pair.

This directly informs final thresholding.

### 3.3 Noise taxonomy

Sample true matches and false/non-match candidate pairs.

Classify:

- punctuation;
- case;
- legal suffix;
- abbreviation;
- token reorder;
- typo;
- transliteration;
- partial name;
- DBA/trade name;
- missing address component;
- address abbreviation;
- number formatting;
- PIN/ZIP missing;
- landmark-style address;
- language/diacritic differences.

Save representative examples to:

```text
experiments/noise_examples.csv
```

Do not manually use external information to resolve ambiguous entities.

## 4. Preprocessing philosophy

Do not replace raw fields.

Store both:

```text
raw_name
raw_address
```

and normalized/derived fields.

This allows the model to compare both original and transformed representations.

## 5. Name normalization

Implement:

```python
normalize_name(text) -> str
```

Pipeline:

1. Unicode normalization.
2. Case normalization.
3. Unicode-aware punctuation handling.
4. Normalize ampersand-like separators to a consistent token.
5. Collapse whitespace.
6. Carefully normalize common legal abbreviations.
7. Generate token list.
8. Generate sorted-token representation.
9. Generate character n-grams.

Important:

- Do not blindly ASCII-strip French characters.
- Preserve a Unicode representation.
- Optionally create a secondary accent-folded representation.
- Do not delete tokens unless validation demonstrates benefit.

Recommended fields:

```text
name_clean
name_folded
name_tokens
name_sorted
name_prefix
name_char2
name_char3
name_char4
```

## 6. Address normalization

Implement:

```python
normalize_address(text) -> str
extract_address_numbers(text) -> list[str]
extract_postal_code(text, country=None) -> optional[str]
```

Store:

```text
address_clean
address_folded
address_tokens
address_sorted
address_numbers
postal_code
```

Address abbreviations should be conservative.

For example, `st` is ambiguous in real-world addresses and should not be blindly rewritten without validation.

Do not use geocoding or external address services.

## 7. Country handling

Country is an observed input feature.

Do:

- preserve raw country;
- create normalized country string;
- compare country equality;
- optionally use country as a blocking key where appropriate.

Do not:

- restrict processing to US/India;
- create a fixed class list that excludes France;
- discard records from an unseen country;
- use external country/business databases.

## 8. Multiple representations

Every record should expose:

```text
raw
normalized
tokenized
token-sorted
character n-grams
numeric address tokens
postal/PIN/ZIP representation
phonetic representation
embedding text
```

This is more robust than trying to produce one "perfect" normalized string.

## 9. Leakage controls

Never create preprocessing rules using validation/test labels.

Allowed:

- statistics derived from training records;
- dictionaries derived only from provided training data;
- frequency counts from the provided dataset.

Not allowed:

- internet lookups;
- external business databases;
- geocoding;
- external entity-resolution APIs.

## 10. Data quality gates

Before blocking:

```text
[ ] Every row has entity_id
[ ] Source prefixes are valid
[ ] Required columns exist
[ ] No accidental duplicate IDs
[ ] Raw strings preserved
[ ] Normalization is deterministic
[ ] No external lookup occurred
[ ] France/test records remain in pipeline
```

## 11. Required outputs

Create:

```text
data/processed/train_source1_processed.parquet
data/processed/train_source2_processed.parquet
data/processed/train_source3_processed.parquet
data/processed/test_source1_processed.parquet
data/processed/test_source2_processed.parquet
data/processed/test_source3_processed.parquet
experiments/eda_report.json
experiments/noise_examples.csv
```

Parquet is recommended for internal processing; final competition outputs remain TSV.

## 12. Acceptance tests

The preprocessing module passes only if:

- running it twice produces identical outputs;
- IDs are unchanged;
- row counts are unchanged;
- raw fields are preserved;
- no test entity disappears;
- multilingual characters do not crash processing;
- null/empty values are handled explicitly.

## 13. Practical optimization

If the dataset is large:

- use Polars or chunked pandas;
- precompute normalized columns once;
- avoid Python loops over full tables;
- use vectorized operations;
- cache character n-grams/phonetic features;
- save intermediate artifacts.

The objective is not merely clean text. The objective is fast generation of reliable signals for blocking and pair classification.
