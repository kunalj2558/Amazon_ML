# 🏆 Amazon ML Challenge 2026 — Championship-Level Implementation Plan
## Business Entity Resolution: The Blueprint to Win

> **This is not a tutorial. This is a battle plan.**  
> Every decision here is driven by the metric (F_0.5), the data characteristics (~1.3 GB train, unseen France country), and what state-of-the-art ER systems actually do.

---

## 🎯 Competition Intelligence Brief

| Dimension | Detail | Implication |
|---|---|---|
| **Metric** | F_0.5 macro-averaged | Precision is 2× more important than recall. Never guess a match unless you're confident |
| **Scale** | ~1.7M test entities across 3 sources | Brute-force comparison is impossible — blocking architecture is non-negotiable |
| **Unseen Country** | France appears only in test | Features must be language-agnostic. Multilingual models are mandatory |
| **Singletons** | S1 entities with zero matches score 1.0 if correctly predicted empty | Being conservative on borderline pairs = free points |
| **Model limit** | ≤ 8B params, MIT/Apache 2.0 license | Rules out GPT-4. Unlocks LLaMA-3-8B, SBERT, DeBERTa, BGE, etc. |
| **No external data** | No geocoding, no business registries, no APIs | You win purely on ML craftsmanship — this is a leveler |
| **Evaluation** | Macro F_0.5 per S1 entity | A single wrong merge on a singleton destroys that entity's score. **Singletons matter enormously** |

### The Core Math That Governs Every Decision:
```
F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)

If Precision = 1.0, Recall = 0.5  → F_0.5 = 0.833  ✅
If Precision = 0.5, Recall = 1.0  → F_0.5 = 0.556  ❌
```
**Lesson:** It is almost always better to predict fewer matches conservatively than to predict many matches aggressively.

---

## 🧠 Winning Strategy in One Sentence

> **Build a 3-tier pipeline: (1) a high-recall multi-modal blocker that catches every true match, (2) a 40+ feature LightGBM pairwise classifier as the workhorse, and (3) a fine-tuned Cross-Encoder as the precision layer — then fuse them with an F_0.5-optimized ensemble.**

---

## 🏗️ Full System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAW DATA INGESTION                           │
│         S1 (ref) + S2 + S3 — Tab-separated TSV files               │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: DEEP EDA                                │
│   Data profiling · Noise pattern taxonomy · Singleton analysis      │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  PHASE 2: PREPROCESSING ENGINE                      │
│   Name normalization · Address normalization · Feature tokens        │
│   Multilingual-safe · Country-agnostic design                       │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│            PHASE 3: 6-PASS MULTI-MODAL BLOCKING                     │
│                                                                     │
│   Pass 1: BM25 / TF-IDF Sparse Retrieval (names + addresses)        │
│   Pass 2: Dense SBERT Bi-Encoder + FAISS ANN (semantic)            │
│   Pass 3: Phonetic Blocking (Double Metaphone + NYSIIS)             │
│   Pass 4: PIN / ZIP Code Exact Match                                │
│   Pass 5: MinHash LSH on Character n-grams                          │
│   Pass 6: Sorted-Token Prefix Blocking                              │
│                                                                     │
│   ──────────────── UNION ALL PASSES ──────────────────             │
│               → candidate_pairs.tsv                                 │
│   Target: Blocking Recall > 97%                                     │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│             PHASE 4: RICH FEATURE ENGINEERING                       │
│   40+ pairwise similarity features per candidate pair               │
│   Name · Address · Structural · Phonetic · Embedding · Meta        │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                PHASE 5: 3-TIER MODEL STACK                          │
│                                                                     │
│   Tier 1: LightGBM (tabular features) ─────────────────────┐       │
│   Tier 2: SBERT Bi-Encoder cosine score ───────────────────┤       │
│   Tier 3: Fine-tuned Cross-Encoder reranker ───────────────┤       │
│                                              Ensemble fusion┘       │
│                           → match probability per pair              │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│           PHASE 6: POST-PROCESSING & THRESHOLD OPTIMIZATION         │
│   F_0.5-aware threshold search · Singleton protection               │
│   Graph consistency · Per-country calibration                       │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│               PHASE 7: VALIDATION & SUBMISSION                      │
│   validate_submission.py · Self-scoring · Package assembly          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Directory Structure (Final)

```
solution/
├── data/
│   ├── raw/                          ← Symlinks or copies of original TSVs
│   └── processed/                    ← Preprocessed / normalized TSVs
├── notebooks/
│   ├── 01_eda.ipynb                  ← Data profiling, noise analysis
│   ├── 02_preprocessing_dev.ipynb    ← Normalization experiments
│   ├── 03_blocking_analysis.ipynb    ← Blocking recall measurement
│   ├── 04_feature_analysis.ipynb     ← Feature importance, correlation
│   └── 05_threshold_tuning.ipynb     ← F_0.5 curve analysis
├── src/
│   ├── config.py                     ← All hyperparameters in one place
│   ├── preprocess.py                 ← Text normalization engine
│   ├── blocking.py                   ← Multi-pass candidate generation
│   ├── features.py                   ← Pairwise similarity features
│   ├── negative_sampler.py           ← Hard negative mining
│   ├── train_lgbm.py                 ← LightGBM training pipeline
│   ├── train_crossencoder.py         ← Cross-encoder fine-tuning
│   ├── ensemble.py                   ← Score fusion
│   ├── threshold.py                  ← F_0.5 threshold optimizer
│   ├── predict.py                    ← Inference: test set → output TSVs
│   └── evaluate.py                   ← Local F_0.5 scorer
├── models/
│   ├── lgbm_model.pkl
│   ├── crossencoder/                 ← Fine-tuned HuggingFace checkpoint
│   └── sbert_index/                  ← FAISS index files
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── requirements.txt
├── README.md
└── Documentation_template.md
```

---

## 🖥️ Training Platform Strategy

| Phase | What You Do | Best Platform | Cost |
|---|---|---|---|
| EDA + Preprocessing | Explore data, build normalizer | Local or Kaggle Notebook | Free |
| Blocking development | Test blocking recall on train subset | Kaggle Notebook (30GB RAM) | Free |
| Feature engineering | Build 40+ features for train pairs | Kaggle Notebook | Free |
| LightGBM training | Train on all positive + hard negative pairs | Google Colab / Kaggle GPU | Free |
| SBERT encoding | Encode all ~3M records with bi-encoder | Kaggle T4 GPU (30GB) | Free |
| Cross-Encoder fine-tuning | Fine-tune on 500K pairs with labels | **Vast.ai RTX 4090 (24GB)** | ~$3–8 total |
| Final inference | Score all test candidates | Google Colab T4 | Free |

### 🔑 Key Platform Tips:
- **Kaggle Notebooks** give you 30h/week of GPU + 30GB RAM — use it for all data processing
- **Google Colab T4 (free)** is sufficient for LightGBM and SBERT encoding
- **Vast.ai** at ~$0.30/hr for RTX 4090 is the best value for Cross-Encoder fine-tuning (6-8 hrs = ~$3)
- Save model checkpoints to **Google Drive** — mount it in Colab/Kaggle to persist between sessions
- Use **mixed precision (fp16)** to fit larger batches on GPU

---

## 📊 PHASE 1: Deep EDA & Data Profiling (Day 1)

### Goal: Build a complete mental model of the data before touching the model

### 1.1 — Data Volume Profiling
- Count records per source (S1, S2, S3) split by country
- Estimate S1 × S2 and S1 × S3 cross-product sizes (the space blocking must reduce)
- Check for duplicate entity_ids within each source
- Measure null/missing rates per column

### 1.2 — Ground Truth Distribution Analysis (CRITICAL)
- How many S1 entities are **singletons** (0 matches)?
- Distribution of match counts: how many S1 have 1, 2, 3, 4+ matches?
- Which source (S2 vs S3) contributes more matches?
- Distribution of matches per S1 by country (US vs India)
- **This governs your threshold strategy entirely**

### 1.3 — Noise Pattern Taxonomy
Sample 200 matched pairs and manually categorize noise types:

| Noise Category | Example | Feature to Build |
|---|---|---|
| Abbreviation | "Corp" vs "Corporation" | Expand abbreviations → exact match |
| Punctuation | "Johnson & Sons" vs "Johnson and Sons" | Strip & normalize |
| Word order | "Bank of India State" vs "State Bank of India" | Token-sort similarity |
| Missing suffix | "Tata Steel" vs "Tata Steel Ltd" | Token subset check |
| Typo | "Relince" vs "Reliance" | Edit distance |
| Transliteration | "Sharma" vs "Sarma" | Phonetic encoding |
| Abbreviation + typo | "Mc D's" vs "McDonalds" | Combined fuzzy |
| Address landmark | "Near HDFC Bank" vs "42, MG Road" | Hard case → embedding needed |
| PIN present vs absent | "560001" vs no PIN | PIN-conditional features |
| State/city missing | "Mumbai 400001" vs "400001 Maharashtra" | Token overlap |

### 1.4 — Key EDA Questions to Answer (Answer these before Phase 2)
1. What fraction of S1 entities are singletons? → If >40%, default to conservative threshold
2. Are US and India entities structurally different? → Informs per-country features
3. Do S2 and S3 have the same noise profile? → May need source-specific features
4. What is the average character length of names and addresses? → Informs tokenization

---

## 🔤 PHASE 2: Preprocessing Engine (Day 1–2)

> **Design principle: Every normalization step must be language-agnostic (no hardcoded English patterns) so French records in the test set are handled correctly.**

### 2.1 — Business Name Normalization Pipeline

**Step-by-step transformations (applied in order):**

```
1. Unicode normalization (NFKD → ASCII where possible, but keep accents for French)
2. Lowercase
3. Remove/replace: punctuation chars → space, except alphanumerics
4. Abbreviation expansion dictionary:
   corp / corp. → corporation
   pvt / pvt. → private
   ltd / ltd. → limited
   inc / inc. → incorporated
   co / co. → company
   llc → limited liability company
   llp → limited liability partnership
   intl / int'l → international
   mfg → manufacturing
   svcs → services
   assoc → associates
   bros → brothers
   hldgs → holdings
   grp → group
   & → and
5. Remove noise tokens: the, of, a, an (context-specific stopwords)
6. Deduplicate consecutive identical tokens
7. Strip extra whitespace
8. Generate: normalized_name, name_tokens (list), name_sorted (token-sorted string)
```

**Generated name representations (all used in features or blocking):**
- `name_clean`: normalized string
- `name_tokens`: list of words
- `name_sorted`: alphabetically sorted tokens joined (for token-sort similarity)
- `name_char3grams`: set of character 3-grams (for MinHash blocking)
- `name_prefix3`: first 3 chars of normalized name (for prefix blocking)
- `name_soundex`: Soundex of first word
- `name_dbl_metaphone`: Double Metaphone of first word

### 2.2 — Address Normalization Pipeline

**Step-by-step transformations:**

```
1. Unicode normalization → lowercase
2. Abbreviation expansion:
   rd / rd. → road
   st / st. → street (careful: "St" can mean Saint too)
   ave / ave. → avenue
   blvd → boulevard
   dr → drive
   ln → lane
   hwy → highway
   fl / flr → floor
   apt → apartment
   ste → suite
   bldg → building
   no. → number
3. Extract and store separately:
   - all_numbers: list of all numeric tokens (house numbers, PINs, ZIPs)
   - pin_code: 5-6 digit string (US ZIP / Indian PIN)
   - country_code: from 'country' column
4. Remove punctuation, normalize whitespace
5. Generate: addr_clean, addr_tokens, addr_sorted, addr_char3grams
```

**Key extracted fields:**
- `pin_code`: Extracted 5-6 digit PIN or ZIP (HIGH SIGNAL for blocking and features)
- `addr_numbers`: All numeric tokens (house numbers, floor numbers)
- `addr_tokens`: Cleaned word list

### 2.3 — Country Handling
- Keep country as a raw string (do NOT one-hot encode)
- Use country as a grouping key for per-country threshold tuning
- Treat France exactly the same as US/India in the pipeline — language-agnostic processing ensures this works automatically

---

## 🔍 PHASE 3: 6-Pass Multi-Modal Blocking (Day 2–3)

> **The blocking recall is the absolute ceiling of your F_0.5 score. If a true match pair is not in your candidate set, it is a permanent miss.**
>
> **Target: Blocking Recall ≥ 97% | Reduction Ratio ≥ 99.9%**

### Blocking Quality Metrics You Must Track:

```
Blocking Recall = # true match pairs in candidates / # total true match pairs
Reduction Ratio = 1 - (# candidate pairs / # total possible pairs)
Candidates per S1 = average # candidates per S1 entity (target: 50–200)

Ideal: Recall ≥ 97%, Candidates per S1 ≤ 200
```

---

### Pass 1: BM25 Sparse Retrieval on Business Names

**Method:**
- Build a BM25 index over all S2 + S3 `name_clean` fields
- Query with each S1 `name_clean`
- Retrieve Top-50 candidates per S1

**Why BM25 over plain TF-IDF:**
- BM25 accounts for document length — short names won't be dominated by long name matches
- More robust for business names of varying lengths

**Implementation approach:**
- Use `rank_bm25` library or `sklearn` TF-IDF with sublinear_tf=True as approximation
- Process in batches of 10,000 S1 records to manage memory
- Use character n-gram analyzer (analyzer='char_wb', ngram_range=(2,4)) — critical for catching typos

**Expected recall contribution: ~70-75% of true pairs covered**

---

### Pass 2: Dense Semantic Retrieval (SBERT Bi-Encoder + FAISS)

**Method:**
- Use a multilingual sentence embedding model: `paraphrase-multilingual-MiniLM-L12-v2` (MIT license, 118M params, 50+ languages)
- Create combined text field: `name_clean + " [SEP] " + addr_clean`
- Encode all S2 + S3 records → dense embeddings (d=384)
- Build FAISS IVF index for fast approximate nearest-neighbor search
- Query each S1 embedding → retrieve Top-50 candidates

**Why this matters:**
- Captures semantic similarity that exact string matching misses
- "State Bank of India" ↔ "SBI" — embedding model understands these are similar
- Handles French text naturally (multilingual model)
- Addresses described via landmark vs street number — embedding captures semantic context

**Why this is separate from Pass 1 (not redundant):**
- BM25 catches exact token matches and typos
- SBERT catches semantic paraphrases and abbreviations
- Their union covers cases neither handles alone

**Expected additional recall from Pass 2: ~10-15% new pairs beyond Pass 1**

---

### Pass 3: Phonetic Blocking (Double Metaphone + NYSIIS)

**Method:**
- Generate Double Metaphone code for first 2 words of `name_clean`
- Generate NYSIIS code for business name
- Blocking key: `country + double_metaphone_code`
- All S1, S2, S3 records with same key are candidate pairs

**Why Double Metaphone over Soundex:**
- Soundex was designed for English surnames only
- Double Metaphone handles names from many language families
- Better for Indian names (Sharma ↔ Sarma), transliterations

**Expected additional recall: ~3-5% new pairs (especially for transliteration variants)**

---

### Pass 4: PIN / ZIP Code Exact Match Blocking

**Method:**
- If `pin_code` is extractable from address:
  - Blocking key: `country + pin_code`
  - All S1 with same PIN as S2/S3 are candidates
- Also do near-match: if first 3 digits of PIN match (broader catchment)

**Why this is extremely high precision:**
- Two businesses with the same PIN code + similar name = very likely match
- Acts as a precision booster for the candidates that come from this pass

**Expected additional recall: ~5-8% new pairs for PIN-bearing addresses**

---

### Pass 5: MinHash LSH on Character n-grams

**Method:**
- Generate character 3-gram set for `name_clean + addr_clean`
- Use MinHash signatures (128 hash functions) + LSH bands
- Set band/row parameters for ~0.5 Jaccard similarity threshold
- Use `datasketch.MinHashLSH` for efficient implementation

**Why this catches what others miss:**
- Character n-grams are very robust to word boundary changes
- "McDonald" vs "McDonalds" — 3-gram overlap is very high
- Different word order, extra tokens — still high 3-gram Jaccard

**Expected additional recall: ~2-4% new pairs (catches word-boundary and ordering variants)**

---

### Pass 6: Sorted-Token Prefix Blocking

**Method:**
- Sort tokens of `name_clean` alphabetically → join
- Take first 4 characters of this sorted string as blocking key
- Key: `country + first4_of_sorted_name`

**Why this catches transpositions:**
- "Bank of India State" and "State Bank of India" have the same sorted token string
- This is a cheap, deterministic pass that costs almost nothing

---

### Blocking Union & Deduplication

```
FINAL CANDIDATE SET = UNION of all 6 passes
Deduplicate (S1_id, S2/S3_id) pairs
Cap: If any S1 entity has >500 candidates, keep top-500 by BM25+SBERT score
Write: candidate_pairs.tsv
```

### Blocking Recall Validation (MANDATORY before Phase 4):
- On training data, compute blocking recall
- **If recall < 95%: debug which pass is missing which type of pair**
- Examine missed pairs manually → add a 7th blocking pass if needed
- Do not proceed to model training until recall is ≥ 95%

---

## ⚙️ PHASE 4: Rich Feature Engineering (Day 3–4)

> For every candidate pair `(S1_record, S2/S3_record)`, compute the following features.
> These 40+ features are the input to your LightGBM classifier.

### Group A: Name String Similarity Features (15 features)

| Feature Name | Method | Library | Why Useful |
|---|---|---|---|
| `name_jaccard_word` | Jaccard on word token sets | custom | Robust to word reordering |
| `name_jaccard_char3` | Jaccard on char 3-gram sets | custom | Typo-tolerant |
| `name_levenshtein_norm` | Normalized edit distance | `rapidfuzz` | Overall edit distance |
| `name_jaro_winkler` | Jaro-Winkler distance | `rapidfuzz` | Great for prefix matches (abbreviations) |
| `name_token_sort_ratio` | FuzzyWuzzy token_sort_ratio | `rapidfuzz` | Word reordering (SBI vs State Bank India) |
| `name_token_set_ratio` | FuzzyWuzzy token_set_ratio | `rapidfuzz` | Subset/superset names |
| `name_partial_ratio` | Best substring match | `rapidfuzz` | DBA names, trade names |
| `name_trigram_ratio` | Character trigram overlap | custom | Fuzzy match |
| `name_longest_common_substr` | LCS length / max_len | custom | Measures common core |
| `name_common_word_count` | Count of exact shared words | custom | Direct overlap count |
| `name_unique_words_s1` | # unique words in S1 name | custom | Length feature |
| `name_unique_words_s23` | # unique words in S2/S3 name | custom | Length feature |
| `name_len_ratio` | len(name1) / len(name2) | custom | Length mismatch signal |
| `name_first_word_match` | Binary: first word exact? | custom | High signal if company proper name starts same |
| `name_soundex_match` | Binary: same Soundex? | `jellyfish` | Phonetic match |

### Group B: Address String Similarity Features (12 features)

| Feature Name | Method | Why Useful |
|---|---|---|
| `addr_jaccard_word` | Jaccard on address word tokens | Overall address similarity |
| `addr_jaccard_char3` | Char 3-gram Jaccard | Typo-tolerant address match |
| `addr_levenshtein_norm` | Normalized edit distance on address | Edit distance |
| `addr_token_sort_ratio` | Token-sorted fuzzy ratio | Address component reordering |
| `addr_token_set_ratio` | Token set fuzzy ratio | Partial address |
| `addr_pin_exact_match` | Binary: same extracted PIN/ZIP? | **Highest signal feature** |
| `addr_pin_prefix3_match` | Binary: same first 3 digits of PIN? | Nearby area |
| `addr_number_overlap` | Jaccard of numeric tokens | House number, floor match |
| `addr_common_word_count` | Count shared address words | Direct word overlap |
| `addr_len_ratio` | Length ratio of addresses | Completeness signal |
| `addr_has_pin_both` | Binary: both records have PIN? | Data quality signal |
| `addr_has_pin_one` | Binary: only one has PIN? | Partial data signal |

### Group C: Phonetic Name Features (4 features)

| Feature Name | Method |
|---|---|
| `name_dbl_metaphone_match` | Binary: same Double Metaphone primary code? |
| `name_dbl_metaphone_secondary_match` | Binary: same Double Metaphone secondary code? |
| `name_nysiis_match` | Binary: same NYSIIS code? |
| `name_phonetic_any_match` | Binary: any of the above phonetic codes match? |

### Group D: Dense Embedding Similarity Features (4 features)

| Feature Name | Method | Model |
|---|---|---|
| `name_sbert_cosine` | Cosine similarity of name embeddings | `all-MiniLM-L6-v2` |
| `addr_sbert_cosine` | Cosine similarity of address embeddings | `all-MiniLM-L6-v2` |
| `combined_sbert_cosine` | Cosine similarity of `name+addr` embeddings | `all-MiniLM-L6-v2` |
| `multilingual_sbert_cosine` | Cosine similarity using multilingual model | `paraphrase-multilingual-MiniLM-L12-v2` |

> **Note:** You already compute SBERT embeddings in Phase 3 blocking. Reuse them — no re-encoding needed. The cosine similarity between the FAISS query result and its retrieved neighbor is already computed; store it during blocking.

### Group E: Structural / Meta Features (8 features)

| Feature Name | Description |
|---|---|
| `country_match` | Binary: same country string? (always 1 if blocking by country — but keep as feature) |
| `source_pair` | Categorical: "S1-S2" or "S1-S3" (different matching difficulty per source) |
| `name_is_subset` | Binary: all name1 tokens appear in name2 (or vice versa)? |
| `name_abbrev_expanded_match` | Binary: after expanding abbreviations, do names exactly match? |
| `blocking_pass_count` | How many blocking passes retrieved this pair? (higher = more evidence) |
| `bm25_rank` | Rank of this candidate in BM25 retrieval (1 = closest match) |
| `sbert_rank` | Rank of this candidate in SBERT retrieval |
| `bm25_sbert_rank_min` | min(bm25_rank, sbert_rank) — best rank from either retrieval |

### Group F: Derived Interaction Features (4 features)

| Feature Name | Description |
|---|---|
| `name_and_pin_both_high` | Binary: name_jaccard_word > 0.7 AND addr_pin_exact_match = 1 |
| `name_high_addr_low` | Binary: name similarity high but address similarity low (suspicious) |
| `name_low_addr_high` | Binary: address similarity high but name low (possibly same location, different biz) |
| `phonetic_and_fuzzy_agree` | Binary: phonetic match AND token_sort_ratio > 0.7 |

### Total: ~47 features per candidate pair

---

## 🤖 PHASE 5: 3-Tier Model Stack (Day 4–7)

### The Philosophy:
- **Tier 1 (LightGBM):** Fast, interpretable, great with tabular features. Your workhorse.
- **Tier 2 (SBERT Bi-Encoder):** Captures semantic similarity globally. Already computed during blocking.
- **Tier 3 (Cross-Encoder):** Re-reads the actual text of both records together — the most powerful pairwise model. Used on top-N candidates only.

---

### 5.1 — Training Data Construction (THE SECRET WEAPON)

#### Positive Pairs:
- All (S1, S2) and (S1, S3) pairs from `train_ground_truth.tsv`
- Label = 1

#### Negative Pairs (Hard Negative Mining — This is what separates winners):
**DO NOT randomly sample negatives. Use hard negatives:**

**Hard negatives** = pairs that your blocking retrieved (they look similar) but are NOT true matches

```
For each S1 entity:
  candidate_negatives = blocking_candidates(S1) - ground_truth_matches(S1)
  
Hard negatives are far more informative than random negatives because:
  - They are what the model will actually see during inference
  - They force the model to discriminate between "looks similar" vs "actually same"
```

#### Class Imbalance Handling:
- Typical ratio: ~1 positive per 50–200 hard negatives
- Strategy: Sample hard negatives at 1:5 (positive:negative) ratio for training
- Use `class_weight='balanced'` or `scale_pos_weight` in LightGBM as a backup

#### Train/Validation Split Strategy:
```
Split at S1 ENTITY LEVEL (not pair level):
  - 70% of S1 entities → training pairs
  - 15% of S1 entities → validation (threshold tuning)
  - 15% of S1 entities → test (final evaluation)

Do NOT split at pair level — this leaks information.
Ensure all 3 countries are represented in each split.
```

---

### 5.2 — Tier 1: LightGBM Classifier

**The Core Model — Fast, powerful, interpretable**

**Hyperparameters to tune:**
```
n_estimators: 500–2000 (use early stopping on validation)
learning_rate: 0.02–0.05
num_leaves: 63–255
min_child_samples: 20–100
feature_fraction: 0.7–0.9
bagging_fraction: 0.8–0.9
bagging_freq: 5
class_weight: balanced (or manual scale_pos_weight)
```

**Training procedure:**
1. Train on 70% training pairs with early stopping (patience=100 rounds)
2. Track validation AUC and validation F_0.5 at each checkpoint
3. Save best model by validation F_0.5 (not AUC — optimize for the actual metric)
4. Analyze feature importance → drop features with near-zero importance
5. Re-train with pruned feature set for final model

**Expected performance:** F_0.5 ~ 0.82–0.88 on validation

---

### 5.3 — Tier 2: SBERT Bi-Encoder Score

**Already computed during Phase 3 blocking.**

The `combined_sbert_cosine` feature (cosine between S1 and S2/S3 combined embeddings) is a standalone strong signal. Use it both:
- As a feature in LightGBM (Group D features)
- As an independent score in the ensemble

**Standalone threshold:** If SBERT cosine > 0.92 → high-confidence match (can be a rule override)

---

### 5.4 — Tier 3: Fine-Tuned Cross-Encoder (THE PRECISION WEAPON)

> **This is what top teams use that others don't. It's the single biggest performance jump.**

**What is a Cross-Encoder?**
Unlike a bi-encoder (which encodes each record independently), a cross-encoder takes BOTH records as input simultaneously and computes attention across them. This means it can model fine-grained interactions between the two texts.

**Model choice:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (Apache 2.0, 22M params)
OR `BAAI/bge-reranker-base` (MIT, 278M params — stronger but needs more GPU memory)

**Fine-tuning procedure:**

```
Input format for cross-encoder:
  "[S1] McDonald's Corp, 42 Main St, New York 10001 [SEP] [S2] McDonalds Inc, 
   42 Main Street, NY 10001"

Labels:
  1.0 for positive pairs (true matches from ground truth)
  0.0 for hard negative pairs

Training:
  Loss: Binary Cross-Entropy
  Optimizer: AdamW, lr=2e-5
  Epochs: 3–5
  Batch size: 64 (fp16 on GPU)
  Warmup: 10% of steps
  Early stopping on validation F_0.5

Inference:
  Apply ONLY to top-100 candidates per S1 entity (from LightGBM or SBERT pre-filtering)
  This makes it feasible — not all 50M candidate pairs, just the top-100 per S1
```

**Expected performance jump:** +3 to +6 F_0.5 points over LightGBM alone

**Training cost:** ~4–6 hours on Vast.ai RTX 4090 (~$2)

---

### 5.5 — Ensemble Fusion Strategy

**Combine all three tiers into a final score:**

```
final_score = α × lgbm_prob + β × sbert_cosine_normalized + γ × crossencoder_prob

Where:
  α, β, γ are weights tuned on validation F_0.5
  Typical starting point: α=0.4, β=0.2, γ=0.4
  Normalize SBERT cosine to [0,1] range before combining

Tuning method:
  Grid search over α, β, γ combinations that sum to 1.0
  Evaluate macro F_0.5 on validation split for each combination
  Pick (α, β, γ) that maximizes validation F_0.5
```

**Alternative ensemble approaches (if time allows):**
- Stacking: train a meta-learner (logistic regression) on [lgbm_prob, sbert_cosine, ce_prob] → match
- Ranking ensemble: for each S1, re-rank candidates by ensemble score, then apply threshold

---

## 🎛️ PHASE 6: Post-Processing & Threshold Optimization (Day 7–8)

### 6.1 — F_0.5-Optimized Threshold Search

> **This is not a binary optimization — it's a continuous search on your actual metric.**

```
For threshold t in {0.20, 0.25, 0.30, ..., 0.80}:
  For each S1 entity in validation:
    predicted_matches = {pair for pair in candidates if final_score(pair) > t}
    
  Compute per-entity precision, recall, F_0.5
  Compute macro F_0.5 across all validation S1 entities
  
Record t* = argmax(macro F_0.5)
```

**Per-Country Threshold Tuning:**
- US businesses may need a different threshold than India businesses
- France (test-only) — you cannot tune on it, so use a single global threshold as a safe default
- If per-country thresholds improve validation F_0.5, apply them

**Per-Entity Confidence Calibration:**
- If a S1 entity has 0 candidates with score > 0.3, predict singleton (empty match list) with high confidence
- If a S1 entity has 1 candidate with score > 0.8, always predict as match

### 6.2 — Singleton Precision Strategy

**Why singletons are gold:**
- A S1 entity with no true matches scores 1.0 if you predict empty list
- Predicting a wrong match for a singleton scores 0.0
- If 40% of entities are singletons, and you get them all right, that's 40% of your macro F_0.5 for free

**Strategy:**
1. Do NOT lower threshold to capture marginal matches — let them be singletons
2. If the best candidate for a S1 entity has `final_score < 0.45`, predict singleton
3. Singletons = conservative prediction = free F_0.5 points

### 6.3 — Graph Consistency Check (Optional, but Powerful)

**After thresholding, apply transitive closure:**
```
If S1-001 → S2-050 (predicted match)
And S1-002 → S2-050 (predicted match)

This is a contradiction — S2-050 can only represent one real business.
Action: Assign S2-050 to the S1 entity with the higher match score.
Keep the other S1 entity as a singleton.
```

This post-processing step removes impossible multi-source assignments and improves precision.

### 6.4 — Rule-Based High-Confidence Overrides

**Apply these rules AFTER the ML model, as overrides:**

```
Rule 1 (Match Override):
  IF name_jaccard_word > 0.85 AND addr_pin_exact_match = 1
  THEN → predict as match (regardless of model score)
  
Rule 2 (Singleton Override):
  IF name_jaccard_word < 0.2 AND combined_sbert_cosine < 0.5
  THEN → do NOT predict as match (regardless of model score)

Rule 3 (Same Business, Different Address):
  IF name_token_set_ratio > 0.90 AND same_country
  THEN → predict as match (common for chains with multiple addresses)
```

---

## ✅ PHASE 7: Validation & Submission (Day 8)

### 7.1 — Local Self-Scoring

Build a local F_0.5 scorer that exactly mirrors the competition metric:
```
For each S1 entity:
  precision = |predicted ∩ ground_truth| / |predicted|    (0 if predicted is empty and ground_truth is empty → 1.0)
  recall    = |predicted ∩ ground_truth| / |ground_truth| (1.0 if both empty)
  f05       = (1.25 × precision × recall) / (0.25 × precision + recall)

macro_f05 = mean(f05 for all S1 entities)
```

**Always compute this on your held-out validation split before submitting.**

### 7.2 — Format Validation

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
```

**Expected output: `PASS` with exit code 0**

### 7.3 — Final Submission Package

```
<teamname>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/               (all source files)
│       ├── README.md          (exact reproduction steps)
│       └── requirements.txt   (pinned versions)
└── Documentation_template.md  (filled methodology doc)
```

---

## 📅 Day-by-Day Execution Schedule

| Day | Phase | Tasks | Platform |
|---|---|---|---|
| **Day 1** | EDA | Load data, count records, sample matched pairs, noise taxonomy, singleton % | Kaggle Notebook |
| **Day 2** | Preprocessing | Write normalizer for names + addresses, verify on samples | Local / Kaggle |
| **Day 3** | Blocking (Passes 1-3) | BM25 + SBERT encoding + phonetic. Measure blocking recall | Kaggle T4 GPU |
| **Day 4** | Blocking (Passes 4-6) | PIN + MinHash + prefix blocking. Union all. Write candidate_pairs.tsv | Kaggle T4 GPU |
| **Day 5** | Feature Engineering | Compute all 47 features for all training candidate pairs | Kaggle CPU/GPU |
| **Day 6** | LightGBM Training | Train + tune + validate. Get first F_0.5 score. **First submission!** | Google Colab |
| **Day 7** | SBERT + Ensemble | Add embedding features. Tune ensemble weights. | Google Colab T4 |
| **Day 8** | Cross-Encoder | Fine-tune cross-encoder. Apply to top-100 candidates | Vast.ai RTX 4090 |
| **Day 9** | Post-processing | Threshold search, singleton strategy, graph consistency | Local |
| **Day 10** | Iteration | Analyze errors, improve blocking/features based on validation mistakes | Local + Colab |
| **Day 11+** | Refinement | Ensemble experiments, per-country tuning, re-submit | Colab / Vast.ai |

---

## 🏅 What Separates Top-3 Teams from the Rest

| Factor | Most Teams | Top Teams (You) |
|---|---|---|
| **Blocking** | Single-pass TF-IDF only | 6-pass multi-modal union |
| **Negatives** | Random negative sampling | Hard negative mining from blocking candidates |
| **Features** | 10-15 basic features | 47 rich features + embedding cosines |
| **Model** | Single LightGBM | 3-tier stack with fine-tuned cross-encoder |
| **Threshold** | Default 0.5 | F_0.5-optimized per-country threshold |
| **Singletons** | Predict conservatively? Not sure | Explicitly model singletons → free F_0.5 points |
| **France** | Might break → hardcoded English logic | Language-agnostic pipeline, multilingual embeddings |
| **Validation** | AUC or accuracy | Actual macro F_0.5 on entity-level holdout |
| **Graph consistency** | Not done | Remove impossible multi-source assignments |

---

## 🔧 Technology Stack & Libraries

| Component | Library | Version | License |
|---|---|---|---|
| Data loading | `polars` (faster than pandas for large TSVs) | ≥0.20 | MIT |
| Fuzzy string | `rapidfuzz` | ≥3.0 | MIT |
| Phonetic | `jellyfish`, `phonetics` | latest | MIT |
| BM25 | `rank_bm25` | latest | Apache 2.0 |
| TF-IDF | `scikit-learn` | ≥1.3 | BSD-3 |
| MinHash LSH | `datasketch` | latest | MIT |
| Embeddings | `sentence-transformers` | ≥2.6 | Apache 2.0 |
| FAISS | `faiss-cpu` / `faiss-gpu` | latest | MIT |
| ML model | `lightgbm` | ≥4.0 | MIT |
| Cross-encoder | `sentence-transformers` CrossEncoder | ≥2.6 | Apache 2.0 |
| GPU training | `transformers` + `torch` | latest | Apache 2.0 / BSD |
| Calibration | `scikit-learn` CalibratedClassifierCV | ≥1.3 | BSD-3 |
| Graph ops | `networkx` | ≥3.0 | BSD-3 |

### All chosen models are ≤ 8B parameters and MIT/Apache 2.0 licensed ✅

---

## 🔁 Iteration Loop (After First Submission)

```
Submit → Get leaderboard score → Error analysis
                  ↓
Score < 0.80  →  Blocking recall too low. Add more blocking passes.
                  Check which true pairs are outside candidate set.
                  
Score 0.80–0.87 → Feature quality. Add embedding features.
                   Try cross-encoder. Tune negatives ratio.
                   
Score 0.87–0.91 → Threshold + singletons. Per-country tuning.
                   Graph consistency. Ensemble weights.
                   
Score > 0.91   →  Fine-grained: try larger cross-encoder (BGE-reranker-large).
                   Experiment with contrastive fine-tuning of bi-encoder on your data.
```

---

## ⚠️ Critical Rules & Pitfalls

| Rule | What Goes Wrong Without It | Fix |
|---|---|---|
| `sep="\t"` always | Silent single-column parse failure | Always specify sep |
| Every S1 entity in output | Submission rejected | Assert row count == S1 count before saving |
| matched ⊆ candidates | Pipeline bug — validator warns | Check subset property before saving |
| No S1-IDs in matches | Submission rejected | Filter all matched IDs starting with "S1-" |
| No external data | Disqualification | Use only provided TSV files |
| Model ≤ 8B params | Violation | All suggested models are well within limit |
| MIT/Apache 2.0 license | Violation | All suggested models verified |
| Split at S1 entity level, not pair level | Data leakage → inflated val score | Entity-level stratified split |
| Optimize threshold on F_0.5, not AUC | AUC optimal ≠ F_0.5 optimal | Use actual metric for threshold tuning |
| Keep France-compatible (no English-only hardcoding) | All France entities score 0.0 | Language-agnostic features + multilingual SBERT |

---

## 📝 Documentation Template Strategy (Filled with Your Approach)

The jury reviews this for top teams. Write it to impress:
- **Section 2:** Clearly state "hybrid 3-tier pipeline: sparse retrieval + dense retrieval + cross-encoder reranking"
- **Section 3:** Give exact blocking recall numbers (e.g., "97.3% recall, 99.8% reduction ratio")
- **Section 4:** List all 47 features with rationale. Show feature importance plot.
- **Section 5:** Show F_0.5 vs threshold curve. Show confusion analysis.
- **Appendix B:** Include blocking recall curve, ensemble weight ablation, per-country scores

---

*Amazon ML Challenge 2026 | Business Entity Resolution | Championship Plan*  
*Created: September 2026*
