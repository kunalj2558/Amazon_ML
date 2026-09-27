# Amazon ML Challenge 2026 — Business Entity Resolution Solution

## Team Setup
**Model license:** All models used are MIT/Apache 2.0 licensed, ≤ 8B parameters.
**External data:** None. Only challenge-provided TSV files are used.

---

## Quick Start — Reproduce End-to-End

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run full pipeline (EDA → Preprocess → Block → Features → Train → Predict)
python run_pipeline.py --mode full --submission-name submission_01

# Step 3: Validate submission
python "D:\Deksto-1\Desktop\Hackthons\Projects\Amazon_ML\6ab10eb3b23ba_student_resource\student_resource\utils\validate_submission.py" \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir "D:\Deksto-1\Desktop\Hackthons\Projects\Amazon_ML\6ab10eb3b23ba_student_resource\student_resource\dataset\test"
```

---

## Pipeline Modes

| Command | What it does |
|---|---|
| `python run_pipeline.py --mode eda` | EDA report only |
| `python run_pipeline.py --mode train` | Preprocess + Block + Features + Train |
| `python run_pipeline.py --mode predict --model-path models/lgbm_*.pkl` | Inference on test set |
| `python run_pipeline.py --mode full` | All steps end-to-end |
| `python run_pipeline.py --mode full --threshold 0.45` | Override tuned threshold |

---

## Source Layout

```
amazon_ml_solution/
├── src/
│   ├── config.py          # All hyperparameters and paths
│   ├── io_utils.py        # Safe TSV read/write
│   ├── preprocess.py      # Name + address normalization
│   ├── blocking.py        # 7-pass candidate generation
│   ├── features.py        # 52 pairwise similarity features
│   ├── train.py           # LightGBM training
│   ├── predict.py         # Inference + threshold + consistency
│   └── evaluate.py        # Exact F_0.5 metric
├── run_pipeline.py        # Master runner
├── models/                # Saved model files
├── experiments/           # Logs, metrics, ablation results
├── output/                # matching_results.tsv + candidate_pairs.tsv
├── submissions/           # Versioned submission snapshots
├── requirements.txt
└── README.md
```

---

## Architecture Summary

### Stage 1: Preprocessing
- Unicode-safe name normalization (preserves French accents)
- Abbreviation expansion (Corp→Corporation, Pvt→Private, etc.)
- Address normalization + postal code extraction
- Phonetic encoding (Double Metaphone)
- Character n-gram sets, sorted token representation

### Stage 2: 7-Pass Multi-Modal Blocking
| Pass | Method | Catches |
|---|---|---|
| A | Exact normalized name | Perfect normalized matches |
| B | Char TF-IDF (ngram 2-5) | Typos, abbreviations |
| C | Token TF-IDF (word 1-2) | Word-level overlap |
| D | Postal/ZIP code exact | Same-location businesses |
| E | Double Metaphone | Transliterations |
| F | MinHash LSH (char 3-gram) | Word-order variants |
| G | Sorted-token prefix | Transpositions |

Target: **blocking recall ≥ 95%** before model training.

### Stage 3: 52-Feature Pairwise Classification
- **Name features (16):** Jaccard, Levenshtein, Jaro-Winkler, token sort/set ratio, partial ratio, char n-gram Jaccard, first token, subset check
- **Address features (13):** Jaccard, edit distance, postal exact/prefix, number overlap, token sort/set ratio
- **Phonetic features (3):** Double Metaphone primary/secondary, any-match
- **Retrieval features (7):** TF-IDF score/rank, token score/rank, best rank/score, passes hit
- **Metadata features (7):** Country match, source (S2/S3), missing indicators
- **Interaction features (6):** High-name-and-postal, name-address agreement/disagreement

### Stage 4: LightGBM Classifier
- Hard negative mining (from blocking candidates, NOT random)
- Entity-level train/val split (no leakage)
- Early stopping on validation loss
- Model selection by validation macro F_0.5

### Stage 5: F_0.5-Optimized Threshold
- Grid search from 0.05 to 0.95 (step 0.01)
- Metric: validation macro F_0.5
- Singletons: predicted empty = 1.0 per entity (conservative threshold preferred)

---

## Key Design Decisions

1. **Never one-hot encode country** — France appears only in test, pipeline is open-set
2. **Hard negatives** — candidates from blocking that are not true matches are far more informative than random negatives
3. **Entity-level split** — pairs from the same S1 entity never appear in both train and validation
4. **Precision bias** — threshold tuned on F_0.5 (not AUC), which weights precision 2× over recall
5. **Singleton protection** — all S1 entities initialized with empty prediction; model must exceed threshold to add a match

---

## Experiment Log

| Submission | Local F_0.5 | Public Score | Changes |
|---|---|---|---|
| submission_01 | TBD | TBD | Baseline pipeline |

---

## Fair Play Declaration

- All entity matching information comes from challenge-provided training data only
- No business registry lookup performed
- No geocoding APIs used
- No commercial entity resolution services used
- No internet-based data augmentation performed
- All models are MIT/Apache 2.0 licensed with ≤ 8B parameters
