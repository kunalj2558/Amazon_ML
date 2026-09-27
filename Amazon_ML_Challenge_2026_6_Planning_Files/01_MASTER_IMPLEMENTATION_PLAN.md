# Amazon ML Challenge 2026 — Master Implementation Plan

## 0. Purpose

This document is the single source of truth for implementing the Business Entity Resolution solution for the Amazon ML Challenge 2026.

The official problem is to match every Source 1 business against matching records in Source 2 and Source 3 despite noisy names and addresses. Source 1 is the deduplicated reference source; a Source 1 entity may have zero, one, or many matches. The test set contains France even though training contains US and India, so the pipeline must not hard-code the training countries.

The leaderboard metric is macro-averaged F0.5. It is precision-heavy, and singleton predictions matter. The official challenge also requires a candidate_pairs.tsv artifact, exact output formatting, a reproducible source package, and a methodology document.

## 1. Non-negotiable competition constraints

- Input/output files are TSV. Always read with `sep="\t"`.
- Every test Source 1 entity must appear exactly once in `matching_results.tsv`.
- `matched_entity_ids` may contain only existing Source 2 / Source 3 test IDs.
- No duplicate IDs inside a match list.
- Final matches must be a subset of `candidate_pairs.tsv`.
- Correctly predicted singletons are rewarded.
- The final metric is macro F0.5:
  `F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)`
- The final submission package must contain:
  - `output/matching_results.tsv`
  - `output/candidate_pairs.tsv`
  - `code/business_entity_resolution/src/`
  - `README.md`
  - pinned `requirements.txt`
  - filled methodology document.
- The model must comply with the stated <=8B parameter and MIT/Apache 2.0 model-license constraints.
- External entity lookup, geocoding, government/business registries, commercial ER APIs, and internet-based data augmentation are prohibited.
- The official validator must be run before submission.
- Maintain version history of all leaderboard submissions.

## 2. Core architecture

```text
RAW TSV
  |
  v
Data Contract + Profiling
  |
  v
Normalization / Field Representations
  |
  v
Multi-pass Candidate Generation
  |       |       |       |
 BM25   char-NN  phonetic  exact/structural
  |       |       |       |
  +-------+-------+-------+
          |
          v
     Candidate Union
          |
          v
     Pair Features
          |
          v
  Pairwise Match Classifier
          |
          +---- optional semantic/reranker score
          |
          v
   F0.5-aware calibration
          |
          v
  Singleton + consistency rules
          |
          v
matching_results.tsv
candidate_pairs.tsv
          |
          v
official validator
```

## 3. Design priorities

Priority order:

1. Never lose true pairs during candidate generation.
2. Keep candidate sets computationally manageable.
3. Maximize precision at the final decision stage because F0.5 is precision-heavy.
4. Explicitly model the empty-match/singleton case.
5. Prevent train/validation leakage.
6. Keep the pipeline language-agnostic enough for unseen France.
7. Make every experiment reproducible.

## 4. What to implement first

### MVP pipeline

Implement this before attempting complex neural models:

1. Robust normalization.
2. Character TF-IDF retrieval.
3. Token-based retrieval.
4. Exact/near exact PIN and numeric-address blocking.
5. RapidFuzz + structural pair features.
6. LightGBM pair classifier.
7. Entity-level validation.
8. F0.5 threshold search.
9. Submission generation and validation.

Only after the MVP is stable should additional semantic models be added.

### Advanced pipeline

Add:

- multilingual sentence embeddings for candidate retrieval;
- embedding cosine features;
- hard-negative mining;
- a cross-encoder only if compute/time permits;
- ensemble fusion only when validation demonstrates improvement.

Do not add a component merely because it is sophisticated. It must improve entity-level validation F0.5 without damaging reproducibility.

## 5. Critical correction to the earlier long-form plan

The official challenge window is only three days. Therefore, an 8–11 day development schedule is not operationally valid for the live challenge.

Use the following three-day execution structure:

### Day 1 — Understand + baseline

- Load all files.
- Validate schema.
- Profile counts/nulls/duplicates.
- Analyze ground-truth match cardinality.
- Quantify singleton rate.
- Build normalization.
- Build baseline candidate generation.
- Build local F0.5 evaluator.
- Produce first valid submission if possible.

### Day 2 — Candidate recall + pair model

- Improve blocking.
- Measure blocking recall on training data.
- Build hard negatives.
- Generate pair features.
- Train LightGBM.
- Tune threshold on entity-level validation.
- Submit.
- Inspect leaderboard feedback.

### Day 3 — Precision refinement + final package

- Add semantic retrieval/features only if justified.
- Tune thresholds/ensemble.
- Analyze false positives and false negatives.
- Improve singleton handling.
- Run complete validation.
- Generate final outputs.
- Run official validator.
- Build reproducible ZIP.
- Submit final version with preserved experiment history.

## 6. Experiment discipline

Every experiment must record:

```text
experiment_id
date/time
code version
data version
blocking configuration
feature set
model configuration
negative sampling configuration
validation split seed
validation F0.5
precision
recall
candidate recall
candidate count
submission score
notes
```

Never overwrite the best model or output without saving the previous version.

## 7. Definition of done

The solution is ready only when:

- all test Source 1 IDs are present exactly once;
- every predicted match exists in test S2/S3;
- no duplicate match IDs exist;
- final matches are contained in candidates;
- local validation uses the official macro-per-S1 F0.5 formulation;
- candidate recall has been measured;
- singleton behavior has been measured;
- no external data was used;
- all dependencies/models satisfy competition constraints;
- official validator returns PASS;
- final ZIP reproduces outputs from raw input.

## 8. Recommended source layout

```text
amazon_ml_solution/
├── data/
│   ├── train/
│   └── test/
├── src/
│   ├── config.py
│   ├── io_utils.py
│   ├── preprocess.py
│   ├── blocking.py
│   ├── features.py
│   ├── train.py
│   ├── predict.py
│   ├── threshold.py
│   ├── evaluate.py
│   └── validate_pipeline.py
├── notebooks/
├── models/
├── experiments/
├── output/
├── README.md
├── requirements.txt
└── Documentation_template.md
```

## 9. Success criterion

Do not define success as "using the most advanced model." Define it as:

> highest reliable validation macro F0.5 under the competition's exact constraints, with a candidate set that contains the true matches and a final decision rule that minimizes false merges.

No model or architecture can guarantee a winning leaderboard position.
