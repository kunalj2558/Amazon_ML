# Engineering Execution, Reproducibility and Methodology Plan

## 1. Objective

Turn the research pipeline into a clean, reproducible competition submission that can survive technical review.

The official final package is not just predictions. It includes outputs, runnable code, requirements, and methodology. The top packages may be reviewed and reproduced.

## 2. Final repository structure

```text
business_entity_resolution/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── io_utils.py
│   ├── preprocess.py
│   ├── blocking.py
│   ├── features.py
│   ├── train.py
│   ├── score.py
│   ├── threshold.py
│   ├── evaluate.py
│   ├── predict.py
│   └── submission.py
├── tests/
│   ├── test_preprocess.py
│   ├── test_blocking.py
│   ├── test_features.py
│   └── test_submission.py
├── models/
├── configs/
├── experiments/
├── output/
├── README.md
├── requirements.txt
└── Documentation_template.md
```

## 3. Configuration-driven implementation

Put all tunable values in one config:

```yaml
blocking:
  char_top_k: ...
  token_top_k: ...
  semantic_top_k: ...
  max_candidates_per_s1: ...

features:
  use_phonetic: true
  use_embeddings: true

model:
  type: lightgbm
  learning_rate: ...
  num_leaves: ...
  n_estimators: ...

threshold:
  global: ...
  country_specific: ...

runtime:
  batch_size: ...
  num_workers: ...
```

Do not scatter magic numbers across Python files.

## 4. Pipeline entry points

Provide:

```bash
python -m src.preprocess
python -m src.blocking
python -m src.train
python -m src.predict
python -m src.evaluate
python -m src.submission
```

Or provide one:

```bash
python run_pipeline.py --config configs/final.yaml
```

The README must show exact commands.

## 5. Determinism

Set:

- random seed;
- NumPy seed;
- LightGBM seed;
- train/validation split seed;
- retrieval configuration;
- model versions.

When GPU components are used, document any nondeterministic operations.

## 6. Memory strategy

For large TSVs:

- use Polars or chunked pandas;
- avoid loading unnecessary duplicate copies;
- use Parquet for intermediate data;
- use sparse matrices for TF-IDF;
- batch embedding inference;
- persist reusable indices.

Never keep multiple full dense representations of every record in RAM unnecessarily.

## 7. Runtime profiling

Record:

```text
preprocessing_time
blocking_time_by_pass
feature_time
training_time
inference_time
submission_generation_time
peak_memory
candidate_count
```

Use these measurements to remove bottlenecks.

## 8. Testing strategy

### Unit tests

Test:

- Unicode normalization;
- missing values;
- abbreviation handling;
- address number extraction;
- candidate deduplication;
- feature calculations;
- empty prediction handling.

### Integration tests

Use a tiny synthetic dataset where expected matches are known.

Verify:

```text
raw -> preprocess -> block -> features -> model -> output
```

### Submission tests

Before any real upload:

```text
test all IDs
test candidate subset property
test duplicates
test empty lists
test exact headers
test tab separators
```

## 9. Methodology document

The official methodology requirements cover:

1. Methodology used.
2. Candidate generation/blocking strategy.
3. Model architecture and feature engineering.
4. Other relevant information.

Write the document as a technical report, not marketing copy.

Recommended structure:

```markdown
# 1. Problem Understanding
# 2. Data Analysis
# 3. Preprocessing
# 4. Candidate Generation
# 5. Feature Engineering
# 6. Model Architecture
# 7. Training and Negative Sampling
# 8. Validation Methodology
# 9. Threshold / Post-processing
# 10. Results and Ablations
# 11. Error Analysis
# 12. Reproducibility
# 13. Constraints and Fair Play
# 14. Conclusion
```

## 10. Results reporting

Never invent results.

Only report measured values:

```text
candidate recall = measured value
candidate count = measured value
validation F0.5 = measured value
precision = measured value
recall = measured value
```

Include ablation tables:

```text
Configuration | Candidate Recall | Precision | Recall | F0.5
```

Useful comparisons:

- exact-name baseline;
- sparse blocking;
- sparse + phonetic;
- sparse + address;
- sparse + semantic;
- LightGBM feature groups;
- LightGBM + reranker;
- threshold strategies.

## 11. License compliance

Maintain:

```text
experiments/model_license_audit.md
```

For every pretrained model:

```text
model name
parameter count
license
source
purpose
```

Use only models compatible with the official constraints.

Do not assume that a library being Apache/MIT automatically means every pretrained model distributed through it has the same license.

## 12. Fair-play audit

Create:

```text
experiments/fair_play_audit.md
```

State explicitly:

- all matching information comes from challenge-provided data;
- no business registry lookup;
- no geocoding;
- no commercial ER API;
- no internet-based entity augmentation;
- no hidden external labels.

## 13. Submission artifact audit

Before ZIP:

```text
output/
  matching_results.tsv
  candidate_pairs.tsv

code/
  business_entity_resolution/
    src/
    README.md
    requirements.txt

Documentation_template.md
```

Test extraction into a clean environment.

The clean-environment test is important: if the solution only runs because of hidden local files, it is not reproducible.

## 14. Three-day operational plan

### Day 1

Morning:

- inspect data;
- verify schema;
- analyze ground truth;
- implement preprocessing.

Afternoon:

- exact + sparse blocking;
- local evaluator;
- baseline model.

Evening:

- first valid submission;
- record experiment.

### Day 2

Morning:

- measure missed true pairs;
- improve blocking;
- generate hard negatives.

Afternoon:

- build full pair features;
- train LightGBM;
- optimize threshold.

Evening:

- second/third submission only after validation;
- inspect score movement.

### Day 3

Morning:

- semantic retrieval/features if needed;
- precision error analysis;
- singleton calibration.

Afternoon:

- final threshold;
- final predictions;
- official validator.

Evening:

- final ZIP;
- reproducibility test;
- final methodology document;
- final submission.

## 15. Team parallelization

If multiple team members are available:

```text
Member A:
EDA + preprocessing

Member B:
blocking + recall diagnostics

Member C:
features + LightGBM

Member D:
validation + submission + documentation
```

Merge through a shared experiment/config convention.

No member should silently change preprocessing or labels without recording it.

## 16. What not to waste time on

Do not prioritize:

- giant language models;
- external data;
- elaborate UI;
- generic accuracy metrics;
- random negative sampling;
- brute-force all-pairs matching;
- unvalidated rule stacks;
- public leaderboard overfitting;
- a neural reranker before the baseline is stable.

The challenge rewards the final entity-level prediction, not architectural complexity.

## 17. Final technical narrative

The strongest defensible narrative is:

> We treat entity resolution as a two-stage retrieval-and-ranking problem. First, multiple complementary blocking mechanisms maximize candidate recall while keeping the search space tractable. Second, heterogeneous pairwise features capture name, address, numeric, phonetic, retrieval and semantic evidence. A supervised ranker estimates match likelihood, and a validation-tuned decision layer converts pair scores into entity-level predictions under the precision-heavy macro F0.5 objective. The complete pipeline is reproducible, uses only challenge-provided data, and produces both final matches and the exact candidate set used for inference.

This should remain factual and only include components actually implemented.

## 18. Final definition of done

```text
ENGINEERING
[ ] clean repository
[ ] pinned dependencies
[ ] deterministic config
[ ] tests pass
[ ] clean-environment reproduction succeeds

ML
[ ] entity-level validation
[ ] measured blocking recall
[ ] hard-negative training
[ ] F0.5 threshold optimization
[ ] singleton analysis
[ ] error analysis

SUBMISSION
[ ] matching_results.tsv
[ ] candidate_pairs.tsv
[ ] exact headers
[ ] every S1 exactly once
[ ] all IDs valid
[ ] final matches subset of candidates
[ ] official validator PASS

COMPLIANCE
[ ] no external lookup
[ ] model parameter/license constraints checked
[ ] no fabricated results
[ ] methodology complete
[ ] version history preserved
```
