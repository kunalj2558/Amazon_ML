# Validation, Thresholding, Error Analysis and Submission Plan

## 1. Objective

Convert pair scores into the exact required entity-level output while optimizing macro F0.5 and protecting against false merges.

The official evaluation is macro-averaged across Source 1 entities. Correct empty predictions for true singletons receive full credit for that entity; false matches on singletons receive zero.

## 2. Exact local metric

For each Source 1 entity:

```python
pred = set(predicted_ids)
true = set(ground_truth_ids)

if not pred and not true:
    f05 = 1.0
else:
    precision = len(pred & true) / len(pred) if pred else 0.0
    recall = len(pred & true) / len(true) if true else 0.0

    if precision == 0 or recall == 0:
        f05 = 0.0
    else:
        f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
```

Then:

```python
macro_f05 = mean(f05 over every S1 entity)
```

Implement this independently of the model code.

## 3. Validation protocol

Use a fixed entity-level validation split.

Never repeatedly change the split based on results.

For every experiment record:

```text
validation_seed
validation_S1_count
macro_F0.5
mean_precision
mean_recall
singleton_F0.5
multi_match_F0.5
```

## 4. Threshold search

For candidate pair score `p`, evaluate thresholds across a fine grid.

Example:

```text
0.05 to 0.95
step 0.01
```

For each threshold:

1. score every candidate;
2. convert scores into per-S1 predicted sets;
3. calculate macro F0.5;
4. calculate singleton performance;
5. record candidate volume.

Select the threshold based on validation F0.5.

Do not use a default 0.5 threshold without testing it.

## 5. Per-country thresholding

Because France is absent from training, do not create a France-specific threshold.

Possible strategy:

```text
US threshold
India threshold
global fallback threshold for unseen countries
```

Only retain country-specific thresholds if they improve validation without creating brittle behavior.

For an unseen country, use the global/fallback rule.

## 6. Candidate-count-aware decisions

Track the score distribution per S1.

Useful diagnostics:

```text
best_score
second_best_score
best_second_gap
number_above_threshold
candidate_count
```

The best-vs-second-best margin can help identify ambiguous matches.

Do not automatically force the best candidate to match.

## 7. Singleton protection

A conservative final layer should ask:

- Is there any candidate with strong evidence?
- Is the top score barely above threshold?
- Is the top score much stronger than the next candidate?
- Does name/address evidence agree?
- Is the entity likely to have no match according to training distributions?

Do not encode a universal rule like "score < 0.45 = singleton" until validation supports it.

## 8. Multiple matches

The task explicitly allows zero, one, or many matches per Source 1 entity.

Therefore, do not impose a one-to-one assignment between S1 and S2/S3.

For each S1:

```text
predict every candidate that passes the final evidence rule
```

subject to the F0.5-optimized decision process.

## 9. Consistency checks

Before finalizing predictions:

- remove impossible self-source matches;
- verify every predicted ID exists in test S2/S3;
- verify no duplicate IDs;
- verify every final match was a candidate;
- verify every S1 has one output row.

Do not apply graph/transitive rules merely because they sound theoretically attractive. Validate any such rule against training data first.

## 10. Error analysis loop

After every meaningful submission:

### If false positives dominate

Inspect:

- weak address evidence;
- chain businesses sharing names;
- common business names;
- postal-code collisions;
- over-aggressive semantic retrieval;
- singleton contamination.

Then raise precision through features/thresholds.

### If false negatives dominate

Inspect:

- candidate pairs absent from blocker;
- transliteration;
- address missingness;
- severe abbreviation;
- word reordering;
- rare names.

Improve blocking before changing the classifier.

### If validation is strong but leaderboard is weak

Investigate:

- validation distribution mismatch;
- threshold overfitting;
- unseen-country behavior;
- candidate recall on test;
- feature preprocessing inconsistency;
- accidental test-time filtering.

Do not repeatedly tune to the public leaderboard without a stable local protocol.

## 11. Submission generation

Generate:

```text
output/matching_results.tsv
output/candidate_pairs.tsv
```

`matching_results.tsv`:

```text
source1_entity_id<TAB>matched_entity_ids
```

`candidate_pairs.tsv`:

```text
source1_entity_id<TAB>candidate_entity_ids
```

Empty lists must remain empty rather than being filled with placeholders.

## 12. Pre-submit validator

Run the official validator:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

Expected status:

```text
PASS
```

Treat any warning/error as blocking the submission until fixed.

## 13. Additional internal validation

Implement:

```python
assert len(matching) == len(test_source1)
assert matching.source1_entity_id.is_unique
assert candidate.source1_entity_id.is_unique
```

For every row:

```python
assert matched_ids <= candidate_ids
assert matched_ids <= valid_s2_s3_ids
assert no_duplicates(matched_ids)
```

Also verify:

```text
all test S1 IDs appear
no extra S1 IDs appear
```

## 14. Submission versioning

Use:

```text
submission_01/
submission_02/
submission_03/
...
```

Each folder should store:

```text
matching_results.tsv
candidate_pairs.tsv
config.json
model_version.txt
local_metrics.json
notes.md
```

Keep a leaderboard log:

```text
submission_id
time
local_F0.5
public_score
major_change
```

The official rules state that teams may make at most five submissions per day and should maintain submission history. Do not waste submissions on unvalidated files.

## 15. Final package

The final ZIP should contain:

```text
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

The code must be capable of reproducing the outputs from the provided data.

## 16. Final checklist

```text
[ ] local macro F0.5 calculated
[ ] threshold frozen
[ ] no leaderboard-driven last-minute untracked change
[ ] candidate recall measured
[ ] singleton behavior reviewed
[ ] all S1 rows present
[ ] all match IDs valid
[ ] matches subset of candidates
[ ] no duplicates
[ ] official validator PASS
[ ] package reproducible
[ ] methodology document complete
[ ] model/license constraints checked
[ ] no external data used
```
