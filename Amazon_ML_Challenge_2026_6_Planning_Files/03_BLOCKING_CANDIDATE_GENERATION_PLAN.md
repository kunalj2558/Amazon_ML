# Candidate Generation / Blocking Implementation Plan

## 1. Objective

Candidate generation defines the maximum achievable recall of the downstream matcher.

The official challenge explicitly asks for `candidate_pairs.tsv` and uses it to analyze blocking quality. Every final match must be present in the candidate set.

Therefore:

> First maximize true-pair coverage; then let the classifier recover precision.

## 2. Formal definition

For each Source 1 entity `s1`, generate:

```text
C(s1) = union of candidate IDs from every blocking pass
```

Only compare:

```text
S1 -> S2
S1 -> S3
```

Never generate S1->S1 matches.

## 3. Measure blocking correctly

On training data:

```text
blocking_recall =
  number of ground-truth pairs contained in candidate set
  /
  total number of ground-truth pairs
```

Also measure:

```text
average candidates per S1
median candidates per S1
p95 candidates per S1
maximum candidates per S1
candidate reduction ratio
recall by source pair
recall by country
recall by noise type if available
```

Do not use a guessed target as proof of quality. Measure the actual trade-off.

## 4. Pass A — exact normalized name

Create an inverted index:

```text
name_clean -> S2/S3 IDs
```

For every S1:

- retrieve exact normalized-name matches;
- add all IDs;
- record `blocking_pass = exact_name`.

This is cheap and high precision.

## 5. Pass B — character TF-IDF retrieval

Build a character n-gram TF-IDF index over S2+S3.

Recommended initial configuration:

```text
analyzer = char_wb
ngram_range = (2, 5)
sublinear_tf = True
min_df = small positive value appropriate to data
```

Retrieve top-K neighbors for each S1 name.

Tune K using training blocking recall rather than copying a fixed K.

Store:

```text
tfidf_rank
tfidf_score
```

## 6. Pass C — token retrieval / BM25

Build a token-level retrieval index.

Use BM25 if memory/runtime is practical. Otherwise use sparse TF-IDF retrieval.

Retrieve top-K by normalized name and optionally combined name+address.

Why both character and token retrieval:

- character retrieval catches typos and formatting noise;
- token retrieval catches word-level overlap and reordering.

## 7. Pass D — numeric/address blocking

Extract postal/PIN/ZIP and address numbers.

Use exact postal match as a candidate generator.

Additional combinations:

```text
country + postal_code
country + postal_code + first_name_token
country + house_number
```

Avoid using postal code alone as a final match rule.

A shared postal code can represent many businesses.

## 8. Pass E — phonetic blocking

Generate phonetic signatures for business names.

Use:

- Double Metaphone where available;
- another validated phonetic representation if useful.

Phonetic blocking should generate candidates, not determine final identity.

It is particularly useful for transliteration and spelling variants.

## 9. Pass F — character MinHash/LSH

Create character n-gram signatures for:

```text
name_clean
name_clean + address_clean
```

Use LSH only if it provides additional recall at acceptable candidate volume.

Measure incremental recall:

```text
recall_after_pass_n - recall_after_pass_(n-1)
```

If a pass adds almost no recall but produces a huge number of candidates, remove or narrow it.

## 10. Pass G — token-sort / prefix retrieval

Create:

```text
name_sorted
name_prefix
```

Use deterministic inverted indices to catch:

- word order changes;
- token subset cases;
- common legal suffix differences.

Again, this is candidate generation only.

## 11. Optional semantic retrieval

If compute is available and the simpler blocker is insufficient, add a multilingual sentence embedding index.

Input:

```text
name_clean + " [SEP] " + address_clean
```

Use ANN search.

Before committing it to the final pipeline, measure:

```text
incremental blocking recall
candidate growth
runtime
memory
```

A semantic model is justified only if it catches real training matches missed by sparse blocking.

## 12. Candidate union

For every S1:

```python
candidate_ids = set()

for blocker in blockers:
    candidate_ids.update(blocker(s1))

candidate_ids = deduplicate(candidate_ids)
```

Store metadata for each candidate:

```text
candidate_entity_id
passes_hit
tfidf_rank
tfidf_score
bm25_rank
bm25_score
semantic_rank
semantic_score
```

The final `candidate_pairs.tsv` should contain the exact candidate set passed into the final matching stage.

## 13. Candidate explosion protection

Never silently truncate candidates.

If a blocker produces thousands of candidates:

1. diagnose the key;
2. identify overly common tokens;
3. apply frequency-aware filtering;
4. use top-K retrieval;
5. record the change;
6. re-measure recall.

If a hard cap is necessary, verify that it does not remove true pairs on training data.

## 14. Hard-negative generation

After blocking:

```text
hard_negatives(s1) =
    candidates(s1) - ground_truth_matches(s1)
```

These are the most useful negatives for pair classification.

Keep some easy negatives only for calibration/robustness; do not let millions of trivial negatives dominate training.

## 15. Candidate set diagnostics

Generate:

```text
experiments/blocking_metrics.csv
experiments/missed_true_pairs.csv
experiments/candidate_size_distribution.csv
```

For every missed training true pair, record:

```text
S1 ID
true S2/S3 ID
name similarity
address similarity
country
which blockers failed
```

This is the fastest way to design the next blocking rule.

## 16. Final candidate artifact

Produce:

```text
output/candidate_pairs.tsv
```

Rules:

- one row per Source 1 entity;
- all candidate IDs are S2/S3;
- no duplicate candidate IDs;
- empty list allowed;
- every final match must occur here.

## 17. Blocking acceptance criteria

Before model training:

```text
[ ] Candidate recall measured on training data
[ ] Recall measured separately for S1-S2 and S1-S3
[ ] Singleton S1 entities included
[ ] Candidate size distribution inspected
[ ] Missed true pairs manually inspected
[ ] No external lookup used
[ ] Final candidate set is reproducible
```

## 18. Key principle

Do not optimize blocking for the smallest candidate set.

Optimize:

```text
high true-pair recall
+
manageable candidate count
+
stable runtime
```

The classifier cannot recover a pair that blocking discarded.
