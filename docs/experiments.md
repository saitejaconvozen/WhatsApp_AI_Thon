# Next-Phase Evaluation

Checkpoint: 2026-09-18. All comparisons run locally; no template is sent to a
model provider or Meta. This is research evidence, not a production gate.

## Chronological Split

The cutoff is the 75th percentile of eligible families' first creation dates:
2026-08-18T11:16:01.240000+00:00. Training families must have every candidate
creation/update timestamp strictly before it. Test families first appear on or
after it. Families spanning the cutoff are purged. Invalid or missing dates
exclude the affected family. This produces 1,242 training families, 423 test
families and 26 boundary exclusions; no normalized family crosses the split.

These are creation/update timestamps, not category-decision timestamps. The
export contains current labels and text, not historical versions. Therefore
this is a chronological snapshot test, not proof of what the system would have
known at the historical cutoff. Both partitions must contain both categories;
the cutoff is not searched to improve scores.

## Compared Methods

All methods use class-balanced logistic regression, random seed 42 and the
same chronological split. Vocabularies and transforms are fitted on training
text only. Header, body, footer and button text are included.

- Word TF-IDF: word unigrams and bigrams, matching the serving baseline.
- Character TF-IDF: character-within-word n-grams of lengths 3 to 5.
- LSA: word TF-IDF reduced to 100 dimensions with TruncatedSVD and normalized.
  This is a local latent-semantic representation, not a pretrained neural
  sentence encoder. Smaller datasets use fewer dimensions.

Implementations use scikit-learn's documented
[TF-IDF vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
and [TruncatedSVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html).

## Measured Results

| Method | Utility precision | Utility recall | False utility | Accuracy |
| --- | --- | --- | --- | --- |
| Word TF-IDF | 77.7% | 71.0% | 52 | 70.2% |
| Character TF-IDF | 79.1% | 74.1% | 50 | 72.6% |
| Local LSA | 69.9% | 82.7% | 91 | 68.1% |

The training-majority baseline scores 39.7% on this holdout. The holdout has
255 utility and 168 marketing labels; this differs substantially from the
random holdout. Do not compare these scores directly with its 82.0% accuracy
or 73.6% utility precision as if the test samples were identical.

## Wording Overlap Audit

A separate training-fitted character TF-IDF representation compares normalized
test bodies with training bodies. At a preselected cosine threshold of 0.90,
20 test families have a close lexical match. IDs and scores are saved in the
report. Batching bounds the similarity matrix size.

All methods are also scored on the same remaining 403 test examples. Utility
precision is 76.4% for words, 78.4% for characters, and 69.3% for LSA. This is
a sensitivity check, not a new independent test or semantic deduplication.
It can miss differently worded paraphrases. No threshold was optimized on the
test labels, and no serving model was replaced.

## Review and Rewrite Validation

The Error Review queue contains the original random-holdout errors: 43 false
utility and 33 false marketing. Those are distinct from this experiment's error
counts. Full components are preserved. Human review status, reason, notes,
business context, candidate draft and manually observed Meta outcome are saved
separately and exported as JSON. Source labels never change.

The existing limited English rewrite remains the only automatic editor. It
requires a supported event and confirmed recipient relationship, removes only
separate promotional fragments and refuses ambiguous mixed content. General
semantic rewriting and policy validation are not implemented by this iteration.

## Remaining Work

- The platform team must review examples and provide factual recipient context.
- Obtain Meta decision timestamps and historical revisions for stronger testing.
- Compare a locally approved pretrained sentence encoder or configured LLM.
- Validate any expanded rewriting with human reviewers and subsequent actual
  Meta decisions. The app records outcomes but does not submit messages.
- Use a new untouched holdout before selecting a production model. These models
  were compared on one test split, and all remain unsuitable for auto-submission.

## Structural Features: a Measured Negative Result (2026-09-18)

Marginal rates on the eligible families suggested that signals `normalized_content`
erases should help. Against a 37% base utility rate: templates with no
placeholders are 20.2% utility and those with three or more are 53.0%; templates
with no buttons are 50.5% utility and those with buttons 30.2%.

A twelve-feature block (placeholder counts, button count, header/footer
presence, body length, emoji, `!`, currency/percent, uppercase ratio, digit
runs, URLs) was added to the serving pipeline as a `FeatureUnion` branch and
measured against text alone on the random grouped holdout, all calibrated:

| Variant | Accuracy | Utility precision | Utility recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Text only | 79.8% | 78.7% | 64.5% | 70.9% |
| Text + all 12 features | 76.7% | 80.9% | 50.8% | 62.4% |
| Text + 3 strongest | 75.9% | 76.8% | 52.5% | 62.3% |
| Text + 3 strongest, log1p | 76.1% | 75.4% | 55.2% | 63.7% |
| Text + placeholder count only | 79.6% | 81.0% | 60.7% | 69.4% |

Every variant is worse than text alone by F1. The chronological split reproduces
it independently: 66.3% accuracy and 69.6% utility precision with the structural
block, against 70.3% and 74.4% without.

The likely explanation is that TF-IDF already carries this signal — normalization
rewrites each placeholder as the token `variable` and each number as `number`, so
term frequency is a proxy for the counts, and the raw scaled counts mostly add
variance. **The features were removed from the serving pipeline.** They remain as
the `Word TF-IDF + structure` experiment arm so the claim can be rechecked rather
than taken on trust.

The marginal rates were real; they were just not additional information. A
per-feature rate difference does not establish that a model lacks the signal.

## Grouping Change (2026-09-18)

Family keys now cover header, body, footer and buttons rather than the body
alone, so templates differing only in their buttons are no longer merged. This
yields 1,923 clean families instead of 1,691 and drops conflicting families from
86 to 75. Measured effect on accuracy is within noise (74.0% against 74.2% on the
requested-UTILITY slice); it is a correctness fix, not an accuracy improvement.

The key is derived at read time. `normalize_record` folds `family` into the
stored record `id`, so changing the stored value would change every id, duplicate
the dataset on the next import of the same export, and orphan every annotation
keyed by record id.
