# First Dataset Audit

Source: the JSON attachment shared in this task. Inspected on 2026-09-18.
The raw attachment is not included in the Git repository.

## Records

The export contains **4,079 draft-template records**, created between
2025-06-04 and 2026-09-17. All records declare `ENGLISH_US`; this does not
guarantee their actual content is entirely English.

| Recorded Meta category | Records |
| --- | ---: |
| Marketing | 2,552 |
| Utility | 1,157 |
| Authentication | 19 |
| Missing | 351 |

| Verification status | Records |
| --- | ---: |
| VERIFIED | 3,621 |
| FAILED | 132 |
| PENDING | 137 |
| REJECTED | 114 |
| IN_PROGRESS | 75 |

## Category Fields

- `messageBody.templateCategory`: the requested/draft category.
- `metaTemplateCategory`: Meta's final category after its verification.
- `verificationStatus`: VERIFIED means successful verification at Meta.

The platform owner confirmed these meanings on 2026-09-18. This resolves the
provenance question for the supplied export; no independent Meta API check
was performed. The owner described review of the body, header, other template
parts and variables/placeholders. That does not establish Meta's internal
algorithm or provide a sentence-level explanation of any individual decision.

There are **1,333 records with requested UTILITY and recorded MARKETING**.
This is a field disagreement, not proof that a previously approved utility
template was later reclassified. The export is a snapshot rather than an
approval history.

There are **3,425 VERIFIED records with a recorded marketing/utility label**:
2,406 marketing and 1,019 utility. These are candidate examples, before
filtering missing content, duplicate families, conflicting labels, and
unsupported message formats. VERIFIED has the owner-confirmed meaning above.
The generic importer also accepts APPROVED; that alias was not confirmed for
this export and is not present in its status counts.

## Message Formats

| Format | Records |
| --- | ---: |
| TEXT | 2,964 |
| MEDIA | 790 |
| CAROUSEL | 205 |
| REPLY | 46 |
| CALL_PERMISSION | 45 |
| OTP_TEMPLATES | 19 |
| LIST | 9 |
| LIMITED_TIME_OFFER | 1 |

The first classifier will use verified TEXT records with recorded binary
categories. Other formats remain browsable, but are excluded from training
because their media or interaction content needs additional handling.
Authentication remains a separate category and is never mapped to marketing.

## First Build

1. Import this nested JSON format without mixing requested and observed labels.
2. Show category distributions, missing labels, statuses, duplicate families,
   and conflicting labels in a local dataset browser.
3. Evaluate a TF-IDF/logistic-regression baseline with grouped holdout data.
4. Show its prediction separately from a conservative English policy checklist.
5. Offer utility drafts and limited, explicit promotional-sentence removal.
6. Keep real records and model artifacts in ignored local storage.

No language model or Meta API is connected in the first local version.
Rule assessments and local-model predictions are not Meta decisions.

## Executed Import and Training

All 4,079 records were imported into local SQLite storage. Among verified TEXT
records with binary labels, 2,539 have non-empty bodies; 2,523 remain after
requiring at least 10 characters in the body.

Grouping normalizes body placeholders, numbers, URLs, whitespace and punctuation.
Across all records there are 2,685 normalized body families and 1,394 repeated
records beyond the first representative per family. This does not detect all
paraphrases or prove that every grouped message has the same business purpose.

Within eligible training candidates, 86 body families have both labels, covering
262 records. These families were excluded. The resulting training pool contains
1,691 distinct families: 1,066 marketing and 625 utility.

The first model used a 25% grouped holdout: 1,268 training and 423 test families.
Accuracy is 82.0%, macro F1 is 80.8%, utility precision is 73.6%, and utility recall
is 78.4%. The holdout confusion matrix is:

| Recorded / Predicted | Marketing | Utility |
| --- | ---: | ---: |
| Marketing | 227 | 43 |
| Utility | 33 | 120 |

The 43 false utility predictions are a useful next error-analysis set. This
first result supports further investigation; it is not sufficient evidence
to automate utility submissions. The app's serving model is refitted on all
1,691 eligible families after evaluation. Its predictions on those same
families are marked as seen in training.

## Open Questions

- Is there an event history with category-decision timestamps?
- Can the platform provide message trigger and recipient relationship context?
- Are real filled-in examples available without customer-identifying details?

## Reference

[Meta's public category definitions](https://whatsappbusiness.com/products/platform-pricing/)
were consulted during the project discussion. Policies need to be versioned
and reviewed as the project develops.
