# Implementation Plan and Progress

Last updated: 2026-09-18. This file is updated as checks actually pass.

## Current Execution Checkpoint

Accuracy-goal continuation: a fixed 2,106 / 702 / 937 train/validation/historical
test partition is saved under `.data/improvement/`. A 36-configuration text
search selected 80.3% requested-Utility and 86.9% overall **validation** accuracy.
Five neural fine-tuning epochs completed. Epoch 3 was selected on requested-
Utility validation accuracy: **81.1%**, with **86.3%** overall validation
accuracy. Later epochs did not improve it. No new test score or 90% claim
follows from these selection numbers; the serving model is unchanged.

Public model tester: [Cloudflare link](https://impressive-guam-graphs-architecture.trycloudflare.com).
Aggregate metrics are at `/results/` on the same hostname.
See [public-hosting.md](public-hosting.md) for its temporary lifetime and privacy boundary.

This checkpoint supersedes the initial implementation history below.

| Work | Status |
| --- | --- |
| Separate learned classification from conversion eligibility | Complete |
| Require human context before edits; preserve transaction facts | Complete for conservative extractive edits |
| Intent and business-context inputs for generation | Complete; preset fallback when hosted AI is off |
| Secure opt-in provider launcher and stricter response validation | Complete |
| Training-only retrieval, bounded 100-template evaluation | Complete; 73% accuracy, not 90% |
| Pilot comparison in main app and separate results site | Complete |
| Regression checks | 142 tests passed; desktop/mobile Compose and pilot checks passed |
| Audit disagreements with platform-owner context | Next; original labels remain unchanged |
| Tune alternative models on validation data, then test on fresh data | Pending; no further paid experiments started |
| Validate generated and rewritten drafts against actual Meta outcomes | Pending |

Detailed results, limitations and launch commands: [llm-pilot.md](llm-pilot.md).

## Initial Implementation History

| Step | Deliverable | Status |
| --- | --- | --- |
| 1 | Inspect the supplied JSON and document label semantics | Complete |
| 2 | Local SQLite storage and nested JSON / CSV / XLSX import | Complete; automated tests passed |
| 3 | Text baseline, grouped evaluation, and policy checklist | Complete; automated tests passed |
| 4 | Dataset browser, template reviewer, and model dashboard | Complete; desktop/mobile browser checks passed |
| 5 | Import supplied data, train, and test end-to-end behavior | Complete; 4,079 imported records, 1,691 training families |
| 6 | Start the app and deliver the local URL and documentation | Complete; http://127.0.0.1:8765, VS Code project opened |

## Agreed Scope

- Preserve requested categories separately from recorded Meta categories.
- Import 4,079 supplied records; keep data in ignored local storage.
- Browse all formats. Train initially on completed, labeled TEXT records.
- Exclude authentication, missing labels, unfinished workflows, conflicting
  families, and unsupported formats from binary classifier training.
- Group normalized message bodies and remove repeated families before evaluation.
- Show model predictions separately from explicit English policy checks.
- Offer grounded utility draft patterns and limited promotional-sentence removal.
- Show category precision, recall, confusion matrix, and sample counts.
- Keep uploads and model execution local; no external LLM or Meta API calls.

## Validation Gates

- [x] Nested export imports with the observed category and status counts.
- [x] Importing the same file twice does not duplicate records.
- [x] Draft categories never substitute for missing recorded Meta categories.
- [x] Pending/rejected and unsupported formats are excluded from training.
- [x] Conflicting families are excluded and train/test families do not overlap.
- [x] Missing business context prevents automatic utility rewrites.
- [x] Upload errors, search, pagination, and review interactions work.
- [x] Browser layout works at desktop and mobile sizes.
- [x] The server responds at the delivered local URL.

## Execution Results

- 28 automated tests passed. Two dependency deprecation warnings remain.
- Browser checks passed at 1440 x 1000 and 390 x 844, with no JavaScript errors
  and no horizontal page overflow. Screenshots are in ignored `artifacts/`.
- Reimporting the supplied file through the browser added no duplicate records.
- Real-data evaluation used 1,268 training families and 423 holdout families.
- Accuracy: 82.0%; majority-class baseline: 63.8%; macro F1: 80.8%.
- Utility precision: 73.6% (120 correct of 163 utility predictions).
- Utility recall: 78.4% (120 of 153 recorded utility examples).
- These are baseline measurements, not a production approval guarantee.

## Next Phase

The first local version is complete. See `docs/HANDOFF.md` to resume in VS Code.
Label provenance was confirmed by the platform owner on 2026-09-18.
Priorities are inspection of false utility predictions, stronger duplicate grouping,
future-time validation, and comparing semantic models against this baseline.

## Next-Phase Progress (2026-09-18)

| Step | Status |
| --- | --- |
| Persist holdout-only predictions and export error examples | Complete |
| Verify export against confusion matrix and real dataset | Complete; 32 tests passed; 43 false utility, 33 false marketing |
| Confirm upstream category and VERIFIED status semantics | Complete; platform owner confirmed 2026-09-18 |
| Manually review error examples against business context | Pending |
| Lexical near-duplicate audit and chronological snapshot validation | Complete; semantic paraphrases and decision-time validation remain |
| Compare local latent-semantic and character models against baseline | Complete; neural encoder / LLM comparison remains |

Run `.venv/bin/python -m templatelab.audit` to regenerate the local error report.
The report is `.data/holdout-errors.json`; no template content was added to docs.
This iteration changes auditability, not model accuracy. No frontend changes
were made, so the earlier browser checks were not rerun.

## Development Website (2026-09-18)

- Complete: default Development view with Progress, Data journey, How it works,
  Results and Code map sections.
- Live dataset/model metrics are separate from manually maintained milestones.
- Plain-language explanations cover label provenance, filtering, TF-IDF,
  logistic regression, holdout evaluation, serving refit and limited rewriting.
- Browser checks passed at 1440 x 1000 and 390 x 844 for every new section,
  keyboard tab navigation, refresh and the existing application workflows.
- No JavaScript errors or horizontal page overflow. Desktop/mobile screenshots
  were inspected. Backend behavior was unchanged; backend tests were not rerun
  in this frontend iteration (last checkpoint: 32 passed).
- Local server started at http://127.0.0.1:8765/#development.

## Error Review and Experiments (2026-09-18)

- Complete: full-component error queue, direction/status/search filters,
  snapshot-scoped annotations, version conflict checks and JSON export.
- Complete: candidate draft editor with existing limited rewrite checks,
  factual context fields and manually recorded Meta outcomes. Source labels
  stay immutable. Old outcomes cannot silently follow edited candidates.
- Complete: chronological split, boundary-family purging, lexical overlap
  screening and three local model comparisons, exposed in Experiments.
- Verified: 40 backend tests passed. Browser checks passed on desktop/mobile,
  including save/reload, discard, unsaved navigation protection and existing
  workflows. Test annotations were restored; no human reviews were fabricated.
- Measured: character TF-IDF utility precision 79.1%, word TF-IDF 77.7%,
  LSA 69.9% on 423 chronological test families. Serving model unchanged.
- Remaining: human error analysis, actual decision-time evaluation,
  pretrained neural model comparison, expanded rewriting and Meta validation.
- Detailed methodology and limitations: `docs/experiments.md`.

## Remaining Limits

The first version uses a classical text model and a limited policy checklist.
It does not inspect media or know Meta's internal reasoning. The provided
labels and VERIFIED status have platform-owner-confirmed provenance, without
an independent Meta API check. Evaluation on the export is not future performance.
