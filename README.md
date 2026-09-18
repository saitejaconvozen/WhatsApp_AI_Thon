# Template Lab

A local workspace for auditing WhatsApp templates, evaluating category
prediction, and reviewing utility eligibility.

## Project Context

The platform team has roughly 10,000 templates and wants to understand the
difference between marketing and utility, predict likely categorization, and
draft appropriate utility messages for real billing and service events.

The first shared JSON contains 4,079 records. Read the
[dataset audit](docs/dataset-audit.md) for its schema and preliminary counts.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m uvicorn templatelab.app:app --host 127.0.0.1 --port 8765
```

Open http://127.0.0.1:8765. Keep the app bound to loopback: this local prototype
does not implement authentication or multi-user access control.

## Data

Local data lives in `.data/`, which is excluded from Git. Imported records
retain requested category, recorded Meta category, verification status, and
content separately. Account identifiers and author email fields are not
imported. Message content can still contain personal information.

The initial baseline is text-only and trained on completed, labeled TEXT
records. It does not inspect linked media or submit templates to Meta.
The policy reviewer is an explicit English-language checklist, not an LLM.
It can flag known patterns but cannot guarantee category approval.

## Use the Workspace

- **Development:** the default view, with completed milestones, next steps,
  live dataset counts, data preparation, model walkthrough, explained evaluation
  results and a code map. Milestones/test counts are dated manual checkpoints,
  not live CI results. Refresh reloads dataset and model metrics.
- **Dataset:** import JSON (including the nested draft export), JSONL, CSV,
  TSV or XLSX; map flat columns; search, filter and inspect records; export CSV.
- **Compose:** check or upload a template, see whether it can become utility and
  convert it, or draft a new utility template from a described task.
- **Template review:** open a record or select a utility draft event; supply
  recipient context; inspect separate checklist and trained-model results.
- **Model lab:** train or retrain the baseline and inspect the grouped holdout
  metrics, confusion matrix and evaluation limitations.
- **Error review:** annotate held-out errors, capture business context, assess
  candidate drafts and record manually observed Meta outcomes. Save or discard
  before changing records. JSON export includes full content; keep it local.
- **Experiments:** compare word TF-IDF, character TF-IDF, local LSA, a
  structural-feature arm and an optional sentence encoder on a date-based
  split, with an additional lower-lexical-overlap subset.

Import from a terminal with:

```bash
.venv/bin/python -m templatelab.ingest /absolute/path/to/templates.json
```

For flat files, use the header-only CSV available from the Import dialog.
`meta_category` must contain recorded Meta decisions, while
`requested_category` stores the draft category. `status` must be VERIFIED or
APPROVED for training eligibility. A column named `category` alone is treated
as requested, never silently as Meta's decision. Missing labels remain unknown.

## VS Code

The project includes tasks for starting the app, running tests, and resuming
this exact Codex conversation. Use **Terminal > Run Task > Codex: Resume this
conversation** after the current desktop turn finishes. This uses Codex CLI
inside VS Code's integrated terminal. Native extension history visibility
has not been verified.

Read [the implementation plan](docs/plan.md) and [handoff notes](docs/HANDOFF.md).

## Validation

```bash
.venv/bin/python -m pytest
```

114 tests passed. The browser smoke test covers import idempotency, filtering,
pagination, record details, actual model training, limited rewrites, and
desktop/mobile layout. It requires Node.js 20+, Playwright, and installed
Google Chrome. With the app running:

```bash
node tests/browser_smoke.cjs /absolute/path/to/the/already-imported-export.json
```

The fixture argument is optional; when provided, it must already be imported
because the test verifies that a repeated upload does not add records.
Use `PLAYWRIGHT_MODULE` to specify a Playwright installation if it is not
available through normal Node module resolution.

## Publishing the Results Site

The local results site at `templatelab.results:app` renders full template
content and must stay on loopback. To publish findings, build the separate
metrics-only site instead:

```bash
.venv/bin/python -m templatelab.publish
```

This writes `site/`: a static build whose `results.json` is assembled through
an explicit field allowlist in `templatelab/publish.py`. Dataset counts,
baseline metrics, the confusion matrix and the experiment comparison are
included. Template names, bodies, headers, footers, buttons, human annotations
and near-duplicate pair IDs are never read into the payload; only the error
*counts* are published. `tests/test_publish.py` asserts that a known token
planted in every record body appears in no generated file.

Preview it locally with `python -m http.server` from `site/`. Committing `site/`
and pushing to GitHub deploys it through `.github/workflows/pages.yml`. The
private dataset never reaches CI: the workflow only uploads the committed build.
Rebuild and recommit `site/` after retraining or importing new data.

## Baseline and Limits

Two exports are imported: a draft export of 4,079 records and a production
export of 9,446. Together they yield **3,745 eligible families** after
filtering and excluding conflicting labels, grouped across header, body,
footer and buttons.

**Meta rarely upgrades, but it does.** Almost every category change runs one
way — a template submitted as utility recorded as marketing. The reverse
happens in about 0.6% of cases (31 in the production export), so the live
question is nearly always whether a utility-intent draft gets downgraded.
That makes the requested-UTILITY slice the headline number:

| Slice | Families | Accuracy | Majority | Utility precision |
| --- | ---: | ---: | ---: | ---: |
| **Requested UTILITY** | 549 | **76.0%** | 50.5% | **82.4%** |
| All holdout families | 937 | 83.9% | 70.4% | 77.2% |
| Requested MARKETING | 388 | 95.1% | 98.7% | n/a |

The production export also carries `revisedMetaTemplateCategory` on 192
records — Meta changing a category after its first ruling. 102 went from
utility to marketing and 58 the other way. That field supersedes
`metaTemplateCategory` when present, and it is direct evidence that labels
move over time.

Predictions are calibrated and thresholded. Between the two thresholds the model
returns `NEEDS_REVIEW` instead of guessing: on the holdout that refers 5.8% of
templates and raises utility precision to 87.0% on the rest. Thresholds are
picked from out-of-fold predictions on training data, never on the holdout.

These metrics are provisional: the data is a historical snapshot, grouping can
miss paraphrases, and future-time performance is not measured.
On 2026-09-18 the platform owner confirmed that `metaTemplateCategory` is Meta's
final category and `VERIFIED` means successful Meta verification. This is
owner-confirmed provenance, not an independent check through Meta's API.

The app is a reviewer prototype, not an automatic submission system. A limited
rewrite only removes separate promotional sentences/buttons when sufficient
transaction context is supplied. It cannot perform general semantic rewriting.
An optional LLM reviewer exists but is disabled unless configured; see below.
There is no Meta API connection.

The generated baseline is persisted locally and becomes stale after new data
is imported. Retraining refreshes it. Only the application's own local model
artifact is loaded; do not replace it with untrusted model files.

## Compose

`/#compose` is the authoring surface. Two tabs:

**Check an existing template.** Paste it or upload a JSON/JSONL/CSV/TSV/XLSX file
(the first record is extracted). You get a verdict, the calibrated utility
probability, and — when the template can become utility — an offer to change it,
showing the probability before and after. Accepting rewrites the draft; declining
leaves it alone.

**Generate a new template.** Describe the task and get a utility draft, scored by
the same model and re-checked by the promotional checklist.

Two invariants hold in both directions, ported from the sibling prototype:

- **The transactional anchor is never removed.** Deleting the clause that names
  the order, invoice, booking or case is the one edit that can never help. A model
  reply that drops it is discarded rather than repaired.
- **It refuses rather than launders.** A purely promotional template returns
  `IRREDUCIBLY_MARKETING` and a promotional *task* is refused outright. Meta
  detects disguised promotions, and the penalty — losing utility messaging — costs
  far more than any per-message saving.

Mixed templates return `SPLIT_RECOMMENDED`: the transactional part as UTILITY, the
promotional rider as a separate MARKETING template for an opted-in audience. A
promotional fragment carrying a `{{placeholder}}` is kept and flagged for a human
rather than deleted, because removing it would lose real business data.

Without an LLM backend, conversion uses the checklist and generation falls back to
the five built-in patterns. With one configured, both use the model and the result
is still re-scored and re-checked — the model is never trusted to police itself.

## Decision Layer

`templatelab/economics.py` turns a probability into a decision. Per-message costs
alone make declaring UTILITY weakly dominant for every p < 1, which is exactly the
behaviour Meta polices, so the module prices being wrong two ways: an explicit
per-template break-even, and a portfolio risk budget that spends an agreed
misclassification allowance where it buys the most saving. The second is the one
to ship — it converts an unpriceable account-health risk into a policy choice.

Measured on the real requested-UTILITY holdout, actual misclassification stayed
inside the budget at every level (7.7% at a 10% budget, 6.9% at 15%, 13.5% at 20%).
At 2–5% it declares nothing rather than exceeding the budget. **The 7.5:1 cost
ratio is a placeholder** — replace it with your real per-country rates before
quoting any saving.

## Optional LLM Reviewer

`/api/review` returns three independent readings that are deliberately never
merged: the regex checklist, the trained model, and — when configured — an LLM
judged against `policy/meta-categories.md`, a versioned paraphrase of Meta's
published definitions with a `checked_on` date.

It is **off by default and makes no network call** until two separate opt-ins
are set:

```bash
export TEMPLATELAB_LLM_BACKEND=anthropic
export TEMPLATELAB_LLM_ALLOW_EGRESS=1     # acknowledges template content leaves this machine
export ANTHROPIC_API_KEY=...
```

The second variable exists because enabling this sends real template content,
which may contain personal information, to a third party. Responses are cached
in `.data/llm-cache.sqlite3` keyed by prompt and model, so reruns cost nothing
and stay reproducible. The reviewer never submits anything to Meta, and its
rationale is a model's reading of the published definitions — not Meta's
decision, and not an explanation of one.

## Holdout Error Audit

```bash
.venv/bin/python -m templatelab.audit --retrain
```

This retrains the baseline and writes `.data/holdout-errors.json`, with separate
false-utility and false-marketing lists. Subsequent exports can omit `--retrain`.
Predictions come from the held-out evaluation, never the model refitted on all
records. Older artifacts require one retraining; stale models are rejected.
The supplied dataset produces 32 false-utility and 65 false-marketing examples.
These are disagreements with recorded labels, not verified policy violations.
The report includes template text and must stay local. An already running app
retains its in-memory model until restarted or retrained through Model lab.

## Next-Phase Evaluation

Run comparisons in Experiments or from the terminal:

```bash
.venv/bin/python -m templatelab.experiments
```

The local report is `.data/experiments.json`. No model is automatically promoted.
See [experiment methodology and results](docs/experiments.md).

Annotations live in a separate SQLite table, keyed by evaluation snapshot and
record ID. Retraining creates a new review queue; old annotations remain stored
but are not automatically carried into the new queue or its export. Annotation
edits do not change dataset revision or retrain the classifier. Simultaneous
edits use version checks. A Meta outcome is a human-entered observation for a
specific draft, not a verified API response. Editing that draft clears its
outcome in the browser. Complete reviews require a reason and notes.
