# Continue in VS Code

Project: `/home/saiteja/Documents/ChatGPT/WhatsApp_AIThon`

Saved Codex conversation: `01a0b369-34ab-7850-b4fc-b648a433b77b`

In VS Code, run **Terminal > Run Task > Codex: Resume this conversation**.
This runs the locally verified `codex resume` command in the integrated
terminal. Start it after the desktop turn finishes so there is one active
writer. This is the CLI route, not a claim that the native extension's history
has been transferred or opened.

Equivalent command:

```bash
/usr/lib/chatgpt/resources/codex resume 01a0b369-34ab-7850-b4fc-b648a433b77b --cd /home/saiteja/Documents/ChatGPT/WhatsApp_AIThon --model gpt-6-astra
```

The shell currently resolves plain `codex` to `/snap/bin/codex`; the user saw
v0.114.0 and an older model list there. The explicit app executable above was
verified as v0.155.0-alpha.9. The VS Code task bypasses that PATH mismatch and
explicitly requests Astra. No global PATH or account configuration was changed.

## Current State

The first local implementation is complete and the app is running at
http://127.0.0.1:8765. VS Code was opened on this project with `docs/plan.md`.
No commit has been made. All project files were created during this session.

- Python FastAPI backend; SQLite local storage; vanilla JavaScript/CSS UI.
- JSON/JSONL/CSV/TSV/XLSX import, nested export adapter, flat column mapping.
- Dataset filters, search, pagination, record inspection and CSV export.
- TF-IDF/logistic regression with grouped holdout evaluation.
- Separate English policy checklist, factual event draft patterns, and limited
  promotional sentence removal. No LLM or Meta API is connected.
- 28 tests passed; browser workflow passed with desktop/mobile screenshots.

## Dataset

The user initially described around 10,000 templates, then supplied a JSON
export containing 4,079 records at:

`/home/saiteja/.codex/attachments/58484601-2eea-417e-8c33-f3848477ac33/Pasted text.txt`

It is already imported into `.data/templates.sqlite3`. The raw attachment is
not tracked or duplicated in the repository. Imported content and model
artifacts are ignored by Git and stay local.

Keep these fields distinct:

- `messageBody.templateCategory`: requested category.
- `metaTemplateCategory`: Meta's final category after verification.
- `verificationStatus`: VERIFIED means successful verification at Meta.

Both meanings were confirmed by the platform owner on 2026-09-18. No independent
Meta API check was performed. APPROVED remains a generic importer alias, not
a confirmed status in this export. Historical model artifacts may retain the
older provenance caveat until retrained; the clarification changes no labels,
eligibility rules or evaluation metrics.

There are 2,552 recorded marketing, 1,157 utility, 19 authentication, and 351
missing category values. 1,333 requested utility records have recorded
marketing labels. This does not prove historical reclassification.

After eligibility checks, normalized-body deduplication and exclusion of 86
conflicting families, 1,691 independent normalized families remain. The
first holdout has 423 families. Utility precision is 73.6%, recall 78.4%,
accuracy 82.0%, macro F1 80.8%, versus majority accuracy 63.8%. These results
do not justify automatically submitting utility recommendations.

## Files and Commands

- `docs/plan.md`: completed steps, checks and next phase.
- `docs/dataset-audit.md`: exact initial data counts and interpretation.
- `templatelab/data.py`: parsing, normalization, eligibility and storage.
- `templatelab/model.py`: baseline, evaluation, persistence and similarity.
- `templatelab/policy.py`: limited policy checks and draft patterns.
- `templatelab/app.py`: local API and web server.
- `static/`: frontend and locally bundled Lucide icons.
- `tests/`: data, policy, API/model tests and browser smoke test.
- `.data/model-report.json`: exact locally generated evaluation metrics.
- `artifacts/`: ignored browser screenshots at desktop/mobile sizes.

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m uvicorn templatelab.app:app --host 127.0.0.1 --port 8765
```

The server was already running at handoff. Start it only if it has stopped;
if another application owns that port, choose a different port. In this Codex
sandbox, API tests and Chrome automation required escalated execution due to
socket restrictions. Ordinary local VS Code execution does not use that sandbox.

## Next Useful Work

Latest continuation: Error Review (`/#errors`) and Experiments (`/#experiments`)
are implemented. `templatelab/error_review.py` stores annotations in a separate
SQLite table, scoped to holdout snapshot; optimistic versions reject stale
edits. `static/errors.js` provides filters, full components, context, candidate
drafts, limited-edit checks, manual Meta outcomes and export. Original labels
are unchanged. Old snapshots remain stored but only current annotations are
shown/exported. Browser test notes were restored after persistence checks.

`templatelab/experiments.py` compares word/character TF-IDF and LSA locally.
Actual run: cutoff 2026-08-18; 1,242 train, 423 test, 26 boundary families purged,
20 test families flagged for lexical overlap. Utility precision: word 77.7%,
character 79.1%, LSA 69.9%. It does not promote a model. Read
`docs/experiments.md` for date-proxy and paraphrase limitations. No external
provider or pretrained neural encoder is configured. Human review, expanded
rewriting and actual Meta validation remain pending, not completed by code.

40 backend tests and expanded desktop/mobile browser checks passed. The server
was restarted on port 8765 for the new endpoints. Development milestones were
updated. `.data/experiments.json` is local and ignored. To rerun from the CLI:
`.venv/bin/python -m templatelab.experiments`.

The default website view is now `/#development`: five sections cover progress,
data preparation, the model/checklist distinction, evaluation and code ownership.
`static/development.js` holds this view; live metrics come from existing summary
and model endpoints. Milestone/test evidence is explicitly a manual checkpoint.
Existing dataset, review and model routes remain available. Browser smoke tests
now cover all five development tabs on desktop/mobile plus keyboard navigation.

Continuation update (2026-09-18): holdout predictions now persist with the model.
`templatelab/audit.py` exports `.data/holdout-errors.json` using only held-out
predictions. Real-data retraining reproduced 43 false-utility and 33
false-marketing examples on 423 families. All 32 tests pass. Older artifacts
need retraining once; stale artifacts cannot be audited. The CLI retraining
does not refresh an already running app's in-memory model. No UI changes
were made in this iteration. Error examples still need human review.

With upstream label/status semantics confirmed by the user, inspect the false
utility predictions, add stronger semantic duplicate checks and a time-based
evaluation, then compare embeddings or an LLM reviewer against this baseline.
LLM integration and general rewriting need a separate implementation and an
explicitly configured service. Do not present the existing checklist as AI
reasoning or Meta's internal explanation.
