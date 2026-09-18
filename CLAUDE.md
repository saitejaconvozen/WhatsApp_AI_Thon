# Template Lab

Classifies WhatsApp Business templates as UTILITY or MARKETING the way Meta
records them, rewrites downgraded ones, and drafts new utility templates. Utility
templates cost less to send, so a template Meta downgrades to marketing costs the
business money on every send.

## Bring it up

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt   # ~1.4GB, CPU torch
.venv/bin/python -m templatelab.ingest path/to/templates.json
```

`.data/` is gitignored and holds the corpus and every trained artefact, so a
fresh clone has none of them. Rebuild in this order — `forms` needs the encoder,
`reward` needs the split manifest `forms` does not touch:

```bash
.venv/bin/python -c "from templatelab.data import Store; from templatelab.model import Baseline; \
  print(Baseline(Store('.data')).train())"      # .data/baseline.joblib
.venv/bin/python -m templatelab.forms           # .data/forms.json
.venv/bin/python -m templatelab.reward          # .data/utility-reward.joblib
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q
```

Tests must run with those two env vars set or they reach the network. There are
218 of them and they should all pass before you change anything.

## Serve the API

```bash
export TEMPLATELAB_API_KEYS="partner:$(openssl rand -hex 16)"
export TEMPLATELAB_ALLOWED_ORIGINS="https://app.example.com"   # only for browser callers
.venv/bin/uvicorn templatelab.demo:app --host 127.0.0.1 --port 8767
```

| Endpoint | Does | Latency |
| --- | --- | --- |
| `POST /api/predict` | classify: category, confidence, band | ~50 ms |
| `POST /api/convert` | rewrite ⇄ classify until utility; returns the round trail | 5-30 s |
| `POST /api/generate` | draft a utility template from a described event | 3-20 s |
| `POST /api/explain` | LLM reasoning with policy clause ids | 3-5 s |
| `POST /api/split` | split mixed content into utility + marketing templates | 5-15 s |

Every endpoint takes either flat fields (`body`, `header`, `footer`, `buttons`,
`requested_category`) or `template_json`, **a JSON string** holding a record in
the platform's export shape. `template_json` wins outright when both are sent;
the flat fields are ignored, not merged. Send the real `requested_category`:
with `UTILITY` the request routes to the blended classifier, without it to the
weaker text-only baseline, and the two disagree.

Keys go in `X-API-Key` or `Authorization: Bearer`; limits are per key, 60/min
classify and 20/min model calls. `/api/convert` and `/api/generate` spend a
provider budget on the server's key, so they must be called from a backend — a
key in browser JavaScript is readable by anyone with devtools.

`.env` is gitignored and loaded by `templatelab/__init__.py`; shell exports win
over it. `TEMPLATELAB_LLM_ALLOW_EGRESS=1` is an acknowledgement that template
text leaves the machine for the provider, not a formality.

## What is measured, and what is not

Read `docs/conversion-research.md` before deciding what to build. The useful
findings here are negative, and several were reached after a wrong answer:

- **Conversion has no training data.** The corpus pairs template text with Meta's
  verdict. It contains no example of a template being rewritten, resubmitted and
  then approved — not few, zero. Conversion and generation are therefore scored
  by our own models, never against Meta.
- **The classifier scores 77.8% held-out, about 67% near its decision boundary**,
  which is where every downgraded template sits. It gates every conversion, so a
  conversion rate is partly a measurement of the gate. Its learning curve is
  still climbing at roughly +3pp per doubling: the ceiling is data volume, not a
  missing feature.
- **74% of the templates the loop cannot convert are refused for having no
  transaction to report.** Those are advertisements. Converting one means
  inventing a transaction that does not exist, which is a false statement to Meta.
  Refusal is the correct outcome, not a failure to fix.
- **A score rewards surface form.** Take 120 templates Meta approved and change
  nothing but filling their placeholders: classifier agreement falls from 94% to
  57%. Blocking fabrication in drafts *lowered* their mean score from 0.87 to
  0.70, because invented reference numbers make a template look more like utility.
- `templatelab/align.py` is kept **because it failed**. Stripping calls to action
  moves a template further from approved wording 65% of the time; Meta approves
  "Call Now or choose an option below to proceed".

## Working rules

**Read the outputs before reporting a rate.** Every defect found in this project
was invisible in the summary statistics and obvious in the first six examples: a
form quoting "91% approved" for a form it matched at 0.16, 376 unedited templates
counted among 506 "conversions", a pre-approved credit offer rewritten into "your
loan application has been approved", a mean gain of +0.049 that was −0.11 once
non-improving cases were included. Print real before/after pairs, including the
bad ones, and report a rate only beside that reading.

**Check whether a measurement is circular.** Template families are keyed by a
hash of normalised text, so "families whose members share wording" is a
tautology. A gate built from hand-written vocabulary measures the vocabulary.

**Watch the control arm.** `templatelab.loop` samples 250 Meta-UTILITY templates
alongside the 250 downgraded ones. The approved half needs no conversion, so
every one the classifier pushes into the loop is a false positive, and every
"conversion" there is repair of our own error. It is the floor on how much of the
headline number is the same artefact.

**Never let a rewrite invent a fact.** The gates in `loop.py` and `draft.py` —
placeholders preserved, a transaction named, nothing promotional, a placeholder
budget — exist because each was breached. Do not relax one to raise a number.

## Layout

| Path | |
| --- | --- |
| `templatelab/demo.py` | the public API; allowlists every response |
| `templatelab/loop.py` | classify → rewrite → classify, with hard gates |
| `templatelab/draft.py` | generation from a described event |
| `templatelab/retrieval.py` | precedents from the **undeduplicated** corpus (3,116 approved vs 1,036 for training) |
| `templatelab/forms.py` | 24 approved template shapes, approval rates 17%–100% |
| `templatelab/reward.py` | learned utility score, 78.5% / AUC 0.858 |
| `templatelab/policy.py` | promotional and transaction vocabularies, each widened against measured coverage |
| `templatelab/serving_candidate.py` | the deployed classifier: TF-IDF + encoder blend, threshold 0.49 |

Batch runs write resumable JSONL under `.data/` with a summary beside them:

```bash
.venv/bin/python -m templatelab.loop --utility 250 --marketing 250 --workers 8
.venv/bin/python -m templatelab.regenerate --limit 500 --workers 8
```

## The open question

Nothing here has been submitted to Meta. Every conversion and generation figure
is one model's opinion of another model's output. Around 20 real submissions with
recorded verdicts would create the first ground truth this problem has had, and
would settle whether the conversion rate is worth 16% or 45%. Until then, present
any number as a candidate for human review, never as approval.
