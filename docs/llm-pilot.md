# Meta Prediction and Utility Authoring

Checkpoint: 2026-09-18.

## Objective and Implemented Behavior

1. Predict Marketing or Utility from historical Meta decisions, using the
   trained text classifier. Its prediction remains visible even when context
   is insufficient for a rewrite. A review band is separate from the label.
2. Offer a service-only edit for a genuine transaction, with human confirmation
   of the recipient relationship. Preserve placeholders and literal transaction
   facts. Ambiguous mixed clauses require human review. Pure promotions and
   authentication messages are not relabeled as Utility.
3. Generate a candidate from intent and business context. The opt-in LLM uses
   both inputs; without it, supported events use explicitly labeled presets.
   Generated text is checked for promotions, an anchor and placeholders, then
   scored by the historical classifier. These checks do not prove every fact
   is preserved or that Meta will approve it. Human review remains required.

The learned classifier, heuristic checklist and optional LLM are separate
signals. Checklist outcomes are not predictions of Meta's internal reasoning.
Conversion is deliberately limited to extractive edits until richer factual
preservation has been validated. The embedding direction is not a decoder.

## Measured Pilot

Provider model: `deepseek-v4-flash-fireworks-api`, through the supplied HTTPS
proxy. One approved run of 100 requested-UTILITY holdout families, selected
reproducibly with recorded-label stratification: 50 Utility, 50 Marketing.
Retrieval vocabulary and examples use training records only, excluding all
holdout IDs and families. No automatic retries; at most four concurrent calls.

| Metric | Historical classifier, same 100 | DeepSeek with retrieved examples |
| --- | ---: | ---: |
| End-to-end accuracy | 69.0% | 73.0% |
| Utility precision | 80.6% | 86.1% |
| Utility recall | 50.0% | 62.0% |

DeepSeek returned all 100 responses; 10 were NEEDS_REVIEW, counted as incorrect
in end-to-end accuracy. There were zero provider failures. The approximate
95% Wilson interval for accuracy is 63.6%-80.7%. This sample is too small to
establish a reliable four-point improvement, and 90% was not reached.

This is an already-inspected historical holdout, not a fresh production test.
Do not equate Utility precision with overall accuracy or omit abstentions.
The earlier 76.0% baseline measurement used all 549 requested-Utility holdout
families, so compare the pilot to 69.0% on its same 100, not directly to 76.0%.
No serving classifier was replaced and no templates were submitted to Meta.

Private artifacts: `.data/deepseek-benchmark-0961aae246e2.json` and
`.data/deepseek-benchmark.json`. The dashboard omits row-level predictions and
sample IDs; the static publisher does not export the private pilot report.

## Try the App

- Authoring: http://127.0.0.1:8767/#compose
- Pilot: http://127.0.0.1:8767/#benchmark
- Separate results: http://127.0.0.1:8766/#deepseek

The running servers have hosted AI disabled. Classification and conservative
edits work locally; generation uses presets. To start another local app with
the supplied provider settings and a private key prompt:

```bash
.venv/bin/python -m templatelab.serve --llm --allow-egress --port 8768
```

Enter a rotated key at the hidden prompt. The key stays in process memory,
not source files or shell history. Hosted review, rewriting and drafting send
content to the configured provider and may incur charges. Only review prompts
apply basic contact/URL redaction; it is not guaranteed anonymization, and
authoring prompts can contain supplied facts. Use approved customer data only.
The app has no multi-user authentication; keep it bound to localhost.

## Next Experiments

1. Review model disagreements and missing-context cases with the platform owner.
   Preserve original Meta labels; record explanations and revisions separately.
2. Reserve a fresh, family-separated, later-time test set. Tune prompts, adapted
   encoders or hybrid classifiers on training/validation data, not this pilot.
3. Compare accuracy, Utility precision/recall, review coverage and subgroup
   performance. Require reliable evidence before replacing the serving model.
4. Collect actual outcomes of human-approved drafts and rewrites. Synthetic
   examples or self-reported LLM confidence cannot replace Meta outcomes.

90% remains an experimental target, not an implemented guarantee. Software
checks: 142 tests passed with two dependency warnings; Playwright verified
Compose and pilot results at 1440px and 390px without JavaScript errors or
horizontal overflow. Screenshots are in ignored `artifacts/`.
