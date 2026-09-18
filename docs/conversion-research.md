# Conversion Research: Submitted Utility, Recorded Marketing

Checked: 2026-09-19. This is an evaluation and implementation plan, not a
claim that Meta has approved any rewritten template.

## Evidence from this corpus

The current eligible text-family set has 3,745 families. Of those, 1,180 were
submitted as UTILITY but recorded by Meta as MARKETING. Family grouping avoids
counting repeated versions as independent observations. The full offline
conversion pass is `.data/conversions-retest-offline.jsonl` (private content) and
`.data/conversions-retest-offline.summary.json` (aggregate only):

| Outcome | Families |
| --- | ---: |
| Needs context or review | 1,011 |
| No transactional fact to preserve under the current guard | 167 |
| Produced a rewrite that the same local model called Utility | 2 |

Prominent reasons were insufficient evidence for the service event (551),
a checklist/local-model disagreement with Meta's recorded Marketing ruling
(326), a missing transactional anchor (166), and promotions tangled with
transactional placeholders (50). These are diagnostic buckets from the current
pipeline, not verified judgments of each template's real business purpose.

A prior hosted-model run on the first 100 of these families produced six
rewrites. The first 100 are not a random sample, and neither six nor two is an
approval rate. The 100-run also returned `ALREADY_UTILITY` 30 times on records
Meta had categorized as Marketing. That contradiction is now fixed in the
conversion logic, and both UI paths now preserve the recorded Meta category.

Some rewritten examples moved a loan-waiver offer into a separate marketing
message while keeping the overdue notice. Others treated a renewal instruction
or generic click wording as promotion. Those calls require human review: a
local score increase is not evidence that the wording is factually faithful or
that Meta will reclassify it. No observed text-changing rewrite pairs with
subsequent Meta category decisions were found in the historical export.

## Published category boundary

Meta's [utility page](https://whatsappbusiness.com/products/conversation-categories/utility/)
includes orders, account alerts, payment reminders, specific feedback and
continuing a conversation the customer began elsewhere. A [Meta-authored
category explainer, effective July 2025](https://kanbro.in/site/Whatsapp-Template-Guidelines.pdf)
(third-party-hosted copy) says Utility must be non-promotional and either
specific to/requested by the user or essential/critical. It explicitly calls
mixed promotional and transactional content Marketing; unclear content also
defaults to Marketing. It lists essential notices that need no order number.
These rules are broader than the project's old placeholder-plus-transaction
regex, but a broader regex alone would also let advertising through.

## Recommended system

1. **Triage before rewriting.** Preserve the original Meta ruling. Extract the
   actual trigger: existing order, payment, account change, requested support,
   opt-in, feedback about a specific interaction, or essential notice. Ask a
   business owner to confirm the recipient relationship, trigger, available
   references, and the meanings of placeholders. If there is no genuine
   Utility event, keep the message Marketing. Do not invent one.
2. **Generate only from verified facts.** For a mixed template, separate the
   promotional objective into its own Marketing template and produce several
   service-only candidates. For a pure campaign with a separate real service
   event, start a *new* Utility draft from that event; do not disguise the
   campaign. Use retrieval of approved examples to guide form, not to copy
   customer facts. Preserve every required amount, date, identifier, and
   placeholder, including information moved to the Marketing half.
3. **Independent gates and a human.** Use exact-value/placeholder checks,
   policy checks, and a separate semantic factuality review. Use the category
   model to rank candidates, but not as proof of Meta acceptance: selecting on
   its own score creates optimization bias. Show the original, service-only
   draft, moved promotion, changed facts, and missing evidence side by side.
4. **Get real counterfactual labels.** Submit a small, consented set of
   human-approved candidates to Meta through the existing platform process.
   Record draft version, original template ID, submission time, requested
   category, first and revised Meta category, status, and any rejection reason.
   This is the first valid measurement of conversion success. Use that feedback
   to train a candidate ranker and measure future-time acceptance, not only
   historical classification accuracy.

## Resources required

- Business owners who can verify event triggers, recipient relationships,
  placeholder meanings, and whether omitted clauses are obligations or offers.
- A consented Meta review path: WABA/template management access or a human
  submission workflow, plus category/status/revision outcomes. No Meta API
  credentials or actual resubmission outcomes are currently available here.
- A privacy-approved LLM endpoint and bounded test budget. Customer template
  content may leave the machine; do not publish raw records or credentials.
- A held-out future cohort and a human factuality sample. Track **Meta Utility
  acceptance rate, factuality, coverage, human-review burden, and account-risk
  incidents** separately from classifier accuracy. The 90% classification goal
  remains unproven and does not imply 90% conversion acceptance.
