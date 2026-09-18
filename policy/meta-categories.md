# Meta template category definitions

version: 2
checked_on: 2026-09-19
source: https://whatsappbusiness.com/products/conversation-categories/utility/
explainer: https://kanbro.in/site/Whatsapp-Template-Guidelines.pdf

**This file is a working paraphrase, not verbatim policy text.** It records how
this project interprets Meta's published categories so that the interpretation
is versioned and reviewable. It is not Meta's internal classifier, and it does
not explain any individual decision. Re-check the source and bump `version`
when Meta changes its published guidance.

## UTILITY

A non-promotional message specific to or requested by the user, or essential
or critical to the user. Meta's category explainer covers orders, account
alerts, opt-in management, specific feedback, conversation continuation, and
essential safety or regulatory notices. The explainer is Meta-authored but the
linked copy is hosted by a third party; verify it against current Meta guidance
before changing production policy.

- U1. Relates to a specific user-requested transaction, account, service or
  interaction, or to an essential or critical need such as safety or a required
  notice. A placeholder is not mandatory for every qualifying use case.
- U2. The recipient's action, request, subscribed service, or essential need
  makes the message expected. A lead-generation invitation is not an existing
  service interaction merely because it names a product.
- U3. Content is confined to the service facts, status, amounts, dates,
  references, or necessary action. It has no persuasive or promotional intent.
- U4. A link or button serves that specific interaction or essential need, not
  a catalogue, storefront, upgrade or promotion.

## MARKETING

Anything that promotes, re-engages or sells, including a message that is
otherwise transactional but carries promotional content alongside it.

- M1. Offers, discounts, coupons, sales, price drops or loyalty rewards.
- M2. Product announcements, new collections, catalogue or storefront links.
- M3. Upsell or cross-sell, including upgrades layered onto a service message.
- M4. Re-engagement: abandoned carts, "we miss you", win-back, or attempts to
  secure a renewal. An account-status notice about a current subscription may
  differ; exact wording and purpose matter.
- M5. Invitations, referrals or anything asking the recipient to bring others.
- M6. **Mixed content is marketing.** A genuine transactional update that also
  carries a promotional sentence or a promotional button is marketing.

## AUTHENTICATION

One-time passcodes and verification codes, and nothing else in the same
message.

## Interpretation notes

- The eligible family set includes a small number of requested-MARKETING
  templates recorded as UTILITY. Most disagreements run the other way, but
  the categories are not deterministic or perfectly symmetric.
- M6 is the rule most often implicated in this project's false-utility errors.
- These are working interpretations, not Meta's internal decision rules or
  proof that a changed draft will be accepted.
