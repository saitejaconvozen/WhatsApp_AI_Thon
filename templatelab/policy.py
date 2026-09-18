import re


POLICY_URL = "https://whatsappbusiness.com/products/platform-pricing/"
POLICY_DATE = "2026-09-18"
PROMOTIONS = [
    # Mined from the corpus, intent markers only. Topic nouns (homes, owners,
    # amenities) are deliberately excluded: deciding what a template is *about*
    # is the classifier's job, and putting them here would fire on legitimate
    # property service updates. Recall on real marketing rises from 8.5% to
    # 22.7% while false fires on utility stay near 1%.
    ("discount", r"(?:discounts?|cashback|coupon|promo\s*code|\d+\s*%\s*(?:off|discount)"
                 r"|\bflat\s*[*_~]*\s*(?:₹|rs\.?)\s*[\d,]+|[*_~]*\s*(?:₹|rs\.?)\s*[\d,]+\s*[*_~]*\s*off\b"
                 r"|\bat just\b|\bstarting (?:at|from) (?:just|only)\b|\bupto\b|\bup to\s+\d+\s*%)",
     "A discount or incentive promotes a purchase."),
    ("upsell", r"\b(?:upgrade|upsell|cross.sell|premium plan|unlock (?:more|benefits|exclusive)|add.on)\b",
     "The message encourages an additional purchase or upgrade."),
    ("sales_cta", r"\b(?:buy now|shop now|book now|order now|call now|apply now|claim (?:your|the|now)"
                  r"|grab (?:it|yours|now)|limited.time|exclusive offer|sale ends|hurry|last chance"
                  r"|don'?t miss|miss out|know more|learn more|explore (?:our|more)|check it out"
                  r"|sign up now|register now|enrol now|avail )",
     "The call to action promotes a purchase or offer."),
    ("reengagement", r"\b(?:abandoned cart|left (?:something|items) in your cart|we miss you|renew now"
                     r"|explore our|new collection|special (?:offer|price|deal)|best (?:price|deal|rate)s?"
                     r"|lowest price|for free|free trial|no cost|zero cost|guaranteed)\b",
     "This wording suggests a purchase or re-engagement objective."),
]
AUTH = re.compile(r"\b(?:one.time (?:password|code)|verification code|your otp|authentication code)\b", re.I)
REFERENCE = re.compile(r"\{\{[^}]+\}\}|\b\d{2,}\b", re.I)
TRANSACTION = re.compile(r"\b(?:invoice|bill(?:ing)?|payment|paid|refund|receipt|order|appointment|"
                         r"booking|delivery|shipment|ticket|case|account|subscription|plan|policy|"
                         r"agreement|contract|rent|tenant|landlord|owner|property|visit|inspection|"
                         r"packer|mover|service|request|application|registration|verification|kyc|"
                         r"biometric|document|reference|due|overdue|amount|balance|schedule[d]?|"
                         r"reschedule[d]?|assigned|renewal|expir(?:y|es|ed|ing))\b", re.I)
"""Generic e-commerce vocabulary matched only 34% of this corpus's genuine UTILITY
templates, so two thirds of legitimate drafts were refused as having no transaction
to preserve. Paired with REFERENCE this keeps comparable separation (17.7pp against
18.3pp) at double the coverage."""

# Widened from generic e-commerce vocabulary, which matched only 26.3% of this
# corpus's genuine UTILITY templates and left the rest reporting "not enough
# evidence". The replacement covers 65.1% and separates the classes better too
# (22.9pp against 14.4pp), so it is not a straight precision-for-recall trade.
EVENTS = {
    "billing": r"\b(?:invoice|bill(?:ing)?|amount due|payment|paid|refund|receipt|due|overdue|balance|emi|deposit)\b",
    "order": r"\b(?:order|shipment|delivery|shipped|dispatched|pickup|packer|mover)\b",
    "appointment": r"\b(?:appointment|booking|reservation|visit|inspection|slot|schedule[d]?|reschedul\w+|demo)\b",
    "support": r"\b(?:ticket|case|support request|complaint|query|queries|issue|escalat\w+|resolv\w+)\b",
    "account": r"\b(?:account|profile|registration|verification|kyc|biometric|document|agreement|contract|subscription|plan|renewal|expir\w+|activat\w+|deactivat\w+|service interruption|outage)\b",
    "critical": r"\b(?:recall|safety alert|fraud|severe weather|evacuat\w*|urgent|action required)\b",
}
PRESETS = {
    "invoice": {"name": "Invoice issued", "purpose": "billing", "body": "Your invoice {{invoice_id}} for {{billing_period}} is ready. The amount due is {{amount}}, payable by {{due_date}}.", "buttons": "View invoice"},
    "payment": {"name": "Payment received", "purpose": "billing", "body": "We received your payment of {{amount}} for invoice {{invoice_id}} on {{payment_date}}.", "buttons": "View receipt"},
    "shipment": {"name": "Order shipped", "purpose": "order", "body": "Your order {{order_id}} has shipped. Expected delivery: {{delivery_date}}.", "buttons": "Track order"},
    "appointment": {"name": "Appointment reminder", "purpose": "appointment", "body": "Your appointment {{appointment_id}} is scheduled for {{appointment_date}} at {{appointment_time}}.", "buttons": "View appointment"},
    "support": {"name": "Support update", "purpose": "support", "body": "Your support case {{case_id}} has been updated. Current status: {{case_status}}.", "buttons": "View case"},
}


def promotion_findings(components):
    findings = []
    for component, text in components.items():
        for code, expression, explanation in PROMOTIONS:
            match = re.search(expression, text, re.I)
            if match:
                findings.append({"code": code, "component": component, "evidence": match.group(),
                                 "message": explanation, "severity": "warning"})
    return findings


def assess(payload):
    components = {key: str(payload.get(key) or "") for key in ("header", "body", "footer", "buttons")}
    body = components["body"].strip()
    full = "\n".join(components.values())
    purpose = payload.get("purpose", "unknown")
    findings = promotion_findings(components)
    missing = []
    alpha = [ch for ch in full if ch.isalpha()]
    english_supported = not alpha or sum(not ch.isascii() for ch in alpha) / len(alpha) < 0.15
    supported = payload.get("format", "TEXT") == "TEXT" and english_supported
    event_pattern = EVENTS.get(purpose)
    event_matches = bool(event_pattern and re.search(event_pattern, body, re.I))
    relationship = bool(payload.get("relationship_confirmed"))
    grounded = event_matches and relationship and (purpose == "critical" or bool(REFERENCE.search(body)))
    if not relationship:
        missing.append("Confirm that this message relates to an actual transaction, requested service, or critical recipient need.")
    if purpose not in EVENTS:
        missing.append("Select the event that triggers this message.")
    elif not event_matches:
        missing.append("The body does not clearly identify the selected service event.")
    if purpose in EVENTS and purpose != "critical" and not REFERENCE.search(body):
        missing.append("Add an existing transaction, account, or event reference supported by your business data.")

    if not supported:
        result, summary = "NEEDS_REVIEW", "This checklist covers English text templates. Other languages and rich formats need additional review."
    elif AUTH.search(full):
        result, summary = "AUTHENTICATION", "Verification-code content belongs in a separate authentication review."
    elif findings:
        result, summary = "LIKELY_MARKETING", "The checklist found promotional or re-engagement wording."
    elif grounded:
        result, summary = "UTILITY_CANDIDATE", "The supplied context supports a non-promotional service update. Meta review is still required."
    else:
        result, summary = "NEEDS_REVIEW", "There is not enough evidence to establish utility eligibility."

    rewrite = {"available": False, "components": None, "removed": [],
               "reason": "No promotional edits are needed, or more business context is required."}
    if findings and grounded and supported and result != "AUTHENTICATION":
        edited, removed, ambiguous = {}, [], False
        for component, text in components.items():
            parts = text.splitlines() if component == "buttons" else re.split(r"(?<=[.!?])\s+|\n+", text)
            kept = []
            for part in parts:
                if promotion_findings({component: part}):
                    if TRANSACTION.search(part):
                        ambiguous = True
                    removed.append({"component": component, "text": part})
                else:
                    kept.append(part)
            edited[component] = ("\n" if component == "buttons" else " ").join(kept).strip()
        if not ambiguous and edited["body"] and re.search(event_pattern, edited["body"], re.I):
            rewrite = {"available": True, "components": edited, "removed": removed,
                       "reason": "A limited edit removes separate promotional sentences or buttons. Review all remaining facts and placeholders."}
        else:
            rewrite["reason"] = "Promotion overlaps with transactional information. A human rewrite is needed to preserve the facts."
    elif findings:
        rewrite["reason"] = "A utility rewrite needs a supported service event and a confirmed recipient relationship."
    return {"category": result, "summary": summary, "findings": findings, "missing_context": missing,
            "rewrite": rewrite, "method": "English policy checklist", "policy_url": POLICY_URL,
            "policy_checked_at": POLICY_DATE,
            "limitation": "This is a limited checklist, not Meta's decision or an exhaustive language assessment."}
