"""Inspect a template, offer to convert it toward UTILITY, or draft a new one.

Two invariants carry over from the sibling prototype and must not be relaxed:

  * Never remove the transactional anchor. Deleting the clause that names the
    order, invoice or booking is the one edit that can never make a template
    more Utility.
  * Refuse rather than launder. A purely promotional template comes back as
    IRREDUCIBLY_MARKETING, never as an invented transactional wrapper. Meta
    detects disguised promotions and the penalty is losing utility messaging
    altogether, which costs far more than any per-message saving.

Every produced draft is re-scored by the trained model and both probabilities are
reported. A rewrite loop that stopped when its own judges agreed would be
optimising against its own evaluators, so the loop is capped and the score is
always surfaced rather than used as a silent stopping rule.
"""

import json
import re

from .llm import Unavailable, BACKENDS, configuration, policy_text, policy_version
from .policy import EVENTS, PRESETS, TRANSACTION, assess, promotion_findings

MAX_REWRITE_ATTEMPTS = 2
COMPONENTS = ("header", "body", "footer", "buttons")

ALREADY_UTILITY = "ALREADY_UTILITY"
CONVERTIBLE = "CONVERTIBLE"
SPLIT_RECOMMENDED = "SPLIT_RECOMMENDED"
IRREDUCIBLY_MARKETING = "IRREDUCIBLY_MARKETING"
NEEDS_CONTEXT = "NEEDS_CONTEXT"


def components_of(record):
    return {key: str(record.get(key) or "") for key in COMPONENTS}


def detect_purpose(text):
    for purpose, pattern in EVENTS.items():
        if re.search(pattern, text, re.I):
            return purpose
    return "unknown"


def has_anchor(components):
    """A transactional anchor is what makes a message a service update at all."""
    return bool(TRANSACTION.search(components.get("body", "")))


def score(record, baseline):
    prediction = baseline.predict(record)
    if not prediction.get("available"):
        return {"available": False, "reason": prediction.get("reason")}
    return {"available": True, "category": prediction["category"],
            "utility_probability": prediction.get("utility_probability"),
            "band": prediction.get("band")}


PLACEHOLDER = re.compile(r"\{\{[^}]+\}\}")


def split_promotional(components):
    """Separate promotional fragments from transactional ones, keeping both.

    A promotional fragment is kept only when it carries a {{placeholder}} —
    deleting it would lose real business data, so a human has to make that call.
    Matching a transaction *word* is not enough: "20% off your next order"
    contains "order" while being purely promotional.
    """
    kept, removed, ambiguous = {}, [], []
    for component, text in components.items():
        parts = text.splitlines() if component == "buttons" else re.split(r"(?<=[.!?])\s+|\n+", text)
        keep = []
        for part in parts:
            if not part.strip():
                continue
            if promotion_findings({component: part}):
                if PLACEHOLDER.search(part):
                    ambiguous.append({"component": component, "text": part})
                    keep.append(part)
                else:
                    removed.append({"component": component, "text": part})
            else:
                keep.append(part)
        kept[component] = ("\n" if component == "buttons" else " ").join(keep).strip()
    return kept, removed, ambiguous


def llm_json(prompt):
    config = configuration()
    if config["backend"] == "disabled" or config["backend"] not in BACKENDS:
        raise Unavailable("No LLM backend is configured.")
    if not config["egress_acknowledged"]:
        raise Unavailable("The egress acknowledgement is not set.")
    text = BACKENDS[config["backend"]](prompt, config["model"])
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise Unavailable("The model did not return JSON.")
    try:
        return json.loads(match.group(0)), config
    except json.JSONDecodeError as exc:
        raise Unavailable(f"The model returned malformed JSON: {exc}") from exc


def rewrite_prompt(components, removed):
    stripped = "\n".join(f"- {r['component']}: {r['text']}" for r in removed) or "- (none found by the checklist)"
    return (
        "You rewrite WhatsApp Business templates so they qualify as UTILITY under "
        "Meta's category rules.\n\nCategory definitions:\n\n" + policy_text() + "\n\n"
        "Template to rewrite:\n"
        + "\n".join(f"{k.capitalize()}: {components[k] or '(empty)'}" for k in COMPONENTS) + "\n\n"
        "A checklist flagged these promotional fragments:\n" + stripped + "\n\n"
        "Rules you must follow:\n"
        "1. Keep every transactional fact and every {{placeholder}} that names the order, "
        "invoice, booking, payment or case. Never delete the anchor.\n"
        "2. Remove promotional content rather than rewording it to hide it. Do not invent a "
        "transaction that is not already in the template.\n"
        "3. If the template has no genuine transactional purpose, say so instead of "
        "manufacturing one.\n\n"
        'Reply with JSON only:\n'
        '{"possible": true|false, "header": "", "body": "", "footer": "", "buttons": "", '
        '"removed": ["the promotional parts you took out"], "reason": "one sentence"}'
    )


def convert(record, baseline, purpose=None, relationship_confirmed=False):
    """Decide whether a template can become UTILITY, and produce it if so."""
    components = components_of(record)
    purpose = purpose or detect_purpose(components["body"])
    checklist = assess({**components, "purpose": purpose,
                        "relationship_confirmed": relationship_confirmed,
                        "format": record.get("format", "TEXT")})
    before = score(record, baseline)
    findings = checklist["findings"]

    if not findings:
        return {"verdict": ALREADY_UTILITY, "purpose": purpose, "checklist": checklist,
                "before": before, "after": None, "utility": None, "split_off": None,
                "removed": [], "method": "checklist",
                "reason": "The checklist found no promotional wording to remove. That is not approval — Meta still decides."}

    if not has_anchor(components):
        return {"verdict": IRREDUCIBLY_MARKETING, "purpose": purpose, "checklist": checklist,
                "before": before, "after": None, "utility": None, "split_off": None,
                "removed": [], "method": "checklist",
                "reason": "The template names no order, invoice, booking, payment or case, so there is nothing transactional to keep. Inventing one would be laundering a promotion, which Meta penalises."}

    kept, removed, ambiguous = split_promotional(components)
    method, reason = "checklist", "Promotional fragments were removed. Every transactional fact was kept."
    try:
        payload, config = llm_json(rewrite_prompt(components, removed))
        if payload.get("possible") is False:
            return {"verdict": IRREDUCIBLY_MARKETING, "purpose": purpose, "checklist": checklist,
                    "before": before, "after": None, "utility": None, "split_off": None,
                    "removed": [], "method": f"llm:{config['backend']}",
                    "reason": str(payload.get("reason", ""))[:400] or "The model judged this template purely promotional."}
        candidate = {key: str(payload.get(key) or "") for key in COMPONENTS}
        if candidate["body"].strip() and has_anchor(candidate):
            kept = candidate
            removed = [{"component": "body", "text": str(t)} for t in payload.get("removed", [])][:8]
            ambiguous = []
            method = f"llm:{config['backend']}"
            reason = str(payload.get("reason", ""))[:400] or reason
        # A model reply that dropped the anchor is discarded, not repaired.
    except Unavailable:
        pass

    if not kept["body"].strip() or not has_anchor(kept):
        return {"verdict": IRREDUCIBLY_MARKETING, "purpose": purpose, "checklist": checklist,
                "before": before, "after": None, "utility": None, "split_off": None,
                "removed": [], "method": method,
                "reason": "Removing the promotional content leaves no transactional message behind."}

    after = score({**record, **kept}, baseline)
    promotional = [r["text"] for r in removed if r.get("text")]
    split_off = ({"body": " ".join(promotional)[:1000],
                  "note": "Send this as a separate MARKETING template to an opted-in audience."}
                 if promotional else None)
    return {"verdict": SPLIT_RECOMMENDED if split_off else CONVERTIBLE,
            "purpose": purpose, "checklist": checklist, "before": before, "after": after,
            "utility": kept, "split_off": split_off, "removed": removed,
            "ambiguous": ambiguous, "method": method, "reason": reason,
            "needs_human": bool(ambiguous),
            "limitation": "A candidate rewrite scored by a local model. Meta decides the category, not this tool."}


def generate_prompt(task, purpose):
    return (
        "You write WhatsApp Business message templates that qualify as UTILITY under "
        "Meta's category rules.\n\nCategory definitions:\n\n" + policy_text() + "\n\n"
        f"The business wants a template for this task:\n{task}\n\n"
        f"Detected service event: {purpose}\n\n"
        "Rules you must follow:\n"
        "1. The message must report on a specific transaction the recipient already has — "
        "an order, invoice, payment, booking, appointment, shipment or support case.\n"
        "2. Use {{placeholder_name}} for every business value. Include at least one "
        "placeholder naming the transaction itself.\n"
        "3. No offers, discounts, upsells, catalogue links or re-engagement wording. "
        "Any button must lead to the transaction, not to a storefront.\n"
        "4. If the described task is promotional rather than transactional, set possible to "
        "false. Do not dress a promotion up as a service message.\n\n"
        'Reply with JSON only:\n'
        '{"possible": true|false, "name": "SHORT_TEMPLATE_NAME", "header": "", "body": "", '
        '"footer": "", "buttons": "", "reason": "one sentence"}'
    )


def nearest_preset(task):
    purpose = detect_purpose(task)
    for key, preset in PRESETS.items():
        if preset["purpose"] == purpose:
            return key, preset
    return None, None


def generate(task, baseline, purpose=None):
    """Draft a UTILITY template for a described task, refusing promotional ones."""
    task = str(task or "").strip()
    if not task:
        raise ValueError("Describe the message you need.")
    purpose = purpose or detect_purpose(task)
    promotional = promotion_findings({"body": task})

    if promotional and purpose == "unknown":
        return {"possible": False, "verdict": IRREDUCIBLY_MARKETING, "purpose": purpose,
                "template": None, "method": "checklist", "score": None,
                "findings": promotional,
                "reason": "This task is promotional and names no transaction. A utility template cannot be written for it; send it as MARKETING to an opted-in audience."}

    method, template, reason = "checklist", None, ""
    try:
        payload, config = llm_json(generate_prompt(task, purpose))
        if payload.get("possible") is False:
            return {"possible": False, "verdict": IRREDUCIBLY_MARKETING, "purpose": purpose,
                    "template": None, "method": f"llm:{config['backend']}", "score": None,
                    "findings": promotional,
                    "reason": str(payload.get("reason", ""))[:400] or "The model judged this task promotional."}
        candidate = {key: str(payload.get(key) or "") for key in COMPONENTS}
        if candidate["body"].strip():
            template = {**candidate, "name": str(payload.get("name", ""))[:120]}
            method = f"llm:{config['backend']}"
            reason = str(payload.get("reason", ""))[:400]
    except Unavailable:
        pass

    if template is None:
        key, preset = nearest_preset(task)
        if preset is None:
            return {"possible": False, "verdict": NEEDS_CONTEXT, "purpose": purpose,
                    "template": None, "method": "checklist", "score": None, "findings": promotional,
                    "reason": "No configured model and no preset matches this event. Name the transaction — invoice, order, appointment, support case or account — or enable an LLM backend."}
        template = {"name": preset["name"], "header": "", "body": preset["body"],
                    "footer": "", "buttons": preset["buttons"]}
        purpose, reason = preset["purpose"], f"Filled from the built-in {key} pattern. Replace every placeholder with real business data."

    drafted = generate_findings_guard(template)
    return {"possible": True, "verdict": CONVERTIBLE if drafted else ALREADY_UTILITY,
            "purpose": purpose, "template": template, "method": method,
            "score": score(template, baseline), "findings": drafted,
            "policy": policy_version(), "reason": reason,
            "limitation": "A drafted candidate, not an approved template. Meta decides the category after submission."}


def generate_findings_guard(template):
    """A draft that trips the promotional checklist is reported, never silently returned clean."""
    return promotion_findings({key: template.get(key, "") for key in COMPONENTS})
