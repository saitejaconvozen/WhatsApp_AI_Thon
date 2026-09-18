"""Optional LLM reviewer.

Disabled unless explicitly configured. Enabling a hosted backend sends template
content — which may contain personal information — to a third party, so it
takes two separate opt-ins: a backend choice and an egress acknowledgement.
Nothing here touches the network until both are set.
"""

import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data import record_content

ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = ROOT / "policy" / "meta-categories.md"

BACKEND_VARIABLE = "TEMPLATELAB_LLM_BACKEND"
EGRESS_VARIABLE = "TEMPLATELAB_LLM_ALLOW_EGRESS"
MODEL_VARIABLE = "TEMPLATELAB_LLM_MODEL"
KEY_VARIABLE = "TEMPLATELAB_LLM_API_KEY"
BASE_URL_VARIABLE = "TEMPLATELAB_LLM_BASE_URL"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_MODELS = {"anthropic": "claude-sonnet-5", "deepseek": "deepseek-chat",
                  "openai_compatible": "deepseek-chat"}
MAX_TOKENS = 512
REQUEST_TIMEOUT = 60.0
# Pool is searched, then trimmed to PER_CLASS examples of each label.
NEIGHBOR_POOL = 40
PER_CLASS = 3

EGRESS_NOTICE = (
    "Template content, which may contain personal information, is sent to the "
    "configured provider. Set " + EGRESS_VARIABLE + "=1 to acknowledge this."
)


class Unavailable(RuntimeError):
    """The reviewer cannot run; surfaced to the caller as a reason, never raised through the API."""


def policy_text():
    if not POLICY_PATH.exists():
        raise Unavailable(f"Missing {POLICY_PATH.name}. The category definitions are required for review.")
    return POLICY_PATH.read_text(encoding="utf-8")


def policy_version():
    text = policy_text()
    version = re.search(r"^version:\s*(\S+)", text, re.M)
    checked = re.search(r"^checked_on:\s*(\S+)", text, re.M)
    return {"version": version.group(1) if version else "unknown",
            "checked_on": checked.group(1) if checked else "unknown"}


def configuration():
    backend = os.environ.get(BACKEND_VARIABLE, "disabled").strip().lower() or "disabled"
    model = os.environ.get(MODEL_VARIABLE, "").strip() or DEFAULT_MODELS.get(backend, DEFAULT_MODEL)
    return {"backend": backend, "model": model,
            "base_url": os.environ.get(BASE_URL_VARIABLE, "").strip() or None,
            "egress_acknowledged": os.environ.get(EGRESS_VARIABLE, "").strip() == "1"}


def balanced_examples(neighbors, per_class=PER_CLASS):
    """Nearest precedents from each label rather than the top-k overall.

    A block of six MARKETING neighbours argues for MARKETING by weight of numbers.
    Taking the closest example from each side frames the decision as the contrast
    it actually is: near-identical wording that Meta put in opposite categories.
    """
    counts, chosen = {}, []
    for neighbor in sorted(neighbors, key=lambda n: -(n.get("similarity") or 0)):
        label = neighbor.get("category")
        if label not in {"MARKETING", "UTILITY"} or counts.get(label, 0) >= per_class:
            continue
        counts[label] = counts.get(label, 0) + 1
        chosen.append(neighbor)
    # Closest example last: it sits nearest the template under review, and models
    # weight the end of a long prompt more heavily.
    return list(reversed(chosen))


def build_prompt(record, neighbors):
    selected = balanced_examples(neighbors)
    labels = {n.get("category") for n in selected}
    examples = "\n\n".join(
        f"Example {i} — Meta recorded {n['category']} (wording similarity {n.get('similarity', 0)}):\n{n['body']}"
        for i, n in enumerate(selected, 1)) or "No comparable labelled examples were found."
    contrast = ("\nThese examples sit on opposite sides of the boundary. Where two are "
                "worded similarly but recorded differently, the difference between them "
                "is what matters.\n" if len(labels) > 1 else "\n")
    return (
        "You classify WhatsApp Business message templates the way Meta records them.\n\n"
        "Category definitions:\n\n" + policy_text() + "\n\n"
        "Previously recorded templates with similar wording, for calibration only. "
        "They are historical labels, not ground truth about this draft:\n\n" + examples + "\n"
        + contrast +
        "\nTemplate under review:\n"
        f"Header: {record.get('header') or '(empty)'}\n"
        f"Body: {record.get('body') or '(empty)'}\n"
        f"Footer: {record.get('footer') or '(empty)'}\n"
        f"Buttons: {record.get('buttons') or '(none)'}\n\n"
        "Reply with JSON only, no prose around it:\n"
        '{"category": "MARKETING" | "UTILITY" | "AUTHENTICATION", '
        '"confidence": 0.0-1.0, '
        '"clauses": ["the rule ids that decided it, e.g. M6"], '
        '"rationale": "two sentences at most"}'
    )


def parse_response(text):
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise Unavailable("The model did not return JSON.")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise Unavailable(f"The model returned malformed JSON: {exc}") from exc
    category = str(payload.get("category", "")).upper()
    if category not in {"MARKETING", "UTILITY", "AUTHENTICATION"}:
        raise Unavailable(f"The model returned an unknown category: {category!r}")
    confidence = payload.get("confidence")
    return {"category": category,
            "confidence": float(confidence) if isinstance(confidence, (int, float)) else None,
            "clauses": [str(c) for c in payload.get("clauses", [])][:8],
            "rationale": str(payload.get("rationale", ""))[:1000]}


class Cache:
    """Keyed by prompt and model so a rerun costs nothing and stays reproducible."""

    def __init__(self, directory):
        self.path = Path(directory) / "llm-cache.sqlite3"
        with sqlite3.connect(self.path, timeout=30) as db:
            db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, payload TEXT NOT NULL)")

    @staticmethod
    def key(prompt, model):
        return hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()

    def get(self, key):
        with sqlite3.connect(self.path, timeout=30) as db:
            row = db.execute("SELECT payload FROM responses WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key, payload):
        with sqlite3.connect(self.path, timeout=30) as db:
            db.execute("INSERT OR REPLACE INTO responses VALUES (?, ?)", (key, json.dumps(payload)))


def post(url, headers, payload):
    """Plain HTTP rather than a provider SDK: no extra dependency on the core path."""
    import httpx
    try:
        response = httpx.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise Unavailable(f"The provider call failed: {exc}") from exc


def anthropic_call(prompt, model):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise Unavailable("ANTHROPIC_API_KEY is not set.")
    data = post("https://api.anthropic.com/v1/messages",
                {"x-api-key": key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
                {"model": model, "max_tokens": MAX_TOKENS,
                 "messages": [{"role": "user", "content": prompt}]})
    blocks = data.get("content") or []
    return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")


def deepseek_call(prompt, model):
    """Any OpenAI-compatible chat endpoint: DeepSeek direct, or a LiteLLM-style proxy."""
    key = os.environ.get(KEY_VARIABLE) or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise Unavailable(f"{KEY_VARIABLE} is not set.")
    base = (os.environ.get(BASE_URL_VARIABLE) or DEEPSEEK_BASE_URL).rstrip("/")
    url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
    data = post(url, {"Authorization": f"Bearer {key}", "content-type": "application/json"},
                {"model": model, "max_tokens": MAX_TOKENS,
                 "messages": [{"role": "user", "content": prompt}]})
    choices = data.get("choices") or []
    if not choices:
        raise Unavailable("The provider returned no choices.")
    return (choices[0].get("message") or {}).get("content") or ""


BACKENDS = {"anthropic": anthropic_call, "deepseek": deepseek_call,
            # Same wire format; a separate name so the report says which endpoint answered.
            "openai_compatible": deepseek_call}


class Reviewer:
    def __init__(self, store, baseline):
        self.store = store
        self.baseline = baseline
        self.cache = Cache(store.directory)

    def status(self):
        config = configuration()
        if config["backend"] == "disabled":
            return {"available": False, "backend": "disabled",
                    "reason": f"No LLM backend is configured. Set {BACKEND_VARIABLE} to enable one.",
                    "policy": policy_version()}
        if config["backend"] not in BACKENDS:
            return {"available": False, "backend": config["backend"],
                    "reason": f"Unknown backend {config['backend']!r}. Known backends: {', '.join(sorted(BACKENDS))}.",
                    "policy": policy_version()}
        if not config["egress_acknowledged"]:
            return {"available": False, "backend": config["backend"],
                    "reason": EGRESS_NOTICE, "policy": policy_version()}
        return {"available": True, "backend": config["backend"], "model": config["model"],
                "policy": policy_version(), "egress": EGRESS_NOTICE}

    def neighbors(self, record, exclude=None):
        """A wide pool, so balanced_examples can find a precedent on each side.

        Baseline.predict returns only its top few by similarity, which on a
        lopsided corpus is often all one label.

        `exclude` is mandatory when scoring held-out rows. The serving model is
        refitted on every eligible family, so the index contains the holdout
        itself: without it, a holdout template retrieves *itself* at similarity
        1.0 with its recorded label attached, and the judge simply reads the
        answer off the prompt. That produced a fake 100% before it was caught.
        """
        bundle = getattr(self.baseline, "bundle", None)
        if not bundle or "text_branch" not in bundle:
            return self.baseline.predict(record).get("neighbors", [])
        vector = bundle["text_branch"].transform([record])
        if not vector.nnz:
            return []
        similarity = (bundle["matrix"] @ vector.T).toarray().ravel()
        records = bundle["records"]
        blocked = exclude or set()
        pool = []
        for i in np.argsort(similarity)[::-1]:
            if similarity[i] <= 0 or records[i]["id"] in blocked:
                continue
            pool.append({"id": records[i]["id"], "name": records[i]["name"], "body": records[i]["body"],
                         "category": records[i]["meta_category"], "similarity": round(float(similarity[i]), 3)})
            if len(pool) >= NEIGHBOR_POOL:
                break
        return pool

    def review(self, record, exclude=None):
        status = self.status()
        if not status["available"]:
            return {**status, "reviewed_at": None}
        config = configuration()
        prompt = build_prompt(record, self.neighbors(record, exclude=exclude))
        key = Cache.key(prompt, config["model"])
        cached = self.cache.get(key)
        if cached:
            return {**cached, "available": True, "cached": True, "policy": status["policy"]}
        try:
            parsed = parse_response(BACKENDS[config["backend"]](prompt, config["model"]))
        except Unavailable as exc:
            return {"available": False, "backend": config["backend"], "reason": str(exc),
                    "policy": status["policy"]}
        payload = {**parsed, "model": config["model"], "backend": config["backend"],
                   "reviewed_at": datetime.now(timezone.utc).isoformat(),
                   "limitation": "A model's reading of the published definitions. Not Meta's decision or its reasoning."}
        self.cache.put(key, payload)
        return {**payload, "available": True, "cached": False, "policy": status["policy"]}


def content_hash(record):
    return hashlib.sha256(record_content(record).encode()).hexdigest()[:16]
