"""Public inference-only demo. No dataset, import, provider or training routes."""

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .compose import IRREDUCIBLY_MARKETING, as_template_json, convert, generate
from .data import Store, normalize_rows, suggest_mapping
from .llm import Reviewer, clause_definitions
from .serving_candidate import Candidate

ROOT = Path(__file__).resolve().parent.parent


PREDICT_LIMIT = 60
# Conversion and drafting call a hosted model, so they get a tighter budget.
LLM_LIMIT = 20


class Template(BaseModel):
    template_json: str = Field(default="", max_length=20000)
    requested_category: Literal["UNKNOWN", "UTILITY", "MARKETING"] = "UNKNOWN"
    header: str = Field(default="", max_length=1000)
    body: str = Field(default="", max_length=6000)
    footer: str = Field(default="", max_length=1000)
    buttons: str = Field(default="", max_length=1000)
    relationship_confirmed: bool = False

    def record(self):
        """Pasted JSON wins over the form fields when both are supplied."""
        values = self.model_dump()
        if self.template_json.strip():
            values = {**values, **from_json(self.template_json)}
        values.pop("template_json", None)
        return {**values, "format": values.get("format", "TEXT")}


def from_json(text):
    """Accept a raw template record in the shape of the platform's own export.

    Reuses normalize_rows, the same parser the importer uses, so the nested
    messageBody form is understood without a second implementation.
    """
    try:
        row = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(422, f"That is not valid JSON: {exc}")
    if isinstance(row, list):
        if len(row) != 1:
            raise HTTPException(422, "Paste a single template object, not a list.")
        row = row[0]
    if not isinstance(row, dict):
        raise HTTPException(422, "Paste a template object.")
    try:
        record = normalize_rows([row], suggest_mapping([row]), "pasted")[0]
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    if not record["body"].strip():
        raise HTTPException(422, "No message body was found in that record.")
    return {key: record.get(key, "") for key in ("header", "body", "footer", "buttons")} | {
        "requested_category": record.get("requested_category", "UNKNOWN"),
        "name": record.get("name", ""), "format": record.get("format", "TEXT")}


class Task(BaseModel):
    task: str = Field(min_length=1, max_length=2000)
    context: str = Field(default="", max_length=4000)
    purpose: str = Field(default="", max_length=40)


def create_demo(data_dir=None, predictor=None):
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*.trycloudflare.com", "localhost", "127.0.0.1", "testserver"])
    model = predictor or Candidate(Store(data_dir or ROOT / ".data"))
    reviewer = Reviewer(Store(data_dir or ROOT / ".data"), getattr(model, "baseline", model))
    recent, llm_recent, lock = deque(), deque(), threading.Lock()

    def throttle(window, limit, what):
        with lock:
            now = time.monotonic()
            while window and now - window[0] >= 60:
                window.popleft()
            if len(window) >= limit:
                raise HTTPException(429, f"Demo {what} limit reached. Try again in one minute.")
            window.append(now)

    @app.middleware("http")
    async def headers(request: Request, call_next):
        # Require bounded, non-chunked request bodies before JSON parsing.
        if request.method == "POST":
            length = request.headers.get("content-length", "")
            if not length.isdigit() or int(length) > 40000:
                return JSONResponse({"detail": "Request exceeds the demo limit."}, status_code=413)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self'; img-src 'self' data:; frame-ancestors 'none'"
        return response

    @app.get("/")
    def home():
        return FileResponse(ROOT / "static" / "demo" / "index.html")

    @app.post("/api/predict")
    def predict(payload: Template):
        if not payload.body.strip() and not payload.template_json.strip():
            raise HTTPException(422, "Enter a message body or paste a template JSON.")
        throttle(recent, PREDICT_LIMIT, "request")
        result = model.predict(payload.record())
        if not result.get("available"):
            raise HTTPException(503, "The local classifier is not ready. Please try again later.")
        # Never return neighbors or explanations containing stored customer text.
        return {"category": result["category"], "utility_probability": result["utility_probability"],
                "band": result.get("band"), "model": result.get("model", "Research classifier"),
                "score_label": result.get("score_label", "Utility probability"),
                "notice": "Research prediction, not Meta approval. 90% accuracy is not established."}

    @app.post("/api/explain")
    def explain_template(payload: Template):
        """Why this category — the one job the LLM does better than anything local.

        Deliberately separate from /api/predict: the local verdict returns in
        milliseconds and must not wait on a call that takes seconds.
        """
        if not payload.body.strip() and not payload.template_json.strip():
            raise HTTPException(422, "Enter a message body or paste a template JSON.")
        throttle(llm_recent, LLM_LIMIT, "explanation")
        record = payload.record()
        verdict = reviewer.review(record)
        if not verdict.get("available"):
            return {"available": False, "reason": verdict.get("reason"),
                    "notice": "Explanations need a configured AI provider."}
        local = model.predict(record)
        definitions = clause_definitions()
        # Allowlisted: rationale and clause ids only. The retrieved precedents
        # that shaped this answer are stored templates and never leave the server.
        return {"available": True, "category": verdict["category"],
                "rationale": verdict.get("rationale"),
                "clauses": [{"id": c, "text": definitions.get(c)} for c in verdict.get("clauses", [])],
                "local_category": local.get("category") if local.get("available") else None,
                "agrees": (local.get("category") == verdict["category"]) if local.get("available") else None,
                "model": verdict.get("model"),
                "notice": "A model's reading of the published definitions, and sometimes of similar "
                          "past templates. Not Meta's decision or its reasoning."}

    @app.post("/api/convert")
    def convert_template(payload: Template):
        if not payload.body.strip() and not payload.template_json.strip():
            raise HTTPException(422, "Enter a message body or paste a template JSON.")
        throttle(llm_recent, LLM_LIMIT, "conversion")
        record = payload.record()
        result = convert(record, model, relationship_confirmed=payload.relationship_confirmed)
        # Allowlisted: everything returned is derived from the submitted text.
        # Stored neighbours and annotations are never in this payload.
        return {"verdict": result["verdict"], "reason": result.get("reason"),
                "purpose": result.get("purpose"), "method": result.get("method"),
                "before": result.get("before"), "after": result.get("after"),
                "utility": result.get("utility"), "split_off": result.get("split_off"),
                "removed": result.get("removed", []), "ambiguous": result.get("ambiguous", []),
                "needs_human": result.get("needs_human", False),
                "disputed_by_model": result.get("disputed_by_model", False),
                "selection": result.get("selection"),
                "findings": [{"code": f["code"], "component": f["component"], "message": f["message"]}
                             for f in (result.get("checklist") or {}).get("findings", [])],
                # Without this a NEEDS_CONTEXT verdict says something is missing
                # but never which thing, leaving no way to act on it.
                "missing_context": (result.get("checklist") or {}).get("missing_context", []),
                "utility_json": as_template_json(result["utility"], record.get("name", ""))
                                if result.get("utility") else None,
                "split_off_json": as_template_json({"body": result["split_off"]["body"]},
                                                   f"{record.get('name','')}_PROMO".lstrip("_"), "MARKETING")
                                  if result.get("split_off") else None,
                "notice": "Candidate edit for human review. Meta decides the category."}

    drafter = {}

    def get_drafter():
        """Built once, on first use. Loading the encoder costs seconds, and a
        deployment without the fine-tuned encoder or forms.json must still be
        able to draft -- it just falls back to the checklist path."""
        if "value" not in drafter:
            try:
                from .draft import Drafter
                drafter["value"] = Drafter(Store(data_dir) if data_dir else Store(ROOT / ".data"))
            except Exception:
                drafter["value"] = None
        return drafter["value"]

    @app.post("/api/generate")
    def generate_template(payload: Task):
        throttle(llm_recent, LLM_LIMIT, "drafting")
        task = payload.task + (("\n" + payload.context) if payload.context.strip() else "")
        maker = get_drafter()
        if maker is not None:
            try:
                drafted = maker.draft(task)
            except Exception:
                drafted = None
            if drafted is not None:
                template = ({"header": "", "body": drafted["body"],
                             "footer": "", "buttons": drafted.get("buttons", ""),
                             "name": drafted.get("name", "")} if drafted.get("possible") else None)
                return {"possible": drafted.get("possible"),
                        "verdict": "DRAFTED" if drafted.get("possible") else IRREDUCIBLY_MARKETING,
                        "purpose": payload.purpose or None, "method": "forms+reward",
                        "template": template, "score": drafted.get("score"),
                        "findings": [], "reason": drafted.get("reason"),
                        # Which shape Meta approves this often, stated only when the
                        # draft actually belongs to that shape.
                        "form": drafted.get("form"),
                        "form_matched": drafted.get("form_matched"),
                        "form_approval_rate": drafted.get("form_approval_rate"),
                        "evidence": drafted.get("evidence"),
                        "template_json": as_template_json(template, template.get("name", ""))
                                         if template else None,
                        "notice": "Drafted candidate, not an approved template. Meta decides the category."}
        try:
            result = generate(payload.task, model, purpose=payload.purpose or None,
                              context=payload.context)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        return {"possible": result.get("possible"), "verdict": result.get("verdict"),
                "purpose": result.get("purpose"), "method": result.get("method"),
                "template": result.get("template"), "score": result.get("score"),
                "findings": [{"code": f["code"], "component": f["component"], "message": f["message"]}
                             for f in (result.get("findings") or [])],
                "reason": result.get("reason"),
                "template_json": as_template_json(result["template"], (result["template"] or {}).get("name", ""))
                                 if result.get("template") else None,
                "notice": "Drafted candidate, not an approved template. Meta decides the category."}

    @app.get("/assets/lucide.min.js")
    def icons():
        return FileResponse(ROOT / "static" / "vendor" / "lucide.min.js", media_type="text/javascript")

    app.mount("/assets", StaticFiles(directory=ROOT / "static" / "demo"), name="assets")
    app.mount("/results", StaticFiles(directory=ROOT / "site", html=True), name="results")
    return app


app = create_demo()
