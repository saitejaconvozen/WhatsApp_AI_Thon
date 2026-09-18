"""Public inference-only demo. No dataset, import, provider or training routes."""

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

from .compose import convert, generate
from .data import Store
from .llm import Reviewer, clause_definitions
from .serving_candidate import Candidate

ROOT = Path(__file__).resolve().parent.parent


PREDICT_LIMIT = 60
# Conversion and drafting call a hosted model, so they get a tighter budget.
LLM_LIMIT = 20


class Template(BaseModel):
    requested_category: Literal["UNKNOWN", "UTILITY", "MARKETING"] = "UNKNOWN"
    header: str = Field(default="", max_length=1000)
    body: str = Field(min_length=1, max_length=6000)
    footer: str = Field(default="", max_length=1000)
    buttons: str = Field(default="", max_length=1000)
    relationship_confirmed: bool = False


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
        if not payload.body.strip():
            raise HTTPException(422, "Enter a message body.")
        throttle(recent, PREDICT_LIMIT, "request")
        result = model.predict({**payload.model_dump(), "format": "TEXT"})
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
        if not payload.body.strip():
            raise HTTPException(422, "Enter a message body.")
        throttle(llm_recent, LLM_LIMIT, "explanation")
        record = {**payload.model_dump(), "format": "TEXT"}
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
        if not payload.body.strip():
            raise HTTPException(422, "Enter a message body.")
        throttle(llm_recent, LLM_LIMIT, "conversion")
        result = convert({**payload.model_dump(), "format": "TEXT"}, model,
                         relationship_confirmed=payload.relationship_confirmed)
        # Allowlisted: everything returned is derived from the submitted text.
        # Stored neighbours and annotations are never in this payload.
        return {"verdict": result["verdict"], "reason": result.get("reason"),
                "purpose": result.get("purpose"), "method": result.get("method"),
                "before": result.get("before"), "after": result.get("after"),
                "utility": result.get("utility"), "split_off": result.get("split_off"),
                "removed": result.get("removed", []), "ambiguous": result.get("ambiguous", []),
                "needs_human": result.get("needs_human", False),
                "findings": [{"code": f["code"], "component": f["component"], "message": f["message"]}
                             for f in (result.get("checklist") or {}).get("findings", [])],
                # Without this a NEEDS_CONTEXT verdict says something is missing
                # but never which thing, leaving no way to act on it.
                "missing_context": (result.get("checklist") or {}).get("missing_context", []),
                "notice": "Candidate edit for human review. Meta decides the category."}

    @app.post("/api/generate")
    def generate_template(payload: Task):
        throttle(llm_recent, LLM_LIMIT, "drafting")
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
                "notice": "Drafted candidate, not an approved template. Meta decides the category."}

    @app.get("/assets/lucide.min.js")
    def icons():
        return FileResponse(ROOT / "static" / "vendor" / "lucide.min.js", media_type="text/javascript")

    app.mount("/assets", StaticFiles(directory=ROOT / "static" / "demo"), name="assets")
    app.mount("/results", StaticFiles(directory=ROOT / "site", html=True), name="results")
    return app


app = create_demo()
