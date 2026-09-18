import csv
import io
import json
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .data import FIELDS, MAX_BYTES, Store, normalize_rows, parse_file, suggest_mapping, summarize
from .model import Baseline
from .policy import PRESETS, assess
from .error_review import ErrorReviews
from .experiments import Experiments
from .llm import Reviewer
from .compose import convert, generate


ROOT = Path(__file__).resolve().parent.parent


class ImportRequest(BaseModel):
    token: str
    mapping: dict[str, str] = Field(default_factory=dict)


class ReviewRequest(BaseModel):
    header: str = Field(default="", max_length=4000)
    body: str = Field(min_length=1, max_length=20000)
    footer: str = Field(default="", max_length=4000)
    buttons: str = Field(default="", max_length=4000)
    purpose: str = "unknown"
    relationship_confirmed: bool = False
    format: str = "TEXT"


class ConvertRequest(ReviewRequest):
    pass


class GenerateRequest(BaseModel):
    task: str = Field(min_length=1, max_length=4000)
    purpose: str = ""


class ErrorDraft(BaseModel):
    header: str = Field(default="", max_length=4000)
    body: str = Field(max_length=20000)
    footer: str = Field(default="", max_length=4000)
    buttons: str = Field(default="", max_length=4000)


class ErrorAnnotation(BaseModel):
    snapshot: str = Field(max_length=64)
    version: int = Field(ge=0)
    status: Literal["pending", "context_needed", "reviewed"] = "pending"
    reason: Literal["unknown", "promotion", "missing_context", "ambiguous_wording", "label_question", "model_error", "other"] = "unknown"
    notes: str = Field(default="", max_length=10000)
    context: str = Field(default="", max_length=10000)
    purpose: Literal["unknown", "billing", "order", "appointment", "support", "account", "critical"] = "unknown"
    relationship_confirmed: bool = False
    draft: ErrorDraft | None = None
    meta_outcome: Literal["not_submitted", "MARKETING", "UTILITY", "AUTHENTICATION", "REJECTED"] = "not_submitted"
    outcome_reference: str = Field(default="", max_length=1000)


def create_app(data_dir=None):
    app = FastAPI(title="Template Lab", docs_url="/api/docs")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"])
    store = Store(data_dir or os.environ.get("TEMPLATE_LAB_DATA_DIR", ROOT / ".data"))
    baseline = Baseline(store)
    error_reviews = ErrorReviews(store, baseline)
    experiments = Experiments(store)
    reviewer = Reviewer(store, baseline)
    staged, stage_lock = {}, threading.Lock()
    app.state.store, app.state.baseline = store, baseline

    @app.middleware("http")
    async def local_requests(request: Request, call_next):
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if origin and urlparse(origin).netloc != request.headers.get("host"):
                return JSONResponse({"detail": "Use the local application to perform this action."}, status_code=403)
            try:
                size = int(request.headers.get("content-length", "0"))
            except ValueError:
                return JSONResponse({"detail": "Invalid request length."}, status_code=400)
            if size > MAX_BYTES + 1024 * 1024:
                return JSONResponse({"detail": "The file exceeds the 25 MB limit."}, status_code=413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
        if request.url.path.startswith("/api"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/health")
    def health():
        return {"status": "ok", "storage": "local", "external_services": False}

    @app.get("/api/summary")
    def summary():
        return summarize(store.records())

    @app.get("/api/records")
    def records(q: str = Query(default="", max_length=500), category: str = "", status: str = "",
                mismatch: bool = False, page: int = Query(default=1, ge=1), page_size: int = Query(default=25, ge=1, le=100)):
        rows = store.records()
        if q:
            rows = [r for r in rows if q.lower() in " ".join((r["name"], r["header"], r["body"], r["footer"], r["buttons"])).lower()]
        if category:
            rows = [r for r in rows if r["meta_category"] == category]
        if status:
            rows = [r for r in rows if r["status"] == status]
        if mismatch:
            rows = [r for r in rows if r["requested_category"] == "UTILITY" and r["meta_category"] == "MARKETING"]
        start = (page - 1) * page_size
        return {"total": len(rows), "page": page, "page_size": page_size, "items": rows[start:start + page_size]}

    @app.get("/api/records/{record_id}")
    def record(record_id: str):
        result = next((r for r in store.records() if r["id"] == record_id), None)
        if not result:
            raise HTTPException(404, "Template not found.")
        return result

    @app.post("/api/import/preview")
    async def preview(file: UploadFile = File(...)):
        content = await file.read(MAX_BYTES + 1)
        await file.close()
        try:
            rows = parse_file(content, file.filename or "templates.json")
            mapping = suggest_mapping(rows)
            sample = normalize_rows(rows[:5], mapping, file.filename or "templates.json")
        except (ValueError, TypeError, AttributeError, KeyError) as exc:
            raise HTTPException(400, str(exc)) from exc
        token = secrets.token_urlsafe(24)
        with stage_lock:
            for key in list(staged):
                if time.monotonic() - staged[key][0] > 600:
                    del staged[key]
            while len(staged) >= 4:
                del staged[next(iter(staged))]
            staged[token] = (time.monotonic(), rows, file.filename or "templates.json")
        nested = all(isinstance(r.get("messageBody"), dict) for r in rows)
        columns = list(dict.fromkeys(str(k) for r in rows[:100] for k in r))
        return {"token": token, "count": len(rows), "nested": nested, "mapping": mapping,
                "columns": columns, "fields": FIELDS, "preview": sample}

    @app.post("/api/import/commit")
    def commit(payload: ImportRequest):
        with stage_lock:
            item = staged.get(payload.token)
        if not item or time.monotonic() - item[0] > 600:
            raise HTTPException(400, "This upload preview expired. Upload the file again.")
        try:
            normalized = normalize_rows(item[1], payload.mapping, item[2])
            if not any(r["body"] for r in normalized):
                raise ValueError("No message bodies were found. Check the Body column mapping.")
            result = store.import_records(normalized)
        except (ValueError, TypeError, AttributeError, KeyError) as exc:
            raise HTTPException(400, str(exc)) from exc
        with stage_lock:
            staged.pop(payload.token, None)
        return {**result, "summary": summarize(store.records())}

    @app.get("/api/export")
    def export():
        output = io.StringIO()
        fields = ["name", "body", "header", "footer", "buttons", "meta_category", "requested_category",
                  "status", "language", "format", "created_at", "source"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in store.records():
            clean = {key: str(row.get(key, "")) for key in fields}
            for key, value in clean.items():
                if value.lstrip().startswith(("=", "+", "-", "@")):
                    clean[key] = "'" + value
            writer.writerow(clean)
        return Response("\ufeff" + output.getvalue(), media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="templates-export.csv"'})

    @app.get("/api/schema")
    def schema():
        return Response(",".join(FIELDS) + "\n", media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="template-columns.csv"'})

    @app.get("/api/presets")
    def presets():
        return PRESETS

    @app.post("/api/review")
    def review(payload: ReviewRequest):
        if not payload.body.strip():
            raise HTTPException(400, "Enter a message body.")
        record = payload.model_dump()
        # Three independent readings, deliberately not reconciled into one verdict.
        return {"policy": assess(record), "prediction": baseline.predict(record),
                "llm": reviewer.review(record)}

    @app.post("/api/convert")
    def convert_template(payload: ConvertRequest):
        if not payload.body.strip():
            raise HTTPException(400, "Enter a message body.")
        record = payload.model_dump()
        return convert(record, baseline, purpose=payload.purpose if payload.purpose != "unknown" else None,
                       relationship_confirmed=payload.relationship_confirmed)

    @app.post("/api/generate")
    def generate_template(payload: GenerateRequest):
        try:
            return generate(payload.task, baseline, purpose=payload.purpose or None)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/extract")
    async def extract(file: UploadFile = File(...)):
        """Pull one template out of an uploaded file so it can be reviewed."""
        content = await file.read()
        if len(content) > MAX_BYTES:
            raise HTTPException(400, "The file exceeds the upload limit.")
        try:
            rows = parse_file(content, file.filename or "upload.json")
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        mapping = suggest_mapping(rows)
        records = normalize_rows(rows[:1], mapping, file.filename or "upload")
        if not records or not records[0]["body"].strip():
            raise HTTPException(400, "No message body was found in the first record of that file.")
        first = records[0]
        return {"template": {key: first.get(key, "") for key in ("name", "header", "body", "footer", "buttons")},
                "requested_category": first.get("requested_category", "UNKNOWN"),
                "meta_category": first.get("meta_category", "UNKNOWN"),
                "format": first.get("format", "TEXT"),
                "records_in_file": len(rows)}

    @app.get("/api/model")
    def model():
        return baseline.status()

    @app.get("/api/llm")
    def llm_status():
        return reviewer.status()

    @app.get("/api/experiments")
    def experiment_status():
        return experiments.status()

    @app.post("/api/experiments/run")
    def run_experiments():
        try:
            return experiments.run()
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/errors")
    def errors():
        try:
            return error_reviews.current()
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/errors/{record_id}")
    def annotate_error(record_id: str, payload: ErrorAnnotation):
        try:
            return error_reviews.save(record_id, payload.model_dump())
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/error-export")
    def export_errors():
        try:
            report = error_reviews.current()
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return Response(json.dumps(report, indent=2), media_type="application/json",
                        headers={"Content-Disposition": 'attachment; filename="error-reviews.json"'})

    @app.post("/api/model/train")
    def train():
        try:
            result = baseline.train()
            result["training"] = False
            return result
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/model/report")
    def model_report():
        state = baseline.status()
        if not state.get("trained"):
            raise HTTPException(404, "Train a model before exporting its report.")
        return Response(json.dumps(state, indent=2), media_type="application/json",
                        headers={"Content-Disposition": 'attachment; filename="model-report.json"'})

    @app.get("/")
    def home():
        return FileResponse(ROOT / "static" / "index.html")

    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
    return app


app = create_app()
