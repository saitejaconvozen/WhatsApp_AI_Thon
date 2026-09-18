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

from .data import Store
from .serving_candidate import Candidate

ROOT = Path(__file__).resolve().parent.parent


class Template(BaseModel):
    requested_category: Literal["UNKNOWN", "UTILITY", "MARKETING"] = "UNKNOWN"
    header: str = Field(default="", max_length=1000)
    body: str = Field(min_length=1, max_length=6000)
    footer: str = Field(default="", max_length=1000)
    buttons: str = Field(default="", max_length=1000)


def create_demo(data_dir=None, predictor=None):
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*.trycloudflare.com", "localhost", "127.0.0.1", "testserver"])
    model = predictor or Candidate(Store(data_dir or ROOT / ".data"))
    recent, lock = deque(), threading.Lock()

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
        with lock:
            now = time.monotonic()
            while recent and now - recent[0] >= 60:
                recent.popleft()
            if len(recent) >= 60:
                raise HTTPException(429, "Demo request limit reached. Try again in one minute.")
            recent.append(now)
            result = model.predict({**payload.model_dump(), "format": "TEXT"})
        if not result.get("available"):
            raise HTTPException(503, "The local classifier is not ready. Please try again later.")
        # Never return neighbors or explanations containing stored customer text.
        return {"category": result["category"], "utility_probability": result["utility_probability"],
                "band": result.get("band"), "model": result.get("model", "Research classifier"),
                "score_label": result.get("score_label", "Utility probability"),
                "notice": "Research prediction, not Meta approval. 90% accuracy is not established."}

    @app.get("/assets/lucide.min.js")
    def icons():
        return FileResponse(ROOT / "static" / "vendor" / "lucide.min.js", media_type="text/javascript")

    app.mount("/assets", StaticFiles(directory=ROOT / "static" / "demo"), name="assets")
    app.mount("/results", StaticFiles(directory=ROOT / "site", html=True), name="results")
    return app


app = create_demo()
