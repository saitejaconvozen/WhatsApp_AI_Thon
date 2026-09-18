"""Separate read-only results website backed by the existing local workspace."""

from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .data import Store, summarize
from .error_review import ErrorReviews
from .experiments import Experiments
from .model import Baseline
from .benchmark import latest_report
from .improve import summary as improvement_summary


ROOT = Path(__file__).resolve().parent.parent


def create_results_app(data_dir=None):
    app = FastAPI(title="Template Lab Results", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"])
    store = Store(data_dir or ROOT / ".data")

    @app.middleware("http")
    async def headers(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
        return response

    @app.get("/api/results")
    def results():
        revision = store.revision()
        baseline = Baseline(store)
        try:
            errors = {"available": True, **ErrorReviews(store, baseline).current()}
        except ValueError as exc:
            errors = {"available": False, "reason": str(exc), "items": []}
        report = {"loaded_at": datetime.now(timezone.utc).isoformat(), "revision": revision,
                  "dataset": summarize(store.records()), "baseline": baseline.status(),
                  "experiments": Experiments(store).status(), "errors": errors,
                  "benchmark": latest_report(store), "improvement": improvement_summary(store)}
        if revision != store.revision():
            raise HTTPException(409, "Dataset changed while loading results. Refresh to load a consistent snapshot.")
        return report

    @app.get("/")
    def home():
        return FileResponse(ROOT / "static" / "results" / "index.html")

    @app.get("/assets/lucide.min.js")
    def icons():
        return FileResponse(ROOT / "static" / "vendor" / "lucide.min.js", media_type="text/javascript")

    app.mount("/assets", StaticFiles(directory=ROOT / "static" / "results"), name="assets")
    return app


app = create_results_app()
