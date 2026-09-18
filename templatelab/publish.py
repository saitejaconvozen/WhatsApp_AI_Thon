"""Build a publishable static results site containing only aggregate metrics.

Every field is copied through an explicit allowlist. Template names, bodies,
headers, footers, buttons, human annotations and near-duplicate pair IDs are
never read into the payload, so the published build cannot expose message
content even if the frontend were changed to look for it.
"""

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .data import Store, summarize
from .error_review import ErrorReviews
from .experiments import Experiments
from .model import Baseline
from .benchmark import latest_report
from .improve import summary as improvement_summary


ROOT = Path(__file__).resolve().parent.parent

DATASET_FIELDS = ("total", "categories", "requested_categories", "statuses", "formats",
                  "languages", "category_mismatches", "families", "duplicate_records",
                  "conflicting_families", "conflicting_records", "candidate_records",
                  "training_families", "training_categories", "missing_bodies")
BASELINE_FIELDS = ("trained_at", "method", "split", "train_families", "test_families",
                   "total_families", "excluded_conflicting_families", "accuracy",
                   "majority_accuracy", "macro_f1", "labels", "confusion_matrix",
                   "per_class", "predicted_utility_count", "group_overlap", "limitations",
                   "feature_version", "thresholds", "slices", "bands")
EXPERIMENT_FIELDS = ("generated_at", "split", "majority_accuracy", "results", "limitations",
                     "skipped")
NEAR_DUPLICATE_FIELDS = ("method", "threshold", "flagged_test_families")

NO_CONTENT_REASON = ("Error examples are excluded from the published build because they "
                     "contain template text. Review them in the local workspace.")


def pick(source, fields):
    return {field: source[field] for field in fields if field in source}


def public_benchmark(store):
    result = latest_report(store)
    if not result["available"]:
        return result
    source = result["report"]
    report = pick(source, ("model", "completed", "max_calls", "complete", "generated_at", "stale",
                           "selection", "retrieval", "retries", "limitations"))
    metrics = source.get("metrics")
    if metrics:
        report["metrics"] = pick(metrics, ("attempted", "failures", "abstentions", "accuracy_95pct_interval",
                                           "target_accuracy", "target_observed"))
        for field in ("end_to_end", "baseline_same_sample"):
            report["metrics"][field] = pick(metrics[field], ("samples", "accuracy", "majority_accuracy",
                                                            "utility_precision", "utility_recall", "recorded_utility"))
    else:
        report["metrics"] = None
    return {"available": True, "report": report}


def build_payload(store):
    """Aggregate-only mirror of the local /api/results response."""
    baseline = Baseline(store)
    status = baseline.status()
    report = status.get("report")
    experiments = Experiments(store).status()
    experiment_report = experiments.get("report")
    if experiment_report:
        experiment_report = {**pick(experiment_report, EXPERIMENT_FIELDS),
                             "near_duplicates": pick(experiment_report["near_duplicates"],
                                                     NEAR_DUPLICATE_FIELDS)}
    try:
        counts = ErrorReviews(store, baseline).current()["counts"]
    except ValueError as exc:
        errors = {"available": False, "reason": str(exc), "items": []}
    else:
        errors = {"available": False, "reason": NO_CONTENT_REASON, "items": [], "counts": counts}
    return {
        "loaded_at": datetime.now(timezone.utc).isoformat(),
        "published": True,
        "dataset": pick(summarize(store.records()), DATASET_FIELDS),
        "baseline": {"trained": status["trained"], "training": False,
                     "stale": status.get("stale", False),
                     "report": pick(report, BASELINE_FIELDS) if report else None},
        "experiments": {"available": experiments["available"], "running": False,
                        "stale": experiments.get("stale", False), "report": experiment_report},
        "errors": errors,
        "benchmark": public_benchmark(store),
        "improvement": improvement_summary(store),
    }


def build(store, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(store)
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    shutil.copyfile(ROOT / "static" / "results" / "results.css", out_dir / "results.css")
    shutil.copyfile(ROOT / "static" / "results" / "results.js", out_dir / "results.js")
    shutil.copyfile(ROOT / "static" / "results" / "benchmark.js", out_dir / "benchmark.js")
    shutil.copyfile(ROOT / "static" / "vendor" / "lucide.min.js", out_dir / "lucide.min.js")
    page = (ROOT / "static" / "results" / "index.html").read_text(encoding="utf-8")
    page = (page.replace('href="/assets/results.css"', 'href="results.css"')
                .replace('src="/assets/lucide.min.js"', 'src="lucide.min.js"')
                .replace('src="/assets/benchmark.js"', 'src="benchmark.js"')
                .replace('<script src="/assets/results.js" defer></script>',
                         '<script>window.RESULTS_ENDPOINT = "results.json";</script>\n'
                         '  <script src="results.js" defer></script>')
                .replace("Local research workspace", "Published metrics")
                .replace("Template content stays on this computer.",
                         "Aggregate metrics only; no template content is published."))
    (out_dir / "index.html").write_text(page, encoding="utf-8")
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Build the publishable metrics-only site.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / ".data")
    parser.add_argument("--out", type=Path, default=ROOT / "site")
    args = parser.parse_args()
    payload = build(Store(args.data_dir), args.out)
    print(json.dumps({"out": str(args.out), "templates": payload["dataset"]["total"],
                      "baseline_trained": payload["baseline"]["trained"],
                      "experiments": payload["experiments"]["available"],
                      "error_items_published": len(payload["errors"]["items"])}, indent=2))


if __name__ == "__main__":
    main()
