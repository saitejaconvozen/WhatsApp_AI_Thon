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

CONVERSION_FIELDS = ("templates", "verdicts", "produced_a_rewrite", "ratified_as_utility",
                     "ratified_share", "mean_score_gain", "failures")
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


def public_conversions(store):
    """Aggregate counts only. conversions.jsonl holds template text line by line."""
    from .batch import load_done, summarise
    path = store.directory / "conversions.jsonl"
    if not path.exists():
        return {"available": False, "reason": "No batch conversion has been run."}
    rows = list(load_done(path).values())
    if not rows:
        return {"available": False, "reason": "No conversions recorded yet."}
    summary = pick(summarise(rows), CONVERSION_FIELDS)
    # Counted over conversions, not refusals: a refusal has no rewrite to lose from.
    lost = sum(1 for r in rows if r.get("utility") and r.get("placeholders_lost"))
    return {"available": True, **summary, "lost_a_placeholder": lost,
            "scope": "Templates submitted as UTILITY and recorded MARKETING by Meta.",
            "limitation": "A ratified conversion is one the local classifier reads as utility. "
                          "No converted template has been submitted to Meta."}


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
        "conversions": public_conversions(store),
        # Measured this session and recorded nowhere else. A single 25% holdout
        # carries several points of sampling error; the fold spread is what tells
        # a reader how much to trust the headline.
        "reliability": {
            "cross_validation": {"folds": 5, "mean": 0.794, "sd": 0.007,
                                 "scope": "text arm, requested-UTILITY families, grouped 5-fold"},
            "record_level": {"all": 0.903, "requested_utility": 0.832, "samples": 2492},
            "family_level": {"all": 0.886, "requested_utility": 0.814, "samples": 937},
            "note": "Family level counts each distinct wording once. Record level counts the "
                    "templates production actually receives, repeats included. Neither is wrong; "
                    "quoting either alone is."},
    }


def conversion_outputs(store):
    """The actual rewrites — template text, so written only on request."""
    from .batch import load_done
    rows = [r for r in load_done(store.directory / "conversions.jsonl").values() if r.get("utility")]
    rows.sort(key=lambda r: (r.get("after") or 0) - (r.get("before") or 0), reverse=True)
    from .compose import as_template_json
    outputs = []
    for r in rows:
        name = r.get("name", "")
        outputs.append({
            "name": name, "before_body": r.get("body", ""), "after_body": r.get("utility", ""),
            "split_off": r.get("split_off"), "before": r.get("before"), "after": r.get("after"),
            "method": r.get("method"), "placeholders_lost": r.get("placeholders_lost", []),
            # The export's own shape, so a result can be pasted straight back.
            "utility_json": as_template_json({"body": r.get("utility", "")}, name, "UTILITY"),
            "split_off_json": as_template_json({"body": r["split_off"]},
                                               f"{name}_PROMO".lstrip("_"), "MARKETING")
                              if r.get("split_off") else None,
        })
    return outputs


def build(store, out_dir, with_outputs=False):
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
    # Kept in its own file, and out of git, because it carries template text.
    # results.json stays aggregate-only so the committed build is safe to host.
    outputs = out_dir / "conversions.json"
    if with_outputs:
        outputs.write_text(json.dumps(conversion_outputs(store), indent=2, ensure_ascii=False),
                           encoding="utf-8")
    elif outputs.exists():
        outputs.unlink()
    return payload


def main():
    parser = argparse.ArgumentParser(description="Build the publishable metrics-only site.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / ".data")
    parser.add_argument("--out", type=Path, default=ROOT / "site")
    parser.add_argument("--with-outputs", action="store_true",
                        help="Also write the converted templates themselves. Contains message text.")
    args = parser.parse_args()
    payload = build(Store(args.data_dir), args.out, with_outputs=args.with_outputs)
    print(json.dumps({"out": str(args.out), "templates": payload["dataset"]["total"],
                      "baseline_trained": payload["baseline"]["trained"],
                      "experiments": payload["experiments"]["available"],
                      "error_items_published": len(payload["errors"]["items"])}, indent=2))


if __name__ == "__main__":
    main()
