"""Bounded, reproducible LLM benchmark on the persisted requested-utility holdout."""

import argparse
import getpass
import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from . import llm
from .data import Store
from .model import Baseline, features, score_rows


def sample_rows(holdout, limit, seed=2026):
    rows = sorted((r for r in holdout if r.get("requested_category") == "UTILITY"), key=lambda r: r["id"])
    if limit < 1 or limit > 100:
        raise ValueError("This pilot permits 1 to 100 templates and at most one call per template.")
    if len(rows) <= limit:
        return rows
    labels = [r["recorded_category"] for r in rows]
    if limit < len(set(labels)):
        return [rows[np.random.default_rng(seed).integers(len(rows))]]
    _, selected = train_test_split(np.arange(len(rows)), test_size=limit, stratify=labels, random_state=seed)
    return [rows[i] for i in sorted(selected)]


def wilson(correct, total):
    if not total:
        return None
    p, z = correct / total, 1.96
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [max(0, center - margin), min(1, center + margin)]


def latest_report(store):
    paths = list(store.directory.glob("deepseek-benchmark-*.json"))
    if not paths:
        return {"available": False}
    report = json.loads(max(paths, key=lambda path: path.stat().st_mtime_ns).read_text())
    report.pop("sample_ids", None)
    if report.get("metrics"):
        report["metrics"].pop("predictions", None)
    report["stale"] = report["dataset_revision"] != store.revision()
    return {"available": True, "report": report}


def summarize(rows, results):
    if len(rows) != len(results):
        raise ValueError("Every sampled row must have a result, including failures.")
    scored = [{**row, "prediction": result.get("category", "ERROR") if result.get("available") else "ERROR"}
              for row, result in zip(rows, results)]
    correct = sum(row["prediction"] == row["recorded_category"] for row in scored)
    successful = [row for row in scored if row["prediction"] != "ERROR"]
    return {"attempted": len(rows), "successful_responses": len(successful),
            "failures": len(rows) - len(successful),
            "abstentions": sum(row["prediction"] == "NEEDS_REVIEW" for row in scored),
            "end_to_end": score_rows(scored), "successful_responses_only": score_rows(successful),
            "baseline_same_sample": score_rows(rows), "accuracy_95pct_interval": wilson(correct, len(rows)),
            "target_accuracy": .90, "target_observed": bool(rows and correct / len(rows) >= .90),
            "predictions": [{"id": row["id"], "recorded_category": row["recorded_category"],
                             "baseline_prediction": row["prediction"], "result": result}
                            for row, result in zip(rows, results)]}


def evaluate(store, limit=100, workers=4):
    baseline = Baseline(store)
    if not baseline.status().get("trained") or baseline.status().get("stale"):
        raise ValueError("A current baseline with persisted holdout predictions is required.")
    revision = store.revision()
    holdout = baseline.bundle.get("holdout", [])
    rows = sample_rows(holdout, limit)
    if not rows:
        raise ValueError("No requested-utility holdout templates are available.")
    exclude = {row["id"] for row in holdout}
    families = {row["family"] for row in holdout}
    training = [r for r in baseline.bundle["records"] if r["id"] not in exclude and r["family"] not in families]
    # Refit the retrieval vocabulary on training only, not the all-data serving index.
    text_branch = features().fit(training)
    from types import SimpleNamespace
    retrieval = SimpleNamespace(bundle={"records": training, "text_branch": text_branch,
                                       "matrix": text_branch.transform(training)})
    reviewer = llm.Reviewer(store, retrieval)
    status = reviewer.status()
    if not status["available"]:
        raise ValueError(status["reason"])
    config = llm.configuration()
    protocol = {"dataset_revision": revision, "trained_at": baseline.bundle["report"]["trained_at"],
                "model": config["model"], "backend": config["backend"], "base_url": config["base_url"],
                "policy": llm.policy_version(), "seed": 2026, "sample_ids": [r["id"] for r in rows],
                "max_calls": len(rows), "max_tokens_per_call": llm.MAX_TOKENS,
                "selection": "Recorded-label-stratified random sample of requested-UTILITY holdout families",
                "retrieval": "Training-only vocabulary/index; every holdout ID and family excluded",
                "retries": 0, "generated_at": datetime.now(timezone.utc).isoformat()}
    run_id = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12]
    path = store.directory / f"deepseek-benchmark-{run_id}.json"
    results = []

    def checkpoint():
        report = {**protocol, "complete": len(results) == len(rows),
                  "completed": len(results), "stale": revision != store.revision(),
                  "metrics": summarize(rows[:len(results)], results) if results else None,
                  "limitations": [
                      "Research pilot on an already-inspected historical holdout, not an untouched external test.",
                      "Failures and abstentions count as incorrect in end-to-end accuracy. Success-only metrics are secondary.",
                      "Confidence values are model self-reports, not calibrated approval probabilities.",
                      "Obvious contact details/URLs are redacted; this does not guarantee anonymity.",
                      "No model promotion, template submission, or automatic rewriting is performed.",
                  ]}
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(path)
        return report

    checkpoint()
    # One initial call discovers authentication/configuration errors before dispatching the batch.
    first = reviewer.review(rows[0], exclude=exclude)
    results.append(first)
    checkpoint()
    print(f"Completed 1/{len(rows)}; available={first.get('available', False)}", flush=True)
    if not first.get("available"):
        print(f"Stopped after initial failure: {first.get('reason')}", flush=True)
        return path, checkpoint()
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 4))) as pool:
        for result in pool.map(lambda row: reviewer.review(row, exclude=exclude), rows[1:]):
            results.append(result)
            checkpoint()
            print(f"Completed {len(results)}/{len(rows)}", flush=True)
    report = checkpoint()
    if not report["stale"]:
        latest = store.directory / "deepseek-benchmark.json"
        temporary = latest.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(latest)
    return path, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / ".data")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--allow-egress", action="store_true")
    parser.add_argument("--prompt-key", action="store_true")
    args = parser.parse_args()
    if not args.allow_egress:
        parser.error("--allow-egress is required; sanitized template text is sent to the configured provider.")
    if not args.base_url.startswith("https://"):
        parser.error("The provider base URL must use HTTPS.")
    os.environ.update({llm.BACKEND_VARIABLE: "openai_compatible", llm.MODEL_VARIABLE: args.model,
                       llm.BASE_URL_VARIABLE: args.base_url, llm.EGRESS_VARIABLE: "1"})
    if args.prompt_key:
        os.environ[llm.KEY_VARIABLE] = getpass.getpass("API key (not saved): ")
    if not os.environ.get(llm.KEY_VARIABLE):
        parser.error("Configure TEMPLATELAB_LLM_API_KEY or use --prompt-key.")
    path, report = evaluate(Store(args.data_dir), args.limit, args.workers)
    print(json.dumps({"path": str(path), "complete": report["complete"],
                      "metrics": {k:v for k,v in (report["metrics"] or {}).items() if k != "predictions"}}, indent=2))


if __name__ == "__main__":
    main()
