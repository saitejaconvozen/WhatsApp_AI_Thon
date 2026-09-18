"""Score the LLM policy judge against the persisted holdout.

The baseline learns what this corpus did; the judge reads Meta's published rules.
Measuring them on the same rows is the only way to tell whether outside knowledge
beats the corpus-lexical ceiling, and their disagreements are the active-learning
signal — where they differ, one of them is wrong.
"""

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .data import Store
from .llm import Cache, Reviewer, build_prompt, configuration
from .model import Baseline, score_rows, slice_metrics

ROOT = Path(__file__).resolve().parent.parent


def check_budget(uncached, limit, yes):
    if uncached > limit and not yes:
        raise SystemExit(f"{uncached} rows need a paid API call, above the --limit of {limit}. "
                         "Re-run with --yes to spend them.")


def prepare(reviewer, rows, exclude=None):
    """Resolve each row's cache key up front so the cost estimate is real, not a guess."""
    model = configuration()["model"]
    prepared = []
    for row in rows:
        key = Cache.key(build_prompt(row, reviewer.neighbors(row, exclude=exclude)), model)
        prepared.append({"row": row, "cached": reviewer.cache.get(key) is not None})
    return prepared


def aggregate(rows, results):
    """rows and results are positionally paired; results come from Reviewer.review."""
    judged, failures, clauses, authentication = [], Counter(), Counter(), 0
    for row, result in zip(rows, results):
        if not result.get("available"):
            failures[result.get("reason", "unknown")] += 1
            continue
        category = result["category"]
        if category == "AUTHENTICATION":
            authentication += 1
        clauses.update(result.get("clauses") or [])
        judged.append({**row, "prediction": category,
                       "baseline_prediction": row["prediction"],
                       "confidence": result.get("confidence"),
                       "rationale": result.get("rationale", "")})

    scored = {r["id"] for r in judged}
    baseline_rows = [r for r in rows if r["id"] in scored]
    disagree = [r for r in judged if r["prediction"] != r["baseline_prediction"]]
    agreement = Counter((r["baseline_prediction"], r["prediction"]) for r in judged)
    return {
        "judged": len(judged),
        "failed": dict(failures),
        "authentication_verdicts": authentication,
        "llm": slice_metrics(judged),
        "baseline_same_rows": slice_metrics(baseline_rows),
        "agreement": {f"baseline={b} llm={l}": n for (b, l), n in sorted(agreement.items())},
        "agreed": sum(n for (b, l), n in agreement.items() if b == l),
        "disagreement": {
            "count": len(disagree),
            "llm_correct": sum(1 for r in disagree if r["prediction"] == r["recorded_category"]),
            "baseline_correct": sum(1 for r in disagree if r["baseline_prediction"] == r["recorded_category"]),
            "neither_correct": sum(1 for r in disagree if r["recorded_category"]
                                   not in {r["prediction"], r["baseline_prediction"]}),
            "ids": [r["id"] for r in disagree],
        },
        "clauses_cited": dict(clauses.most_common()),
    }


def evaluate(store, limit=100, use_all=False, workers=4, sleep=0.0, yes=False):
    baseline = Baseline(store)
    if not baseline.status()["trained"]:
        raise SystemExit("Train a baseline before evaluating the judge against it.")
    holdout = (baseline.bundle or {}).get("holdout")
    if not holdout:
        raise SystemExit("This baseline stores no holdout predictions. Retrain it once.")

    reviewer = Reviewer(store, baseline)
    status = reviewer.status()
    if not status["available"]:
        raise SystemExit(status["reason"])

    # The serving model is refitted on every eligible family, so the retrieval
    # index contains the holdout. Without this, a held-out template retrieves
    # itself at similarity 1.0 with its recorded label and the judge reads the
    # answer off the prompt — that produced a fake 100% on the first run.
    exclude = {row["id"] for row in holdout}
    sample = holdout if use_all else holdout[:limit]
    prepared = prepare(reviewer, sample, exclude=exclude)
    uncached = sum(1 for p in prepared if not p["cached"])
    print(f"{len(sample)} rows selected; {len(sample) - uncached} cached, {uncached} need a call.")
    check_budget(uncached, limit, yes)

    def run(item):
        if sleep and not item["cached"]:
            time.sleep(sleep)
        return reviewer.review(item["row"], exclude=exclude)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(run, prepared))

    report = {**aggregate(sample, results), "model": status.get("model"),
              "backend": status.get("backend"), "policy": status.get("policy"),
              "holdout_families": len(holdout), "sampled": len(sample),
              "retrieval": "training families only; all holdout ids excluded",
              "limitation": "The judge reads published definitions. Neither side is Meta's decision."}
    return report


def summarise(report):
    lines = [f"backend {report['backend']} / model {report['model']}",
             f"judged {report['judged']} of {report['sampled']} sampled"]
    if report["failed"]:
        lines.append(f"failures: {report['failed']}")
    for name in ("requested_utility", "all"):
        llm, base = report["llm"].get(name), report["baseline_same_rows"].get(name)
        if not llm or not base:
            continue
        lines.append(f"\n{name} (n={llm['samples']})")
        lines.append(f"  {'':<10}{'accuracy':>10}{'utilP':>9}{'utilR':>9}")
        for label, metrics in (("llm", llm), ("baseline", base)):
            precision = "n/a" if metrics["utility_precision"] is None else f"{metrics['utility_precision'] * 100:.1f}%"
            recall = "n/a" if metrics["utility_recall"] is None else f"{metrics['utility_recall'] * 100:.1f}%"
            lines.append(f"  {label:<10}{metrics['accuracy'] * 100:9.1f}%{precision:>9}{recall:>9}")
    d = report["disagreement"]
    lines.append(f"\nagreed on {report['agreed']}, disagreed on {d['count']} "
                 f"(llm right {d['llm_correct']}, baseline right {d['baseline_correct']}, "
                 f"neither {d['neither_correct']})")
    if report["clauses_cited"]:
        lines.append("clauses cited: " + ", ".join(f"{k}={v}" for k, v in report["clauses_cited"].items()))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / ".data")
    parser.add_argument("--limit", type=int, default=100, help="Rows to score, and the paid-call cap.")
    parser.add_argument("--all", action="store_true", help="Score every holdout row; needs --yes.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sleep", type=float, default=0.0, help="Delay before each uncached call.")
    parser.add_argument("--yes", action="store_true", help="Authorise calls beyond --limit.")
    args = parser.parse_args()
    report = evaluate(Store(args.data_dir), limit=args.limit, use_all=args.all,
                      workers=args.workers, sleep=args.sleep, yes=args.yes)
    path = args.data_dir / "llm-eval.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(summarise(report))
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
