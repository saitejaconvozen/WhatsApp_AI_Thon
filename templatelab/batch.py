"""Run every marketing template through conversion and keep the results.

Writes one JSON object per template as it goes, so a run that is interrupted
loses nothing and a rerun skips what is already done. Results stay in .data/,
which is ignored by git: every row contains template text.

Success is what the classifier ratifies, not what the checklist waved through.
Counting conversions alone rewards a permissive filter, which is how an
advertisement once came back as a "utility version" with one line deleted.
"""

import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .compose import convert
from .data import Store, training_records
from .model import Baseline

PLACEHOLDER = re.compile(r"\{\{[^}]+\}\}")
ROOT = Path(__file__).resolve().parent.parent


def load_done(path):
    if not path.exists():
        return {}
    done = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # a half-written final line from an interrupted run
        done[row["id"]] = row
    return done


def summarise(rows):
    total = len(rows)
    converted = [r for r in rows if r.get("utility")]
    ratified = [r for r in converted if r.get("after_category") == "UTILITY"]
    moved = [r["after"] - r["before"] for r in ratified
             if r.get("after") is not None and r.get("before") is not None]
    verdicts = {}
    for r in rows:
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
    return {
        "templates": total,
        "verdicts": verdicts,
        "produced_a_rewrite": len(converted),
        "ratified_as_utility": len(ratified),
        "ratified_share": round(len(ratified) / total, 4) if total else 0.0,
        "mean_score_gain": round(sum(moved) / len(moved), 4) if moved else None,
        "failures": sum(1 for r in rows if r.get("error")),
    }


def run(store, out, limit=None, workers=4, requested_utility_only=True):
    baseline = Baseline(store)
    from .serving_candidate import Candidate
    model = Candidate(store)

    records, _ = training_records(store.records())
    targets = [r for r in records if r["meta_category"] == "MARKETING"]
    if requested_utility_only:
        # Templates the business asked to send as utility and Meta downgraded:
        # the only ones where a conversion changes anything.
        targets = [r for r in targets if r.get("requested_category") == "UTILITY"]
    done = load_done(out)
    pending = [r for r in targets if r["id"] not in done][:limit]
    print(f"{len(targets)} marketing templates | {len(done)} already done | {len(pending)} to run", flush=True)

    lock = threading.Lock()
    handle = out.open("a", encoding="utf-8")

    def one(record):
        row = {"id": record["id"], "name": record.get("name", ""),
               "requested_category": record.get("requested_category"),
               "meta_category": record["meta_category"], "body": record.get("body", "")}
        try:
            d = convert({**record, "format": "TEXT"}, model, relationship_confirmed=True)
        except Exception as exc:
            row.update({"verdict": "ERROR", "error": str(exc)[:300]})
        else:
            before, after = d.get("before") or {}, d.get("after") or {}
            row.update({
                "verdict": d["verdict"], "method": d.get("method"), "reason": d.get("reason"),
                "before": before.get("utility_probability"), "after": after.get("utility_probability"),
                "after_category": after.get("category"),
                "utility": (d.get("utility") or {}).get("body"),
                "split_off": (d.get("split_off") or {}).get("body"),
                # Only meaningful when a rewrite exists. Computing it for a refusal
                # marks every placeholder "lost" simply because there is no rewrite,
                # which once reported 774 losses against 21 conversions.
                "placeholders_lost": sorted(
                    set(PLACEHOLDER.findall(record.get("body", "")))
                    - set(PLACEHOLDER.findall((d.get("utility") or {}).get("body", "")))
                    - set(PLACEHOLDER.findall((d.get("split_off") or {}).get("body", ""))))
                if d.get("utility") else [],
            })
        with lock:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
        return row

    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for index, _ in enumerate(pool.map(one, pending), 1):
                if index % 25 == 0:
                    print(f"  {index}/{len(pending)}", flush=True)
    finally:
        handle.close()

    rows = list(load_done(out).values())
    report = summarise(rows)
    (out.with_suffix(".summary.json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / ".data")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Templates to run this pass.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--all-marketing", action="store_true",
                        help="Include templates that were submitted as marketing too.")
    args = parser.parse_args()
    out = args.out or args.data_dir / "conversions.jsonl"
    run(Store(args.data_dir), out, limit=args.limit, workers=args.workers,
        requested_utility_only=not args.all_marketing)


if __name__ == "__main__":
    main()
