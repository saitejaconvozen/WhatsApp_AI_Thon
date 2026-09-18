"""Recover what a downgraded template was trying to say, then draft it afresh.

Conversion edits the existing wording and inherits its problems: the promotional
framing is load-bearing, and removing it tends to leave a template that is
neither an advertisement nor a service message. This takes the other route --
read the template for the business event underneath it, discard the wording
entirely, and draft a new template from that intent using the forms Meta
approves (templatelab/draft.py).

The step that makes this safe is the first one. If a template's only purpose is
to promote something, the recovered intent says so, and the drafter refuses it.
A painting advertisement does not become a service message by being described
and rewritten; it becomes a described advertisement, and is declined. That is
the intended behaviour, not a gap: the pipeline must not launder a template into
a category Meta would be right to reject.
"""

import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .data import Store, training_records
from .draft import Drafter, script_of
from .llm import BACKENDS, Cache, Unavailable, configuration
from .recast import PLACEHOLDER, parse_rewrite

ROOT = Path(__file__).resolve().parent.parent

CONTEXT_PROMPT = """Read this WhatsApp Business template and describe what it is for.

Template:
{body}

Placeholders it uses: {placeholders}

Describe the business event underneath the wording -- what has happened, or what
the recipient already has, that makes this message worth sending. Do not repeat
the template's phrasing. Do not soften it.

If the template's purpose is to promote a service, advertise an offer, drive an
upgrade, or re-engage someone who has no pending transaction, say that plainly.
Do not describe an advertisement as a service update.

Describe only what the template actually states. Do not infer that an order,
booking, appointment or application exists because one plausibly might. An offer
the recipient has not accepted is not a transaction: "pre-approved for up to
X" means nobody applied for anything, and "we found a property you may like"
means nothing was booked. If the only thing that has happened is that the
business decided to make an offer, then service_intent is false.

Return JSON only:
{{"service_intent": true|false,
  "context": "one or two sentences describing what the message must tell the recipient",
  "reason": "one sentence on why it is or is not a service message"}}"""


def context_for(record, cache, config):
    body = record.get("body") or ""
    placeholders = sorted(set(PLACEHOLDER.findall(body)))
    prompt = CONTEXT_PROMPT.format(body=body, placeholders=", ".join(placeholders) or "(none)")
    key = Cache.key(prompt, config["model"])
    hit = cache.get(key)
    if hit:
        return hit
    raw = BACKENDS[config["backend"]](prompt, config["model"])
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        raise Unavailable("The model did not return JSON for the context.")
    payload = json.loads(match.group(0))
    parsed = {"service_intent": bool(payload.get("service_intent")),
              "context": str(payload.get("context") or "")[:800],
              "reason": str(payload.get("reason") or "")[:400]}
    cache.put(key, parsed)
    return parsed


def regenerate(record, drafter, cache, config, rounds=3, target=0.60):
    row = {"id": record["id"], "name": record.get("name", ""), "body": record.get("body", "")}
    try:
        recovered = context_for(record, cache, config)
    except (Unavailable, json.JSONDecodeError, ValueError) as exc:
        return {**row, "verdict": "ERROR", "reason": str(exc)[:200]}

    row["placeholder_budget"] = len(set(PLACEHOLDER.findall(row["body"]))) + 1
    row["script"] = script_of(row["body"])
    row["context"] = recovered["context"]
    row["service_intent"] = recovered["service_intent"]
    if not recovered["service_intent"]:
        # Declined here rather than drafted and rejected later: the intent itself
        # is promotional, so there is nothing for the drafter to write.
        return {**row, "verdict": "PROMOTIONAL_INTENT", "reason": recovered["reason"]}

    # The new template may declare its own placeholders, but not more of them
    # than the original carried (plus one, for a recipient name the original may
    # have left out). More than that is the drafter inventing facts.
    budget = len(set(PLACEHOLDER.findall(row["body"]))) + 1
    try:
        drafted = drafter.draft(recovered["context"], rounds=rounds, target=target,
                                placeholder_budget=budget, script=script_of(row["body"]))
    except Unavailable as exc:
        return {**row, "verdict": "ERROR", "reason": str(exc)[:200]}

    if not drafted.get("possible"):
        return {**row, "verdict": "REFUSED", "reason": (drafted.get("reason") or "")[:300],
                "score_trail": drafted.get("score_trail")}

    # A draft below target is reported as WEAK, not counted as a success. The
    # drafter returns its best clean attempt even when refinement never reached
    # the bar, and calling that "drafted" would report a 0.39 template beside a
    # 0.92 one as the same outcome.
    verdict = "DRAFTED" if (drafted.get("score") or 0) >= target else "WEAK"
    return {**row, "verdict": verdict, "drafted": drafted["body"],
            "drafted_name": drafted.get("name", ""), "buttons": drafted.get("buttons", ""),
            "score": drafted.get("score"), "score_trail": drafted.get("score_trail"),
            "rounds_used": drafted.get("rounds_used"),
            "form": drafted.get("form"), "form_matched": drafted.get("form_matched"),
            "form_approval_rate": drafted.get("form_approval_rate"),
            "evidence": drafted.get("evidence"),
            "placeholders_before": len(set(PLACEHOLDER.findall(row["body"]))),
            "placeholders_after": len(set(PLACEHOLDER.findall(drafted["body"])))}


def load_done(path):
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[row["id"]] = row
    return done


def summarise(rows, target=0.60):
    drafted = [r for r in rows if r.get("verdict") == "DRAFTED"]
    weak = [r for r in rows if r.get("verdict") == "WEAK"]
    strong = [r for r in drafted if r.get("form_matched")]
    scores = [r["score"] for r in drafted if r.get("score") is not None]
    return {
        "templates": len(rows),
        "drafted": len(drafted),
        "drafted_share": round(len(drafted) / len(rows), 4) if rows else 0.0,
        "weak_below_target": len(weak),
        "promotional_intent": sum(1 for r in rows if r.get("verdict") == "PROMOTIONAL_INTENT"),
        "refused_by_drafter": sum(1 for r in rows if r.get("verdict") == "REFUSED"),
        "errors": sum(1 for r in rows if r.get("verdict") == "ERROR"),
        "backed_by_a_matched_form": len(strong),
        "mean_score": round(sum(scores) / len(scores), 4) if scores else None,
        "scored_above_target": sum(1 for s in scores if s >= target),
    }


def run(store, out, limit=None, workers=4, rounds=3, target=0.60):
    config = configuration()
    if config["backend"] == "disabled":
        raise SystemExit("No LLM backend configured; set TEMPLATELAB_LLM_BACKEND.")
    drafter = Drafter(store)
    drafter.centroids  # build once here, not concurrently inside worker threads
    cache, lock = Cache(store.directory), threading.Lock()

    records, _ = training_records(store.records())
    targets = [r for r in records if r.get("requested_category") == "UTILITY"
               and r["meta_category"] == "MARKETING"]
    # Latin script only, by request. The 28 Indic-script templates need their own
    # handling: the promotional checklist and the reward model are both English-
    # trained, so a score on Kannada or Hindi prose is not trustworthy even when
    # the draft looks right.
    indic = [r for r in targets if script_of(r.get("body") or "")]
    targets = [r for r in targets if not script_of(r.get("body") or "")]
    print(f"skipping {len(indic)} non-Latin templates", flush=True)
    done = load_done(out)
    pending = [r for r in targets if r["id"] not in done][:limit]
    print(f"{len(targets)} downgraded templates | {len(done)} done | {len(pending)} this pass", flush=True)

    handle = out.open("a", encoding="utf-8")

    def one(record):
        row = regenerate(record, drafter, cache, config, rounds=rounds, target=target)
        with lock:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
        return row

    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for i, _ in enumerate(pool.map(one, pending), 1):
                if i % 20 == 0:
                    print(f"  {i}/{len(pending)}", flush=True)
    finally:
        handle.close()

    report = summarise(list(load_done(out).values()), target)
    out.with_suffix(".summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / ".data")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--target", type=float, default=0.60)
    args = parser.parse_args()
    run(Store(args.data_dir), args.out or args.data_dir / "regenerated.jsonl",
        limit=args.limit, workers=args.workers, rounds=args.rounds, target=args.target)


if __name__ == "__main__":
    main()
