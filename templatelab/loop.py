"""Rewrite a template until the classifier calls it UTILITY, or give up saying why.

The caller classifies a template, gets MARKETING, and asks for a conversion. This
runs the loop: rewrite, classify, and if it is still MARKETING feed that verdict
and its specific reasons back to the model and try again.

Two things bound it. It stops after `max_rounds`, because a loop against a
probabilistic judge will otherwise wander. And every round must pass the same
hard gates regardless of what the classifier says: no placeholder may be lost, a
transaction must still be named, and no promotional wording may appear. Without
those a loop like this converges by learning what the classifier rewards -- an
advertisement acquires a reference number and a past-tense verb and passes --
which is laundering, not conversion.

The classifier is roughly 78% accurate, and about 67% near its decision boundary
where these templates sit. A UTILITY verdict here is this model's opinion, not
Meta's decision, and the round-by-round trail is returned so a caller can judge
for themselves rather than trust the final label.
"""

import json
import re

from .compose import PLACEHOLDER, as_template_json
from .llm import BACKENDS, Cache, Unavailable, configuration
from .policy import REFERENCE, TRANSACTION, promotion_findings
from .recast import parse_rewrite
from .retrieval import Examples

MAX_ROUNDS = 4
COMPONENTS = ("header", "body", "footer", "buttons")

REWRITE = """You rewrite WhatsApp Business templates so Meta classifies them as UTILITY.

A utility template reports on something the recipient ALREADY has -- an order,
invoice, booking, appointment or agreement -- and names it specifically. It says
what happened: "has been dispatched", "we received", "was scheduled". A marketing
template invites the recipient to start something new: "get", "explore",
"book now", or offers a discount.

A call to action is NOT the problem. Meta approves "Call Now or choose an option
below to proceed". Keep links and buttons.

{examples}
The template to rewrite:

{body}

{verdict}

Rules:
- Keep EVERY placeholder exactly as written: {placeholders}
- Keep every business fact. Invent no order number, date, amount or reference.
- Do not add a discount, offer, or anything that promotes a purchase.
- Do not claim a transaction the original does not claim. If the original only
  says an offer was extended, it has no order to report.
- If this template has no underlying transaction at all, it cannot become a
  utility template. Say so rather than inventing one.

Return JSON only:
{{"possible": true|false, "body": "the rewritten template", "reason": "one sentence"}}"""


def reasons(text, placeholders):
    """Why this text still reads as marketing, in terms a rewrite can act on."""
    notes = []
    findings = promotion_findings({"body": text})
    for finding in findings:
        notes.append(f'Promotional wording "{finding["evidence"]}": {finding["message"]}'
                     if finding.get("evidence") else finding["message"])
    if not TRANSACTION.search(text):
        notes.append("It names no transaction (order, invoice, booking, appointment, agreement).")
    if not REFERENCE.search(text):
        notes.append("It carries no reference identifying which specific one.")
    if re.search(r"\b(?:get|grab|avail|explore|discover)\b|\b(?:book|call|apply|schedule) now\b", text, re.I):
        notes.append("It invites the recipient to start something NEW; report on an existing one instead.")
    if not re.search(r"\b(?:has|have) been\b|\bwas\b|\bwe (?:have )?received\b", text, re.I):
        notes.append("Nothing is reported as already having happened.")
    lost = sorted(set(placeholders) - set(PLACEHOLDER.findall(text)))
    if lost:
        notes.append(f"These placeholders were dropped and must be restored: {', '.join(lost)}")
    return notes


def gates(text, placeholders):
    """Constraints no round may break, whatever the classifier says."""
    lost = sorted(set(placeholders) - set(PLACEHOLDER.findall(text)))
    promotional = [f["code"] for f in promotion_findings({"body": text})]
    return {
        "placeholders_lost": lost,
        "has_anchor": bool(TRANSACTION.search(text)) and bool(REFERENCE.search(text)),
        "promotional": promotional,
        "ok": not lost and bool(TRANSACTION.search(text)) and bool(REFERENCE.search(text))
              and not promotional,
    }


def classify(model, components, requested="UTILITY"):
    record = {**{k: components.get(k, "") for k in COMPONENTS},
              "format": "TEXT", "requested_category": requested}
    result = model.predict(record)
    return {"category": result.get("category"),
            "confidence": result.get("utility_probability"),
            "available": bool(result.get("available"))}


def convert(record, model, store, max_rounds=MAX_ROUNDS, reasoning="", examples=None):
    """Loop until the classifier says UTILITY, or until the rounds run out."""
    config = configuration()
    if config["backend"] == "disabled":
        raise Unavailable("No language model is configured for conversion.")
    cache = Cache(store.directory)

    # Real precedents beat abstract rules: the model is shown the approved
    # templates closest to this one, and the closest downgraded one, so it can
    # see where the line falls rather than infer it from a description.
    shown = ""
    if examples is not None:
        found = examples.for_template(str(record.get("body") or ""),
                                      exclude=record.get("id"), floor=0.30)
        shown = examples.render(found)

    components = {k: str(record.get(k) or "") for k in COMPONENTS}
    body = components["body"]
    if not body.strip():
        raise ValueError("The template has no message body.")
    placeholders = sorted(set(PLACEHOLDER.findall(body)))

    before = classify(model, components)
    # Caller-supplied reasoning from their own classify call seeds round one, so
    # the model is not re-deriving what the caller already knows.
    why = [reasoning.strip()] if reasoning.strip() else reasons(body, placeholders)
    current, trail, best = body, [], None

    for index in range(max_rounds):
        verdict = ("The classifier reads this as MARKETING"
                   + (f" (utility score {trail[-1]['confidence']:.2f})." if trail and trail[-1].get("confidence") is not None else ".")
                   + "\n\nWhy:\n- " + "\n- ".join(why)) if why else \
                  "The classifier reads this as MARKETING."
        prompt = REWRITE.format(body=current, verdict=verdict, examples=shown,
                                placeholders=", ".join(placeholders) or "(none)")
        key = Cache.key(prompt, config["model"])
        answer = cache.get(key)
        if not answer:
            answer = parse_rewrite(BACKENDS[config["backend"]](prompt, config["model"]))
            cache.put(key, answer)

        if not answer.get("possible") or not (answer.get("body") or "").strip():
            trail.append({"round": index + 1, "outcome": "refused",
                          "reason": (answer.get("reason") or "")[:400]})
            break

        candidate = answer["body"].strip()
        check = gates(candidate, placeholders)
        scored = classify(model, {**components, "body": candidate})
        trail.append({"round": index + 1,
                      "category": scored["category"], "confidence": scored["confidence"],
                      "gates_passed": check["ok"], "body": candidate,
                      "issues": [] if check["ok"] else reasons(candidate, placeholders)[:4]})

        if check["ok"] and scored["category"] == "UTILITY":
            best = ({**components, "body": candidate}, scored, answer)
            break
        current, why = candidate, reasons(candidate, placeholders)
        if not check["ok"] and check["placeholders_lost"]:
            # A dropped placeholder is the one failure worth restating verbatim.
            why.insert(0, f"Restore these exactly: {', '.join(check['placeholders_lost'])}")

    if best is None:
        return {"converted": False, "category": "MARKETING",
                "confidence": before["confidence"], "reasoning": why,
                "template": None, "template_json": None, "rounds": trail,
                "reason": ("No rewrite satisfied both the classifier and the constraints. "
                           "Where the template has no underlying transaction, this is the "
                           "correct outcome rather than a failure.")}

    final, scored, answer = best
    return {"converted": True, "category": scored["category"],
            "confidence": scored["confidence"],
            "confidence_before": before["confidence"],
            "reasoning": [str(answer.get("reason") or "")[:400]],
            "template": final,
            "template_json": as_template_json(final, str(record.get("name") or "")),
            "rounds": trail}


def run(store, out, utility=250, marketing=250, workers=6, max_rounds=4, seed=0):
    """Classify a sample, and run the conversion loop on whatever comes back MARKETING.

    The Meta-UTILITY half is the control. Those templates need no conversion, so
    every one the classifier sends into the loop is a false positive -- and a
    conversion produced there is work done on a template that was already fine.
    """
    import random, threading
    from concurrent.futures import ThreadPoolExecutor
    from .data import training_records
    from .serving_candidate import Candidate

    model = Candidate(store)
    examples = Examples(store)
    records, _ = training_records(store.records())
    intended = [r for r in records if r.get("requested_category") == "UTILITY"]
    rng = random.Random(seed)
    approved = [r for r in intended if r["meta_category"] == "UTILITY"]
    downgraded = [r for r in intended if r["meta_category"] == "MARKETING"]
    rng.shuffle(approved); rng.shuffle(downgraded)
    sample = ([{**r, "_set": "meta_utility"} for r in approved[:utility]]
              + [{**r, "_set": "meta_marketing"} for r in downgraded[:marketing]])

    done = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[row["id"]] = row
    pending = [r for r in sample if r["id"] not in done]
    print(f"{len(sample)} sampled | {len(done)} done | {len(pending)} to run", flush=True)

    lock = threading.Lock()
    handle = out.open("a", encoding="utf-8")

    def one(record):
        row = {"id": record["id"], "name": record.get("name", ""),
               "set": record["_set"], "meta_category": record["meta_category"],
               "body": record.get("body", "")}
        first = classify(model, {k: str(record.get(k) or "") for k in COMPONENTS})
        row.update({"classified": first["category"], "confidence": first["confidence"]})
        if first["category"] == "UTILITY":
            row["outcome"] = "already_utility"
        else:
            try:
                result = convert(record, model, store, max_rounds=max_rounds,
                                 examples=examples)
            except (Unavailable, ValueError) as exc:
                row.update({"outcome": "error", "reason": str(exc)[:200]})
            else:
                row.update({"outcome": "converted" if result["converted"] else "not_converted",
                            "final_confidence": result["confidence"],
                            "converted_body": (result.get("template") or {}).get("body"),
                            "rounds": len(result["rounds"]),
                            "round_trail": [{k: v for k, v in r.items() if k != "body"}
                                            for r in result["rounds"]]})
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

    rows = []
    for line in out.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    report = summarise(rows)
    out.with_suffix(".summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def summarise(rows):
    out = {}
    for name in ("meta_utility", "meta_marketing"):
        group = [r for r in rows if r.get("set") == name]
        if not group:
            continue
        sent = [r for r in group if r.get("classified") == "MARKETING"]
        out[name] = {
            "templates": len(group),
            "classifier_said_UTILITY": sum(1 for r in group if r.get("classified") == "UTILITY"),
            "classifier_accuracy": round(
                sum(1 for r in group if r.get("classified") == r.get("meta_category")) / len(group), 4),
            "sent_to_conversion": len(sent),
            "converted": sum(1 for r in sent if r.get("outcome") == "converted"),
            "not_converted": sum(1 for r in sent if r.get("outcome") == "not_converted"),
            "errors": sum(1 for r in sent if r.get("outcome") == "error"),
            "mean_rounds": round(sum(r.get("rounds", 0) for r in sent) / len(sent), 2) if sent else None,
        }
    return out


def main():
    import argparse
    from pathlib import Path
    from .data import Store
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / ".data")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--utility", type=int, default=250)
    parser.add_argument("--marketing", type=int, default=250)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-rounds", type=int, default=4)
    args = parser.parse_args()
    run(Store(args.data_dir), args.out or args.data_dir / "loop-run.jsonl",
        utility=args.utility, marketing=args.marketing,
        workers=args.workers, max_rounds=args.max_rounds)


if __name__ == "__main__":
    main()
