"""Rewrite a downgraded template into the shape Meta approves for its form.

Earlier conversion asked a model to make a template "sound like utility" and let
our own classifier decide whether it had worked. Both halves were weak: the
target was abstract, and the classifier is only 67% accurate near the decision
boundary, which is where every downgraded template sits.

Here the target is concrete. Meta's approved templates fall into recurring forms
(templatelab/forms.py), and approval rates differ sharply between them -- one
form holds 86 approvals and no failures, another 27 approvals against 131
failures. Each template is rewritten toward the approved exemplars of its own
form, which gives the model something specific to imitate rather than a mood to
evoke.

The recipe in the prompt is measured, not invented. Comparing approved against
downgraded templates within a matched business event, approved ones presuppose
something that already exists (+16.5pp), name a specific artifact (+14.7pp) and
report in the past tense (+8.3pp), while downgraded ones invite a new action
(-17.7pp) about a generic topic noun (-17.2pp).
"""

import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .align import ApprovedIndex
from .data import Store, training_records
from .forms import is_service_message
from .llm import BACKENDS, Cache, Unavailable, configuration
from .policy import REFERENCE, TRANSACTION, promotion_findings
from .reward import Reward

ROOT = Path(__file__).resolve().parent.parent
PLACEHOLDER = re.compile(r"\{\{[^}]+\}\}")

PROMPT = """You rewrite WhatsApp Business templates so Meta classifies them as UTILITY.

Meta approved all of these templates. They are the shape your answer must take:

{exemplars}

What separates Meta's approved utility templates from ones it downgrades to
marketing, measured across this corpus:
- Approved templates report on something the recipient ALREADY has: a booking,
  an invoice, an agreement, an appointment. They presuppose it exists.
- Approved templates name that specific thing, not a generic topic. "your rental
  agreement {{{{agreement_id}}}}", not "your property".
- Approved templates report what happened, often in the past tense: "has been
  drafted", "we received", "was completed".
- Downgraded templates invite the recipient to start something NEW: "get",
  "schedule now", "book now", "explore".
- A call to action is NOT the problem. Meta approves "Call Now or choose an
  option below to proceed". Do not remove links or buttons.

Rewrite this template:

{body}

Rules:
- Keep EVERY placeholder exactly as written: {placeholders}
- Keep every business fact. Do not invent an order number, date or amount.
- Do not add a discount, offer, or any wording that promotes a purchase.
- If this template does not report on a transaction the recipient already has,
  do not rewrite it. It is genuinely marketing; say so.

Return JSON only:
{{"possible": true|false, "body": "the rewritten template", "reason": "one sentence"}}"""

REFINE = """Your previous rewrite scored {score:.2f} out of 1.00 against a model trained on
templates Meta approved as utility. {verdict}

Your previous attempt:
{previous}

{critique}

Rewrite it again, addressing that specifically. The same rules hold: keep every
placeholder ({placeholders}), invent no facts, add nothing promotional. If the
template genuinely cannot become a service message, say so instead of forcing it.

Return JSON only:
{{"possible": true|false, "body": "the rewritten template", "reason": "one sentence"}}"""


def critique(text):
    """Name what is still wrong, in the terms the corpus actually separates on."""
    notes = []
    if not TRANSACTION.search(text):
        notes.append("It still names no transaction (order, invoice, booking, appointment).")
    if not REFERENCE.search(text):
        notes.append("It carries no reference identifying which one.")
    if promotion_findings({"body": text}):
        notes.append("Promotional wording remains.")
    if re.search(r"\b(?:get|grab|avail|explore)\b|\bto schedule\b|\b(?:book|call|apply) now\b", text, re.I):
        notes.append("It still invites a NEW action; report on an existing one instead.")
    if not re.search(r"\b(?:has|have) been\b|\bwas\b|\bwe received\b", text, re.I):
        notes.append("Nothing is reported as already having happened.")
    return ("Specific problems:\n- " + "\n- ".join(notes)) if notes else (
        "No rule fired, so the wording is simply further from the approved corpus than it could be.")


def parse_rewrite(text):
    """Parse the rewrite response.

    llm.parse_response cannot be reused: it validates a classification payload
    and rejects anything without a known `category`, which a rewrite has no
    reason to carry.
    """
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise Unavailable("The model did not return JSON.")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise Unavailable(f"The model returned malformed JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise Unavailable("Expected a JSON object.")
    # Pass through the optional fields the drafting prompt asks for. Returning a
    # fixed three-key dict silently dropped every name and button label.
    return {"possible": bool(payload.get("possible")),
            "body": str(payload.get("body") or "")[:4000],
            "reason": str(payload.get("reason") or "")[:1000],
            "name": str(payload.get("name") or "")[:120],
            "buttons": str(payload.get("buttons") or "")[:120]}


def call(prompt, cache, config):
    key = Cache.key(prompt, config["model"])
    hit = cache.get(key)
    if hit:
        return hit
    parsed = parse_rewrite(BACKENDS[config["backend"]](prompt, config["model"]))
    cache.put(key, parsed)
    return parsed


def gates(text, placeholders):
    """Hard constraints. These are not quality judgements; they are the things a
    rewrite must not do regardless of how well it scores."""
    lost = sorted(set(placeholders) - set(PLACEHOLDER.findall(text)))
    return {"placeholders_lost": lost,
            "has_anchor": bool(TRANSACTION.search(text)) and bool(REFERENCE.search(text)),
            "promotional": bool(promotion_findings({"body": text}))}


def recast(record, index, exemplars, cache, config, reward=None, rounds=3, target=0.60):
    body = record.get("body") or ""
    placeholders = sorted(set(PLACEHOLDER.findall(body)))
    prompt = PROMPT.format(
        exemplars="\n\n".join(f"--- approved example {i+1} ---\n{e}" for i, e in enumerate(exemplars)),
        body=body, placeholders=", ".join(placeholders) or "(none)")
    row = {"id": record["id"], "name": record.get("name", ""), "body": body}
    start = reward.score(body) if reward else None

    best, trail = None, []
    for attempt in range(rounds):
        try:
            answer = call(prompt, cache, config)
        except Unavailable as exc:
            return {**row, "verdict": "ERROR", "reason": str(exc)[:200], "start_score": start}
        if not answer.get("possible") or not (answer.get("body") or "").strip():
            if best is None:
                return {**row, "verdict": "REFUSED", "start_score": start,
                        "reason": (answer.get("reason") or "")[:300], "rounds_used": attempt + 1}
            break
        candidate = answer["body"].strip()
        checks = gates(candidate, placeholders)
        score = reward.score(candidate) if reward else 0.0
        trail.append(round(score, 4))
        clean = not checks["placeholders_lost"] and checks["has_anchor"] and not checks["promotional"]
        # Keep the best CLEAN candidate; a high score never buys a broken template.
        if clean and (best is None or score > best[1]):
            best, answer_best = (candidate, score), answer
        if clean and score >= target:
            break
        prompt = REFINE.format(
            score=score, previous=candidate, critique=critique(candidate),
            placeholders=", ".join(placeholders) or "(none)",
            verdict="That is below what approved templates typically score."
                    if score < target else "That clears the bar, but the constraints below still fail.")

    if best is None:
        return {**row, "verdict": "REJECTED", "start_score": start,
                "reason": "No attempt satisfied the hard constraints.",
                "score_trail": trail, "rounds_used": len(trail)}
    new, final_score = best
    answer = answer_best
    checks = gates(new, placeholders)
    _, before = index.nearest(body)
    _, after = index.nearest(new)
    return {**row, "verdict": "REWROTE", "rewritten": new,
            "reason": (answer.get("reason") or "")[:300], **checks,
            "before_similarity": round(before, 4), "after_similarity": round(after, 4),
            "gain": round(after - before, 4),
            "start_score": round(start, 4) if start is not None else None,
            "score": round(final_score, 4),
            "score_gain": round(final_score - start, 4) if start is not None else None,
            "score_trail": trail, "rounds_used": len(trail),
            "accepted": True}


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


def run(store, out, limit=None, workers=4, rounds=3, target=0.60):
    config = configuration()
    if config["backend"] == "disabled":
        raise SystemExit("No LLM backend configured; set TEMPLATELAB_LLM_BACKEND.")
    forms = json.loads((store.directory / "forms.json").read_text(encoding="utf-8"))

    records, _ = training_records(store.records())
    intended = [r for r in records if r.get("requested_category") == "UTILITY"]
    approved = [r for r in intended if r["meta_category"] == "UTILITY"]
    index = ApprovedIndex(approved)

    targets = [r for r in intended
               if r["meta_category"] == "MARKETING" and is_service_message(r)]
    done = load_done(out)
    pending = [r for r in targets if r["id"] not in done][:limit]
    print(f"{len(targets)} service messages | {len(done)} done | {len(pending)} this pass", flush=True)

    # Exemplars come from the form each template most resembles; a template is
    # imitating templates Meta approved for its own kind of message.
    by_form = {f["form"]: f for f in forms["forms"]}
    assignments = forms.get("assignments") or {}
    if not assignments:
        raise SystemExit("forms.json has no assignments; rerun `python -m templatelab.forms`.")
    fallback = max(forms["forms"], key=lambda f: f["approval_rate"])
    reward = Reward(store)
    reward.model  # fail now, not inside a worker thread
    cache, lock = Cache(store.directory), threading.Lock()
    handle = out.open("a", encoding="utf-8")

    def one(record):
        _, similarity = index.nearest(record.get("body") or "")
        # The template's own form, not the best-performing one. Records are
        # plain dicts and carry no form field; the mapping lives in forms.json.
        form = by_form.get(assignments.get(record["id"])) or fallback
        row = recast(record, index, form["exemplars"][:3], cache, config,
                     reward=reward, rounds=rounds, target=target)
        row["form_approval_rate"] = form["approval_rate"]
        row["nearest_approved_before"] = round(similarity, 4)
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

    rows = list(load_done(out).values())
    accepted = [r for r in rows if r.get("accepted")]
    gains = [r["gain"] for r in accepted if r.get("gain") is not None]
    report = {
        "attempted": len(rows),
        "rewrote": len(accepted),
        "rewrote_share": round(len(accepted) / len(rows), 4) if rows else 0.0,
        "refused": sum(1 for r in rows if r.get("verdict") == "REFUSED"),
        "rejected_by_gates": sum(1 for r in rows if r.get("verdict") == "REJECTED"),
        "errors": sum(1 for r in rows if r.get("verdict") == "ERROR"),
        "mean_similarity_gain": round(sum(gains) / len(gains), 4) if gains else None,
        "moved_closer": sum(1 for g in gains if g > 0),
        "moved_further": sum(1 for g in gains if g < 0),
    }
    scored = [r for r in accepted if r.get("score") is not None]
    if scored:
        starts = [r["start_score"] for r in scored if r.get("start_score") is not None]
        report.update({
            "mean_reward_before": round(sum(starts) / len(starts), 4) if starts else None,
            "mean_reward_after": round(sum(r["score"] for r in scored) / len(scored), 4),
            "reward_improved": sum(1 for r in scored if (r.get("score_gain") or 0) > 0),
            "cleared_target": sum(1 for r in scored if r["score"] >= target),
            "mean_rounds": round(sum(r.get("rounds_used", 1) for r in scored) / len(scored), 2),
        })
    out.with_suffix(".summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / ".data")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3, help="Refinement attempts per template.")
    parser.add_argument("--target", type=float, default=0.60, help="Reward score to stop at.")
    args = parser.parse_args()
    run(Store(args.data_dir), args.out or args.data_dir / "recast.jsonl",
        limit=args.limit, workers=args.workers, rounds=args.rounds, target=args.target)


if __name__ == "__main__":
    main()
