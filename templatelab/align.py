"""Move a downgraded template toward wording Meta has already approved.

MEASURED RESULT: the mechanical path in this module does not work. Stripping
questions and calls to action moves a template further from approved wording in
65% of cases (mean gain -0.11 over 423 edited templates), because Meta's own
approved utility templates contain those constructions at a similar rate. The
aggregate signal that suggested it -- approved templates ask questions 8% of the
time against 16% -- does not survive being applied per template. The module is
kept because the ApprovedIndex and the tiering are reusable, and because the
negative result is worth not rediscovering.

The rewriter asks a model to invent text that sounds like utility, then asks our
own classifier whether it succeeded. Both halves are weak: "sounds like utility"
is not a target, and the classifier is only 67% accurate near the decision
boundary, which is exactly where every downgraded template sits.

This module replaces both. The target is a specific Meta-approved template, and
the test is how close the result gets to one. That test is grounded in Meta's own
record: across 5,087 pairs of near-identical templates in this corpus, Meta gave
the same verdict 98.6% of the time at similarity 0.95+, and 78.9% at 0.80-0.90.
Similarity to approved wording predicts Meta's verdict. Our classifier, here,
does not.

Nothing here proves Meta would approve an aligned template. It reports how far a
template moved toward wording Meta accepted, which is a measurable claim; the
approval itself can only be established by submitting.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .data import Store, normalized_content, training_records
from .policy import TRANSACTION, promotion_findings

ROOT = Path(__file__).resolve().parent.parent
PLACEHOLDER = re.compile(r"\{\{[^}]+\}\}")
REFERENCE = re.compile(r"\{\{[^}]+\}\}|\d{2,}")

# Measured on this corpus: Meta's approved utility templates ask questions in 8%
# of cases against 16% of downgraded ones, and read as past-tense reports 26% of
# the time against 11%. These patterns remove the solicitation, never the report.
QUESTION = re.compile(r"[^.!?\n]*\?\s*")
CALL_TO_ACTION = re.compile(
    r"\b(?:call|click|tap|book|order|buy|shop|visit|apply|register|sign\s?up|subscribe|"
    r"claim|grab|hurry|explore|discover|check\s+out|don'?t\s+miss|avail|enroll)\b"
    r"[^.!?\n]*(?:now|today|here|below|link|us|online)\b[^.!?\n]*[.!?]?", re.I)
URGENCY = re.compile(r"\b(?:limited (?:time|period|offer)|last chance|hurry|ends (?:today|soon)|"
                     r"only \d+ (?:days?|hours?) left|while stocks last)\b[^.!?\n]*[.!?]?", re.I)


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def is_solicitation(part):
    """A fragment that asks the recipient to start something new, rather than
    reporting something that already happened."""
    return bool(QUESTION.fullmatch(part.strip() + " ") or CALL_TO_ACTION.search(part)
                or URGENCY.search(part) or promotion_findings({"body": part}))


def strip_solicitation(body):
    """Keep the report, drop the pitch — but never drop a business value.

    A fragment carrying a placeholder or a transaction reference is kept even
    when it reads promotionally: removing it would change what the message
    tells the recipient, which is laundering rather than reclassifying.
    """
    kept, dropped = [], []
    for part in sentences(body):
        if is_solicitation(part) and not (PLACEHOLDER.search(part) or REFERENCE.search(part)):
            dropped.append(part)
        else:
            kept.append(part)
    return " ".join(kept).strip(), dropped


class ApprovedIndex:
    """Meta-approved utility templates, searchable by wording."""

    def __init__(self, approved):
        self.approved = approved
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=40000)
        self.matrix = self.vectorizer.fit_transform(
            [normalized_content(r["body"]) for r in approved])

    def nearest(self, body):
        if not (body or "").strip():
            return None, 0.0
        scores = (self.vectorizer.transform([normalized_content(body)]) @ self.matrix.T).toarray()[0]
        best = int(scores.argmax())
        return self.approved[best], float(scores[best])


def align(record, index):
    """Strip the solicitation and report how far the result moved."""
    body = record.get("body") or ""
    before_twin, before = index.nearest(body)
    edited, dropped = strip_solicitation(body)
    after_twin, after = index.nearest(edited)

    lost = sorted(set(PLACEHOLDER.findall(body)) - set(PLACEHOLDER.findall(edited)))
    anchor = bool(TRANSACTION.search(edited)) and bool(REFERENCE.search(edited))
    clean = not promotion_findings({"body": edited})

    return {
        "id": record["id"], "name": record.get("name", ""),
        "body": body, "aligned": edited, "dropped": dropped,
        "before_similarity": round(before, 4), "after_similarity": round(after, 4),
        "gain": round(after - before, 4),
        "exemplar": (after_twin or {}).get("name", ""),
        "exemplar_body": (after_twin or {}).get("body", ""),
        "placeholders_lost": lost,
        "has_anchor": anchor, "no_promotional_wording": clean,
        # Two gates, deliberately separate. The substantive one asks whether the
        # result is a service message at all: every business value intact, a
        # transaction named, no promotional wording left. The evidential one asks
        # how much Meta's own record backs it. A template can pass the first and
        # fail the second simply because this corpus holds no approved template
        # for its use case -- that is an absence of precedent, not a bad result,
        # and collapsing the two would relabel guesses as evidence.
        "substantive_ok": (not lost) and anchor and clean,
        "tier": tier_for(after, (not lost) and anchor and clean),
        "aligned_ok": (not lost) and anchor and clean and after >= 0.70,
    }


def tier_for(similarity, substantive):
    """How strong the evidence is that Meta would accept the aligned template."""
    if not substantive:
        return "not-a-service-message"
    if similarity >= 0.80:
        return "precedent-backed"     # Meta agrees with itself 79-99% this close
    if similarity >= 0.70:
        return "precedent-adjacent"   # 67-75% agreement; worth submitting early
    return "policy-backed"            # no precedent exists; rests on the profile alone


def run(store, out, threshold=0.70):
    records, _ = training_records(store.records())
    intended = [r for r in records if r.get("requested_category") == "UTILITY"]
    approved = [r for r in intended if r["meta_category"] == "UTILITY"]
    downgraded = [r for r in intended if r["meta_category"] == "MARKETING"]
    index = ApprovedIndex(approved)

    rows = [align(r, index) for r in downgraded]
    edited = [r for r in rows if r["dropped"]]
    ok = [r for r in rows if r["aligned_ok"]]
    tiers = {}
    for r in rows:
        tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
    moved = [r for r in rows if r["gain"] > 0]
    report = {
        "downgraded": len(rows),
        "aligned": len(ok),
        "aligned_share": round(len(ok) / len(rows), 4),
        "substantively_converted": sum(1 for r in rows if r["substantive_ok"]),
        "substantively_converted_share": round(
            sum(1 for r in rows if r["substantive_ok"]) / len(rows), 4),
        "tiers": tiers,
        # Averaged over every edited template, never over the ones that improved.
        # Reporting the improvers alone showed +0.049 and hid the real result:
        # stripping calls to action moves templates AWAY from approved wording
        # 65% of the time, because approved utility templates say "click the
        # link below" just as often as downgraded ones do.
        "edited": len(edited),
        "moved_closer": len(moved),
        "moved_further": sum(1 for r in edited if r["gain"] < 0),
        "mean_gain_over_edited": round(float(np.mean([r["gain"] for r in edited])), 4) if edited else 0.0,
        "blocked_by": {
            "lost a placeholder": sum(1 for r in rows if r["placeholders_lost"]),
            "no transactional anchor": sum(1 for r in rows if not r["has_anchor"]),
            "promotional wording remains": sum(1 for r in rows if not r["no_promotional_wording"]),
            "below similarity threshold": sum(1 for r in rows if r["after_similarity"] < threshold),
        },
    }
    out.write_text(json.dumps({"summary": report, "rows": rows}, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"written to {out}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / ".data")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.70)
    args = parser.parse_args()
    run(Store(args.data_dir), args.out or args.data_dir / "aligned.json", args.threshold)


if __name__ == "__main__":
    main()
