"""Find a downgraded template's closest Meta-approved twin.

The rewriter asks a model to invent wording that might pass. This asks a
different question: has Meta already approved wording almost identical to this?
For 111 of the 1,180 downgraded templates in this corpus the answer is yes, and
22 of those are word-for-word identical to something Meta accepted.

That matters because a precedent is Meta's own ruling, not a prediction of one.
Our classifier is 93% accurate on templates far from the decision boundary and
67% on the contested ones, which is exactly where these sit — so near the
boundary its opinion is worth less than Meta's recorded decision.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .data import Store, normalized_content, training_records

ROOT = Path(__file__).resolve().parent.parent
IDENTICAL = 0.995
NEAR = 0.90
LOOSE = 0.85
PLACEHOLDER = re.compile(r"\{\{[^}]+\}\}")


def tier_for(similarity):
    if similarity >= IDENTICAL:
        return "identical"
    if similarity >= NEAR:
        return "near"
    if similarity >= LOOSE:
        return "loose"
    return "none"


def compatible(downgraded, approved):
    """A precedent is only usable if it carries the same business values.

    Matching wording is not enough: a twin that drops {{Account_Number}} cannot
    stand in for a template that needs it.
    """
    return not (set(PLACEHOLDER.findall(downgraded)) - set(PLACEHOLDER.findall(approved)))


def find(store, limit=None):
    records, _ = training_records(store.records())
    intended = [r for r in records if r.get("requested_category") == "UTILITY"]
    approved = [r for r in intended if r["meta_category"] == "UTILITY"]
    downgraded = [r for r in intended if r["meta_category"] == "MARKETING"][:limit]
    if not approved or not downgraded:
        raise ValueError("Need both approved and downgraded templates to match against.")

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=40000)
    approved_matrix = vectorizer.fit_transform([normalized_content(r["body"]) for r in approved])
    target_matrix = vectorizer.transform([normalized_content(r["body"]) for r in downgraded])

    matches = []
    for start in range(0, target_matrix.shape[0], 256):
        block = (target_matrix[start:start + 256] @ approved_matrix.T).toarray()
        for offset, scores in enumerate(block):
            record = downgraded[start + offset]
            # Rank by similarity, then take the best that keeps every placeholder.
            for index in np.argsort(scores)[::-1][:5]:
                twin, similarity = approved[index], float(scores[index])
                if similarity < LOOSE:
                    break
                if compatible(record["body"], twin["body"]):
                    matches.append({
                        "id": record["id"], "name": record.get("name", ""),
                        "downgraded_body": record["body"], "approved_body": twin["body"],
                        "approved_name": twin.get("name", ""), "similarity": round(similarity, 4),
                        "tier": tier_for(similarity),
                        "identical": normalized_content(record["body"]) == normalized_content(twin["body"]),
                    })
                    break
    matches.sort(key=lambda m: -m["similarity"])
    return {"downgraded": len(downgraded), "approved_pool": len(approved),
            "matched": len(matches),
            "tiers": {tier: sum(1 for m in matches if m["tier"] == tier)
                      for tier in ("identical", "near", "loose")},
            "matches": matches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / ".data")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    store = Store(args.data_dir)
    report = find(store, limit=args.limit)
    out = args.out or args.data_dir / "precedents.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "matches"}, indent=2))
    print(f"written to {out}")


if __name__ == "__main__":
    main()
