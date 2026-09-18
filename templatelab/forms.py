"""Group Meta's approved utility templates into the forms they actually take.

Precedent matching asks whether a specific approved twin exists, and finds one
for 187 of the 1,180 downgraded templates. This asks a looser and more useful
question: what recurring *shapes* does Meta approve, and how safe is each one?

The answer is that approval is strongly form-dependent. One form -- offering
appointment slots and asking the recipient to confirm a preferred one -- holds
86 approved templates and zero downgraded ones. Another, asking for feedback,
holds 27 approved against 131 downgraded. A template's odds are set as much by
the shape it takes as by any single word in it.

That gives a rewriting target that precedent matching cannot: pour a template's
content into a form Meta approves at a high rate, rather than hunt for a twin.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from .data import Store, training_records
from .policy import REFERENCE, TRANSACTION, promotion_findings

ROOT = Path(__file__).resolve().parent.parent
FORMS = 24


def is_service_message(record):
    """Reports on a transaction the recipient already has, and sells nothing.

    This is the gate that decides whether a downgraded template is worth
    rewriting at all. Templates failing it are advertisements that happen to
    share vocabulary with service messages -- a painting offer sitting in the
    property-listing form -- and no rewrite makes them utility.
    """
    body = record.get("body") or ""
    return (bool(TRANSACTION.search(body)) and bool(REFERENCE.search(body))
            and not promotion_findings({"body": body}))


def label_terms(matrix, names, members, others, count=5):
    inside = np.asarray(matrix[members].mean(0)).ravel()
    outside = np.asarray(matrix[others].mean(0)).ravel()
    return [str(t) for t in names[np.argsort(inside - outside)[::-1][:count]]]


def build(store, encoder_dir, forms=FORMS, seed=0):
    records, _ = training_records(store.records())
    intended = [r for r in records if r.get("requested_category") == "UTILITY"]
    approved = [r for r in intended if r["meta_category"] == "UTILITY"]
    downgraded = [r for r in intended if r["meta_category"] == "MARKETING"]

    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(str(encoder_dir), device="cpu")
    embed = lambda rs: encoder.encode([r["body"] for r in rs], batch_size=64,
                                      normalize_embeddings=True, show_progress_bar=False)
    approved_vectors, downgraded_vectors = embed(approved), embed(downgraded)

    model = KMeans(n_clusters=forms, n_init=10, random_state=seed).fit(approved_vectors)
    assignment = np.argmax(downgraded_vectors @ model.cluster_centers_.T, axis=1)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=8000, stop_words="english")
    matrix = vectorizer.fit_transform([r["body"].lower() for r in approved])
    names = np.array(vectorizer.get_feature_names_out())

    out = []
    for index in range(forms):
        members = np.where(model.labels_ == index)[0]
        landed = [downgraded[i] for i in np.where(assignment == index)[0]]
        if not len(members):
            continue
        service = [r for r in landed if is_service_message(r)]
        rate = len(members) / (len(members) + len(landed)) if landed else 1.0
        out.append({
            "form": index,
            "terms": label_terms(matrix, names, members, np.where(model.labels_ != index)[0]),
            "approved": len(members), "downgraded": len(landed),
            "approval_rate": round(rate, 4),
            "downgraded_service_messages": len(service),
            # Resubmitting a service message unchanged should land near its form's
            # base rate; rewriting it to match that form's approved variants is
            # what the headroom above the floor buys.
            "expected_at_base_rate": round(len(service) * rate, 1),
            "exemplars": [r["body"] for r in [approved[i] for i in members[:3]]],
        })
    # Persist which form each downgraded template landed in. Without this the
    # taxonomy is a report nobody can act on: a rewriter cannot fetch "this
    # template's own form" from a list of cluster summaries.
    assignments = {downgraded[i]["id"]: int(form) for i, form in enumerate(assignment)}
    out.sort(key=lambda f: -f["approval_rate"])

    service_total = sum(f["downgraded_service_messages"] for f in out)
    floor = sum(f["expected_at_base_rate"] for f in out)
    return {
        "downgraded": len(downgraded), "approved": len(approved),
        "service_messages": service_total,
        "floor": round(floor), "floor_share": round(floor / len(downgraded), 4),
        "ceiling": service_total, "ceiling_share": round(service_total / len(downgraded), 4),
        "forms": out,
        "assignments": assignments,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / ".data")
    parser.add_argument("--encoder", type=Path, default=None)
    parser.add_argument("--forms", type=int, default=FORMS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    store = Store(args.data_dir)
    report = build(store, args.encoder or args.data_dir / "improvement" / "encoder-selected", args.forms)
    out = args.out or args.data_dir / "forms.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "forms"}, indent=2))
    for f in report["forms"][:5] + report["forms"][-3:]:
        print(f"  {f['approval_rate']:>5.0%}  {f['approved']:>4} approved / {f['downgraded']:>4} down"
              f"  {', '.join(f['terms'])[:46]}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
