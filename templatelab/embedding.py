"""A utility direction in embedding space, and a rewriter that optimises along it.

The idea this implements: embed templates so utility and marketing separate, take
the direction between them, and use displacement along that direction to steer a
rewrite.

One part of that idea cannot be built as stated. Sentence embeddings have no
inverse — there is no decoder from a vector back to text, because the encoder
discards everything needed to reconstruct it. So the direction is not used to
*generate* text. It is used as the objective and the verifier: a language model
proposes candidate rewrites, each candidate is embedded, and the one that moves
furthest along the utility direction while keeping the transactional anchor wins.
That closes the same loop without needing a decoder that does not exist.

The direction is fitted with Linear Discriminant Analysis rather than a centroid
difference. A centroid difference points at whatever separates the class means,
including variance directions that carry no label information; LDA divides by the
within-class scatter, so it points at what actually discriminates.

There are no observed rewrite pairs in this corpus — 2,387 templates appear in
several versions, only 5 changed category, and none of those changed their text —
so the direction cannot be learned from real edits. It is class geometry, and it
describes what utility text looks like on average, not what any specific template
needs.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import record_content, training_records

ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MIN_PER_CLASS = 20


class Unavailable(RuntimeError):
    """The encoder is not installed or cannot load; reported, never raised at a user."""


def load_encoder(name=ENCODER_NAME):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise Unavailable("sentence-transformers is not installed.") from exc
    try:
        return SentenceTransformer(name)
    except Exception as exc:
        raise Unavailable(f"Could not load {name}: {exc}") from exc


def encode(texts, encoder=None, batch_size=64):
    encoder = encoder or load_encoder()
    vectors = np.asarray(encoder.encode(list(texts), batch_size=batch_size,
                                        show_progress_bar=False, normalize_embeddings=True))
    return vectors.astype(np.float64)


@dataclass
class Direction:
    """A fitted utility direction plus the scale needed to read scores off it."""

    vector: np.ndarray
    utility_mean: float
    marketing_mean: float
    encoder_name: str
    train_size: int

    def score(self, vectors):
        """Signed position along the direction; higher is more utility-like."""
        raw = np.atleast_2d(vectors) @ self.vector
        spread = self.utility_mean - self.marketing_mean
        if abs(spread) < 1e-12:
            return raw
        # Rescaled so 0 sits at the marketing mean and 1 at the utility mean, which
        # makes a displacement readable as "fraction of the gap closed".
        return (raw - self.marketing_mean) / spread

    def separation(self, vectors, labels):
        """Cohen's d between the classes along this direction."""
        scores = self.score(vectors)
        utility = scores[np.asarray(labels) == "UTILITY"]
        marketing = scores[np.asarray(labels) == "MARKETING"]
        if len(utility) < 2 or len(marketing) < 2:
            return 0.0
        pooled = np.sqrt((utility.var(ddof=1) + marketing.var(ddof=1)) / 2)
        return float((utility.mean() - marketing.mean()) / pooled) if pooled > 1e-12 else 0.0

    def to_dict(self):
        return {"vector": self.vector.tolist(), "utility_mean": self.utility_mean,
                "marketing_mean": self.marketing_mean, "encoder_name": self.encoder_name,
                "train_size": self.train_size}

    @classmethod
    def from_dict(cls, payload):
        return cls(np.asarray(payload["vector"], dtype=float), payload["utility_mean"],
                   payload["marketing_mean"], payload["encoder_name"], payload["train_size"])


def fit_direction(vectors, labels, encoder_name=ENCODER_NAME, shrinkage=1e-3):
    """LDA direction: inv(within-class scatter) @ (mean difference)."""
    vectors = np.asarray(vectors, dtype=float)
    labels = np.asarray(labels)
    utility, marketing = vectors[labels == "UTILITY"], vectors[labels == "MARKETING"]
    if len(utility) < MIN_PER_CLASS or len(marketing) < MIN_PER_CLASS:
        raise Unavailable(f"Need at least {MIN_PER_CLASS} examples of each category to fit a direction.")
    difference = utility.mean(axis=0) - marketing.mean(axis=0)
    centred = np.vstack([utility - utility.mean(axis=0), marketing - marketing.mean(axis=0)])
    scatter = centred.T @ centred / max(1, len(centred) - 2)
    # Shrinkage keeps the solve stable when dimensions outnumber examples.
    scatter.flat[:: scatter.shape[0] + 1] += shrinkage * np.trace(scatter) / scatter.shape[0]
    vector = np.linalg.solve(scatter, difference)
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        raise Unavailable("The fitted direction is degenerate.")
    vector = vector / norm
    return Direction(vector, float((utility @ vector).mean()), float((marketing @ vector).mean()),
                     encoder_name, len(vectors))


def build(store, encoder=None, records=None):
    """Fit a direction over every eligible training family."""
    records = records if records is not None else training_records(store.records())[0]
    if not records:
        raise Unavailable("No eligible training families.")
    encoder = encoder or load_encoder()
    vectors = encode([record_content(r) for r in records], encoder)
    labels = [r["meta_category"] for r in records]
    direction = fit_direction(vectors, labels, getattr(encoder, "_model_name", ENCODER_NAME))
    return direction, vectors, np.asarray(labels), records


def save(direction, path):
    Path(path).write_text(json.dumps(direction.to_dict()), encoding="utf-8")


def load(path):
    path = Path(path)
    if not path.exists():
        raise Unavailable("No direction has been fitted yet.")
    return Direction.from_dict(json.loads(path.read_text(encoding="utf-8")))


def displacement(before, after, direction):
    """How far a rewrite moved along the utility direction, in gap-fractions."""
    # Rounded first so the three numbers a reader sees actually add up.
    start = round(float(direction.score(before)[0]), 4)
    end = round(float(direction.score(after)[0]), 4)
    return {"before": start, "after": end, "moved": round(end - start, 4)}


def evaluate(store, encoder=None, seed=42):
    """Does the embedding space beat the TF-IDF baseline on the same split?

    Uses the same GroupShuffleSplit settings as Baseline.train, so the numbers
    line up with the model report rather than describing a different holdout.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit

    from .model import as_text, pipeline

    records = training_records(store.records())[0]
    encoder = encoder or load_encoder()
    vectors = encode([record_content(r) for r in records], encoder)
    labels = np.array([r["meta_category"] for r in records])
    groups = np.array([r["family"] for r in records])

    splitter = GroupShuffleSplit(n_splits=30, test_size=0.25, random_state=seed)
    train_idx, test_idx = next((a, b) for a, b in splitter.split(vectors, labels, groups)
                               if len(set(labels[a])) == 2 and len(set(labels[b])) == 2)
    requested = np.array([r.get("requested_category", "UNKNOWN") for r in records])

    def scores(predicted, mask):
        truth, guess = labels[test_idx][mask], predicted[mask]
        utility = guess == "UTILITY"
        hit = (guess == "UTILITY") & (truth == "UTILITY")
        return {"samples": int(mask.sum()),
                "accuracy": float(np.mean(truth == guess)),
                "utility_precision": float(hit.sum() / utility.sum()) if utility.any() else None,
                "utility_recall": float(hit.sum() / (truth == "UTILITY").sum()) if (truth == "UTILITY").any() else None}

    results = {}
    embedded = CalibratedClassifierCV(
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        method="sigmoid", cv=5)
    embedded.fit(vectors[train_idx], labels[train_idx])
    results["embedding"] = embedded.predict(vectors[test_idx])

    lexical = pipeline(calibrated=True)
    lexical.fit([records[i] for i in train_idx], labels[train_idx])
    results["tfidf"] = lexical.predict([records[i] for i in test_idx])

    direction = fit_direction(vectors[train_idx], labels[train_idx],
                              getattr(encoder, "_model_name", ENCODER_NAME))
    projected = direction.score(vectors[test_idx]).ravel()
    results["direction"] = np.where(projected >= 0.5, "UTILITY", "MARKETING")

    live = requested[test_idx] == "UTILITY"
    report = {
        "encoder": direction.encoder_name,
        "families": len(records), "train": len(train_idx), "test": len(test_idx),
        "separation_cohens_d": round(direction.separation(vectors[test_idx], labels[test_idx]), 3),
        "direction_auc": round(float(roc_auc_score(labels[test_idx] == "UTILITY", projected)), 4),
        "methods": {name: {"all": scores(predicted, np.ones(len(test_idx), bool)),
                           "requested_utility": scores(predicted, live)}
                    for name, predicted in results.items()},
    }
    return report, direction


def main():
    import argparse
    from .data import Store

    parser = argparse.ArgumentParser(description="Fit and evaluate the utility direction.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / ".data")
    parser.add_argument("--encoder", default=ENCODER_NAME)
    args = parser.parse_args()
    store = Store(args.data_dir)
    try:
        report, direction = evaluate(store, load_encoder(args.encoder))
    except Unavailable as exc:
        raise SystemExit(str(exc))
    save(direction, args.data_dir / "utility-direction.json")
    print(f"encoder {report['encoder']}  families {report['families']}  "
          f"train/test {report['train']}/{report['test']}")
    print(f"direction separation (Cohen's d) {report['separation_cohens_d']}   AUC {report['direction_auc']}")
    for slice_name in ("requested_utility", "all"):
        print(f"\n{slice_name} (n={report['methods']['tfidf'][slice_name]['samples']})")
        print(f"{'method':<14}{'accuracy':>10}{'utilP':>9}{'utilR':>9}")
        for name in ("tfidf", "embedding", "direction"):
            m = report["methods"][name][slice_name]
            up = f"{m['utility_precision']*100:8.1f}%" if m["utility_precision"] is not None else "     n/a"
            ur = f"{m['utility_recall']*100:8.1f}%" if m["utility_recall"] is not None else "     n/a"
            print(f"  {name:<12}{m['accuracy']*100:9.1f}%{up}{ur}")
    (args.data_dir / "embedding-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()


def rank_candidates(original, candidates, direction, encoder=None, keeps_anchor=None):
    """Order rewrites by how far they move along the utility direction.

    Movement alone is not enough to accept one: the direction points at what
    utility text looks like on average, and the shortest path there is often to
    delete the transactional detail that makes the message worth sending. So a
    candidate that drops the anchor is ranked but marked ineligible, and the
    caller picks from eligible candidates only.
    """
    encoder = encoder or load_encoder()
    texts = [original] + list(candidates)
    vectors = encode(texts, encoder)
    scores = direction.score(vectors).ravel()
    base = float(scores[0])
    ranked = []
    for index, candidate in enumerate(candidates, start=1):
        eligible = True if keeps_anchor is None else bool(keeps_anchor(candidate))
        ranked.append({"text": candidate, "score": round(float(scores[index]), 4),
                       "moved": round(float(scores[index]) - base, 4), "eligible": eligible})
    ranked.sort(key=lambda row: (row["eligible"], row["moved"]), reverse=True)
    return {"base_score": round(base, 4), "candidates": ranked}
