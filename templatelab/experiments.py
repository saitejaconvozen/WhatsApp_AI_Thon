"""Offline chronological comparisons; never replaces the serving model."""

import argparse
import json
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler, normalize

from .data import (Store, candidate, family_key, normalized_content, record_content,
                   structural_features, training_records)


def date_value(value):
    try:
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    except (ValueError, TypeError, AttributeError):
        return None


def chronological_split(records):
    clean, _ = training_records(records)
    eligible = {r["family"]: r for r in clean}
    spans, invalid = defaultdict(list), set()
    for row in records:
        key = family_key(row)
        if candidate(row) and key in eligible:
            created = date_value(row["created_at"])
            updated = date_value(row["updated_at"]) if row["updated_at"] else created
            if created is None or updated is None or updated < created:
                invalid.add(key)
            else:
                spans[key].extend([created, updated])
    spans = {key: values for key, values in spans.items() if key not in invalid}
    if len(spans) < 30:
        raise ValueError("Chronological evaluation needs at least 30 clean families with valid dates.")
    starts = sorted(min(values) for values in spans.values())
    cutoff = starts[int(len(starts) * .75)]
    train = [eligible[key] for key, dates in spans.items() if max(dates) < cutoff]
    test = [eligible[key] for key, dates in spans.items() if min(dates) >= cutoff]
    if min(len(train), len(test)) < 5 or any(len({r["meta_category"] for r in part}) < 2 for part in (train, test)):
        raise ValueError("The fixed date cutoff did not yield enough families and both classes on each side.")
    return train, test, {"cutoff": cutoff.isoformat(), "train_families": len(train),
                        "test_families": len(test), "excluded_invalid_date_families": len(invalid),
                        "purged_boundary_families": len(spans) - len(train) - len(test),
                        "group_overlap": len({r["family"] for r in train} & {r["family"] for r in test})}


def metrics(actual, predicted):
    labels = ["MARKETING", "UTILITY"]
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=labels, zero_division=0)
    return {"accuracy": float(accuracy_score(actual, predicted)), "macro_f1": float(np.mean(f1)),
            "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
            "utility_precision": float(precision[1]), "utility_recall": float(recall[1]),
            "utility_support": int(support[1]), "predicted_utility": int(sum(p == "UTILITY" for p in predicted)),
            "samples": len(actual)}


METHODS = ("Word TF-IDF", "Character TF-IDF", "Latent semantic (LSA)",
           "Word TF-IDF + structure", "Sentence embeddings")

ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Unavailable(RuntimeError):
    """A comparison arm that cannot run here; recorded rather than raised to the caller."""


def encode(texts):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise Unavailable(
            "sentence-transformers is not installed. Install it to run the neural comparison; "
            "the encoder downloads once and then runs locally."
        ) from exc
    try:
        model = SentenceTransformer(ENCODER_NAME)
    except Exception as exc:  # offline, missing cache, corrupt download
        raise Unavailable(f"Could not load {ENCODER_NAME} locally: {exc}") from exc
    return normalize(np.asarray(model.encode(list(texts), batch_size=32, show_progress_bar=False)))


def build_features(method, train, test, train_text, test_text):
    if method == "Sentence embeddings":
        return encode(train_text), encode(test_text), None
    vectorizer = TfidfVectorizer(ngram_range=(3, 5) if method == "Character TF-IDF" else (1, 2),
                                 analyzer="char_wb" if method == "Character TF-IDF" else "word",
                                 max_features=25000, sublinear_tf=True)
    x_train = vectorizer.fit_transform(train_text)
    x_test = vectorizer.transform(test_text)
    if method == "Latent semantic (LSA)":
        dimensions = min(100, x_train.shape[0] - 1, x_train.shape[1] - 1)
        if dimensions < 2:
            raise Unavailable("Not enough vocabulary for latent-semantic comparison.")
        svd = TruncatedSVD(n_components=dimensions, random_state=42)
        return normalize(svd.fit_transform(x_train)), normalize(svd.transform(x_test)), dimensions
    if method == "Word TF-IDF + structure":
        # Measured as worse than text alone; kept as a reproducible arm so the
        # claim in docs/experiments.md can be rechecked rather than trusted.
        scaler = StandardScaler()
        train_numeric = scaler.fit_transform([structural_features(r) for r in train])
        test_numeric = scaler.transform([structural_features(r) for r in test])
        return (hstack([x_train, csr_matrix(train_numeric)]).tocsr(),
                hstack([x_test, csr_matrix(test_numeric)]).tocsr(), None)
    return x_train, x_test, None


def compare(records):
    train, test, split = chronological_split(records)
    train_text = [normalized_content(record_content(r)) for r in train]
    test_text = [normalized_content(record_content(r)) for r in test]
    y_train, y_test = [r["meta_category"] for r in train], np.array([r["meta_category"] for r in test])
    # Similarity is fitted on training bodies, and processed in batches to bound memory.
    near_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=25000)
    train_bodies = near_vectorizer.fit_transform([normalized_content(r["body"]) for r in train])
    test_bodies = near_vectorizer.transform([normalized_content(r["body"]) for r in test])
    neighbors, scores = [], []
    for start in range(0, len(test), 128):
        similarities = (test_bodies[start:start + 128] @ train_bodies.T).toarray()
        neighbors.extend(similarities.argmax(axis=1).tolist())
        scores.extend(similarities.max(axis=1).tolist())
    threshold = .90
    distinct = np.array(scores) < threshold
    pairs = [{"test_id": test[i]["id"], "train_id": train[neighbors[i]]["id"],
              "similarity": round(score, 4)} for i, score in enumerate(scores) if score >= threshold]
    results, skipped = [], []
    for method in METHODS:
        try:
            x_train, x_test, dimensions = build_features(method, train, test, train_text, test_text)
        except Unavailable as exc:
            skipped.append({"method": method, "reason": str(exc)})
            continue
        classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        classifier.fit(x_train, y_train)
        predictions = classifier.predict(x_test)
        results.append({"method": method, "dimensions": dimensions,
                        "all": metrics(y_test, predictions),
                        "lower_overlap": metrics(y_test[distinct], predictions[distinct]) if any(distinct) else None})
    majority = max(set(y_train), key=y_train.count)
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "split": split,
            "majority_accuracy": float(np.mean(y_test == majority)), "results": results,
            "skipped": skipped,
            "near_duplicates": {"method": "Training-fitted character TF-IDF cosine on normalized bodies",
                                "threshold": threshold, "flagged_test_families": len(pairs), "pairs": pairs},
            "limitations": [
                "Dates are template creation/update proxies, not Meta decision timestamps. Historical label availability cannot be established from this snapshot.",
                "The cutoff is the 75th percentile of family first dates. Families spanning the cutoff are purged, including later updates.",
                "Lower-overlap scores exclude test bodies with cosine similarity >= 0.90 to a training body; this threshold was fixed before evaluation, not tuned on labels.",
                "Lexical overlap screening does not detect every paraphrase. Lower-overlap results are not an independent final test set.",
                "LSA is a local low-rank text representation, not pretrained neural sentence embeddings or an LLM.",
                "Comparisons are exploratory. No winner is promoted and no automatic submissions are enabled.",
            ]}


class Experiments:
    def __init__(self, store):
        self.store, self.lock = store, threading.Lock()
        self.path = store.directory / "experiments.json"

    def status(self):
        if not self.path.exists():
            return {"available": False, "running": self.lock.locked()}
        report = json.loads(self.path.read_text())
        return {"available": True, "running": self.lock.locked(),
                "stale": report["dataset_revision"] != self.store.revision(), "report": report}

    def run(self):
        if not self.lock.acquire(blocking=False):
            raise ValueError("An experiment is already running.")
        try:
            revision = self.store.revision()
            report = compare(self.store.records())
            if revision != self.store.revision():
                raise ValueError("Dataset changed during evaluation. Run again on the new snapshot.")
            report["dataset_revision"] = revision
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
            temporary.replace(self.path)
            return self.status()
        finally:
            self.lock.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / ".data")
    args = parser.parse_args()
    result = Experiments(Store(args.data_dir)).run()
    print(json.dumps({"split": result["report"]["split"], "results": result["report"]["results"]}, indent=2))
