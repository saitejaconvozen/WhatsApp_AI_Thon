import json
import threading
from collections import Counter
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from .data import (normalized_content, record_content, structural_features,
                   training_records)


LABEL_PROVENANCE = "The platform owner confirmed on 2026-09-18 that metaTemplateCategory is Meta's final category and VERIFIED means successful Meta verification. No independent Meta API check was performed."

# Bumped whenever features or grouping change. Store.revision() only hashes record
# ids, so it cannot notice a pipeline change; without this an artifact trained on
# the old features would be served and audited as if it were current.
FEATURE_VERSION = 2

TARGET_UTILITY_PRECISION = 0.85


def as_text(records):
    return [normalized_content(record_content(r)) for r in records]


def as_numeric(records):
    return np.asarray([structural_features(r) for r in records], dtype=float)


def features():
    """Text only.

    Structural counts (placeholders, buttons, emoji) were measured as a
    FeatureUnion branch here and made every metric worse — TF-IDF already
    encodes them through its `variable` and `number` tokens. The comparison is
    kept as an experiments arm rather than deleted; see docs/experiments.md.
    """
    return Pipeline([
        ("select", FunctionTransformer(as_text)),
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=25000, sublinear_tf=True)),
    ])


def pipeline(calibrated=False):
    classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    if calibrated:
        # class_weight="balanced" skews raw scores, so thresholds are only
        # meaningful after calibration.
        classifier = CalibratedClassifierCV(classifier, method="sigmoid", cv=5)
    return Pipeline([("features", features()), ("classifier", classifier)])


def operating_point(probabilities, truth, target=TARGET_UTILITY_PRECISION, positive="UTILITY"):
    """Lowest threshold whose predictions reach the target precision.

    Ascending order means the first qualifying threshold is the one that keeps
    the most recall. Falls back to 1.0 when no threshold reaches the target,
    which abstains rather than predicting at a precision nobody asked for.
    """
    for threshold in np.unique(np.round(probabilities, 3)):
        predicted = probabilities >= threshold
        if not predicted.any():
            continue
        if float(np.mean(truth[predicted] == positive)) >= target:
            return float(threshold)
    return 1.0


def banded(probabilities, low, high):
    return np.where(probabilities >= high, "UTILITY",
                    np.where(probabilities < low, "MARKETING", "NEEDS_REVIEW"))


def score_rows(rows):
    if not rows:
        return None
    truth = [r["recorded_category"] for r in rows]
    predicted = [r["prediction"] for r in rows]
    hits = sum(1 for t, p in zip(truth, predicted) if t == p)
    utility_predicted = [r for r in rows if r["prediction"] == "UTILITY"]
    utility_actual = [r for r in rows if r["recorded_category"] == "UTILITY"]
    correct_utility = sum(1 for r in utility_predicted if r["recorded_category"] == "UTILITY")
    majority = Counter(truth).most_common(1)[0][1]
    return {
        "samples": len(rows),
        "accuracy": hits / len(rows),
        "majority_accuracy": majority / len(rows),
        "utility_precision": (correct_utility / len(utility_predicted)) if utility_predicted else None,
        "utility_recall": (correct_utility / len(utility_actual)) if utility_actual else None,
        "recorded_utility": len(utility_actual),
    }


def slice_metrics(holdout):
    """Meta never upgrades, so requested-UTILITY is the only slice with a live decision."""
    return {
        "all": score_rows(holdout),
        "requested_utility": score_rows([r for r in holdout if r.get("requested_category") == "UTILITY"]),
        "requested_marketing": score_rows([r for r in holdout if r.get("requested_category") == "MARKETING"]),
    }


def band_metrics(holdout):
    decided = [r for r in holdout if r["band"] != "NEEDS_REVIEW"]
    correct = sum(1 for r in decided if r["band"] == r["recorded_category"])
    utility = [r for r in decided if r["band"] == "UTILITY"]
    correct_utility = sum(1 for r in utility if r["recorded_category"] == "UTILITY")
    return {
        "coverage": (len(decided) / len(holdout)) if holdout else 0.0,
        "decided": len(decided),
        "referred": len(holdout) - len(decided),
        "accuracy_when_decided": (correct / len(decided)) if decided else None,
        "utility_precision_when_decided": (correct_utility / len(utility)) if utility else None,
    }


class Baseline:
    def __init__(self, store):
        self.store = store
        self.lock = threading.Lock()
        self.bundle = None
        path = store.directory / "baseline.joblib"
        if path.exists():
            try:
                # Only this application's generated local artifact is loaded; model uploads are not accepted.
                self.bundle = joblib.load(path)
            except (OSError, ValueError, EOFError, KeyError):
                self.bundle = None

    def status(self):
        if not self.bundle:
            return {"trained": False, "training": self.lock.locked()}
        report = {**self.bundle["report"]}
        report["limitations"] = [LABEL_PROVENANCE if "upstream semantics need confirmation" in item else item
                                 for item in report.get("limitations", [])]
        outdated = self.bundle.get("feature_version") != FEATURE_VERSION
        return {"trained": True, "training": self.lock.locked(),
                "stale": self.bundle["revision"] != self.store.revision() or outdated,
                "outdated_features": outdated, "report": report}

    def train(self):
        if not self.lock.acquire(blocking=False):
            raise ValueError("Training is already running.")
        try:
            revision = self.store.revision()
            records, conflicts = training_records(self.store.records())
            counts = Counter(r["meta_category"] for r in records)
            if len(records) < 30 or min(counts.get("MARKETING", 0), counts.get("UTILITY", 0)) < 5:
                raise ValueError("Training needs at least 30 distinct eligible text families, including 5 in each category.")
            x = records
            y = np.array([r["meta_category"] for r in records])
            groups = np.array([r["family"] for r in records])
            split = GroupShuffleSplit(n_splits=30, test_size=0.25, random_state=42)
            selected = next(((a, b) for a, b in split.split(x, y, groups)
                             if len(set(y[a])) == 2 and len(set(y[b])) == 2), None)
            if selected is None:
                raise ValueError("Could not create a holdout containing both classes. Add more independent examples.")
            train_idx, test_idx = selected

            # Thresholds come from out-of-fold predictions across the whole
            # training set. A single held-back slice was tried first and gave
            # thresholds that missed their precision target on the holdout,
            # because a few hundred validation rows do not pin down the tail of
            # the probability distribution. Tuning on the holdout itself would
            # report an operating point that cannot be reproduced at all.
            folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
            out_of_fold = cross_val_predict(
                pipeline(calibrated=True), [x[i] for i in train_idx], y[train_idx],
                cv=folds, method="predict_proba")
            utility_column = list(np.unique(y[train_idx])).index("UTILITY")
            tuning_probability = out_of_fold[:, utility_column]
            high = operating_point(tuning_probability, y[train_idx])
            low = 1.0 - operating_point(1.0 - tuning_probability, y[train_idx], positive="MARKETING")
            # On well-separated data both sides clear the target with room to
            # spare and the thresholds cross. There is no ambiguous zone to
            # refer, so collapse to a single cut rather than leaving an
            # inverted band that would label every record on both rules.
            low = min(low, high)

            evaluation = pipeline(calibrated=True)
            evaluation.fit([x[i] for i in train_idx], y[train_idx])
            predictions = evaluation.predict([x[i] for i in test_idx])
            probability = evaluation.predict_proba([x[i] for i in test_idx])[
                :, list(evaluation.classes_).index("UTILITY")]
            bands = banded(probability, low, high)
            holdout = [{"id": records[i]["id"], "family": records[i]["family"],
                        "name": records[i]["name"], "body": records[i]["body"],
                        "header": records[i].get("header", ""),
                        "footer": records[i].get("footer", ""),
                        "buttons": records[i].get("buttons", []),
                        "requested_category": records[i].get("requested_category", "UNKNOWN"),
                        "recorded_category": str(y[i]), "prediction": str(predicted),
                        "utility_probability": round(float(p), 4), "band": str(b)}
                       for i, predicted, p, b in zip(test_idx, predictions, probability, bands)]
            labels = ["MARKETING", "UTILITY"]
            precision, recall, f1, support = precision_recall_fscore_support(
                y[test_idx], predictions, labels=labels, zero_division=0)
            majority = Counter(y[train_idx]).most_common(1)[0][0]
            report = {
                "trained_at": datetime.now(timezone.utc).isoformat(), "method": "TF-IDF + logistic regression",
                "split": "25% grouped holdout; one representative per normalized body family",
                "train_families": len(train_idx), "test_families": len(test_idx),
                "total_families": len(records), "excluded_conflicting_families": len(conflicts),
                "accuracy": float(accuracy_score(y[test_idx], predictions)),
                "majority_accuracy": float(np.mean(y[test_idx] == majority)),
                "macro_f1": float(np.mean(f1)), "labels": labels,
                "confusion_matrix": confusion_matrix(y[test_idx], predictions, labels=labels).tolist(),
                "per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]),
                                      "f1": float(f1[i]), "support": int(support[i])} for i, label in enumerate(labels)},
                "predicted_utility_count": int(np.sum(predictions == "UTILITY")),
                "group_overlap": len(set(groups[train_idx]) & set(groups[test_idx])),
                "feature_version": FEATURE_VERSION,
                "thresholds": {"utility": high, "marketing": low,
                               "target_utility_precision": TARGET_UTILITY_PRECISION},
                "slices": slice_metrics(holdout),
                "bands": band_metrics(holdout),
                "limitations": [
                    LABEL_PROVENANCE,
                    "Meta never upgraded a requested MARKETING template to UTILITY in this export, so the requested-UTILITY slice is the only one carrying a real decision.",
                    "Grouping covers header, body, footer and buttons, but may still miss paraphrased duplicates.",
                    "This is a random grouped holdout, not a future-time or external validation set.",
                    "Only TEXT records are used. Images, carousels and other rich formats are excluded.",
                    "Thresholds come from a validation slice of the training data. Probabilities are calibrated against recorded labels, not Meta approval odds.",
                    "After evaluation, the serving model is refitted on all eligible families.",
                ],
            }
            serving = pipeline(calibrated=True)
            serving.fit(x, y)
            text_branch = serving.named_steps["features"]
            bundle = {"pipeline": serving, "records": records,
                      "matrix": text_branch.named_steps["tfidf"].transform(as_text(x)),
                      "text_branch": text_branch, "revision": revision,
                      "feature_version": FEATURE_VERSION, "thresholds": {"utility": high, "marketing": low},
                      "report": report, "holdout": holdout}
            if revision != self.store.revision():
                raise ValueError("The dataset changed during training. Please train again on the new snapshot.")
            temporary = self.store.directory / "baseline.joblib.tmp"
            joblib.dump(bundle, temporary)
            temporary.replace(self.store.directory / "baseline.joblib")
            (self.store.directory / "model-report.json").write_text(json.dumps(report, indent=2))
            self.bundle = bundle
            return self.status()
        finally:
            self.lock.release()

    def predict(self, record):
        bundle = self.bundle
        if not bundle:
            return {"available": False, "reason": "Train a baseline in Model lab to add a dataset-based prediction.", "neighbors": []}
        if bundle["revision"] != self.store.revision():
            return {"available": False, "reason": "The dataset changed. Retrain the baseline before using predictions.", "neighbors": []}
        if record.get("format", "TEXT") != "TEXT":
            return {"available": False, "reason": "This baseline only supports TEXT templates.", "neighbors": []}
        if bundle.get("feature_version") != FEATURE_VERSION:
            return {"available": False, "reason": "This baseline predates the current features. Retrain before using predictions.", "neighbors": []}
        vector = bundle["text_branch"].transform([record])
        if vector.nnz == 0:
            return {"available": False, "reason": "This text has no recognized vocabulary. Human review is needed.", "neighbors": []}
        predicted = str(bundle["pipeline"].predict([record])[0])
        probability = float(bundle["pipeline"].predict_proba([record])[
            0, list(bundle["pipeline"].classes_).index("UTILITY")])
        thresholds = bundle["thresholds"]
        band = str(banded(np.array([probability]), thresholds["marketing"], thresholds["utility"])[0])
        similarity = (bundle["matrix"] @ vector.T).toarray().ravel()
        nearest = np.argsort(similarity)[::-1][:4]
        neighbors = [{"id": bundle["records"][i]["id"], "name": bundle["records"][i]["name"],
                      "body": bundle["records"][i]["body"], "category": bundle["records"][i]["meta_category"],
                      "similarity": round(float(similarity[i]), 3)} for i in nearest if similarity[i] > 0]
        seen = any(normalized_content(r["body"]) == normalized_content(record.get("body", "")) for r in bundle["records"])
        return {"available": True, "category": predicted, "neighbors": neighbors, "seen_family": seen,
                "utility_probability": round(probability, 4), "band": band, "thresholds": thresholds,
                "requested_marketing": record.get("requested_category") == "MARKETING",
                "method": "Local text and structure baseline",
                "limitation": "Prediction from historical labels; not a Meta decision."}
