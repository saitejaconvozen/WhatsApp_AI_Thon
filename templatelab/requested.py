"""Conditional classifier for drafts submitted as UTILITY.

The requested category exists before Meta's decision. The conditional rule is
only applicable when the sender actually supplies this field; it is never
inferred from the text or filled from the final Meta label.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from .data import Store
from .improve import candidate, fixed_split, metrics, ranking
from .model import Baseline


def conditional_predictions(records, utility_predictions):
    """Use the specialized model only for records submitted as UTILITY."""
    if len(records) != len(utility_predictions):
        raise ValueError("Every validation record must have a prediction.")
    return np.where(np.array([r.get("requested_category") == "MARKETING" for r in records]),
                    "MARKETING", utility_predictions)


def configurations():
    for features in ("word", "char", "combined"):
        for normalize in (True, False):
            for c in (.3, 1., 3.):
                for classifier in ("svm", "logistic"):
                    for class_weight in (None, "balanced"):
                        yield {"features": features, "normalize": normalize, "C": c,
                               "classifier": classifier, "class_weight": class_weight}


def run(store):
    baseline = Baseline(store)
    if not baseline.status().get("trained") or baseline.status().get("stale"):
        raise ValueError("A current baseline with a persisted holdout is required.")
    directory = store.directory / "improvement"
    train, validation, test, manifest = fixed_split(baseline.bundle, directory)
    path = directory / "requested-validation.json"
    if path.exists():
        raise ValueError("Conditional experiment already exists; inspect it before another run.")
    target_train = [r for r in train if r.get("requested_category") == "UTILITY"]
    label_counts = {label: sum(r["meta_category"] == label for r in target_train)
                    for label in ("MARKETING", "UTILITY")}
    if min(label_counts.values()) < 20:
        raise ValueError("Insufficient examples in the requested-UTILITY training group.")
    report = {"fingerprint": manifest["fingerprint"], "train": len(train),
              "specialized_train": len(target_train), "validation": len(validation), "test": len(test),
              "test_evaluated": False, "complete": False, "results": [],
              "input_contract": "requested_category must be supplied before Meta's decision",
              "baseline_validation": metrics(validation,
                  joblib.load(directory / "lexical-validation.joblib")["model"].predict(validation))}
    best, best_model = None, None
    for config in configurations():
        model = candidate(config).fit(target_train, [r["meta_category"] for r in target_train])
        pred = conditional_predictions(validation, model.predict(validation))
        scores = metrics(validation, pred)
        row = {"config": config, "validation": scores}
        report["results"].append(row)
        if best is None or ranking(scores) > ranking(best["validation"]):
            best, best_model = row, model
            print("New best: " + json.dumps(row), flush=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(path)
    joblib.dump({"model": best_model, "config": best["config"],
                "fingerprint": manifest["fingerprint"]}, directory / "requested-validation.joblib")
    report["selected"] = best
    report["complete"] = True
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(path)
    print("Selected on validation only: " + json.dumps(best), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / ".data")
    args = parser.parse_args()
    run(Store(args.data_dir))


if __name__ == "__main__":
    main()
