"""Validation-only comparison of specialized text and supervised encoder scores."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV

from .data import Store
from .improve import candidate, fixed_split, metrics, ranking
from .model import Baseline
from .requested import conditional_predictions


WEIGHTS = (0., .25, .5, .75, 1.)
THRESHOLDS = (.35, .4, .45, .5, .55, .6, .65)


def compare(validation, text_probability, neural_probability):
    if len(text_probability) != len(validation) or len(neural_probability) != len(validation):
        raise ValueError("Every validation record needs both scores.")
    results = []
    truth = np.array([r["meta_category"] for r in validation])
    utility = np.array([r.get("requested_category") == "UTILITY" for r in validation])
    text_label = text_probability >= .5
    neural_label = neural_probability >= .5
    for weight in WEIGHTS:
        blended = (1 - weight) * text_probability + weight * neural_probability
        for threshold in THRESHOLDS:
            predicted = np.where(blended >= threshold, "UTILITY", "MARKETING")
            scores = metrics(validation, conditional_predictions(validation, predicted))
            results.append({"neural_weight": weight, "threshold": threshold, "validation": scores})
    best = max(results, key=lambda r: ranking(r["validation"]))
    return {"selected": best, "results": results,
            "disagreements_on_requested_utility": int(np.sum(utility & (text_label != neural_label))),
            "oracle_accuracy_on_requested_utility": float(np.mean(
                (text_label[utility] == (truth[utility] == "UTILITY")) |
                (neural_label[utility] == (truth[utility] == "UTILITY"))))}


def run(store):
    baseline = Baseline(store)
    if not baseline.status().get("trained") or baseline.status().get("stale"):
        raise ValueError("A current baseline is required.")
    directory = store.directory / "improvement"
    path = directory / "requested-blend.json"
    if path.exists():
        raise ValueError("Blend experiment already exists; inspect its report.")
    train, validation, test, manifest = fixed_split(baseline.bundle, directory)
    selected = joblib.load(directory / "requested-validation.joblib")
    if selected["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Specialized model belongs to another dataset revision.")
    target_train = [r for r in train if r.get("requested_category") == "UTILITY"]
    calibrated = CalibratedClassifierCV(candidate(selected["config"]), method="sigmoid", cv=3)
    calibrated.fit(target_train, [r["meta_category"] for r in target_train])
    text_probability = calibrated.predict_proba(validation)[:, list(calibrated.classes_).index("UTILITY")]
    saved = np.load(directory / "validation-probabilities.npz")
    if list(saved["ids"]) != [r["id"] for r in validation]:
        raise ValueError("Neural probabilities are not aligned with the fixed validation records.")
    neural_probability = saved["neural"]
    compared = compare(validation, text_probability, neural_probability)
    report = {"fingerprint": manifest["fingerprint"], "train": len(train),
              "validation": len(validation), "test": len(test), "complete": True,
              "test_evaluated": False, "source_models": ["Requested-UTILITY text SVM, calibrated on train only",
                                                      "Supervised encoder selected on validation epoch 3"],
              "limitations": ["Oracle accuracy is a diagnostic upper bound, not a usable model.",
                              "35 blend settings were selected on validation and require independent testing."],
              **compared}
    path.write_text(json.dumps(report, indent=2))
    joblib.dump({"model": calibrated, "fingerprint": manifest["fingerprint"]},
                directory / "requested-calibrated.joblib")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / ".data")
    args = parser.parse_args()
    run(Store(args.data_dir))


if __name__ == "__main__":
    main()
