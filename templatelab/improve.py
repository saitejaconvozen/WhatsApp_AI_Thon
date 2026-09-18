"""Validation-selected text models; the persisted historical holdout stays fixed."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from .data import Store, normalized_content, record_content
from .model import Baseline, score_rows


def texts(records, normalize=False):
    values = [record_content(r) for r in records]
    return [normalized_content(v) for v in values] if normalize else values


def fingerprint(records):
    fields = ("id", "family", "meta_category", "requested_category", "header", "body", "footer", "buttons")
    payload = [{key: r.get(key) for key in fields} for r in sorted(records, key=lambda r: r["id"])]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def fixed_split(bundle, directory):
    records = sorted(bundle["records"], key=lambda r: r["id"])
    test_families = {r["family"] for r in bundle["holdout"]}
    development = [r for r in records if r["family"] not in test_families]
    test = [r for r in records if r["family"] in test_families]
    path = directory / "improvement-split.json"
    digest = fingerprint(records)
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["fingerprint"] != digest or set(manifest["test_ids"]) != {r["id"] for r in test}:
            raise ValueError("Dataset or holdout changed. Preserve the previous experiment before creating a new protocol.")
        train_ids, validation_ids = set(manifest["train_ids"]), set(manifest["validation_ids"])
        train = [r for r in development if r["id"] in train_ids]
        validation = [r for r in development if r["id"] in validation_ids]
    else:
        strata = [r["meta_category"] + ":" + r.get("requested_category", "UNKNOWN") for r in development]
        # Rare requested-category strata cannot support a stratified split.
        from collections import Counter
        if min(Counter(strata).values()) < 2:
            strata = [r["meta_category"] for r in development]
        train, validation = train_test_split(development, test_size=.25, random_state=20260918, stratify=strata)
        manifest = {"fingerprint": digest, "seed": 20260918,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "train_ids": [r["id"] for r in train], "validation_ids": [r["id"] for r in validation],
                    "test_ids": [r["id"] for r in test],
                    "selection_metric": "requested-UTILITY validation accuracy; all-validation accuracy breaks ties",
                    "test_status": "Previously inspected historical holdout; not an untouched production test"}
        path.write_text(json.dumps(manifest, indent=2))
    partitions = [train, validation, test]
    if any(not part for part in partitions):
        raise ValueError("Every partition must contain records.")
    groups = [{r["family"] for r in part} for part in partitions]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError("Template families overlap between partitions.")
    if sum(map(len, partitions)) != len(records):
        raise ValueError("Split does not cover the dataset exactly once.")
    return train, validation, test, manifest


def metrics(records, predicted):
    rows = [{"recorded_category": r["meta_category"], "prediction": str(p),
             "requested_category": r.get("requested_category")} for r, p in zip(records, predicted)]
    if len(rows) != len(records) or len(predicted) != len(records):
        raise ValueError("Every record must have exactly one prediction.")
    return {"all": score_rows(rows),
            "requested_utility": score_rows([r for r in rows if r["requested_category"] == "UTILITY"])}


def ranking(scores):
    return scores["requested_utility"]["accuracy"], scores["all"]["accuracy"]


def summary(store):
    directory = store.directory / "improvement"
    result = {"available": False, "stages": []}
    for filename, name in (("lexical-validation.json", "Text model search"),
                           ("neural-validation.json", "Neural fine-tuning"),
                           ("requested-validation.json", "Requested-category experiment"),
                           ("requested-blend.json", "Neural and text combination")):
        path = directory / filename
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        selected = report.get("selected") or {}
        result["available"] = True
        result["stages"].append({"name": name, "complete": report["complete"],
                                 "train": report["train"], "validation": report["validation"],
                                 "test": report["test"], "test_evaluated": report["test_evaluated"],
                                 "candidates": len(report.get("results", report.get("epochs", []))),
                                 "selected_validation": selected.get("validation"),
                                 "selected_epoch": selected.get("epoch")})
    return result


def candidate(config):
    # Keep the transformer importable when this module is run with `python -m`.
    from .improve import texts as text_transform
    word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=50000)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
                           min_df=2, max_features=80000)
    vectorizer = {"word": word, "char": char,
                  "combined": FeatureUnion([("word", word), ("char", char)])}[config["features"]]
    if config["classifier"] == "svm":
        classifier = LinearSVC(C=config["C"], class_weight=config["class_weight"], random_state=42, max_iter=5000)
    else:
        classifier = LogisticRegression(C=config["C"], class_weight=config["class_weight"],
                                        solver="liblinear", max_iter=2000, random_state=42)
    return Pipeline([("text", FunctionTransformer(text_transform, kw_args={"normalize": config["normalize"]})),
                     ("features", vectorizer), ("classifier", classifier)])


def configurations():
    for features in ("word", "char", "combined"):
        for normalize in (True, False):
            for c in (.3, 1., 3.):
                for classifier in ("svm", "logistic"):
                    yield {"features": features, "normalize": normalize, "C": c,
                           "classifier": classifier, "class_weight": None}


def run(store):
    baseline = Baseline(store)
    status = baseline.status()
    if not status.get("trained") or status.get("stale"):
        raise ValueError("A current baseline and its persisted holdout are required.")
    directory = store.directory / "improvement"
    directory.mkdir(exist_ok=True)
    train, validation, test, manifest = fixed_split(baseline.bundle, directory)
    path = directory / "lexical-validation.json"
    if path.exists():
        raise ValueError("This experiment already exists; inspect its evidence instead of overwriting it.")
    report = {"fingerprint": manifest["fingerprint"], "train": len(train), "validation": len(validation),
              "test": len(test), "complete": False, "test_evaluated": False, "results": []}
    best, best_model = None, None
    for config in configurations():
        model = candidate(config)
        model.fit(train, [r["meta_category"] for r in train])
        scores = metrics(validation, model.predict(validation))
        row = {"config": config, "validation": scores}
        report["results"].append(row)
        if best is None or ranking(scores) > ranking(best["validation"]):
            best, best_model = row, model
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(path)
        print(json.dumps(row), flush=True)
    report["selected"] = best
    report["complete"] = True
    joblib.dump({"model": best_model, "config": best["config"], "fingerprint": manifest["fingerprint"]},
                directory / "lexical-validation.joblib")
    path.write_text(json.dumps(report, indent=2))
    print("Selected on validation only: " + json.dumps(best), flush=True)
    return report


def rebuild_selected(store):
    """Recreate only the selected validation artifact, without another search."""
    directory = store.directory / "improvement"
    report = json.loads((directory / "lexical-validation.json").read_text())
    train, validation, _, manifest = fixed_split(Baseline(store).bundle, directory)
    if not report["complete"] or report["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("A complete, matching selection report is required.")
    config = report["selected"]["config"]
    model = candidate(config).fit(train, [r["meta_category"] for r in train])
    if metrics(validation, model.predict(validation)) != report["selected"]["validation"]:
        raise ValueError("Rebuilt model did not reproduce its validation scores.")
    joblib.dump({"model": model, "config": config, "fingerprint": manifest["fingerprint"]},
                directory / "lexical-validation.joblib")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / ".data")
    args = parser.parse_args()
    run(Store(args.data_dir))


if __name__ == "__main__":
    main()
