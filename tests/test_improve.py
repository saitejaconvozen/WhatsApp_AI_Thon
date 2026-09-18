from copy import deepcopy

import pytest

from templatelab.improve import candidate, fingerprint, fixed_split, metrics
from templatelab.finetune import select_blend
import numpy as np


def bundle():
    records = [{"id": str(i), "family": str(i), "meta_category": "UTILITY" if i % 2 else "MARKETING",
                "requested_category": "UTILITY", "body": "Invoice due" if i % 2 else "Buy now"} for i in range(80)]
    return {"records": records, "holdout": records[-20:]}


def test_fixed_partitions_are_complete_reproducible_and_family_disjoint(tmp_path):
    first = fixed_split(bundle(), tmp_path)
    second = fixed_split(bundle(), tmp_path)
    assert [{r["id"] for r in part} for part in first[:3]] == [{r["id"] for r in part} for part in second[:3]]
    assert sum(len(part) for part in first[:3]) == 80
    assert len(first[2]) == 20
    assert not ({r["family"] for r in first[0]} & {r["family"] for r in first[1]})


def test_changed_text_or_label_invalidates_experiment(tmp_path):
    data = bundle()
    fixed_split(data, tmp_path)
    changed = deepcopy(data)
    changed["records"][0]["body"] = "Different"
    assert fingerprint(data["records"]) != fingerprint(changed["records"])
    with pytest.raises(ValueError, match="changed"):
        fixed_split(changed, tmp_path)


def test_metrics_count_review_and_failures_as_incorrect():
    records = bundle()["records"][:4]
    report = metrics(records, ["MARKETING", "NEEDS_REVIEW", "ERROR", "UTILITY"])
    assert report["all"]["accuracy"] == .5
    with pytest.raises(ValueError):
        metrics(records, ["MARKETING"])


def test_candidate_uses_content_not_recorded_or_requested_category():
    records = bundle()["records"]
    model = candidate({"features": "combined", "normalize": False, "classifier": "svm", "C": 1, "class_weight": None})
    model.fit(records, [r["meta_category"] for r in records])
    altered = [{**r, "meta_category": "WRONG", "requested_category": "WRONG"} for r in records]
    assert list(model.predict(records)) == list(model.predict(altered))


def test_blend_selection_does_not_require_test_labels():
    records = bundle()["records"][:4]
    neural = np.array([.1, .9, .1, .9])
    lexical = np.array([.9, .1, .9, .1])
    choice = select_blend(records, neural, lexical)
    assert choice["validation"]["requested_utility"]["accuracy"] == 1
    assert choice["neural_weight"] > .5
