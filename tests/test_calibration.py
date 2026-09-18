import numpy as np
import pytest

from templatelab.data import Store, family_key, structural_features, training_records, FEATURE_NAMES
from templatelab.model import (FEATURE_VERSION, Baseline, band_metrics, banded, operating_point,
                               slice_metrics)
from test_experiments import dated_records


def test_structural_features_are_fixed_length_and_counted():
    record = {"body": "Hi {{1}}, invoice {{1}} and {{2}} total $40! 🎉 https://x.co",
              "header": "Bill", "footer": "", "buttons": "Pay\nHelp"}
    values = dict(zip(FEATURE_NAMES, structural_features(record)))
    assert len(structural_features(record)) == len(FEATURE_NAMES)
    assert values["placeholder_count"] == 3 and values["distinct_placeholders"] == 2
    assert values["button_count"] == 2 and values["has_header"] == 1 and values["has_footer"] == 0
    assert values["emoji_count"] == 1 and values["exclamation_count"] == 1
    assert values["url_count"] == 1 and values["symbol_count"] == 1
    assert structural_features({"body": ""})[FEATURE_NAMES.index("uppercase_ratio")] == 0


def test_family_key_separates_templates_that_differ_only_in_buttons():
    base = {"body": "Your order {{1}} shipped.", "header": "", "footer": "", "family_id": ""}
    quiet = {**base, "buttons": ""}
    promotional = {**base, "buttons": "Shop more"}
    assert family_key(quiet) != family_key(promotional)
    assert family_key(quiet) == family_key({**quiet})


def test_grouping_keeps_families_the_body_only_key_would_merge():
    records = dated_records()
    for index, record in enumerate(records):
        record["buttons"] = "Shop now" if index % 2 else ""
    clean, _ = training_records(records)
    assert len({r["family"] for r in clean}) == len(clean)
    assert len(clean) > len({r["family"] for r in records if r.get("family")}) / 2


def test_operating_point_picks_the_lowest_threshold_reaching_target():
    probabilities = np.array([0.1, 0.4, 0.6, 0.8, 0.9])
    truth = np.array(["MARKETING", "MARKETING", "UTILITY", "UTILITY", "UTILITY"])
    assert operating_point(probabilities, truth, target=1.0) == pytest.approx(0.6)
    # Unreachable targets abstain rather than predicting at a lower precision.
    assert operating_point(np.array([0.5]), np.array(["MARKETING"]), target=0.9) == 1.0


def test_band_never_labels_inside_the_review_window():
    probabilities = np.array([0.1, 0.5, 0.95])
    assert list(banded(probabilities, 0.4, 0.9)) == ["MARKETING", "NEEDS_REVIEW", "UTILITY"]


def test_band_metrics_count_referrals():
    holdout = [{"band": "UTILITY", "recorded_category": "UTILITY", "prediction": "UTILITY"},
               {"band": "NEEDS_REVIEW", "recorded_category": "MARKETING", "prediction": "UTILITY"},
               {"band": "MARKETING", "recorded_category": "MARKETING", "prediction": "MARKETING"}]
    result = band_metrics(holdout)
    assert result["decided"] == 2 and result["referred"] == 1
    assert result["accuracy_when_decided"] == 1.0
    assert result["utility_precision_when_decided"] == 1.0


def test_slice_metrics_separate_the_trivial_requested_marketing_cases():
    holdout = [{"requested_category": "MARKETING", "recorded_category": "MARKETING",
                "prediction": "MARKETING", "band": "MARKETING"},
               {"requested_category": "UTILITY", "recorded_category": "UTILITY",
                "prediction": "UTILITY", "band": "UTILITY"},
               {"requested_category": "UTILITY", "recorded_category": "MARKETING",
                "prediction": "UTILITY", "band": "UTILITY"}]
    slices = slice_metrics(holdout)
    assert slices["requested_marketing"]["accuracy"] == 1.0
    assert slices["requested_utility"]["samples"] == 2
    assert slices["requested_utility"]["utility_precision"] == 0.5
    assert slices["all"]["samples"] == 3


def test_training_produces_calibrated_probabilities_and_thresholds(tmp_path):
    store = Store(tmp_path)
    store.import_records(dated_records())
    report = Baseline(store).train()["report"]
    assert report["feature_version"] == FEATURE_VERSION
    thresholds = report["thresholds"]
    assert 0.0 <= thresholds["marketing"] <= thresholds["utility"] <= 1.0
    for row in Baseline(store).bundle["holdout"]:
        assert 0.0 <= row["utility_probability"] <= 1.0
        assert row["band"] in {"UTILITY", "MARKETING", "NEEDS_REVIEW"}
        assert "requested_category" in row
    assert report["slices"]["all"]["samples"] == report["test_families"]


def test_old_feature_version_is_stale_and_refuses_prediction(tmp_path):
    store = Store(tmp_path)
    store.import_records(dated_records())
    baseline = Baseline(store)
    baseline.train()
    baseline.bundle["feature_version"] = FEATURE_VERSION - 1
    assert baseline.status()["stale"] is True
    assert baseline.status()["outdated_features"] is True
    assert baseline.predict({"body": "Your invoice is due.", "format": "TEXT"})["available"] is False
