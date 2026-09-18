from types import SimpleNamespace

import pytest

from templatelab.audit import error_report


def baseline(bundle, **status):
    return SimpleNamespace(bundle=bundle, status=lambda: status)


@pytest.mark.parametrize("status", [{"trained": False}, {"trained": True, "stale": True}])
def test_requires_current_model(status):
    with pytest.raises(ValueError, match="current dataset"):
        error_report(baseline(None, **status))


def test_old_bundle_requires_retraining():
    with pytest.raises(ValueError, match="Retrain once"):
        error_report(baseline({}, trained=True))


def test_error_directions_and_correct_predictions():
    rows = [{"recorded_category": actual, "prediction": predicted}
            for actual, predicted in [("MARKETING", "UTILITY"), ("UTILITY", "MARKETING"),
                                      ("UTILITY", "UTILITY")]]
    report = error_report(baseline({"report": {"trained_at": "example"}, "revision": "abc",
                                    "holdout": rows}, trained=True))
    assert report["false_utility"] == [rows[0]]
    assert report["false_marketing"] == [rows[1]]
    assert report["test_families"] == 3
