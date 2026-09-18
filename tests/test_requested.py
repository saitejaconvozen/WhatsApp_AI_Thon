import numpy as np
import pytest

from templatelab.requested import conditional_predictions


def test_conditional_rule_uses_only_predecision_request():
    rows = [{"requested_category": "MARKETING", "meta_category": "UTILITY"},
            {"requested_category": "UTILITY", "meta_category": "MARKETING"}]
    assert list(conditional_predictions(rows, ["UTILITY", "UTILITY"])) == ["MARKETING", "UTILITY"]
    altered = [{**r, "meta_category": "OTHER"} for r in rows]
    assert np.array_equal(conditional_predictions(rows, ["UTILITY", "UTILITY"]),
                          conditional_predictions(altered, ["UTILITY", "UTILITY"]))
    with pytest.raises(ValueError):
        conditional_predictions(rows, ["UTILITY"])
