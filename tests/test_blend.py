import numpy as np
import pytest

from templatelab.blend import compare


def test_blend_uses_same_validation_rows_and_does_not_score_unknown_truth():
    records = [{"id": str(i), "meta_category": "UTILITY" if i % 2 else "MARKETING",
                "requested_category": "UTILITY" if i < 8 else "MARKETING"} for i in range(10)]
    text = np.array([.8 if i % 2 else .2 for i in range(10)])
    neural = 1 - text
    result = compare(records, text, neural)
    assert result["selected"]["validation"]["all"]["samples"] == 10
    assert result["disagreements_on_requested_utility"] == 8
    assert result["oracle_accuracy_on_requested_utility"] == 1
    with pytest.raises(ValueError):
        compare(records, text[:-1], neural)
