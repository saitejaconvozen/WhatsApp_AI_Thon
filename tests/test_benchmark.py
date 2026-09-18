import pytest
import json
from types import SimpleNamespace

from templatelab import benchmark
from templatelab.benchmark import sample_rows, summarize, wilson
from templatelab.llm import Cache, parse_response, redact, Unavailable


def rows():
    return [{"id": str(i), "requested_category": "UTILITY" if i < 200 else "MARKETING",
             "recorded_category": "UTILITY" if i % 2 else "MARKETING", "prediction": "UTILITY"}
            for i in range(240)]


def test_sample_is_reproducible_stratified_and_only_requested_utility():
    selected = sample_rows(rows(), 100)
    assert selected == sample_rows(list(reversed(rows())), 100)
    assert len({row["id"] for row in selected}) == 100
    assert all(row["requested_category"] == "UTILITY" for row in selected)
    assert sum(row["recorded_category"] == "UTILITY" for row in selected) == 50


@pytest.mark.parametrize("limit", [0, -1, 101])
def test_hard_call_limit(limit):
    with pytest.raises(ValueError):
        sample_rows(rows(), limit)


def test_failures_and_abstentions_cannot_inflate_accuracy():
    sample = rows()[:4]
    results = [{"available": True, "category": sample[0]["recorded_category"]},
               {"available": False, "reason": "timeout"},
               {"available": True, "category": "NEEDS_REVIEW"},
               {"available": True, "category": sample[3]["recorded_category"]}]
    report = summarize(sample, results)
    assert report["end_to_end"]["accuracy"] == .5
    assert report["end_to_end"]["samples"] == 4
    assert report["failures"] == 1 and report["abstentions"] == 1
    assert not report["target_observed"]
    assert report["baseline_same_sample"]["samples"] == 4
    assert wilson(2, 4)[0] < .5 < wilson(2, 4)[1]
    with pytest.raises(ValueError):
        summarize(sample, results[:1])


def test_contact_redaction_preserves_placeholder_and_amount():
    text = redact("Email jane@example.com, phone +91 98765 43210, visit https://example.com/customer/123; invoice {{id}}: Rs 200.")
    assert "jane@example.com" not in text and "98765" not in text and "customer/123" not in text
    assert "{{id}}" in text and "Rs 200" in text


def test_cache_namespace_includes_endpoint(monkeypatch):
    monkeypatch.setenv("TEMPLATELAB_LLM_BASE_URL", "https://one.example")
    first = Cache.key("prompt", "same-model")
    monkeypatch.setenv("TEMPLATELAB_LLM_BASE_URL", "https://two.example")
    assert Cache.key("prompt", "same-model") != first


@pytest.mark.parametrize("text", ['{"category":"UTILITY","confidence":5}', '{"category":"UTILITY","clauses":null}'])
def test_response_types_are_validated(text):
    with pytest.raises(Unavailable):
        parse_response(text)


def test_retrieval_excludes_entire_holdout_and_stops_after_first_failure(tmp_path, monkeypatch):
    holdout = [{**rows()[i], "family": f"held-{i}"} for i in range(4)]
    training = {"id": "train", "family": "train"}
    bundle = {"holdout": holdout, "records": [*holdout, training,
              {"id": "other-version", "family": "held-0"}], "report": {"trained_at": "today"}}
    monkeypatch.setattr(benchmark, "Baseline", lambda store: SimpleNamespace(
        bundle=bundle, status=lambda: {"trained": True, "stale": False}))
    seen = []
    class Features:
        def fit(self, records):
            assert records == [training]
            return self
        def transform(self, records):
            assert records == [training]
            return []
    class Reviewer:
        def __init__(self, store, retrieval):
            assert retrieval.bundle["records"] == [training]
        def status(self):
            return {"available": True}
        def review(self, record, exclude):
            assert exclude == {r["id"] for r in holdout}
            seen.append(record)
            return {"available": False, "reason": "provider unavailable"}
    monkeypatch.setattr(benchmark, "features", Features)
    monkeypatch.setattr(benchmark.llm, "Reviewer", Reviewer)
    store = SimpleNamespace(directory=tmp_path, revision=lambda: "revision")
    path, report = benchmark.evaluate(store, limit=4)
    assert len(seen) == 1
    assert not report["complete"] and report["metrics"]["failures"] == 1
    assert json.loads(path.read_text())["completed"] == 1
    dashboard = benchmark.latest_report(store)
    assert "sample_ids" not in dashboard["report"]
    assert "predictions" not in dashboard["report"]["metrics"]
