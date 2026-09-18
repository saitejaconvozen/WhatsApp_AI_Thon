import json

import pytest

from templatelab.data import Store
from templatelab.experiments import Experiments
from templatelab.model import Baseline
from templatelab.publish import build, build_payload
from templatelab import publish
from test_experiments import dated_records

SECRET = "Zarquon"


@pytest.fixture
def store(tmp_path):
    records = dated_records()
    for record in records:
        record["name"] = f"{SECRET}_TEMPLATE_{record['id'][:6]}"
        record["body"] = f"{record['body']} Contact {SECRET} support."
    store = Store(tmp_path / "data")
    store.import_records(records)
    return store


def test_untrained_workspace_still_builds(store, tmp_path):
    payload = build(store, tmp_path / "site")
    assert payload["baseline"]["trained"] is False
    assert payload["experiments"]["available"] is False
    assert payload["dataset"]["total"] == len(dated_records())


def test_published_build_excludes_template_content(store, tmp_path):
    Baseline(store).train()
    Experiments(store).run()
    out = tmp_path / "site"
    payload = build(store, out)

    assert payload["baseline"]["report"]["accuracy"] >= 0
    assert payload["experiments"]["report"]["near_duplicates"]["flagged_test_families"] >= 0
    assert payload["errors"]["available"] is False
    assert payload["errors"]["items"] == []
    assert set(payload["errors"]["counts"]) == {"total", "false_utility", "false_marketing", "reviewed"}

    for path in out.rglob("*"):
        if path.is_file():
            assert SECRET not in path.read_text(encoding="utf-8", errors="ignore"), path

    published = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert published == payload
    assert "pairs" not in published["experiments"]["report"]["near_duplicates"]
    assert "sources" not in published["dataset"]


def test_static_site_assets_are_self_contained(store, tmp_path):
    out = tmp_path / "site"
    build(store, out)
    page = (out / "index.html").read_text(encoding="utf-8")
    assert '/assets/' not in page
    assert 'window.RESULTS_ENDPOINT = "results.json"' in page
    for asset in ("results.css", "results.js", "benchmark.js", "lucide.min.js", ".nojekyll"):
        assert (out / asset).exists()


def test_benchmark_publishing_omits_private_rows_and_endpoint(store, monkeypatch):
    score = {"accuracy": .73, "samples": 100, "body": SECRET}
    monkeypatch.setattr(publish, "latest_report", lambda store: {"available": True, "report": {
        "model": "test-model", "base_url": SECRET, "sample_ids": [SECRET],
        "metrics": {"end_to_end": score, "baseline_same_sample": score,
                    "predictions": [{"rationale": SECRET}], "failures": 0}}})
    payload = publish.public_benchmark(store)
    assert SECRET not in json.dumps(payload)
    assert payload["report"]["metrics"]["end_to_end"]["accuracy"] == .73
