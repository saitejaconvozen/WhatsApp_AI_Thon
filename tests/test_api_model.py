import json

import pytest
from fastapi.testclient import TestClient

from templatelab.app import create_app
from templatelab.data import normalize_rows, suggest_mapping


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        yield client


def test_import_search_and_idempotency(client):
    rows = [{"name": "Example invoice", "body": "Your invoice {{id}} is ready.", "meta_category": "UTILITY", "status": "VERIFIED"}]
    for expected_added in [1, 0]:
        preview = client.post('/api/import/preview', files={"file": ("a.json", json.dumps(rows), "application/json")}).json()
        response = client.post('/api/import/commit', json={"token": preview["token"], "mapping": preview["mapping"]})
        assert response.status_code == 200
        assert response.json()["added"] == expected_added
    assert client.get('/api/records?q=invoice&category=UTILITY').json()["total"] == 1
    assert client.get('/api/records?q=missing').json()["total"] == 0
    assert client.get('/api/summary').json()["total"] == 1
    assert client.get('/api/model').json()["trained"] is False


def test_upload_validation_and_unknown_record(client):
    assert client.post('/api/import/preview', files={"file": ("a.json", b"[", "application/json")}).status_code == 400
    assert client.post('/api/import/commit', json={"token": "expired"}).status_code == 400
    assert client.get('/api/records/not-real').status_code == 404
    assert client.post('/api/review', json={"body": "  "}).status_code == 400


def test_cross_origin_writes_are_rejected(client):
    assert client.post('/api/model/train', json={}, headers={"Origin": "https://outside.example"}).status_code == 403


def test_csv_export_neutralizes_formulas(client):
    rows = [{"name": "=1+1", "body": "+formula", "meta_category": "UTILITY"}]
    client.app.state.store.import_records(normalize_rows(rows, suggest_mapping(rows), "example.json"))
    exported = client.get('/api/export').text
    assert "'=1+1" in exported
    assert "'+formula" in exported


def test_model_evaluation_persistence_and_staleness(client):
    services = ['electricity', 'water', 'broadband', 'maintenance', 'insurance', 'storage', 'parking', 'cleaning', 'membership', 'hosting', 'software', 'school', 'rental', 'utilities', 'transport', 'security', 'repairs', 'laundry', 'catering', 'consulting']
    rows = []
    for service in services:
        rows.append({"body": f"Your {service} invoice {{{{id}}}} is due on {{{{date}}}}.", "meta_category": "UTILITY", "status": "VERIFIED"})
        rows.append({"body": f"Get an exclusive {service} discount. Shop now and save.", "meta_category": "MARKETING", "status": "VERIFIED"})
    client.app.state.store.import_records(normalize_rows(rows, suggest_mapping(rows), "synthetic-test.json"))
    response = client.post('/api/model/train', json={})
    assert response.status_code == 200, response.text
    report = response.json()["report"]
    assert report["group_overlap"] == 0
    assert report["train_families"] + report["test_families"] == 40
    assert sum(sum(row) for row in report["confusion_matrix"]) == report["test_families"]
    from templatelab.audit import error_report
    from templatelab.model import Baseline
    persisted = Baseline(client.app.state.store)
    audit = error_report(persisted)
    assert audit["test_families"] == report["test_families"]
    assert len(audit["false_utility"]) == report["confusion_matrix"][0][1]
    assert len(audit["false_marketing"]) == report["confusion_matrix"][1][0]
    assert len({r["family"] for r in persisted.bundle["holdout"]}) == report["test_families"]
    prediction = client.post('/api/review', json={"body": "Your hosting invoice {{id}} is due on {{date}}."}).json()["prediction"]
    assert prediction["available"]
    assert prediction["seen_family"]
    from templatelab.model import Baseline
    assert Baseline(client.app.state.store).status()["trained"]
    new_rows = [{"body": "Your entirely new invoice {{id}} is ready.", "meta_category": "UTILITY", "status": "VERIFIED"}]
    client.app.state.store.import_records(normalize_rows(new_rows, suggest_mapping(new_rows), "new.json"))
    assert client.get('/api/model').json()["stale"]
    assert not client.post('/api/review', json={"body": "Your invoice is due."}).json()["prediction"]["available"]
