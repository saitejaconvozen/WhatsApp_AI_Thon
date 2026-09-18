import pytest
from fastapi.testclient import TestClient

from templatelab.app import create_app
from templatelab.model import FEATURE_VERSION


@pytest.fixture
def error_client(tmp_path):
    app = create_app(tmp_path)
    app.state.baseline.bundle = {
        "revision": app.state.store.revision(), "feature_version": FEATURE_VERSION,
        "report": {"trained_at": "2026-09-18"},
        "holdout": [{"id": "example", "family": "family", "name": "Example", "header": "Header",
                     "body": "Invoice {{id}}. Upgrade today.", "footer": "Footer", "buttons": "View",
                     "recorded_category": "MARKETING", "prediction": "UTILITY"}],
    }
    with TestClient(app) as client:
        yield client


def payload(client):
    report = client.get('/api/errors').json()
    return {"snapshot": report["snapshot"], "version": 0, "status": "reviewed",
            "reason": "promotion", "notes": "Separate upsell in body", "context": "Invoice event"}


def test_save_export_persistence_and_labels(error_client):
    client = error_client
    revision = client.app.state.store.revision()
    p = payload(client)
    saved = client.post('/api/errors/example', json=p)
    assert saved.status_code == 200
    assert saved.json()["version"] == 1
    report = client.get('/api/error-export').json()
    assert report["counts"]["reviewed"] == 1
    assert report["items"][0]["recorded_category"] == "MARKETING"
    assert report["items"][0]["annotation"]["notes"] == p["notes"]
    assert client.app.state.store.revision() == revision
    from templatelab.error_review import ErrorReviews
    assert ErrorReviews(client.app.state.store, client.app.state.baseline).current()["counts"]["reviewed"] == 1
    assert client.post('/api/errors/example', json=p).status_code == 409


def test_snapshot_and_unknown_record(error_client):
    p = payload(error_client)
    assert error_client.post('/api/errors/missing', json=p).status_code == 404
    assert error_client.post('/api/errors/example', json={**p, "snapshot": "old"}).status_code == 409
    assert error_client.post('/api/errors/example', json={**p, "notes": ""}).status_code == 409
    assert error_client.post('/api/errors/example', json={**p, "status": "invalid"}).status_code == 422
    error_client.app.state.baseline.bundle["revision"] = "old"
    assert error_client.get('/api/errors').status_code == 409
    assert error_client.get('/api/error-export').status_code == 409


def test_meta_outcome_requires_draft_and_reference(error_client):
    p = {**payload(error_client), "meta_outcome": "UTILITY"}
    assert error_client.post('/api/errors/example', json=p).status_code == 409
    p.update(draft={"body": "Your invoice {{id}} is ready."}, outcome_reference="decision-123")
    assert error_client.post('/api/errors/example', json=p).status_code == 200
    p.update(version=1, draft={"body": "Your changed invoice {{id}} is ready."})
    assert error_client.post('/api/errors/example', json=p).status_code == 409


def test_annotations_are_snapshot_scoped(error_client):
    p = payload(error_client)
    assert error_client.post('/api/errors/example', json=p).status_code == 200
    error_client.app.state.baseline.bundle["report"]["trained_at"] = "2026-09-19"
    report = error_client.get('/api/errors').json()
    assert report["snapshot"] != p["snapshot"]
    assert report["counts"]["reviewed"] == 0


def test_error_writes_reject_external_origin(error_client):
    assert error_client.post('/api/errors/example', json=payload(error_client),
                             headers={"Origin": "https://outside.example"}).status_code == 403
