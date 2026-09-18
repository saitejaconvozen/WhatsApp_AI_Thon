from fastapi.testclient import TestClient
from templatelab.demo import create_demo


class Predictor:
    def predict(self, record):
        return {"available": True, "category": "UTILITY", "utility_probability": .8, "band": "UTILITY",
                "neighbors": [{"body": "PRIVATE CUSTOMER DATA"}]}


def test_public_inference_does_not_expose_stored_templates():
    with TestClient(create_demo(predictor=Predictor())) as client:
        response = client.post('/api/predict', json={"body": "Your invoice {{id}} is due."})
        assert response.status_code == 200
        assert response.json()["category"] == "UTILITY"
        assert "PRIVATE" not in response.text and "neighbors" not in response.json()
        for route in ('/api/records', '/api/export', '/.env', '/.data/templates.sqlite3', '/docs'):
            assert client.get(route).status_code == 404


def test_demo_validation_and_limit():
    with TestClient(create_demo(predictor=Predictor())) as client:
        assert client.post('/api/predict', json={"body": " "}).status_code == 422
        assert client.post('/api/predict', json={"body": "x" * 6001}).status_code == 422
        for _ in range(60):
            assert client.post('/api/predict', json={"body": "Invoice due"}).status_code == 200
        assert client.post('/api/predict', json={"body": "Invoice due"}).status_code == 429
