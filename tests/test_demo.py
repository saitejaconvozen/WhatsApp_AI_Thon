import json

import pytest
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


def test_convert_and_generate_never_return_stored_templates(monkeypatch):
    """Both new routes are LLM-backed; neither may echo the retrieval corpus."""
    from templatelab import demo

    monkeypatch.setattr(demo, "convert", lambda record, model, **kw: {
        "verdict": "SPLIT_RECOMMENDED", "reason": "ok", "purpose": "order", "method": "checklist",
        "before": {"available": True, "utility_probability": .3},
        "after": {"available": True, "utility_probability": .7},
        "utility": {"body": "Your order {{1}} shipped."}, "split_off": {"body": "20% off"},
        "removed": [{"component": "body", "text": "20% off"}], "ambiguous": [], "needs_human": False,
        "checklist": {"findings": [{"code": "discount", "component": "body",
                                    "message": "promo", "evidence": "PRIVATE CUSTOMER DATA"}],
                      "rewrite": {"components": {"body": "PRIVATE CUSTOMER DATA"}}},
        "neighbors": [{"body": "PRIVATE CUSTOMER DATA"}]})
    monkeypatch.setattr(demo, "generate", lambda task, model, **kw: {
        "possible": True, "verdict": "ALREADY_UTILITY", "purpose": "billing", "method": "checklist",
        "template": {"name": "X", "body": "Your invoice {{1}} is ready.", "buttons": "View"},
        "score": {"available": True, "utility_probability": .8}, "findings": [],
        "reason": "drafted", "neighbors": [{"body": "PRIVATE CUSTOMER DATA"}]})

    with TestClient(demo.create_demo(predictor=Predictor())) as client:
        converted = client.post('/api/convert', json={"body": "Your order {{1}} shipped. 20% off!"})
        assert converted.status_code == 200
        assert converted.json()["verdict"] == "SPLIT_RECOMMENDED"
        assert "PRIVATE" not in converted.text and "neighbors" not in converted.json()

        drafted = client.post('/api/generate', json={"task": "tell them the invoice is ready"})
        assert drafted.status_code == 200
        assert drafted.json()["template"]["name"] == "X"
        assert "PRIVATE" not in drafted.text and "neighbors" not in drafted.json()


def test_llm_routes_have_their_own_tighter_budget(monkeypatch):
    from templatelab import demo

    monkeypatch.setattr(demo, "generate", lambda task, model, **kw: {"possible": False, "verdict": "NEEDS_CONTEXT"})
    with TestClient(demo.create_demo(predictor=Predictor())) as client:
        for _ in range(demo.LLM_LIMIT):
            assert client.post('/api/generate', json={"task": "invoice"}).status_code == 200
        assert client.post('/api/generate', json={"task": "invoice"}).status_code == 429
        # The cheaper local classifier keeps its own, larger allowance.
        assert client.post('/api/predict', json={"body": "Invoice due"}).status_code == 200


def explaining(monkeypatch, verdict):
    from templatelab import demo
    monkeypatch.setattr(demo.Reviewer, "review", lambda self, record, **kw: verdict)
    return demo.create_demo(predictor=Predictor())


def test_explanation_returns_reasoning_and_never_the_precedents(monkeypatch):
    """The retrieved examples shape the answer but are stored customer text."""
    app = explaining(monkeypatch, {
        "available": True, "category": "UTILITY", "rationale": "Names a specific invoice.",
        "clauses": ["U1", "U3"], "confidence": 1.0, "model": "test-model",
        "neighbors": [{"body": "PRIVATE CUSTOMER DATA"}]})
    with TestClient(app) as client:
        data = client.post('/api/explain', json={"body": "Your invoice {{id}} is due."}).json()
        assert data["available"] is True and data["rationale"] == "Names a specific invoice."
        assert [c["id"] for c in data["clauses"]] == ["U1", "U3"]
        # Clause ids resolve to their real policy text, not bare codes.
        assert "transaction" in data["clauses"][0]["text"]
        assert "PRIVATE" not in client.post('/api/explain', json={"body": "x"}).text
        # Self-reported confidence is not exposed; it is routinely 1.0 and uninformative.
        assert "confidence" not in data


def test_explanation_flags_disagreement_with_the_scoring_model(monkeypatch):
    app = explaining(monkeypatch, {"available": True, "category": "MARKETING",
                                   "rationale": "Carries an offer.", "clauses": ["M6"]})
    with TestClient(app) as client:
        data = client.post('/api/explain', json={"body": "Your order {{1}} shipped. 20% off!"}).json()
        # Predictor fixture always says UTILITY, so this must be reported as a split.
        assert data["agrees"] is False
        assert data["local_category"] == "UTILITY" and data["category"] == "MARKETING"


def test_explanation_degrades_clearly_without_a_backend(monkeypatch):
    app = explaining(monkeypatch, {"available": False, "reason": "No LLM backend is configured."})
    with TestClient(app) as client:
        response = client.post('/api/explain', json={"body": "Your invoice {{id}} is due."})
        assert response.status_code == 200
        assert response.json()["available"] is False
        assert "backend" in response.json()["reason"]


def test_every_policy_clause_resolves_to_text():
    from templatelab.llm import clause_definitions
    definitions = clause_definitions()
    assert set(definitions) == {"U1", "U2", "U3", "U4", "M1", "M2", "M3", "M4", "M5", "M6"}
    assert all(len(text) > 20 for text in definitions.values())
    assert "**" not in "".join(definitions.values())


def test_conversion_says_which_context_is_missing(monkeypatch):
    """A NEEDS_CONTEXT verdict is unactionable unless it names the gap."""
    from templatelab import demo
    monkeypatch.setattr(demo, "convert", lambda record, model, **kw: {
        "verdict": "NEEDS_CONTEXT", "reason": "not enough evidence", "method": "checklist",
        "before": {"available": True, "utility_probability": None}, "after": None,
        "utility": None, "split_off": None, "removed": [], "ambiguous": [], "needs_human": False,
        "checklist": {"findings": [],
                      "missing_context": ["Confirm that this message relates to an actual transaction.",
                                          "Select the event that triggers this message."]}})
    with TestClient(demo.create_demo(predictor=Predictor())) as client:
        data = client.post('/api/convert', json={"body": "Can we call you about property search?"}).json()
        assert data["verdict"] == "NEEDS_CONTEXT"
        assert len(data["missing_context"]) == 2
        assert "transaction" in data["missing_context"][0]


PASTED = json.dumps({
    "templateName": "PERMISSION",
    "messageBody": {"type": "TEXT", "templateCategory": "UTILITY",
                    "header": "", "body": "Hi {{bodyVar1}}, your invoice {{id}} is ready.",
                    "footer": "", "interactionConfiguration": {"interactionType": "NONE"}},
    "verificationStatus": "VERIFIED", "metaTemplateCategory": "UTILITY"})


def test_pasted_json_is_parsed_into_components():
    from templatelab.demo import from_json
    parsed = from_json(PASTED)
    assert parsed["body"].startswith("Hi {{bodyVar1}}")
    assert parsed["requested_category"] == "UTILITY"
    assert parsed["name"] == "PERMISSION"


@pytest.mark.parametrize("text, fragment", [
    ("not json at all", "valid JSON"),
    ('[{"a":1},{"b":2}]', "single template"),
    ('{"messageBody": {"body": "   "}}', "No message body"),
])
def test_bad_pasted_json_is_rejected_clearly(text, fragment):
    from fastapi import HTTPException
    from templatelab.demo import from_json
    with pytest.raises(HTTPException) as caught:
        from_json(text)
    assert fragment in caught.value.detail


def test_json_input_and_output_round_trip(monkeypatch):
    from templatelab import demo
    monkeypatch.setattr(demo, "convert", lambda record, model, **kw: {
        "verdict": "CONVERTIBLE", "reason": "ok", "method": "checklist",
        "before": {"available": True, "utility_probability": .4},
        "after": {"available": True, "utility_probability": .8},
        "utility": {"header": "", "body": "Your invoice {{id}} is ready.", "footer": "", "buttons": "View\nHelp"},
        "split_off": None, "removed": [], "ambiguous": [], "needs_human": False, "checklist": {}})
    with TestClient(demo.create_demo(predictor=Predictor())) as client:
        data = client.post('/api/convert', json={"template_json": PASTED}).json()
        assert data["verdict"] == "CONVERTIBLE"
        rendered = data["utility_json"]
        assert rendered["templateName"] == "PERMISSION"
        assert rendered["messageBody"]["templateCategory"] == "UTILITY"
        assert rendered["messageBody"]["body"] == "Your invoice {{id}} is ready."
        # Buttons come back as a list, matching the export shape rather than a blob.
        assert rendered["messageBody"]["buttons"] == ["View", "Help"]


def test_generation_returns_a_pasteable_template_json(monkeypatch):
    from templatelab import demo
    monkeypatch.setattr(demo, "generate", lambda task, model, **kw: {
        "possible": True, "verdict": "ALREADY_UTILITY", "purpose": "billing", "method": "checklist",
        "template": {"name": "INVOICE_READY", "header": "", "body": "Your invoice {{id}} is ready.",
                     "footer": "", "buttons": "View invoice"},
        "score": {"available": True, "utility_probability": .8}, "findings": [], "reason": ""})
    with TestClient(demo.create_demo(predictor=Predictor())) as client:
        data = client.post('/api/generate', json={"task": "invoice is ready"}).json()
        assert data["template_json"]["templateName"] == "INVOICE_READY"
        assert data["template_json"]["messageBody"]["buttons"] == ["View invoice"]


def test_conversion_reports_how_the_rewrite_was_chosen(monkeypatch):
    """Selection detail is how a reader tells a chosen rewrite from the only one."""
    from templatelab import demo
    monkeypatch.setattr(demo, "convert", lambda record, model, **kw: {
        "verdict": "SPLIT_RECOMMENDED", "reason": "ok", "method": "llm-select:openai_compatible",
        "before": {"available": True, "utility_probability": .3},
        "after": {"available": True, "utility_probability": .77},
        "utility": {"body": "Your {{plan}} renewal is due {{date}}."}, "split_off": {"body": "20% off"},
        "removed": [], "ambiguous": [], "needs_human": False, "checklist": {},
        "selection": {"candidates": 5, "eligible": 3, "moved": 0.42}})
    with TestClient(demo.create_demo(predictor=Predictor())) as client:
        data = client.post('/api/convert', json={"body": "20% off your {{plan}} renewal {{date}}."}).json()
        assert data["selection"] == {"candidates": 5, "eligible": 3, "moved": 0.42}
        assert data["method"].startswith("llm-select")
