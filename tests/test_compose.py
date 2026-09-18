import json

import pytest

from templatelab import compose, llm
from templatelab.data import Store
from templatelab.model import Baseline
from test_experiments import dated_records


@pytest.fixture(autouse=True)
def no_backend(monkeypatch):
    for variable in (llm.BACKEND_VARIABLE, llm.EGRESS_VARIABLE, llm.MODEL_VARIABLE,
                     "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(variable, raising=False)


@pytest.fixture
def baseline(tmp_path):
    store = Store(tmp_path)
    store.import_records(dated_records())
    trained = Baseline(store)
    trained.train()
    return trained


def enable_llm(monkeypatch, reply):
    monkeypatch.setenv(llm.BACKEND_VARIABLE, "deepseek")
    monkeypatch.setenv(llm.EGRESS_VARIABLE, "1")
    monkeypatch.setitem(llm.BACKENDS, "deepseek",
                        lambda prompt, model: reply if isinstance(reply, str) else json.dumps(reply))


MIXED = {"header": "", "body": "Your order {{1}} has shipped. 20% off your next order!",
         "footer": "", "buttons": "Track order\nShop now", "format": "TEXT"}
PURE_PROMO = {"header": "", "body": "Festive Sale is LIVE! 70% off everything.",
              "footer": "", "buttons": "Shop now", "format": "TEXT"}
CLEAN = {"header": "", "body": "Your order {{1}} has shipped. Expected delivery {{2}}.",
         "footer": "", "buttons": "Track order", "format": "TEXT"}


def test_clean_template_needs_no_conversion(baseline):
    result = compose.convert(CLEAN, baseline, relationship_confirmed=True)
    assert result["verdict"] == compose.ALREADY_UTILITY
    assert result["utility"] is None


def test_purely_promotional_template_is_refused_not_laundered(baseline):
    result = compose.convert(PURE_PROMO, baseline)
    assert result["verdict"] == compose.IRREDUCIBLY_MARKETING
    assert result["utility"] is None
    assert "launder" in result["reason"] or "nothing transactional" in result["reason"]


def test_mixed_template_splits_and_keeps_the_anchor(baseline):
    result = compose.convert(MIXED, baseline, relationship_confirmed=True)
    assert result["verdict"] == compose.SPLIT_RECOMMENDED
    assert "{{1}}" in result["utility"]["body"]
    assert "20%" not in result["utility"]["body"]
    assert "Shop now" not in result["utility"]["buttons"]
    assert "Track order" in result["utility"]["buttons"]
    assert result["split_off"] is not None
    assert result["before"]["available"] and result["after"]["available"]


def test_model_reply_that_drops_the_anchor_is_discarded(baseline, monkeypatch):
    # The anchor guard must not be repairable by a confident model reply.
    enable_llm(monkeypatch, {"possible": True, "header": "", "body": "Thank you for being a customer.",
                             "footer": "", "buttons": "", "removed": [], "reason": "cleaned"})
    result = compose.convert(MIXED, baseline, relationship_confirmed=True)
    assert "{{1}}" in result["utility"]["body"]
    assert result["method"] == "checklist"


def test_model_may_refuse_a_conversion(baseline, monkeypatch):
    enable_llm(monkeypatch, {"possible": False, "reason": "Purely promotional."})
    result = compose.convert(MIXED, baseline, relationship_confirmed=True)
    assert result["verdict"] == compose.IRREDUCIBLY_MARKETING
    assert result["method"].startswith("llm:")


def test_llm_rewrite_is_used_when_it_keeps_the_anchor(baseline, monkeypatch):
    enable_llm(monkeypatch, {"possible": True, "header": "", "body": "Your order {{1}} has shipped.",
                             "footer": "", "buttons": "Track order",
                             "removed": ["20% off your next order!"], "reason": "removed the offer"})
    result = compose.convert(MIXED, baseline, relationship_confirmed=True)
    assert result["method"] == "llm:deepseek"
    assert result["utility"]["body"] == "Your order {{1}} has shipped."
    assert result["split_off"] is not None


def test_generation_falls_back_to_a_preset_without_a_backend(baseline):
    result = compose.generate("tell the customer their invoice is ready", baseline)
    assert result["possible"] is True
    assert result["method"] == "checklist"
    assert "{{" in result["template"]["body"]
    assert result["purpose"] == "billing"


def test_generation_refuses_a_promotional_task(baseline):
    result = compose.generate("announce 50% off our new collection", baseline)
    assert result["possible"] is False
    assert result["verdict"] == compose.IRREDUCIBLY_MARKETING


def test_generation_without_a_recognised_event_asks_for_context(baseline):
    result = compose.generate("say something nice to our customers", baseline)
    assert result["possible"] is False
    assert result["verdict"] == compose.NEEDS_CONTEXT


def test_generated_draft_is_rescored_and_checked(baseline, monkeypatch):
    enable_llm(monkeypatch, {"possible": True, "name": "ORDER_SHIPPED", "header": "",
                             "body": "Your order {{order_id}} shipped on {{date}}.",
                             "footer": "", "buttons": "Track order", "reason": "drafted"})
    result = compose.generate("tell them their order shipped", baseline)
    assert result["method"] == "llm:deepseek"
    assert result["template"]["name"] == "ORDER_SHIPPED"
    assert result["findings"] == []
    assert result["score"]["available"] is True


def test_a_generated_draft_that_trips_the_checklist_is_reported(baseline, monkeypatch):
    # The model is not trusted to police itself; the checklist runs on its output.
    enable_llm(monkeypatch, {"possible": True, "name": "X", "header": "",
                             "body": "Your order {{1}} shipped. Buy now for 10% off.",
                             "footer": "", "buttons": "", "reason": "drafted"})
    result = compose.generate("tell them their order shipped", baseline)
    assert result["findings"], "promotional wording in a generated draft must be surfaced"
    assert not result["possible"]
    assert result["template"] is None


def test_classification_is_available_even_when_rewrite_needs_context(baseline):
    result = compose.convert(MIXED, baseline)
    assert result["before"]["available"]
    assert result["verdict"] == compose.NEEDS_CONTEXT
    assert result["utility"] is None


def test_authentication_is_not_called_utility(baseline):
    result = compose.convert({"body": "Your verification code is {{1}}."}, baseline)
    assert result["verdict"] == "AUTHENTICATION"


def test_generation_does_not_invent_order_from_discount(baseline):
    result = compose.generate("Offer 20% off the next order", baseline)
    assert not result["possible"]
    assert result["template"] is None


def test_literal_transaction_facts_are_not_deleted(baseline):
    record = {"body": "Your invoice 123 is due tomorrow; upgrade today. Your account remains active."}
    result = compose.convert(record, baseline, relationship_confirmed=True)
    assert result["utility"] is None
    assert result["verdict"] == compose.NEEDS_CONTEXT


def test_llm_cannot_drop_placeholders_or_keep_promotion(baseline, monkeypatch):
    enable_llm(monkeypatch, {"possible": True, "body": "Your order shipped. Buy now for 50% off.", "removed": []})
    result = compose.convert(MIXED, baseline, relationship_confirmed=True)
    assert result["method"] == "checklist"
    assert "{{1}}" in result["utility"]["body"]
    assert "Buy now" not in result["utility"]["body"]


def test_empty_task_is_rejected(baseline):
    with pytest.raises(ValueError):
        compose.generate("   ", baseline)


def test_generation_passes_business_context_to_provider(baseline, monkeypatch):
    prompts = []
    enable_llm(monkeypatch, {})
    def reply(prompt, model):
        prompts.append(prompt)
        return json.dumps({"possible": True, "body": "Your invoice {{invoice_id}} is ready."})
    monkeypatch.setitem(llm.BACKENDS, "deepseek", reply)
    result = compose.generate("Notify the customer", baseline, context="Existing customer; invoice is ready.")
    assert result["possible"]
    assert "Existing customer; invoice is ready." in prompts[0]


@pytest.mark.parametrize("payload", [{"possible": "true"}, {"possible": True, "body": {"text": "invoice"}}])
def test_malformed_generation_schema_falls_back_safely(baseline, monkeypatch, payload):
    enable_llm(monkeypatch, payload)
    result = compose.generate("Tell them their invoice is ready", baseline)
    assert result["method"] == "checklist"


@pytest.mark.parametrize("raw, expected", [
    ("Track order", "Track order"),
    ('[{"type":"URL","text":"View receipt","url":"x"}]', "View receipt"),
    ([{"type": "URL", "text": "View booking"}, {"text": "Help"}], "View booking\nHelp"),
    ({"text": "Solo"}, "Solo"),
    ("[broken json", "[broken json"),
    (None, ""),
])
def test_button_labels_survive_every_shape_a_model_returns(raw, expected):
    """A serialised button list once reached the UI as a raw JSON blob."""
    from templatelab.compose import button_text
    assert button_text(raw) == expected


def test_anchor_needs_a_transaction_and_a_reference():
    """A topic word alone is not a transaction the recipient already has."""
    from templatelab.compose import has_anchor
    assert has_anchor({"body": "Your rent receipt {{id}} is ready."})
    assert has_anchor({"body": "Your order 12345 has shipped."})
    assert not has_anchor({"body": "Rent a property today!"})
    assert not has_anchor({"body": "Your order has shipped."})
