import json
import re

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


def test_meta_downgrade_cannot_be_called_already_utility(baseline):
    record = {**CLEAN, "requested_category": "UTILITY", "meta_category": "MARKETING"}
    result = compose.convert(record, baseline, relationship_confirmed=True)
    assert result["verdict"] == compose.NEEDS_CONTEXT
    assert result["disputed_by_meta"] is True
    assert result["utility"] is None
    assert "Meta recorded" in result["reason"]


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


def test_a_refusal_only_decides_when_the_checklist_cannot_convert(baseline, monkeypatch):
    """This asserted the opposite until the veto was measured as harmful.

    A model refusal on a separable template used to return IRREDUCIBLY_MARKETING
    and discard a working conversion. It now defers to the checklist there, and
    only decides when the checklist has produced nothing.
    """
    enable_llm(monkeypatch, {"possible": False, "reason": "Purely promotional."})
    assert compose.convert(MIXED, baseline, relationship_confirmed=True)["verdict"] == compose.SPLIT_RECOMMENDED
    refused = compose.convert(PURE_PROMO, baseline, relationship_confirmed=True)
    assert refused["verdict"] == compose.IRREDUCIBLY_MARKETING


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


def test_clean_wording_the_model_calls_marketing_is_reported_as_disputed(baseline, monkeypatch):
    """The checklist matches wording; only the model can judge intent.

    "Can we call you about your queries" trips the support vocabulary while being
    lead generation, which Meta records as marketing.
    """
    class Marketing:
        def predict(self, record):
            return {"available": True, "category": "MARKETING", "utility_probability": .26, "band": "MARKETING"}
    result = compose.convert({"header": "", "body": "Hi {{1}}, can we call you about your queries?",
                              "footer": "", "buttons": "", "format": "TEXT"}, Marketing(),
                             relationship_confirmed=True)
    assert result["verdict"] == compose.ALREADY_UTILITY
    assert result["disputed_by_model"] is True
    assert "needing a human" in result["reason"]


def test_agreement_is_not_reported_as_a_dispute(baseline):
    result = compose.convert(CLEAN, baseline, relationship_confirmed=True)
    assert result["verdict"] == compose.ALREADY_UTILITY
    assert result["disputed_by_model"] is False


def test_the_model_cannot_veto_a_working_checklist_conversion(baseline, monkeypatch):
    """A refusal used to return immediately, discarding a conversion that worked.

    Measured on 68 held-out promo-tripping templates, that veto cut successful
    conversions from 15 to 3.
    """
    enable_llm(monkeypatch, {"possible": False, "reason": "Reads as promotional."})
    result = compose.convert(MIXED, baseline, relationship_confirmed=True)
    assert result["verdict"] == compose.SPLIT_RECOMMENDED
    assert "{{1}}" in result["utility"]["body"]
    assert result["method"] == "checklist"


def test_the_model_still_decides_when_the_checklist_has_nothing(baseline, monkeypatch):
    enable_llm(monkeypatch, {"possible": False, "reason": "Purely promotional."})
    result = compose.convert(PURE_PROMO, baseline, relationship_confirmed=True)
    assert result["verdict"] == compose.IRREDUCIBLY_MARKETING


TANGLED = {"header": "", "body": "Hi {{1}}, 20% off your {{plan_name}} renewal due on {{date}}.",
           "footer": "", "buttons": "", "format": "TEXT"}


def test_tangled_clause_is_rewritten_and_the_claim_split_out(baseline, monkeypatch):
    """Deletion cannot resolve a promotion welded to a placeholder."""
    enable_llm(monkeypatch, {"possible": True, "header": "",
                             "body": "Hi {{1}}, your {{plan_name}} renewal is due on {{date}}.",
                             "footer": "", "buttons": "",
                             "promotional": ["20% off your renewal"], "reason": "separated"})
    result = compose.convert(TANGLED, baseline, relationship_confirmed=True)
    assert result["verdict"] in {compose.CONVERTIBLE, compose.SPLIT_RECOMMENDED}
    assert "{{plan_name}}" in result["utility"]["body"] and "{{date}}" in result["utility"]["body"]
    assert "20%" not in result["utility"]["body"]
    assert result["method"].startswith("llm-untangle")
    assert result["split_off"] is not None


@pytest.mark.parametrize("reply, why", [
    ({"possible": True, "body": "Thanks for being a customer.", "header": "", "footer": "", "buttons": "",
      "promotional": []}, "anchor"),
    ({"possible": True, "body": "Your {{plan_name}} renewal is due. 20% off!", "header": "", "footer": "",
      "buttons": "", "promotional": []}, "checklist"),
    ({"possible": False, "reason": "Nothing transactional."}, "transactional"),
])
def test_an_unsafe_untangle_is_refused_not_accepted(baseline, monkeypatch, reply, why):
    """A model-authored rewrite passes the same guards as a deletion."""
    enable_llm(monkeypatch, reply)
    result = compose.convert(TANGLED, baseline, relationship_confirmed=True)
    assert result["verdict"] == compose.NEEDS_CONTEXT
    assert result["utility"] is None


CANDIDATES = {"candidates": [
    {"header": "", "body": "Hi {{1}}, your {{plan_name}} renewal is due on {{date}}.", "footer": "", "buttons": ""},
    {"header": "", "body": "Hi {{1}}, renewal of {{plan_name}} falls due {{date}}.", "footer": "", "buttons": ""},
    {"header": "", "body": "Thanks for being with us.", "footer": "", "buttons": ""},
    {"header": "", "body": "Hi {{1}}, your {{plan_name}} renewal is due {{date}}. 20% off!", "footer": "", "buttons": ""},
], "promotional": ["20% off your renewal"], "reason": "separated"}


def test_selection_picks_among_candidates_and_drops_unsafe_ones(baseline, monkeypatch):
    """Several rewrites are proposed; the direction chooses between the safe ones."""
    enable_llm(monkeypatch, CANDIDATES)
    result = compose.convert(TANGLED, baseline, relationship_confirmed=True)
    assert result["method"].startswith("llm-select")
    body = result["utility"]["body"]
    assert "{{plan_name}}" in body and "{{date}}" in body and "20%" not in body
    # Four proposed, two survive: one drops the anchor, one keeps the promotion.
    assert result["selection"]["candidates"] == 4
    assert result["selection"]["eligible"] == 2


def test_selection_refuses_when_no_candidate_is_safe(baseline, monkeypatch):
    enable_llm(monkeypatch, {"candidates": [
        {"header": "", "body": "Thanks for being a customer.", "footer": "", "buttons": ""},
        {"header": "", "body": "Your {{plan_name}} renewal. Buy now for 20% off!", "footer": "", "buttons": ""},
    ], "promotional": []})
    result = compose.convert(TANGLED, baseline, relationship_confirmed=True)
    assert result["verdict"] == compose.NEEDS_CONTEXT
    assert result["utility"] is None


def test_selection_is_optional_when_no_direction_is_fitted(baseline, monkeypatch):
    """Without the fitted direction the first safe candidate is taken, not an error."""
    monkeypatch.setattr(compose, "_selector", (None, None))
    enable_llm(monkeypatch, CANDIDATES)
    result = compose.convert(TANGLED, baseline, relationship_confirmed=True)
    assert result["method"].startswith("llm-select")
    assert result["selection"]["moved"] is None
    assert "{{plan_name}}" in result["utility"]["body"]


def test_the_model_must_ratify_the_edit(baseline, monkeypatch):
    """A checklist sees only wording it has vocabulary for.

    Deleting one recognised phrase from an advertisement once produced a
    "utility version" that was still the advertisement. The model judges the
    whole template, so its verdict is the gate.
    """
    class StillMarketing:
        def predict(self, record):
            return {"available": True, "category": "MARKETING", "utility_probability": .30, "band": "MARKETING"}
    result = compose.convert(MIXED, StillMarketing(), relationship_confirmed=True)
    assert result["verdict"] == compose.NEEDS_CONTEXT
    assert result["utility"] is None
    assert "still reads the result as marketing" in result["reason"]


def test_a_ratified_edit_is_returned(baseline):
    result = compose.convert(MIXED, baseline, relationship_confirmed=True)
    assert result["verdict"] == compose.SPLIT_RECOMMENDED
    assert result["after"]["category"] == "UTILITY"


@pytest.mark.parametrize("text", [
    "Apply for your home loan and get *₹10,000* OFF on legal verification charges",
    "Call now or schedule a callback to claim your benefits",
    "Tap below to know more about the fund",
    "Don't miss out — at just ₹999",
])
def test_rebuilt_promotional_vocabulary_sees_real_marketing(text):
    from templatelab.policy import promotion_findings
    assert promotion_findings({"body": text}), text


@pytest.mark.parametrize("text", [
    "Your invoice {{1}} is due on {{2}}.",
    "Your order {{1}} has shipped. Expected delivery {{2}}.",
    "Your property visit {{visit_id}} is confirmed.",
])
def test_rebuilt_vocabulary_does_not_fire_on_service_updates(text):
    from templatelab.policy import promotion_findings
    assert not promotion_findings({"body": text}), text


def test_a_dropped_placeholder_is_refused(baseline, monkeypatch):
    """4 of 26 conversions silently lost a business value before this guard."""
    enable_llm(monkeypatch, {"candidates": [
        {"header": "", "body": "Your {{plan_name}} renewal is due.", "footer": "", "buttons": ""}],
        "promotional": ["20% off"]})
    result = compose.convert(TANGLED, baseline, relationship_confirmed=True)
    assert result["verdict"] == compose.NEEDS_CONTEXT
    assert "{{date}}" in result["reason"] and "business data" in result["reason"]


def test_a_placeholder_may_move_into_the_promotional_half(baseline, monkeypatch):
    """Splitting is allowed to relocate a value; only losing one is not."""
    enable_llm(monkeypatch, {"candidates": [
        {"header": "", "body": "Your {{plan_name}} renewal is due on {{date}}.", "footer": "", "buttons": ""}],
        "promotional": ["20% off for {{1}}"]})
    result = compose.convert(TANGLED, baseline, relationship_confirmed=True)
    assert result["verdict"] == compose.SPLIT_RECOMMENDED
    assert "{{1}}" in result["split_off"]["body"]


def test_sample_values_fit_the_name_or_the_sentence():
    """Most placeholders here are named bodyVarN, so context decides the value."""
    from templatelab.compose import fill_by_rule
    named = fill_by_rule("Your EMI of {{EMI_Amount}} on A/c {{Account_Number}} due {{Due_Date}}. Pay: {{Payment_Link}}")
    assert "₹" in named and "October" in named and "https://" in named and "{{" not in named

    generic = fill_by_rule("Hi {{bodyVar1}}, your payment of {{bodyVar2}} is due. Pay here: {{bodyVar3}}")
    assert "Rahul" in generic and "₹" in generic and "https://" in generic

    # Two links in one message must not read as the same URL.
    twice = fill_by_rule("Pay here: {{a}}. Check your booking anytime: {{b}}")
    links = re.findall(r"https://\S+", twice)
    assert len(links) == 2 and links[0] != links[1]


def test_a_fill_that_stops_reading_as_utility_is_rejected(baseline, monkeypatch):
    """A filled template that drifted into marketing is worse than an unfilled one."""
    enable_llm(monkeypatch, {"header": "", "body": "Hi Rahul, 50% OFF your renewal! Buy now.",
                             "footer": "", "buttons": ""})
    components = {"header": "", "body": "Hi {{1}}, your invoice {{2}} is due.", "footer": "", "buttons": ""}
    filled, how = compose.fill_sample_values(components, baseline)
    assert how != "llm"
    assert "50% OFF" not in filled["body"]


def test_a_fill_leaving_placeholders_behind_is_rejected(baseline, monkeypatch):
    enable_llm(monkeypatch, {"header": "", "body": "Hi Rahul, your invoice {{2}} is due.",
                             "footer": "", "buttons": ""})
    components = {"header": "", "body": "Hi {{1}}, your invoice {{2}} is due.", "footer": "", "buttons": ""}
    filled, how = compose.fill_sample_values(components, baseline)
    assert how != "llm"
