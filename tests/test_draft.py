"""Drafting a utility template from a description.

Generation has been deployed and unmeasured for this project's whole life. These
tests do not claim Meta would approve a draft -- only submission establishes that
-- but they pin the properties that are checkable without it: the hard
constraints hold, an inherently promotional task is refused rather than dressed
up, a losing template form is never chosen, and a weak first draft is refined
rather than returned.
"""

import json

import pytest

from templatelab import draft as draft_module


class FakeReward:
    """Scores by keyword so a test can steer the refinement loop deterministically."""

    def __init__(self, scores=None):
        self.scores = scores or {}
        self.default = 0.9

    def embed(self, texts):
        import numpy as np
        return np.array([[float(len(t) % 7), 1.0, 0.5] for t in texts])

    def score(self, text):
        for needle, value in self.scores.items():
            if needle in text:
                return value
        return self.default


@pytest.fixture
def drafter(tmp_path, monkeypatch):
    forms = {"forms": [
        {"form": 0, "terms": ["slots", "confirm"], "approval_rate": 1.0, "approved": 86,
         "downgraded": 0, "exemplars": ["Your booking {{id}} has been scheduled."]},
        {"form": 1, "terms": ["feedback", "survey"], "approval_rate": 0.17, "approved": 27,
         "downgraded": 131, "exemplars": ["How was your experience?"]},
        {"form": 2, "terms": ["agreement", "draft"], "approval_rate": 0.66, "approved": 25,
         "downgraded": 13, "exemplars": ["Your agreement {{id}} draft is ready."]},
    ]}
    (tmp_path / "forms.json").write_text(json.dumps(forms), encoding="utf-8")

    class Store:
        directory = tmp_path
    monkeypatch.setattr(draft_module, "Reward", lambda store: FakeReward())
    monkeypatch.setenv("TEMPLATELAB_LLM_BACKEND", "openai_compatible")
    monkeypatch.setenv("TEMPLATELAB_LLM_API_KEY", "test-key")
    return draft_module.Drafter(Store())


def reply(**payload):
    return lambda prompt, model: json.dumps(payload)


def test_a_losing_form_is_never_chosen(drafter):
    """The feedback form is approved 17% of the time; drafting into it would be
    choosing a shape Meta rejects five times in six."""
    assert all(f["approval_rate"] >= 0.60 for f in drafter.safe)
    assert ["feedback", "survey"] not in [f["terms"] for f in drafter.safe]


def test_a_promotional_task_is_refused_not_dressed_up(drafter, monkeypatch):
    monkeypatch.setitem(draft_module.BACKENDS, "openai_compatible",
                        reply(possible=False, reason="Inherently promotional."))
    result = drafter.draft("Promote our painting service with 20% off")
    assert result["possible"] is False
    assert "promotional" in result["reason"].lower()


def test_a_draft_without_a_reference_is_rejected(drafter, monkeypatch):
    """No reference means nothing identifies which transaction it reports on."""
    monkeypatch.setitem(draft_module.BACKENDS, "openai_compatible",
                        reply(possible=True, body="Your order has shipped.", reason="ok"))
    result = drafter.draft("Tell them the order shipped", rounds=1)
    assert result["possible"] is False


def test_a_complete_draft_is_returned_with_its_form_evidence(drafter, monkeypatch):
    monkeypatch.setitem(draft_module.BACKENDS, "openai_compatible",
                        reply(possible=True, name="ORDER_SHIPPED",
                              body="Your order {{order_id}} has been shipped on {{date}}.",
                              buttons="Track order", reason="Reports an existing order."))
    result = drafter.draft("Tell them order shipped", rounds=1)
    assert result["possible"] is True
    assert result["has_anchor"] and result["has_placeholder"] and not result["promotional"]
    assert result["name"] == "ORDER_SHIPPED"
    assert result["buttons"] == "Track order"
    # The draft carries Meta's historical rate for the shape it follows.
    assert result["form_approved"] == 86 and result["form_downgraded"] == 0


def test_a_weak_draft_is_refined_rather_than_returned(drafter, monkeypatch):
    drafter.reward.scores = {"WEAK": 0.10, "STRONG": 0.95}
    bodies = iter([
        json.dumps({"possible": True, "body": "WEAK: your order {{id}} was sent.", "reason": "a"}),
        json.dumps({"possible": True, "body": "STRONG: your order {{id}} has been shipped.", "reason": "b"}),
    ])
    monkeypatch.setitem(draft_module.BACKENDS, "openai_compatible",
                        lambda prompt, model: next(bodies))
    result = drafter.draft("Tell them order shipped", rounds=3, target=0.60)
    assert result["possible"] is True
    assert "STRONG" in result["body"], "the loop must keep the better-scoring draft"
    assert result["rounds_used"] == 2
    assert result["score_trail"] == [0.10, 0.95]


def test_placeholders_are_required_so_the_template_can_vary(drafter, monkeypatch):
    monkeypatch.setitem(draft_module.BACKENDS, "openai_compatible",
                        reply(possible=True, body="Your order 12345 has been shipped.", reason="ok"))
    result = drafter.draft("Tell them order shipped", rounds=1)
    assert result["possible"] is False


def test_an_unmatched_form_does_not_lend_its_approval_rate(drafter, monkeypatch):
    """The ten forms above 60% approval do not span every task. A support-ticket
    draft sits 0.34 from the nearest of them -- reporting that form's 91% record
    for it would be inventing evidence for the draft."""
    monkeypatch.setattr(draft_module.Drafter, "classify_draft",
                        lambda self, body: (self.safe[0], 0.16))
    monkeypatch.setitem(draft_module.BACKENDS, "openai_compatible",
                        reply(possible=True, body="Your order {{id}} has been shipped.", reason="ok"))
    result = drafter.draft("Tell them the support ticket is resolved", rounds=1)
    assert result["possible"] is True
    assert result["form_matched"] is False
    assert result["form_approval_rate"] is None
    assert result["form_approved"] is None
    assert "no historical approval rate applies" in result["evidence"]


def test_a_matched_form_does_lend_its_approval_rate(drafter, monkeypatch):
    monkeypatch.setattr(draft_module.Drafter, "classify_draft",
                        lambda self, body: (self.safe[0], 0.72))
    monkeypatch.setitem(draft_module.BACKENDS, "openai_compatible",
                        reply(possible=True, body="Your booking {{id}} has been scheduled.", reason="ok"))
    result = drafter.draft("Confirm their booking slot", rounds=1)
    assert result["form_matched"] is True
    assert result["form_approved"] == 86 and result["form_downgraded"] == 0
    assert "approved 86 of 86 times" in result["evidence"]


def test_form_evidence_is_measured_on_the_draft_not_the_task(drafter, monkeypatch):
    """A task description and a template body sit in different regions of the
    encoder's space, so the task's own similarity must not decide the evidence:
    the same drafts that match a task at 0.20 match their own form at 0.51."""
    seen = {}
    monkeypatch.setattr(draft_module.Drafter, "choose_form",
                        lambda self, task: (self.safe[1], 0.05))
    def classify(self, body):
        seen["body"] = body
        return self.safe[0], 0.55
    monkeypatch.setattr(draft_module.Drafter, "classify_draft", classify)
    monkeypatch.setitem(draft_module.BACKENDS, "openai_compatible",
                        reply(possible=True, body="Your booking {{id}} has been scheduled.", reason="ok"))
    result = drafter.draft("anything at all", rounds=1)
    assert seen["body"] == "Your booking {{id}} has been scheduled."
    # The low-approval form guided the wording; the evidence comes from the
    # form the finished draft actually belongs to.
    assert result["form_matched"] is True
    assert result["form_approved"] == 86


def test_a_kannada_template_is_not_answered_in_english(drafter, monkeypatch):
    """Half the downgraded templates carry Indic script while their stored
    language says ENGLISH_US. A draft the recipient cannot read is not a better
    template, so a script mismatch fails the constraints like any other."""
    english = json.dumps({"possible": True, "reason": "ok",
                          "body": "Your loan account {{id}} has an installment due."})
    # Real Indic templates in this corpus are mixed: Kannada or Hindi prose
    # around English lending terms and ASCII placeholders.
    kannada = json.dumps({"possible": True, "reason": "ok",
                          "body": "ಪ್ರಿಯ ಗ್ರಾಹಕರೇ, ನಿಮ್ಮ Loan A/c {{id}} EMI ಬಾಕಿ ಇದೆ."})
    replies = iter([english, kannada])
    monkeypatch.setitem(draft_module.BACKENDS, "openai_compatible",
                        lambda prompt, model: next(replies))
    result = drafter.draft("EMI due", rounds=2, script="Kannada")
    assert result["possible"] is True
    assert result["kept_script"] is True
    assert "ನಿಮ್ಮ" in result["body"], "the English attempt must be rejected and retried"


def test_script_is_read_from_the_body_not_the_language_field():
    assert draft_module.script_of("ನಿಮ್ಮ ಸಾಲದ ಖಾತೆ") == "Kannada"
    assert draft_module.script_of("आपका ऋण खाता") == "Devanagari"
    assert draft_module.script_of("Your loan account {{id}}") is None
    # Emoji and placeholders must not be mistaken for a script.
    assert draft_module.script_of("Hi {{1}} 👋 your order shipped") is None
