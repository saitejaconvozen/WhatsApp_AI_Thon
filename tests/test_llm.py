import json

import pytest

from templatelab import llm
from templatelab.data import Store
from templatelab.llm_eval import aggregate, check_budget, prepare


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for variable in (llm.BACKEND_VARIABLE, llm.EGRESS_VARIABLE, llm.MODEL_VARIABLE,
                     "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(variable, raising=False)


@pytest.fixture
def reviewer(tmp_path):
    class NoBaseline:
        def predict(self, record):
            return {"available": False, "neighbors": []}
    return llm.Reviewer(Store(tmp_path), NoBaseline())


def exploding_backend(prompt, model):
    raise AssertionError("A disabled or unacknowledged reviewer must not call a backend.")


def test_disabled_by_default_and_makes_no_call(reviewer, monkeypatch):
    monkeypatch.setitem(llm.BACKENDS, "anthropic", exploding_backend)
    status = reviewer.status()
    assert status["available"] is False
    assert status["backend"] == "disabled"
    assert reviewer.review({"body": "Your invoice is ready."})["available"] is False


def test_backend_without_egress_acknowledgement_makes_no_call(reviewer, monkeypatch):
    monkeypatch.setenv(llm.BACKEND_VARIABLE, "anthropic")
    monkeypatch.setitem(llm.BACKENDS, "anthropic", exploding_backend)
    status = reviewer.status()
    assert status["available"] is False
    assert llm.EGRESS_VARIABLE in status["reason"]
    assert reviewer.review({"body": "Your invoice is ready."})["available"] is False


def test_unknown_backend_is_reported_not_raised(reviewer, monkeypatch):
    monkeypatch.setenv(llm.BACKEND_VARIABLE, "nowhere")
    monkeypatch.setenv(llm.EGRESS_VARIABLE, "1")
    assert reviewer.status()["available"] is False


def test_review_parses_caches_and_does_not_recall(reviewer, monkeypatch):
    monkeypatch.setenv(llm.BACKEND_VARIABLE, "anthropic")
    monkeypatch.setenv(llm.EGRESS_VARIABLE, "1")
    calls = []

    def backend(prompt, model):
        calls.append(prompt)
        return 'Sure: {"category":"MARKETING","confidence":0.8,"clauses":["M6"],"rationale":"Mixed."}'

    monkeypatch.setitem(llm.BACKENDS, "anthropic", backend)
    record = {"header": "", "body": "Invoice {{1}} is due. Upgrade now!", "footer": "", "buttons": ""}
    first = reviewer.review(record)
    assert first["category"] == "MARKETING" and first["clauses"] == ["M6"] and first["cached"] is False
    second = reviewer.review(record)
    assert second["cached"] is True and len(calls) == 1
    assert "M6" in calls[0] and "Upgrade now!" in calls[0]


@pytest.mark.parametrize("text", ["no json here", "{not json}", '{"category":"SOMETHING"}'])
def test_bad_responses_are_reported_not_raised(reviewer, monkeypatch, text):
    monkeypatch.setenv(llm.BACKEND_VARIABLE, "anthropic")
    monkeypatch.setenv(llm.EGRESS_VARIABLE, "1")
    monkeypatch.setitem(llm.BACKENDS, "anthropic", lambda prompt, model: text)
    assert reviewer.review({"body": "Invoice is due."})["available"] is False


def test_policy_definitions_are_versioned_and_in_the_prompt():
    version = llm.policy_version()
    assert version["version"] != "unknown" and version["checked_on"] != "unknown"
    prompt = llm.build_prompt({"body": "Order {{1}} shipped."}, [])
    assert "UTILITY" in prompt and "M6" in prompt and "Order {{1}} shipped." in prompt


@pytest.mark.parametrize("backend", ["anthropic", "deepseek"])
def test_every_backend_is_blocked_without_the_egress_opt_in(reviewer, monkeypatch, backend):
    monkeypatch.setenv(llm.BACKEND_VARIABLE, backend)
    monkeypatch.setitem(llm.BACKENDS, backend, exploding_backend)
    assert reviewer.review({"body": "Your invoice is ready."})["available"] is False


@pytest.mark.parametrize("backend, expected", [("anthropic", "claude-sonnet-5"), ("deepseek", "deepseek-chat")])
def test_each_backend_has_its_own_default_model(monkeypatch, backend, expected):
    monkeypatch.setenv(llm.BACKEND_VARIABLE, backend)
    assert llm.configuration()["model"] == expected
    monkeypatch.setenv(llm.MODEL_VARIABLE, "override-1")
    assert llm.configuration()["model"] == "override-1"


def test_deepseek_parses_its_response_shape(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    captured = {}

    def fake_post(url, headers, payload):
        captured.update(url=url, headers=headers, payload=payload)
        return {"choices": [{"message": {"content": '{"category":"UTILITY"}'}}]}

    monkeypatch.setattr(llm, "post", fake_post)
    assert llm.deepseek_call("prompt text", "deepseek-chat") == '{"category":"UTILITY"}'
    assert captured["url"].endswith("/chat/completions")
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["payload"]["messages"][0]["content"] == "prompt text"


def test_anthropic_parses_its_response_shape(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    captured = {}

    def fake_post(url, headers, payload):
        captured.update(headers=headers)
        return {"content": [{"type": "thinking", "text": "ignored"}, {"type": "text", "text": "kept"}]}

    monkeypatch.setattr(llm, "post", fake_post)
    assert llm.anthropic_call("prompt", "claude-sonnet-5") == "kept"
    assert captured["headers"]["x-api-key"] == "test-key"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"


@pytest.mark.parametrize("backend, call", [("anthropic", "anthropic_call"), ("deepseek", "deepseek_call")])
def test_missing_key_is_reported_not_raised(backend, call):
    with pytest.raises(llm.Unavailable, match="is not set"):
        getattr(llm, call)("prompt", "model")


def test_deepseek_without_choices_is_unavailable(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(llm, "post", lambda *a, **k: {"choices": []})
    with pytest.raises(llm.Unavailable, match="no choices"):
        llm.deepseek_call("prompt", "deepseek-chat")


def test_retrieval_balances_labels_rather_than_taking_the_top_k():
    lopsided = ([{"category": "MARKETING", "body": f"promo {i}", "similarity": 0.9 - i / 100}
                 for i in range(10)]
                + [{"category": "UTILITY", "body": "order shipped", "similarity": 0.4}])
    selected = llm.balanced_examples(lopsided, per_class=2)
    assert [n["category"] for n in selected].count("UTILITY") == 1
    assert [n["category"] for n in selected].count("MARKETING") == 2
    # Closest precedent sits last, nearest the template under review.
    assert selected[-1]["similarity"] == 0.9
    prompt = llm.build_prompt({"body": "Order {{1}} shipped."}, lopsided)
    assert "Meta recorded UTILITY" in prompt and "Meta recorded MARKETING" in prompt
    assert "opposite sides of the boundary" in prompt


def test_single_label_pool_omits_the_contrast_claim():
    only_marketing = [{"category": "MARKETING", "body": "promo", "similarity": 0.8}]
    assert "opposite sides of the boundary" not in llm.build_prompt({"body": "x"}, only_marketing)


def holdout_row(identifier, requested, recorded, predicted):
    return {"id": identifier, "requested_category": requested,
            "recorded_category": recorded, "prediction": predicted}


def test_evaluation_scores_both_judges_on_the_same_rows():
    rows = [holdout_row("a", "UTILITY", "UTILITY", "MARKETING"),
            holdout_row("b", "UTILITY", "MARKETING", "MARKETING"),
            holdout_row("c", "MARKETING", "MARKETING", "MARKETING")]
    results = [{"available": True, "category": "UTILITY", "clauses": ["U1"], "confidence": 0.9},
               {"available": True, "category": "MARKETING", "clauses": ["M6"], "confidence": 0.7},
               {"available": False, "reason": "rate limited"}]
    report = aggregate(rows, results)

    assert report["judged"] == 2 and report["failed"] == {"rate limited": 1}
    # The failed row is excluded from BOTH sides, so the comparison stays paired.
    assert report["llm"]["all"]["samples"] == report["baseline_same_rows"]["all"]["samples"] == 2
    assert report["llm"]["all"]["accuracy"] == 1.0
    assert report["baseline_same_rows"]["all"]["accuracy"] == 0.5
    assert report["disagreement"] == {"count": 1, "llm_correct": 1, "baseline_correct": 0,
                                      "neither_correct": 0, "ids": ["a"]}
    assert report["agreed"] == 1
    assert report["clauses_cited"] == {"U1": 1, "M6": 1}
    assert report["llm"]["requested_utility"]["samples"] == 2
    # The only requested-MARKETING row failed, so that slice has nothing to score.
    assert report["llm"]["requested_marketing"] is None


def test_evaluation_counts_authentication_verdicts_as_errors():
    rows = [holdout_row("a", "UTILITY", "UTILITY", "UTILITY")]
    report = aggregate(rows, [{"available": True, "category": "AUTHENTICATION", "clauses": []}])
    assert report["authentication_verdicts"] == 1
    assert report["llm"]["all"]["accuracy"] == 0.0


def test_budget_refuses_paid_calls_beyond_the_limit():
    with pytest.raises(SystemExit, match="above the --limit"):
        check_budget(uncached=250, limit=100, yes=False)
    assert check_budget(uncached=250, limit=100, yes=True) is None
    assert check_budget(uncached=10, limit=100, yes=False) is None


def test_prepare_reports_which_rows_are_already_cached(reviewer, monkeypatch):
    monkeypatch.setenv(llm.BACKEND_VARIABLE, "deepseek")
    monkeypatch.setenv(llm.EGRESS_VARIABLE, "1")
    monkeypatch.setitem(llm.BACKENDS, "deepseek",
                        lambda prompt, model: '{"category":"UTILITY","clauses":[]}')
    rows = [{"id": "a", "body": "Order {{1}} shipped."}, {"id": "b", "body": "Half price today!"}]
    assert [item["cached"] for item in prepare(reviewer, rows)] == [False, False]
    reviewer.review(rows[0])
    assert [item["cached"] for item in prepare(reviewer, rows)] == [True, False]


def test_retrieval_excludes_the_row_being_scored(tmp_path):
    """The serving index contains the holdout, so a held-out row retrieves itself.

    Left unchecked that put the row's own recorded label into the prompt and the
    judge scored a fake 100%. This exclusion is the only thing preventing it.
    """
    from templatelab.model import Baseline
    from templatelab.llm import Reviewer
    from test_experiments import dated_records

    store = Store(tmp_path)
    store.import_records(dated_records())
    baseline = Baseline(store)
    baseline.train()
    scored = baseline.bundle["records"][0]
    reviewer = Reviewer(store, baseline)

    unfiltered = reviewer.neighbors(scored)
    assert unfiltered[0]["id"] == scored["id"]
    assert unfiltered[0]["similarity"] == pytest.approx(1.0, abs=1e-6)

    filtered = reviewer.neighbors(scored, exclude={scored["id"]})
    assert scored["id"] not in [n["id"] for n in filtered]
    assert filtered, "excluding the row itself must still leave real precedents"
