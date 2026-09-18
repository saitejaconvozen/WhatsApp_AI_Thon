import pytest

from templatelab.economics import (CostModel, DEFAULT_RISK_BUDGET, RISK_BUDGETS, allocate,
                                   decide, pure_per_message_costs, risk_curve)


def portfolio():
    return [
        {"id": "payment", "name": "Payment received", "p_utility": 0.97, "monthly_volume": 400000},
        {"id": "shipped", "name": "Order shipped", "p_utility": 0.95, "monthly_volume": 250000},
        {"id": "booking", "name": "Booking confirmed", "p_utility": 0.92, "monthly_volume": 120000},
        {"id": "reminder", "name": "Due reminder", "p_utility": 0.84, "monthly_volume": 90000},
        {"id": "mixed", "name": "Shipped plus offer", "p_utility": 0.55, "monthly_volume": 60000},
        {"id": "winback", "name": "We miss you", "p_utility": 0.12, "monthly_volume": 300000},
    ]


@pytest.mark.parametrize("p_utility", [0.0, 0.05, 0.3, 0.5, 0.8, 0.99])
def test_pure_per_message_cost_makes_declaring_utility_weakly_dominant(p_utility):
    """The reason the risk budget exists, encoded as a test.

    Strip out rework and the account-health penalty and declaring UTILITY is never
    worse, whatever the template says. A system that optimised this number alone
    would declare everything UTILITY and walk the account into a restriction.
    """
    costs = pure_per_message_costs()
    assert costs.cost_declare_utility(p_utility) <= costs.cost_declare_marketing()
    assert costs.breakeven_utility_probability() == 0.0
    assert decide(p_utility, costs)["declare"] == "UTILITY"


def test_pricing_the_consequence_restores_a_real_threshold():
    costs = CostModel()
    threshold = costs.breakeven_utility_probability()
    assert 0.8 < threshold < 0.95
    assert decide(0.99, costs)["declare"] == "UTILITY"
    assert decide(0.50, costs)["declare"] == "MARKETING"
    # Mirrors the sibling prototype's p_marketing threshold of 0.134.
    assert threshold == pytest.approx(1 - 0.134, abs=0.002)


@pytest.mark.parametrize("budget", RISK_BUDGETS)
def test_risk_budget_is_never_exceeded(budget):
    plan = allocate(portfolio(), risk_budget=budget)
    assert plan["expected_misclassification_rate"] <= budget + 1e-9


def test_risk_curve_is_monotone_in_the_budget():
    curve = risk_curve(portfolio())
    shares = [row["utility_share"] for row in curve]
    savings = [row["projected_saving"] for row in curve]
    assert shares == sorted(shares)
    assert savings == sorted(savings)
    assert curve[0]["risk_budget"] < curve[-1]["risk_budget"]


def test_a_purely_promotional_template_is_never_declared_utility():
    plans = {p["id"]: p for p in allocate(portfolio(), risk_budget=max(RISK_BUDGETS))["plans"]}
    assert plans["winback"]["declare"] == "MARKETING"
    assert plans["payment"]["declare"] == "UTILITY"


def test_higher_volume_wins_when_probabilities_tie():
    candidates = [
        {"id": "anchor", "p_utility": 0.99, "monthly_volume": 100},
        {"id": "small", "p_utility": 0.80, "monthly_volume": 10},
        {"id": "large", "p_utility": 0.80, "monthly_volume": 100},
    ]
    # Budget admits the anchor and exactly one of the tied pair.
    plans = {p["id"]: p for p in allocate(candidates, risk_budget=0.105)["plans"]}
    assert plans["large"]["declare"] == "UTILITY"
    assert plans["small"]["declare"] == "MARKETING"


def test_volumes_are_optional_and_fall_back_to_equal_weights():
    candidates = [{"id": "a", "p_utility": 0.98}, {"id": "b", "p_utility": 0.97}]
    plan = allocate(candidates, risk_budget=DEFAULT_RISK_BUDGET)
    assert plan["volumes"] == "equal weights (none supplied)"
    assert plan["total_volume"] == 2
    assert plan["declared_utility"] == 2


def test_an_impossible_budget_declares_nothing_rather_than_cheating():
    # The best template carries 3% risk, so a 2% budget cannot be met at all.
    plan = allocate(portfolio(), risk_budget=0.02)
    assert plan["declared_utility"] == 0
    assert plan["projected_saving"] == 0
    assert plan["utility_share"] == 0.0


def test_saving_is_measured_against_declaring_everything_marketing():
    plan = allocate(portfolio(), risk_budget=DEFAULT_RISK_BUDGET)
    assert plan["baseline_cost_all_marketing"] == pytest.approx(plan["total_volume"] * 7.5)
    assert plan["projected_saving"] == pytest.approx(
        plan["baseline_cost_all_marketing"] - plan["projected_cost"])
    assert 0 < plan["projected_saving"] < plan["baseline_cost_all_marketing"]
