"""What to declare, given money and risk, rather than what the model thinks.

The model produces a calibrated p_utility. Nothing upstream says what to do with
it, and argmax at 0.5 is wrong when the two errors cost very different amounts.

THE UNCOMFORTABLE MATH. Declaring MARKETING costs the marketing rate, certainly.
Declaring UTILITY costs the utility rate when Meta agrees, and the marketing rate
anyway when it does not. On per-message cost alone that second option is weakly
better for every p_utility > 0, however promotional the template is. A system
that optimises per-message cost therefore declares everything UTILITY, which is
exactly the behaviour Meta polices: it warns, then withdraws the 24-hour
reclassification notice, and can stop utility messaging altogether.

The dominance only disappears once being wrong is priced. Two ways to do that:

  1. Per-template: price reclassification, rework and account health explicitly,
     so the threshold is arguable instead of assumed.
  2. Portfolio: pick how much aggregate misclassification the business is willing
     to be caught with, and spend that budget where it buys the most saving.

The second is what should ship. It converts an unknowable account-health penalty
into a policy choice compliance can sign off on. `pure_per_message_costs` exists
only to demonstrate the dominance in tests; it is never a default.

Probabilities here are p_utility, matching the calibrated model. The sibling
prototype this is ported from used p_marketing, so every threshold is mirrored:
our "declare UTILITY above 0.866" is their "declare UTILITY below 0.134".
"""

from dataclasses import dataclass

MARKETING_COST = 7.5
UTILITY_COST = 1.0
DEFAULT_RISK_BUDGET = 0.05
RISK_BUDGETS = (0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20)


@dataclass
class CostModel:
    marketing: float = MARKETING_COST
    utility: float = UTILITY_COST
    # Reclassification cost beyond the rate difference: rework, resubmission
    # delay, campaign disruption. Per-message units.
    rework: float = 2.0
    # Account-health penalty, amortised per message. Deliberately large; it stands
    # in for a tail risk (losing utility messaging) that cannot be priced directly.
    health_penalty: float = 40.0

    def cost_declare_marketing(self):
        return self.marketing

    def cost_declare_utility(self, p_utility):
        reclassified = self.marketing + self.rework + self.health_penalty
        return p_utility * self.utility + (1.0 - p_utility) * reclassified

    def breakeven_utility_probability(self):
        """Minimum p_utility at which declaring UTILITY beats declaring MARKETING.

        Solving p*u + (1-p)*(m+r+h) = m. Note the direction: this is a floor on
        p_utility, where the p_marketing formulation gives a ceiling.
        """
        reclassified = self.marketing + self.rework + self.health_penalty
        denominator = reclassified - self.utility
        if denominator <= 0:
            return 0.0
        return max(0.0, min(1.0, (reclassified - self.marketing) / denominator))

    def saving_per_message(self):
        return self.marketing - self.utility


def pure_per_message_costs():
    """Costs with the consequence of being wrong stripped out.

    Only for demonstrating that this is the degenerate case: with no rework and no
    account-health penalty, declaring UTILITY is weakly dominant everywhere, so the
    threshold collapses to zero and every template is declared UTILITY.
    """
    return CostModel(rework=0.0, health_penalty=0.0)


def decide(p_utility, costs=None):
    costs = costs or CostModel()
    threshold = costs.breakeven_utility_probability()
    return {
        "p_utility": round(float(p_utility), 3),
        "threshold": round(threshold, 3),
        "declare": "UTILITY" if p_utility >= threshold else "MARKETING",
        "expected_cost_if_utility": round(costs.cost_declare_utility(p_utility), 2),
        "expected_cost_if_marketing": round(costs.cost_declare_marketing(), 2),
    }


def _weighted(candidates):
    """Volumes are optional; equal weights let the allocator run on any holdout."""
    supplied = any(c.get("monthly_volume") is not None for c in candidates)
    rows = []
    for candidate in candidates:
        volume = float(candidate.get("monthly_volume") or 0.0) if supplied else 1.0
        rows.append({"id": str(candidate.get("id", "")),
                     "name": str(candidate.get("name", ""))[:90],
                     "p_utility": float(candidate["p_utility"]),
                     "monthly_volume": volume})
    return rows, supplied


def allocate(candidates, risk_budget=DEFAULT_RISK_BUDGET, costs=None):
    """Choose which templates to declare UTILITY, subject to a risk budget.

    risk_budget is the share of declared-utility volume the business accepts
    being genuinely MARKETING in Meta's eyes.

    Ordering is descending p_utility, ties broken by descending volume. That is
    not a heuristic: saving per unit of risk is (m-u) * p/(1-p), which increases
    monotonically in p_utility, so taking templates in this order spends the
    budget in strictly decreasing order of value.

    The accepted set is the longest prefix of that order whose running
    misclassification ratio stays within budget. A prefix rather than a
    best-effort scan, because the ratio is non-decreasing along this ordering, so
    a prefix keeps the risk curve monotone in the budget — a curve that dipped as
    the budget loosened would be indefensible in front of compliance.
    """
    costs = costs or CostModel()
    rows, volumes_supplied = _weighted(candidates)
    ranked = sorted(rows, key=lambda r: (-r["p_utility"], -r["monthly_volume"]))

    utility_volume = risk_volume = 0.0
    accepted = set()
    for index, row in enumerate(ranked):
        volume = row["monthly_volume"]
        candidate_utility = utility_volume + volume
        candidate_risk = risk_volume + volume * (1.0 - row["p_utility"])
        if candidate_utility <= 0 or (candidate_risk / candidate_utility) > risk_budget:
            break
        utility_volume, risk_volume = candidate_utility, candidate_risk
        accepted.add(index)

    plans = []
    for index, row in enumerate(ranked):
        declared = index in accepted
        plans.append({**row,
                      "p_utility": round(row["p_utility"], 3),
                      "monthly_volume": row["monthly_volume"],
                      "declare": "UTILITY" if declared else "MARKETING",
                      "expected_monthly_saving": round(
                          row["monthly_volume"] * row["p_utility"] * costs.saving_per_message()
                          if declared else 0.0, 2),
                      "risk_contribution": round(
                          row["monthly_volume"] * (1.0 - row["p_utility"]) if declared else 0.0, 4)})

    total_volume = sum(row["monthly_volume"] for row in rows)
    baseline = total_volume * costs.marketing
    projected = sum(plan["monthly_volume"] * (costs.utility if plan["declare"] == "UTILITY"
                                              else costs.marketing) for plan in plans)
    return {
        "risk_budget": risk_budget,
        "volumes": "supplied" if volumes_supplied else "equal weights (none supplied)",
        "declared_utility": sum(1 for plan in plans if plan["declare"] == "UTILITY"),
        "templates": len(plans),
        "declared_utility_volume": round(utility_volume, 2),
        "total_volume": round(total_volume, 2),
        "utility_share": round(utility_volume / total_volume, 4) if total_volume else 0.0,
        "expected_misclassification_rate": round(risk_volume / utility_volume, 4) if utility_volume else 0.0,
        "baseline_cost_all_marketing": round(baseline, 2),
        "projected_cost": round(projected, 2),
        "projected_saving": round(baseline - projected, 2),
        "plans": plans,
    }


def risk_curve(candidates, budgets=RISK_BUDGETS, costs=None):
    """Utility share and saving across risk budgets: the trade-off compliance signs off."""
    curve = []
    for budget in budgets:
        plan = allocate(candidates, risk_budget=budget, costs=costs)
        curve.append({"risk_budget": budget,
                      "utility_share": plan["utility_share"],
                      "declared_utility": plan["declared_utility"],
                      "expected_misclassification_rate": plan["expected_misclassification_rate"],
                      "projected_saving": plan["projected_saving"]})
    return curve
