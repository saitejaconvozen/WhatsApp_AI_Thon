"""Behavioural regressions that aggregate metrics cannot catch.

These exist because a headline metric can hold steady, or even improve, while the
model's behaviour on the decisive comparison inverts. Two instances so far:

  - The sibling prototype's contamination augmentation left cross-validated
    accuracy flat at 0.792 while the clean transactional sentence flipped to
    scoring *more* promotional than the same sentence plus a promo rider.
  - In this repo, adding structural features raised utility precision from 78.7%
    to 80.9% while F1 fell from 70.9% to 62.4%.

Accuracy would have passed both. A minimal pair would have failed both.

Assertions are on relative ordering with a margin, not on absolute cut-offs. The
fixture model is trained on twenty synthetic services, so its absolute
probabilities carry no meaning beyond this file; what must hold is that adding a
promotional rider to a transactional template moves it away from UTILITY.
"""

import pytest

from templatelab.data import Store
from templatelab.model import Baseline
from test_experiments import dated_records

# Observed separation on this fixture is ~0.36. Requiring 0.15 leaves room for
# incidental pipeline changes while still failing an inversion or a collapse.
MINIMUM_SEPARATION = 0.15


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    store = Store(tmp_path_factory.mktemp("minimal-pairs"))
    store.import_records(dated_records())
    baseline = Baseline(store)
    baseline.train()
    return baseline


def utility_probability(baseline, body, buttons=""):
    prediction = baseline.predict({"body": body, "header": "", "footer": "",
                                   "buttons": buttons, "format": "TEXT"})
    assert prediction["available"], prediction.get("reason")
    return prediction["utility_probability"]


PAIRS = [
    ("Your electricity invoice {{id}} is due.",
     "Your electricity invoice {{id}} is due. Buy today and get exclusive discounts."),
    ("Your water invoice {{id}} is due.",
     "Your water invoice {{id}} is due. Buy water today and get exclusive discounts."),
]


@pytest.mark.parametrize("clean,contaminated", PAIRS)
def test_promotional_rider_moves_a_template_away_from_utility(trained, clean, contaminated):
    clean_probability = utility_probability(trained, clean)
    contaminated_probability = utility_probability(trained, contaminated)
    assert clean_probability > contaminated_probability, (
        "Adding a promotional rider made the template look MORE like utility. "
        "This is the inversion the module docstring describes.")
    assert clean_probability - contaminated_probability >= MINIMUM_SEPARATION


def test_clean_and_promotional_templates_sit_on_opposite_sides(trained):
    # Loose bounds deliberately: observed values are 0.848 and 0.154, so these
    # fail on a real collapse rather than on ordinary drift.
    assert utility_probability(trained, "Your parking invoice {{id}} is due.") > 0.6
    assert utility_probability(trained, "Buy hosting today and get exclusive discounts.") < 0.4


def test_an_unseen_transactional_template_is_not_penalised_for_being_unseen(trained):
    """Guards FINDINGS §2: short text unlike the training set scoring as promotional."""
    seen = utility_probability(trained, "Your electricity invoice {{id}} is due.")
    unseen = utility_probability(trained, "Your parking invoice {{id}} is due.")
    assert abs(seen - unseen) < 0.25
