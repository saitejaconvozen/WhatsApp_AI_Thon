import numpy as np
import pytest

from templatelab import embedding


def synthetic(n=200, dimensions=24, separation=2.0, seed=0):
    """Two Gaussian blobs offset along axis 0, with correlated nuisance directions.

    The nuisance covariance is the point: a centroid difference would be dragged
    by it, while LDA divides it out. The tests below rely on that difference.
    """
    rng = np.random.default_rng(seed)
    mix = rng.normal(size=(dimensions, dimensions))
    covariance = mix @ mix.T / dimensions
    noise = rng.multivariate_normal(np.zeros(dimensions), covariance, size=2 * n)
    offset = np.zeros(dimensions)
    offset[0] = separation
    vectors = np.vstack([noise[:n] + offset, noise[n:]])
    labels = np.array(["UTILITY"] * n + ["MARKETING"] * n)
    return vectors, labels


def test_direction_separates_the_classes():
    vectors, labels = synthetic()
    direction = embedding.fit_direction(vectors, labels)
    assert direction.separation(vectors, labels) > 1.0
    scores = direction.score(vectors).ravel()
    assert scores[labels == "UTILITY"].mean() > scores[labels == "MARKETING"].mean()


def test_scores_are_scaled_to_the_class_gap():
    vectors, labels = synthetic()
    direction = embedding.fit_direction(vectors, labels)
    scores = direction.score(vectors).ravel()
    # 0 is the marketing mean and 1 the utility mean, so a displacement reads as
    # a fraction of the gap closed.
    assert scores[labels == "MARKETING"].mean() == pytest.approx(0.0, abs=0.05)
    assert scores[labels == "UTILITY"].mean() == pytest.approx(1.0, abs=0.05)


def test_lda_beats_a_centroid_difference_under_correlated_noise():
    vectors, labels = synthetic()
    lda = embedding.fit_direction(vectors, labels)
    centroid = vectors[labels == "UTILITY"].mean(axis=0) - vectors[labels == "MARKETING"].mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)

    def cohens_d(vector):
        projected = vectors @ vector
        a, b = projected[labels == "UTILITY"], projected[labels == "MARKETING"]
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        return abs(a.mean() - b.mean()) / pooled

    assert cohens_d(lda.vector) > cohens_d(centroid)


def test_too_few_examples_is_reported_not_raised_as_a_crash():
    vectors, labels = synthetic(n=5)
    with pytest.raises(embedding.Unavailable, match="at least"):
        embedding.fit_direction(vectors, labels)


def test_direction_round_trips_through_disk(tmp_path):
    vectors, labels = synthetic()
    direction = embedding.fit_direction(vectors, labels)
    path = tmp_path / "direction.json"
    embedding.save(direction, path)
    restored = embedding.load(path)
    assert np.allclose(restored.vector, direction.vector)
    assert np.allclose(restored.score(vectors), direction.score(vectors))


def test_missing_direction_file_is_reported():
    with pytest.raises(embedding.Unavailable, match="No direction"):
        embedding.load("/nonexistent/direction.json")


def test_displacement_reports_movement_along_the_direction():
    vectors, labels = synthetic()
    direction = embedding.fit_direction(vectors, labels)
    marketing = vectors[labels == "MARKETING"][:1]
    utility = vectors[labels == "UTILITY"][:1]
    moved = embedding.displacement(marketing, utility, direction)
    assert moved["moved"] == pytest.approx(moved["after"] - moved["before"], abs=1e-6)
    assert moved["moved"] > 0


def test_ranking_puts_anchor_preserving_candidates_first(monkeypatch):
    """A candidate that moves furthest but drops the anchor must not win.

    Deleting the transactional detail is the shortest path toward the utility
    region and the one edit that can never help, so eligibility outranks movement.
    """
    vectors, labels = synthetic()
    direction = embedding.fit_direction(vectors, labels)
    utility_point = vectors[labels == "UTILITY"].mean(axis=0)
    marketing_point = vectors[labels == "MARKETING"].mean(axis=0)
    midpoint = (utility_point + marketing_point) / 2

    lookup = {"original": marketing_point, "drops anchor": utility_point, "keeps {{1}}": midpoint}
    monkeypatch.setattr(embedding, "encode",
                        lambda texts, encoder=None, batch_size=64: np.vstack([lookup[t] for t in texts]))

    ranked = embedding.rank_candidates(
        "original", ["drops anchor", "keeps {{1}}"], direction, encoder=object(),
        keeps_anchor=lambda text: "{{" in text)
    assert ranked["candidates"][0]["text"] == "keeps {{1}}"
    assert ranked["candidates"][0]["eligible"] is True
    # The ineligible one genuinely moved further; it is ranked, not hidden.
    assert ranked["candidates"][1]["moved"] > ranked["candidates"][0]["moved"]
    assert ranked["candidates"][1]["eligible"] is False


def test_encoder_absence_is_reported_not_raised(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("sentence_transformers"):
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(embedding.Unavailable, match="not installed"):
        embedding.load_encoder()
