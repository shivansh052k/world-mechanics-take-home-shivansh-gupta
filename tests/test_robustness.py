"""Subspace metrics, re-expression between standardized spaces, stratified differences."""
import numpy as np
import pytest

from vjepa_physics.robustness import (
    null_metrics, orthonormal, re_express, rescale, stratified_difference, subspace_metrics,
)


def test_principal_angles_hand_case():
    t = np.radians(30)
    qa = np.eye(3)[:, :2]
    qb = np.column_stack([[1, 0, 0], [0, np.cos(t), np.sin(t)]])
    m = subspace_metrics(qa, qb)
    np.testing.assert_allclose(m["angles_degrees"], [0.0, 30.0], atol=1e-10)
    assert m["overlap_a_on_b"] == pytest.approx((1 + np.cos(t) ** 2) / 2)
    assert m["grassmann"] == pytest.approx(t)


def test_random_overlap_is_k_over_d():
    """The physics paper's chance level: a random k_B-dim subspace overlaps a fixed k_A-dim one by k_A / d."""
    d, k_a, k_b, draws = 256, 8, 4, 2000
    qa = orthonormal(np.random.default_rng(1).standard_normal((d, k_a)))
    overlaps = null_metrics(qa, np.eye(d), k_b, lambda q: q, draws, seed=0)["overlap_a_on_b"]
    assert abs(overlaps.mean() - k_a / d) < 4 * overlaps.std() / np.sqrt(draws)


def test_weight_re_expression_keeps_predictions():
    """Weights moved into another standardized space predict the same values up to a constant."""
    rng = np.random.default_rng(0)
    x = rng.normal(3.0, 2.0, (100, 6)) * rng.uniform(0.5, 5.0, 6)
    own_mean, own_scale = x.mean(axis=0), x.std(axis=0)
    other_mean, other_scale = x[:40].mean(axis=0), x[:40].std(axis=0)
    w = rng.normal(size=(6, 1))
    before = ((x - own_mean) / own_scale) @ w
    after = ((x - other_mean) / other_scale) @ rescale(w, own_scale, other_scale, "weights")
    assert np.ptp(after - before) < 1e-10 * np.abs(before).max()
    q = re_express(w, own_scale, other_scale, "weights")
    np.testing.assert_allclose(q.T @ q, np.eye(1), atol=1e-12)


def test_stratified_difference_hand_case():
    errors = np.array([1.0, 2.0, 3.0, 5.0, 10.0, 0.0, 7.0])
    flagged = np.array([0, 0, 1, 1, 1, 0, 0], bool)
    strata = np.array(["A", "A", "A", "A", "B", "B", "C"])  # C has no flagged clip and is skipped
    result = stratified_difference(errors, flagged, strata, n_resamples=200, seed=0)
    assert result["difference"] == pytest.approx(2 / 3 * 2.5 + 1 / 3 * 10.0)  # weights = flagged clips per stratum
    assert result["strata"] == ["A", "B"]
    with pytest.raises(ValueError):
        stratified_difference(errors, np.zeros(7, bool), strata, n_resamples=10, seed=0)