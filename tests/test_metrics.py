"""Probe metrics against sklearn and hand-worked cases."""
import numpy as np
import pytest
from sklearn.metrics import r2_score

from vjepa_physics.metrics import angles_from_sincos, circular_errors, r2, sincos_targets


def test_r2_matches_sklearn_single_and_multi_column():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(50, 2))
    p = y + rng.normal(scale=0.3, size=y.shape)
    assert r2(y[:, 0], p[:, 0]) == pytest.approx(r2_score(y[:, 0], p[:, 0]), abs=1e-12)
    assert r2(y, p) == pytest.approx(r2_score(y, p), abs=1e-12)  # (sin, cos): mean of the column scores


def test_r2_refuses_a_constant_target():
    with pytest.raises(ValueError):
        r2(np.ones(5), np.arange(5.0))


def test_circular_error_wraps_around_zero():
    errors = circular_errors(np.array([350.0, 0.0, 90.0]), np.array([10.0, 180.0, 270.0]))
    np.testing.assert_allclose(errors, [20.0, 180.0, 180.0])


def test_sincos_round_trip_ignores_length():
    theta = np.arange(0.0, 360.0, 5.625)  # the direction set's angle grid
    back = angles_from_sincos(3.0 * sincos_targets(theta))  # a probe's (sin, cos) need not have unit length
    assert circular_errors(theta, back).max() < 1e-9
    with pytest.raises(ValueError):
        angles_from_sincos(np.zeros((1, 2)))  # (0, 0) has no angle