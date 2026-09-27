"""Distance-scale readings for the speed / acceleration confound."""
import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score

from vjepa_physics.confounds import (
    CLIP_SECONDS, as_distance, clip_distance, expected_slope, reading_verdict, weighted_balanced_accuracy,
)
from vjepa_physics.geometry import distance_travelled


def test_distance_scale_matches_kinematics():
    table = {"speed_mps": np.array([2.0, 0.0]), "acceleration_mps2": np.array([0.0, 4.0])}
    expected = distance_travelled(table["speed_mps"], table["acceleration_mps2"], CLIP_SECONDS)
    np.testing.assert_allclose(clip_distance(table), expected)
    np.testing.assert_allclose(as_distance("speed", 2.0), expected[0])
    np.testing.assert_allclose(as_distance("acceleration", 4.0), expected[1])


@pytest.mark.parametrize(("variable", "tau", "slope"), [
    ("speed", 0.0, 0.0), ("speed", 0.5, 1.0), ("speed", 1.0, 2.0),  # speed at tau on accelerating clips: 2 tau / T
    ("acceleration", 0.25, 2.0), ("acceleration", 0.5, 1.0), ("acceleration", 1.0, 0.5),  # T / (2 tau)
])
def test_expected_slopes(variable, tau, slope):
    assert expected_slope(variable, tau) == pytest.approx(slope)


def test_reading_rule_thresholds():
    assert reading_verdict((0.0, 0.2)) == "reads acceleration"
    assert reading_verdict((0.6, 1.1)) == "reads a speed / distance quantity"
    assert reading_verdict((0.4, 0.7)) == "mixed"


def test_weighted_balanced_accuracy_matches_sklearn():
    rng = np.random.default_rng(0)
    y, p, w = rng.random(200) < 0.4, rng.random(200) < 0.5, rng.uniform(0.1, 2.0, 200)
    expected = balanced_accuracy_score(y, p, sample_weight=w)
    assert weighted_balanced_accuracy(y, p, w) == pytest.approx(expected, abs=1e-12)