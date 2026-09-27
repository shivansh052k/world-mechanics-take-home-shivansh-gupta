"""Probe fitting: only the fit rows are ever seen, and a linear target is recovered."""
import numpy as np

from vjepa_physics.metrics import r2
from vjepa_physics.probes import fit_probe


def synthetic(n_train: int = 200, n_val: int = 100, d: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(2.0, 3.0, (n_train + n_val, d))
    y = x @ rng.normal(size=d) + 1.5
    roles = np.array(["train"] * n_train + ["val_seen"] * n_val)
    return x, y, roles


def test_validation_rows_never_reach_the_fit():
    x, y, roles = synthetic()
    val = roles != "train"
    probe = fit_probe(x, y, roles)
    x2, y2 = x.copy(), y.copy()
    x2[val] = 1e6 * np.random.default_rng(1).normal(size=x2[val].shape)  # garbage in every validation row
    y2[val] = -y2[val]
    again = fit_probe(x2, y2, roles)
    assert probe.n_fit == again.n_fit == int((~val).sum())
    assert probe.alpha == again.alpha
    np.testing.assert_array_equal(probe.scaler.mean_, again.scaler.mean_)  # the z-scoring saw train rows only
    np.testing.assert_array_equal(probe.ridge.coef_, again.ridge.coef_)


def test_noiseless_linear_target_is_recovered():
    x, y, roles = synthetic()
    val = roles != "train"
    assert r2(y[val], fit_probe(x, y, roles).predict(x[val])) > 0.999


def test_readouts_can_be_fit_on_validation_rows_only():
    x, y, roles = synthetic()
    assert fit_probe(x, y, roles, fit_roles=("val_seen",)).n_fit == int((roles == "val_seen").sum())