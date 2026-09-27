"""Nullspace removal: projections remove exactly the chosen directions; removing the label-covariance direction(s)
leaves a ridge probe nothing to read; control subspaces stay inside the data's span."""
import numpy as np
import pytest

from vjepa_physics.nullspace import (
    covariance_basis, project_out, random_span_basis, train_ridge, train_scaler, train_span,
)
from vjepa_physics.robustness import orthonormal


def test_project_out_removes_exactly_the_basis():
    rng = np.random.default_rng(0)
    q = orthonormal(rng.standard_normal((30, 5)))
    z = rng.standard_normal((40, 30))
    projected = project_out(z, q, 5)
    assert np.abs(projected @ q).max() < 1e-12
    assert np.abs((z - projected) - (z @ q) @ q.T).max() < 1e-12  # only the basis part was taken away
    assert project_out(z, q, 0) is z  # round 1: nothing removed, no arithmetic


@pytest.mark.parametrize("outputs", [1, 2])  # 2 = direction's (sin, cos)
def test_removing_the_covariance_directions_erases_linear_readout(outputs):
    rng = np.random.default_rng(outputs)
    x = rng.normal(size=(300, 20))
    roles = np.array(["train"] * 200 + ["val_seen"] * 100)
    y = (x @ rng.normal(size=(20, outputs)) + 0.1 * rng.normal(size=(300, outputs))).squeeze()
    z = train_scaler(x, roles).transform(x)
    basis = covariance_basis(z, y, roles)
    assert basis.shape == (20, outputs)
    erased = project_out(z, basis, outputs)
    train = roles == "train"
    y_train = y[train].reshape(int(train.sum()), -1)
    assert np.abs(erased[train].T @ (y_train - y_train.mean(axis=0))).max() < 1e-9  # train cross-covariance = 0
    assert np.abs(train_ridge(erased, y, roles).coef_).max() < 1e-10  # so every ridge probe reads nothing


def test_random_control_directions_stay_in_the_train_span():
    x = np.random.default_rng(0).normal(size=(15, 40))  # fewer train rows than features: a proper subspace
    roles = np.array(["train"] * 15)
    span = train_span(train_scaler(x, roles).transform(x), roles)
    assert span.shape[1] == 14  # centred rows: rank n - 1
    q = random_span_basis(span, 4, seed=0)
    np.testing.assert_allclose(q.T @ q, np.eye(4), atol=1e-12)
    assert np.abs(q - span @ (span.T @ q)).max() < 1e-12
    np.testing.assert_array_equal(q, random_span_basis(span, 4, seed=0))  # seeded: the same draw every time