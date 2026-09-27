"""Behavior readout: a 16-bin multinomial logistic classifier of the label value, fit on validation clips at one
site, and the geometry of its output distributions — square-root probabilities on the unit sphere, Hellinger and
Bhattacharyya distances, and log / exp maps at a base point (the tangent plane Goodfire fits the behavior spline in)."""
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from vjepa_physics.manifolds import EXACT, PERIOD, loop_coords, open_coords
from vjepa_physics.probes import fit_probe

N_BINS = 16
VALUES_PER_BIN = 4  # 64 grid values -> 16 bins; bin edges fall between grid values
N_GRID_VALUES = 64
READOUT_ROLES = ("val_seen", "val_unseen")  # D-16: readouts are fit on validation clips only
CS = np.logspace(-4, 4, 9)  # inverse L2 strengths tried by cross-validation
CV_FOLDS = 5
MAX_ITER = 10_000
DENSE_POINTS = 2001  # grid of curve values searched for the nearest behavior-curve point


def value_bins(value_index: np.ndarray) -> np.ndarray:
    """Bin 0...15 of each clip: value_index // 4. Raises ValueError outside the 64-value grid."""
    v = np.asarray(value_index)
    if v.min() < 0 or v.max() >= N_GRID_VALUES:
        raise ValueError(f"value indices must lie in 0...{N_GRID_VALUES - 1}")
    return v // VALUES_PER_BIN


@dataclass(frozen=True)
class BinReadout:
    """A fitted bin readout: validation z-scoring, the cross-validated multinomial logistic fit, clips seen, and
    whether every solver run converged."""

    scaler: StandardScaler
    model: LogisticRegressionCV
    n_fit: int
    converged: bool

    @property
    def c(self) -> float:
        return float(self.model.C_)

    @property
    def c_edge(self) -> str | None:
        """"lower" or "upper" if the chosen C is an end of CS, else None."""
        if self.c == CS[0]:
            return "lower"
        if self.c == CS[-1]:
            return "upper"
        return None

    def probabilities(self, x: np.ndarray) -> np.ndarray:
        """(n, d) raw features -> (n, 16) bin probabilities, columns in bin order 0...15."""
        return self.model.predict_proba(self.scaler.transform(np.asarray(x, dtype=np.float64)))


def fit_bin_readout(
    x: np.ndarray, bins: np.ndarray, roles: np.ndarray, seed: int, fit_roles: tuple[str, ...] = READOUT_ROLES
) -> BinReadout:
    """Fit on the rows whose role is in `fit_roles` only: z-score, then LogisticRegressionCV (L2, lbfgs, full
    multinomial loss) with C chosen by shuffled stratified 5-fold log-loss. Raises ValueError if a bin is missing
    from the fit rows."""
    rows = np.isin(np.asarray(roles), fit_roles)
    x_fit, y_fit = np.asarray(x, dtype=np.float64)[rows], np.asarray(bins)[rows]
    scaler = StandardScaler().fit(x_fit)
    model = LogisticRegressionCV(
        Cs=CS, cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=seed), l1_ratios=(0.0,),
        scoring="neg_log_loss", solver="lbfgs", max_iter=MAX_ITER, use_legacy_attributes=False,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(scaler.transform(x_fit), y_fit)
    if not np.array_equal(model.classes_, np.arange(N_BINS)):
        raise ValueError(f"fit rows hold bins {model.classes_.tolist()}, need 0...{N_BINS - 1}")
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return BinReadout(scaler, model, int(rows.sum()), converged)


def hellinger(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hellinger distance over the last axis: |√p - √q| / √2 (0 = same distribution, 1 = disjoint)."""
    return np.linalg.norm(np.sqrt(p) - np.sqrt(q), axis=-1) / np.sqrt(2.0)


def bhattacharyya(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Bhattacharyya distance over the last axis: -log Σ √(p q)."""
    return -np.log(np.clip(np.sqrt(p * q).sum(axis=-1), 1e-300, None))


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def sphere_log(base: np.ndarray, points: np.ndarray) -> np.ndarray:
    """(n, D) tangent vectors at the unit base point: direction of each (normalized) point, length = its great-circle
    angle to the base."""
    b, x = unit(base), unit(points)
    c = np.clip(x @ b, -1.0, 1.0)
    u = x - c[:, None] * b
    n = np.linalg.norm(u, axis=1)
    scale = np.divide(np.arccos(c), n, out=np.zeros_like(n), where=n > 0)
    return u * scale[:, None]


def sphere_exp(base: np.ndarray, tangents: np.ndarray) -> np.ndarray:
    """(n, D) points on the unit sphere reached from the unit base point along each tangent vector (inverse of
    sphere_log for tangents orthogonal to the base with length < π)."""
    b, v = unit(base), np.asarray(tangents, dtype=np.float64)
    n = np.linalg.norm(v, axis=1)
    sinc = np.divide(np.sin(n), n, out=np.ones_like(n), where=n > 0)
    return np.cos(n)[:, None] * b + sinc[:, None] * v

def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def readout_map(readout: BinReadout) -> tuple[np.ndarray, np.ndarray]:
    """Raw-space form of a bin readout: probabilities = softmax(x @ weights + offset), weights (d, 16), offset (16,)."""
    weights = readout.model.coef_.T / readout.scaler.scale_[:, None]
    offset = readout.model.intercept_ - readout.scaler.mean_ @ weights
    return weights, offset


def map_probabilities(weights: np.ndarray, offset: np.ndarray, x: np.ndarray) -> np.ndarray:
    """(n, d) raw features -> (n, 16) bin probabilities through a saved raw-space map."""
    return softmax(np.asarray(x, dtype=np.float64) @ weights + offset)


def two_stage_features(
    x: np.ndarray, targets: np.ndarray, bins: np.ndarray, roles: np.ndarray, seed: int,
    fit_roles: tuple[str, ...] = READOUT_ROLES,
) -> np.ndarray:
    """(clips, F) inputs of the two-stage readout: a ridge readout's outputs with their squares and products.

    Fit rows get out-of-fold ridge outputs (shuffled stratified folds on the bins); every other row gets the output
    of the ridge fit on all fit rows (stacking; in-fold and out-of-fold outputs carry different noise).
    """
    x, y = np.asarray(x, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    bins = np.asarray(bins)
    rows = np.flatnonzero(np.isin(np.asarray(roles), fit_roles))
    out = np.empty((len(x), 1 if y.ndim == 1 else y.shape[1]))
    tag = np.full(len(rows), "fold")
    for fit_idx, held_idx in StratifiedKFold(CV_FOLDS, shuffle=True, random_state=seed).split(rows, bins[rows]):
        probe = fit_probe(x[rows[fit_idx]], y[rows[fit_idx]], tag[fit_idx], fit_roles=("fold",))
        out[rows[held_idx]] = probe.predict(x[rows[held_idx]]).reshape(len(held_idx), -1)
    other = np.setdiff1d(np.arange(len(x)), rows)
    full = fit_probe(x, y, roles, fit_roles=fit_roles)
    out[other] = full.predict(x[other]).reshape(len(other), -1)
    return PolynomialFeatures(2, include_bias=False).fit_transform(out)


@dataclass(frozen=True)
class BehaviorCurve:
    """Behavior manifold: exact curve in the tangent plane at `base`, mapped back to the unit sphere of √p."""

    kind: str  # "open" or "loop"
    base: np.ndarray  # (16,) unit
    tangent: Callable[[np.ndarray], np.ndarray]  # (n,) label values -> (n, 16) tangent vectors at base

    def points(self, values: np.ndarray) -> np.ndarray:
        """(n,) label values -> (n, 16) unit vectors (their squares are distributions)."""
        return sphere_exp(self.base, self.tangent(np.atleast_1d(np.asarray(values, dtype=np.float64))))


def behavior_curve(kind: str, values: np.ndarray, sqrt_centroids: np.ndarray) -> BehaviorCurve:
    """Goodfire-style behavior curve: each value's mean √p normalized to the unit sphere, log map at their normalized
    mean, exact interpolation in the label (natural cubic; periodic cubic for a loop), exp map back."""
    s = unit(sqrt_centroids)
    base = unit(s.mean(axis=0))
    fit = open_coords if kind == "open" else loop_coords
    return BehaviorCurve(kind, base, fit(np.asarray(values, dtype=np.float64), sphere_log(base, s), EXACT))


def curve_grid(kind: str, values: np.ndarray) -> np.ndarray:
    """Values searched for the nearest curve point: the whole circle (loop) or the fitted value range (open)."""
    if kind == "loop":
        return np.linspace(0.0, PERIOD, DENSE_POINTS, endpoint=False)
    return np.linspace(values[0], values[-1], DENSE_POINTS)


def nearest_on_curve(sqrt_p: np.ndarray, curve: BehaviorCurve, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per row of √p (unit vectors): Hellinger distance |√p - q| / √2 to the nearest curve point q on `grid`, and
    that point's label value."""
    q = curve.points(grid)
    s = np.asarray(sqrt_p, dtype=np.float64)
    d2 = (s**2).sum(axis=1)[:, None] + (q**2).sum(axis=1)[None, :] - 2.0 * s @ q.T
    i = d2.argmin(axis=1)
    return np.sqrt(np.clip(d2[np.arange(len(s)), i], 0.0, None) / 2.0), grid[i]