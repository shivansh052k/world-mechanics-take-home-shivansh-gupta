"""Linear probes on pooled activations: features, targets, a ridge fit on train clips only, and scores."""
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from vjepa_physics.extraction import SITES
from vjepa_physics.metrics import angles_from_sincos, circular_mae, mae, r2, sincos_targets

ALPHAS = np.logspace(-3, 7, 41)  # probe alpha grid
FIT_ROLE = "train"  # the only role any probe is fitted on
NO_SIGNAL_R2 = 0.05  # val-seen R² at or below which an upper-edge alpha is expected

def site_features(activations: np.ndarray, site: str, flatten: bool = False) -> np.ndarray:
    """Probe features of every clip at one site, as float64.

    `activations` is (clips, 26, 8, 1024), e.g. load_joined's memory-mapped array; only `site` is read.
    Default: (clips, 1024), the mean of the 8 per-time-step means = the mean over all 2048 tokens (every time
    step has 256 tokens). flatten=True: (clips, 8192), the 8 time steps side by side (a diagnostic only).
    """
    per_step = np.asarray(activations[:, SITES.index(site)], dtype=np.float64)
    if flatten:
        return per_step.reshape(len(per_step), -1)
    return per_step.mean(axis=1)


def probe_targets(variable: str, labels: np.ndarray) -> np.ndarray:
    """Direction: (clips, 2) = (sin, cos) of the angle in degrees. Speed, acceleration: (clips,) = the label."""
    if variable == "direction":
        return sincos_targets(labels)
    return np.asarray(labels, dtype=np.float64)


@dataclass(frozen=True)
class Probe:
    """A fitted probe: the train z-scoring, the ridge fit, and how many train clips it saw."""

    scaler: StandardScaler
    ridge: RidgeCV
    n_fit: int

    @property
    def alpha(self) -> float:
        return float(self.ridge.alpha_)

    @property
    def alpha_edge(self) -> str | None:
        """"lower" or "upper" if the chosen alpha is an end of the grid, else None."""
        alphas = np.asarray(self.ridge.alphas, dtype=float)
        if self.alpha == alphas[0]:
            return "lower"
        if self.alpha == alphas[-1]:
            return "upper"
        return None

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.ridge.predict(self.scaler.transform(x))


def fit_probe(
    x: np.ndarray,
    y: np.ndarray,
    roles: np.ndarray,
    alphas: np.ndarray = ALPHAS,
    fit_roles: tuple[str, ...] = (FIT_ROLE,),
) -> Probe:
    """Fit a probe on the rows whose role is in `fit_roles` only (default: train): z-score, then RidgeCV (efficient
    leave-one-out, one alpha for all outputs).

    `x`, `y` and `roles` cover the same clips in the same order; other rows are never seen by the scaler or the
    ridge. The steering readouts pass the validation roles. Raises ValueError on mismatched lengths or if no row
    has a fit role.
    """
    roles = np.asarray(roles)
    if not len(x) == len(y) == len(roles):
        raise ValueError(f"lengths differ: x {len(x)}, y {len(y)}, roles {len(roles)}")
    fit_rows = np.isin(roles, fit_roles)
    if not fit_rows.any():
        raise ValueError(f"no rows with roles {fit_roles} to fit on")
    scaler = StandardScaler().fit(x[fit_rows])
    ridge = RidgeCV(alphas=alphas, fit_intercept=True).fit(scaler.transform(x[fit_rows]), y[fit_rows])
    return Probe(scaler, ridge, int(fit_rows.sum()))


def alpha_verdict(edge: str | None, val_seen_r2: float) -> str:
    """The alpha rule: "ok" inside the grid; "no_signal" for an upper edge with val-seen R² <= NO_SIGNAL_R2
    (expected: leave-one-out then predicts the train mean); "failure" for a lower edge, or an upper edge
    with val-seen R² above NO_SIGNAL_R2."""
    if edge is None:
        return "ok"
    if edge == "upper" and val_seen_r2 <= NO_SIGNAL_R2:
        return "no_signal"
    return "failure"


def probe_scores(variable: str, labels: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """DATA.md's metrics. Direction: R² on (sin, cos) and circular MAE in degrees. Others: R² and MAE."""
    if variable == "direction":
        return {
            "r2": r2(probe_targets(variable, labels), pred),
            "circular_mae": circular_mae(labels, angles_from_sincos(pred)),
        }
    return {"r2": r2(labels, pred), "mae": mae(labels, pred)}



def clip_folds(groups: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Fold label per sample: every group (clip) goes to exactly one fold.

    The sorted distinct groups are put in a seeded random order and dealt round robin, so fold sizes (in groups)
    differ by at most one and the result depends only on the groups and the seed.
    """
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < n_folds:
        raise ValueError(f"{len(unique)} groups cannot fill {n_folds} folds")
    fold_of_group = np.empty(len(unique), dtype=np.int64)
    fold_of_group[np.random.default_rng(seed).permutation(len(unique))] = np.arange(len(unique)) % n_folds
    return fold_of_group[np.searchsorted(unique, groups)]


@dataclass(frozen=True)
class GroupedRidge:
    """A ridge fit whose alpha was chosen by grouped K-fold CV: weights (k, d), intercept (k,), CV curve."""

    alpha: float
    alpha_edge: str | None
    coef: np.ndarray
    intercept: np.ndarray
    cv_mse: np.ndarray  # mean squared held-out error per alpha
    n_fit: int

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64) @ self.coef.T + self.intercept


def centred_solution(n, sx, sy, xtx, xty):
    """Means and the eigendecomposition of a ridge problem, from sums: n, sum x, sum y, XᵀX, Xᵀy (uncentred)."""
    mx, my = sx / n, sy / n
    gram = xtx - n * np.outer(mx, mx)
    cross = xty - n * np.outer(mx, my)
    lam, v = np.linalg.eigh(gram)
    return mx, my, np.clip(lam, 0.0, None), v, v.T @ cross


def grouped_cv_ridge(x: np.ndarray, y: np.ndarray, folds: np.ndarray, alphas: np.ndarray = ALPHAS) -> GroupedRidge:
    """Ridge regression with an unpenalised intercept; alpha by K-fold CV over the given fold labels.

    CV score = mean squared error over all held-out samples and outputs (as RidgeCV's), ties to the smaller
    alpha; then a final fit on all rows with that alpha. Exact: each fold's training problem is the total minus
    that fold (per-fold sums), centred on its own training mean, and one eigendecomposition serves every alpha.
    `x` is used as given (standardise it first). y may be (n,) or (n, k); coef is always (k, d).
    """
    x = np.asarray(x, dtype=np.float64)
    y2 = np.asarray(y, dtype=np.float64).reshape(len(x), -1)
    folds = np.asarray(folds)
    labels = np.unique(folds)
    if len(labels) < 2 or len(folds) != len(x):
        raise ValueError("need at least two folds and one fold label per row")

    parts = []
    for f in labels:
        rows = folds == f
        xf, yf = x[rows], y2[rows]
        parts.append((rows, int(rows.sum()), xf.sum(axis=0), yf.sum(axis=0), xf.T @ xf, xf.T @ yf))
    total = [sum(p[i] for p in parts) for i in range(1, 6)]

    alphas = np.asarray(alphas, dtype=np.float64)
    sse = np.zeros(len(alphas))
    for rows, *stats in parts:
        mx, my, lam, v, vb = centred_solution(*[t - s for t, s in zip(total, stats)])
        projected = (x[rows] - mx) @ v
        for j, alpha in enumerate(alphas):
            residual = y2[rows] - (projected @ (vb / (lam + alpha)[:, None]) + my)
            sse[j] += float((residual**2).sum())
    cv_mse = sse / y2.size
    best = int(np.argmin(cv_mse))  # first minimum: ties go to the smaller alpha

    mx, my, lam, v, vb = centred_solution(*total)
    w = v @ (vb / (lam + alphas[best])[:, None])  # (d, k)
    edge = "lower" if best == 0 else "upper" if best == len(alphas) - 1 else None
    return GroupedRidge(float(alphas[best]), edge, w.T, my - mx @ w, cv_mse, len(x))

@dataclass(frozen=True)
class NestedCV:
    """Out-of-fold ridge predictions and, per outer fold, the alpha chosen inside it."""

    predictions: np.ndarray  # same shape as y; every row predicted by a fit that never saw its group
    folds: np.ndarray  # outer fold per row
    alphas: np.ndarray  # (n_folds,)
    alpha_edges: tuple[str | None, ...]


def nested_cv_predictions(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, n_folds: int, seed: int, alphas: np.ndarray = ALPHAS
) -> NestedCV:
    """Grouped out-of-fold ridge predictions with alpha chosen inside each outer training part.

    Outer folds: clip_folds(groups, n_folds, seed). For outer fold f, grouped_cv_ridge picks alpha on the other
    folds' rows only (inner folds clip_folds(those groups, n_folds, seed + 1 + f)) and predicts fold f. So the score
    of the returned predictions is not the one alpha was selected on. x is used as given (standardise it first).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)
    if not len(x) == len(y) == len(groups):
        raise ValueError("x, y and groups must cover the same rows")
    outer = clip_folds(groups, n_folds, seed)
    predictions = np.full(y.shape, np.nan)
    chosen, edges = [], []
    for f in range(n_folds):
        held = outer == f
        inner = clip_folds(groups[~held], n_folds, seed + 1 + f)
        fit = grouped_cv_ridge(x[~held], y[~held], inner, alphas)
        predictions[held] = fit.predict(x[held]).reshape(predictions[held].shape)
        chosen.append(fit.alpha)
        edges.append(fit.alpha_edge)
    return NestedCV(predictions, outer, np.array(chosen), tuple(edges))


def label_permutations(n: int, count: int, seed: int) -> np.ndarray:
    """(count, n) permutations of n rows, drawn in order from one generator seeded with `seed`.

    Raises RuntimeError if a permutation is the identity (it would leave the labels unshuffled).
    """
    rng = np.random.default_rng(seed)
    permutations = np.stack([rng.permutation(n) for _ in range(count)])
    if (permutations == np.arange(n)).all(axis=1).any():
        raise RuntimeError("a permutation is the identity")
    return permutations