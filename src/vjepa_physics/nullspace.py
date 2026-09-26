"""Iterative nullspace probing: fit a ridge probe, remove its weight directions, fit again; plus control subspaces.

Everything happens in one train-standardized space: the scaler is fit once on the train rows and never refit,
so every removed direction and every projection lives in the same coordinates.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from vjepa_physics.probes import ALPHAS, FIT_ROLE, probe_scores

NULL_R2 = 0.1  # val-seen R² below this = no longer readable (shuffled-label null); also the no-signal alpha limit
SPAN_TOLERANCE = 1e-10  # singular values below this x the largest are treated as zero (train span)
RANK_TOLERANCE = 1e-8  # a new weight block needs smallest / largest singular value above this
EMPTY_TOLERANCE = 1e-12  # weights whose part outside the removed subspace is smaller than this (relative) are rejected
EXHAUSTION_RATIO = 1e-12  # |W_k| <= this x |W_1|: train cross-covariance exhausted, later rounds are undefined
REDUNDANCY_FRACTIONS = (0.9, 0.5)  # redundancy counts: rounds with R² at or above these fractions of round 1's


def train_scaler(x: np.ndarray, roles: np.ndarray) -> StandardScaler:
    """z-scoring fit on the train rows only (the same computation as fit_probe's scaler)."""
    fit_rows = np.asarray(roles) == FIT_ROLE
    if not fit_rows.any():
        raise ValueError("no train rows to fit on")
    return StandardScaler().fit(x[fit_rows])


def project_out(z: np.ndarray, basis: np.ndarray, n_cols: int) -> np.ndarray:
    """Rows of z with their components along the first n_cols (orthonormal) basis columns removed: z (I - QQᵀ).

    With n_cols = 0 nothing is removed and z itself is returned (no arithmetic, so round 1 matches a plain probe).
    """
    if n_cols == 0:
        return z
    q = basis[:, :n_cols]
    return z - (z @ q) @ q.T


def grid_edge(alpha: float, alphas: np.ndarray) -> str | None:
    """"lower" or "upper" if alpha is an end of the grid, else None."""
    if alpha == alphas[0]:
        return "lower"
    if alpha == alphas[-1]:
        return "upper"
    return None


def nullspace_alpha_verdict(edge: str | None, val_seen_r2: float) -> str:
    """Alpha rule for every nullspace fit: an upper edge is expected once the variable is no longer readable
    (val-seen R² < NULL_R2) -> "no_signal"; an upper edge above it, or any lower edge -> "failure"."""
    if edge is None:
        return "ok"
    if edge == "upper" and val_seen_r2 < NULL_R2:
        return "no_signal"
    return "failure"


def extend_basis(basis: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Add the directions of w (d, m) to an orthonormal basis (d, j) -> (d, j + m) basis, leak, rank ratio.

    leak = |Qᵀw| / |w|: how much of the new probe already lies in the removed subspace (≈ 0 expected, since a
    ridge fit on projected rows has its weights in their span). rank ratio = smallest / largest singular value of
    the new block (1.0 when m = 1). Raises ValueError if w is (almost) inside the basis or not of full rank m.
    """
    norm = np.linalg.norm(w)
    if norm == 0:
        raise ValueError("probe weights are all zero; nothing to remove")
    inside = basis.T @ w
    rest = w - basis @ inside
    rest -= basis @ (basis.T @ rest)  # second Gram-Schmidt pass keeps the basis orthonormal to rounding
    if np.linalg.norm(rest) <= EMPTY_TOLERANCE * norm:
        raise ValueError("probe weights lie inside the removed subspace")
    s = np.linalg.svd(rest, compute_uv=False)
    rank_ratio = float(s.min() / s.max())
    if rank_ratio <= RANK_TOLERANCE:
        raise ValueError(f"new weight block is rank deficient (singular value ratio {rank_ratio:.2e})")
    q, _ = np.linalg.qr(rest)
    return np.hstack([basis, q]), float(np.linalg.norm(inside) / norm), rank_ratio


@dataclass(frozen=True)
class NullspaceRun:
    """Probe sequence of one run. Round k (0-based) was fit with the first k * dims_per_round basis columns removed."""

    basis: np.ndarray  # (d, columns) orthonormal, in removal order
    weights: np.ndarray  # (rounds, d, m), standardized space
    intercepts: np.ndarray  # (rounds, m)
    alphas: np.ndarray  # (rounds,)
    alpha_edges: tuple[str | None, ...]  # per round
    predictions: np.ndarray  # (rounds, n_predicted, m), rows = predict_rows in order
    leaks: np.ndarray  # (rounds,) |Qᵀw| / |w| against the columns removed before that round's fit
    rank_ratios: np.ndarray  # (rounds,) smallest / largest singular value of the weight block
    n_fit: int
    dims_per_round: int
    exhausted_round: int | None  # 1-based round whose probe was ~0 (run stopped there); None = not exhausted


def run_rounds(
    z: np.ndarray,
    y: np.ndarray,
    roles: np.ndarray,
    predict_rows: np.ndarray,
    n_rounds: int,
    basis: np.ndarray | None = None,
    alphas: np.ndarray = ALPHAS,
) -> NullspaceRun:
    """Fit up to n_rounds ridge probes (RidgeCV, efficient leave-one-out, train rows only) on standardized features z.

    basis=None: iterative nullspace -- each probe's weight directions are added to the removed subspace before
    the next round (m = 1 per round for (n,) targets, 2 for (sin, cos)). The run stops at the first round whose
    weights are <= EXHAUSTION_RATIO x round 1's: the train cross-covariance is exhausted, so that probe is ~0 and its
    direction is rounding noise; the round is recorded (leak and rank ratio NaN) but nothing is added to the basis.
    With a fixed `basis` (control): round k removes its first k * m columns, all n_rounds are run.
    Only `predict_rows` are predicted. Raises ValueError on mismatched lengths, no train rows, or a short basis.
    """
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    roles = np.asarray(roles)
    if not len(z) == len(y) == len(roles) == len(predict_rows):
        raise ValueError("z, y, roles and predict_rows must cover the same clips")
    fit_rows = roles == FIT_ROLE
    if not fit_rows.any():
        raise ValueError("no train rows to fit on")
    m = 1 if y.ndim == 1 else y.shape[1]
    if basis is not None and basis.shape[1] < (n_rounds - 1) * m:
        raise ValueError(f"basis has {basis.shape[1]} columns; {(n_rounds - 1) * m} needed")
    removed = np.empty((z.shape[1], 0)) if basis is None else np.asarray(basis, dtype=np.float64)

    weights, intercepts, chosen, edges, preds, leaks, ratios = [], [], [], [], [], [], []
    exhausted = None
    for k in range(n_rounds):
        n_removed = k * m
        zk = project_out(z, removed, n_removed)
        ridge = RidgeCV(alphas=alphas, fit_intercept=True).fit(zk[fit_rows], y[fit_rows])
        w = np.reshape(ridge.coef_, (m, -1)).T  # (d, m)
        weights.append(w)
        intercepts.append(np.atleast_1d(ridge.intercept_).astype(np.float64))
        chosen.append(float(ridge.alpha_))
        edges.append(grid_edge(float(ridge.alpha_), np.asarray(alphas, dtype=float)))
        preds.append(np.reshape(ridge.predict(zk[predict_rows]), (-1, m)))
        if basis is None:
            if k > 0 and np.linalg.norm(w) <= EXHAUSTION_RATIO * np.linalg.norm(weights[0]):
                exhausted = k + 1
                leaks.append(float("nan"))
                ratios.append(float("nan"))
                break
            removed, leak, ratio = extend_basis(removed, w)
        else:
            used = removed[:, :n_removed]
            leak = float(np.linalg.norm(used.T @ w) / np.linalg.norm(w))
            s = np.linalg.svd(w, compute_uv=False)
            ratio = float(s.min() / s.max())
        leaks.append(leak)
        ratios.append(ratio)

    kept = removed if basis is None else removed[:, : (n_rounds - 1) * m]
    return NullspaceRun(
        kept, np.stack(weights), np.stack(intercepts), np.array(chosen), tuple(edges), np.stack(preds),
        np.array(leaks), np.array(ratios), int(fit_rows.sum()), m, exhausted,
    )


def composite_maps(scaler: StandardScaler, run: NullspaceRun) -> tuple[np.ndarray, np.ndarray]:
    """Raw-space maps: probe k's prediction on a raw feature row x is x @ maps[k] + offsets[k].

    maps[k] = diag(1/σ) P_{k-1} W_k (P = I - QQᵀ over the columns removed before round k), offsets[k] = b_k - μ maps[k].
    """
    m = run.dims_per_round
    maps = np.empty_like(run.weights)
    offsets = np.empty_like(run.intercepts)
    for k in range(len(run.weights)):
        projected = project_out(run.weights[k].T, run.basis, k * m).T  # P is symmetric: (W_kᵀ P)ᵀ = P W_k
        maps[k] = projected / np.asarray(scaler.scale_)[:, None]
        offsets[k] = run.intercepts[k] - np.asarray(scaler.mean_) @ maps[k]
    return maps, offsets


def train_span(z: np.ndarray, roles: np.ndarray) -> np.ndarray:
    """(d, r) orthonormal basis of the span of the standardized train rows, ordered by variance (principal axes).

    The train rows of z are centred (the scaler was fit on them), so the right singular vectors are the principal
    axes; directions with singular value <= SPAN_TOLERANCE x the largest are dropped (rank r = n_train - 1 expected).
    """
    fit_rows = np.asarray(roles) == FIT_ROLE
    _, s, vt = np.linalg.svd(np.asarray(z, dtype=np.float64)[fit_rows], full_matrices=False)
    return vt[s > SPAN_TOLERANCE * s[0]].T


def random_span_basis(span: np.ndarray, n_cols: int, seed: int) -> np.ndarray:
    """(d, n_cols) orthonormal random directions inside `span`: a seeded Gaussian rotation of the span, then QR.

    QR orthonormalizes column by column, so the first j columns span the first j random directions: the removed
    subspaces are nested across rounds.
    """
    if n_cols > span.shape[1]:
        raise ValueError(f"span has {span.shape[1]} dimensions; {n_cols} requested")
    gaussian = np.random.default_rng(seed).standard_normal((span.shape[1], n_cols))
    q, _ = np.linalg.qr(span @ gaussian)
    return q

def round_scores(variable: str, labels: np.ndarray, predictions: np.ndarray) -> list[dict[str, float]]:
    """probe_scores for every round; predictions (rounds, n, m) belong to the clips in `labels`, in order."""
    return [probe_scores(variable, labels, p[:, 0] if p.shape[1] == 1 else p) for p in predictions]


def first_true(mask: np.ndarray) -> int | None:
    """1-based round of the first True; None if there is none."""
    hits = np.flatnonzero(np.asarray(mask, dtype=bool))
    return int(hits[0]) + 1 if hits.size else None


def curve_summary(r2: np.ndarray, dims_per_round: int, threshold: float = NULL_R2) -> dict:
    """K = first round (1-based) with R² < threshold, and what the curve does after it.

    k None = never below within the run (reported as "> rounds"). dims_before_k = dimensions removed before
    round K was fit. last_round_at_or_above and max_r2_after_k show any rise after K.
    """
    c = np.asarray(r2, dtype=np.float64)
    k = first_true(c < threshold)
    above = np.flatnonzero(c >= threshold)
    after = c[k:] if k is not None else c[:0]  # rounds K + 1, K + 2, ...
    return {
        "k": k,
        "capped": k is None,
        "dims_before_k": None if k is None else (k - 1) * dims_per_round,
        "last_round_at_or_above": int(above[-1]) + 1 if above.size else None,
        "max_r2_after_k": float(after.max()) if after.size else None,
        "round_of_max_after_k": int(k + after.argmax()) + 1 if after.size else None,
    }


def redundancy_counts(r2: np.ndarray, fractions: tuple[float, ...] = REDUNDANCY_FRACTIONS) -> dict:
    """Per fraction f of the round-1 R²: rounds from round 1 before the curve first drops below f x R²_1
    ("consecutive") and all rounds at or above it ("total")."""
    c = np.asarray(r2, dtype=np.float64)
    counts = {}
    for f in fractions:
        limit = f * c[0]
        first = first_true(c < limit)
        counts[str(f)] = {"limit": float(limit), "consecutive": len(c) if first is None else first - 1,
                          "total": int((c >= limit).sum())}
    return counts