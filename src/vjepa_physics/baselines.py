"""Reference scores for the probes: a physics fit to the tracked disk (ceiling)."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vjepa_physics.geometry import frame_times, pixel_to_world
from vjepa_physics.metrics import angles_from_sincos
from vjepa_physics.video import load_clip

FPS = 24  # every clip (DATA.md; metadata check)
# Motion models: the powers of t each one fits (p(t) = sum of coefficient x t^power, the t^2 term halved).
MOTION_MODELS = {"quadratic": (0, 1, 2), "linear": (0, 1), "from_rest": (0, 2)}


def physics_fit(
    centre: np.ndarray, visible: np.ndarray, model: str = "quadratic", fps: int = FPS
) -> dict[str, float]:
    """Least-squares fit of p(t) = p0 + v t + a t^2 / 2 to one clip's tracked disk centres, frame k at t = k / fps.

    `centre` (frames, 2) px (col, row); `visible` (frames,) bool selects the frames used (fully visible disk: a
    cut-off disk biases the centroid). model "linear" fits p0 + v t (a = 0); "from_rest" fits p0 + a t^2 / 2
    (v = 0). Returns speed = |v| at t = 0 (m/s), acceleration = |a| (m/s^2), theta = direction of the fitted
    displacement from the first to the last used frame (degrees in [0, 360), 0 = right, 90 = up). Raises
    ValueError for an unknown model, mismatched shapes, fewer used frames than parameters + 1, or a non-finite
    centre on a used frame.
    """
    if model not in MOTION_MODELS:
        raise ValueError(f"unknown model {model!r}, expected one of {tuple(MOTION_MODELS)}")
    centre = np.asarray(centre, dtype=float)
    visible = np.asarray(visible, dtype=bool)
    if centre.ndim != 2 or centre.shape[1] != 2 or visible.shape != centre.shape[:1]:
        raise ValueError(f"centre {centre.shape} and visible {visible.shape}: expected (frames, 2) and (frames,)")

    powers = MOTION_MODELS[model]
    t = frame_times(fps, len(centre))[visible]
    design = np.column_stack([t**p / 2 if p == 2 else t**p for p in powers])
    if len(t) < design.shape[1] + 1:
        raise ValueError(f"{len(t)} used frames, need at least {design.shape[1] + 1} for a {model} fit")
    x, y = pixel_to_world(centre[visible, 0], centre[visible, 1])
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("non-finite disk centre on a used frame")

    coef, *_ = np.linalg.lstsq(design, np.column_stack([x, y]), rcond=None)  # one row per power; columns x, y
    v = coef[powers.index(1)] if 1 in powers else np.zeros(2)
    a = coef[powers.index(2)] if 2 in powers else np.zeros(2)
    dx, dy = (design[-1] - design[0]) @ coef  # fitted displacement, first to last used frame
    theta = angles_from_sincos(np.array([[dy, dx]]))[0]
    return {"speed": float(np.hypot(*v)), "acceleration": float(np.hypot(*a)), "theta": float(theta)}


# Pixel floor: ridge regression on raw pixels, solved through the Gram matrix (clips << pixels).
FLOOR_ALPHAS = np.logspace(-6, 3, 37)  # relative: multiplied by the mean diagonal of the centred train Gram
FIT_ROLE = "train"


def pixel_matrix(paths: list[Path], time_average: bool = False) -> np.ndarray:
    """Decoded clips as unsigned-integer feature rows, one per path.

    Full: (clips, 16 * 256 * 256 * 3) uint8, the raw RGB frames. time_average=True: (clips, 256 * 256 * 3) uint16,
    the sum over the 16 frames (the time-averaged frame times 16, kept integer so its Gram matrix stays exact;
    divide that Gram by 16² = 256 to get the mean frame's).
    """
    rows = []
    for path in paths:
        clip = load_clip(path)
        rows.append(clip.sum(axis=0, dtype=np.uint16).reshape(-1) if time_average else clip.reshape(-1))
    return np.stack(rows)


def exact_gram(x: np.ndarray, chunk: int = 1 << 16) -> np.ndarray:
    """(n, d) unsigned-integer features -> (n, n) Gram matrix X Xᵀ in float64, exactly.

    Built from feature chunks converted to float64. Every product and every partial sum is an integer below
    2^53, so float64 holds each exactly and the result does not depend on chunking or summation order. Raises
    ValueError for a non-unsigned-integer array or if d * max(x)^2 could reach 2^53.
    """
    if x.dtype.kind != "u" or x.ndim != 2:
        raise ValueError(f"expected a 2-D unsigned integer array, got {x.dtype} {x.shape}")
    n, d = x.shape
    peak = int(x.max())
    if d * peak * peak >= 2**53:
        raise ValueError(f"{d} features with max {peak}: sums could exceed 2^53, the Gram would not be exact")
    gram = np.zeros((n, n))
    for start in range(0, d, chunk):
        block = x[:, start:start + chunk].astype(np.float64)
        gram += block @ block.T
    return gram


@dataclass(frozen=True)
class KernelFit:
    """A ridge fit through the Gram matrix: chosen alpha (absolute), its grid position, train size, predictions."""

    alpha: float
    alpha_edge: str | None
    n_fit: int
    predictions: np.ndarray  # (clips, k); NaN on rows that were not asked for
    loo_mse: float = float("nan")  # exact leave-one-out mean squared error at the chosen alpha (train rows)


def kernel_ridge(
    gram: np.ndarray, y: np.ndarray, roles: np.ndarray, predict: np.ndarray, relative_alphas: np.ndarray = FLOOR_ALPHAS
) -> KernelFit:
    """Ridge regression with an unpenalised intercept, given only the Gram matrix of the features over all clips.

    Fit on train rows only; alpha by exact leave-one-out over train (mean squared error over rows and outputs,
    one alpha for all outputs, ties to the smaller alpha, as RidgeCV). Features are centred on the train mean
    through the Gram matrix, which with an unpenalised intercept changes nothing but numerics; the hat matrix is
    then J/n + K(K + aI)^-1, so the leave-one-out shortcut is exact. Alphas = relative_alphas x mean diagonal
    of the centred train Gram. Predictions only for rows where `predict` is True.
    """
    roles = np.asarray(roles)
    train = np.flatnonzero(roles == FIT_ROLE)
    if not len(train):
        raise ValueError("no train rows to fit on")
    if gram.shape != (len(roles), len(roles)) or len(y) != len(roles) or predict.shape != roles.shape:
        raise ValueError("gram, y, roles and predict must cover the same clips")

    row_mean = gram[:, train].mean(axis=1)  # x_i . mu_train for every clip
    centred = gram[:, train] - row_mean[:, None] - row_mean[train][None, :] + row_mean[train].mean()
    k_train = centred[train]
    lam, q = np.linalg.eigh(k_train)
    lam = np.clip(lam, 0.0, None)  # a centred Gram is positive semi-definite; clip round-off negatives

    y_train = np.asarray(y, dtype=np.float64)[train].reshape(len(train), -1)
    y_mean = y_train.mean(axis=0)
    qy = q.T @ (y_train - y_mean)
    alphas = np.asarray(relative_alphas) * np.trace(k_train) / len(train)
    best_score, best, best_gap = -np.inf, 0, float("nan")
    for i, alpha in enumerate(alphas):
        shrink = lam / (lam + alpha)
        fitted = y_mean + q @ (shrink[:, None] * qy)
        hat = 1.0 / len(train) + (q**2) @ shrink
        score = -np.mean(((y_train - fitted) / (1.0 - hat)[:, None]) ** 2)
        if score > best_score:
            best_score, best, best_gap = score, i, float((1.0 - hat).min())

    alpha = float(alphas[best])
    dual = q @ (qy / (lam + alpha)[:, None])  # (K + alpha I)^-1 (y - mean)
    predictions = np.full((len(roles), y_train.shape[1]), np.nan)
    predictions[predict] = y_mean + centred[predict] @ dual
    edge = "lower" if best == 0 else "upper" if best == len(alphas) - 1 else None
    return KernelFit(
        alpha, edge, len(train), predictions.reshape((len(roles), *np.shape(y)[1:])), -best_score, best_gap,
    )


GAMMA_FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)  # RBF width grid: multiples of the median-distance gamma


def squared_distances(x: np.ndarray) -> np.ndarray:
    """(n, d) rows -> (n, n) squared Euclidean distances in float64 (|a|² + |b|² - 2 a·b, round-off negatives
    clipped to 0, exact zeros on the diagonal)."""
    x = np.asarray(x, dtype=np.float64)
    sq = (x**2).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)
    np.maximum(d2, 0.0, out=d2)
    np.fill_diagonal(d2, 0.0)
    return d2


def median_gamma(d2_train: np.ndarray) -> float:
    """Median heuristic: 1 / (2 σ²), σ = median distance between distinct train rows."""
    upper = d2_train[np.triu_indices(len(d2_train), k=1)]
    return float(1.0 / (2.0 * np.median(np.sqrt(upper)) ** 2))


@dataclass(frozen=True)
class RBFFit:
    """RBF kernel ridge: the chosen fit, its gamma, the median-heuristic gamma, and the LOO error per gamma factor."""

    fit: KernelFit
    gamma: float
    gamma_median: float
    gamma_edge: str | None
    loo_mse: np.ndarray  # (len(gamma_factors),), each at its own best alpha
    min_one_minus_hat: float = float("nan")  # smallest 1 - h_ii at the chosen alpha (LOO divides by it)


def rbf_kernel_ridge(
    x: np.ndarray, y: np.ndarray, roles: np.ndarray, predict: np.ndarray,
    gamma_factors: tuple[float, ...] = GAMMA_FACTORS, relative_alphas: np.ndarray = FLOOR_ALPHAS,
) -> RBFFit:
    """Kernel ridge with an RBF kernel exp(-gamma |a - b|²) and an unpenalised intercept (kernel_ridge on its Gram).

    gamma = factor x the median-distance gamma of the train rows; gamma and alpha both by exact leave-one-out on
    train only (lowest LOO error; ties to the earlier factor). Predictions only for rows where `predict` is True.
    """
    roles = np.asarray(roles)
    d2 = squared_distances(x)
    train = roles == FIT_ROLE
    base = median_gamma(d2[np.ix_(train, train)])
    fits = [kernel_ridge(np.exp(-f * base * d2), y, roles, predict, relative_alphas) for f in gamma_factors]
    loo = np.array([f.loo_mse for f in fits])
    best = int(np.argmin(loo))
    edge = "lower" if best == 0 else "upper" if best == len(fits) - 1 else None
    return RBFFit(fits[best], gamma_factors[best] * base, base, edge, loo)