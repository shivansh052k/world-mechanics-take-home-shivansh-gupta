"""Activation manifolds: per-value train centroids in the train-standardized space, their PCA, and a curve through
them parameterized by the label value (open for speed and acceleration, a closed loop in degrees for direction),
plus leave-one-centroid-out (LOCO) errors for choosing the PCA dimension and the smoothing."""
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline, make_interp_spline, make_smoothing_spline

from vjepa_physics.probes import FIT_ROLE

PERIOD = 360.0  # direction labels are degrees on a circle
KINDS = ("open", "loop")
LINE = "line"  # open curves: the least-squares straight line (the smoothing spline's lam -> infinity limit)
EXACT = "exact"  # exact interpolation: natural cubic (open) or periodic cubic (loop)
RANK_TOLERANCE = 1e-10  # PCA axes with singular value <= this x the largest are dropped

PCA_DIMS = (1, 2, 3, 4, 6, 8, 12, 16, None)  # LOCO grid of PCA dimensions, ascending; None = every axis
LAM_FACTORS = np.logspace(-7, 2, 13)  # open curves: lam = factor x (value range)³, near-interpolation to near-line
HARMONICS = tuple(range(1, 13))  # loop curves: trig polynomial degrees H
SELECTION_TOLERANCE = 0.01  # settings within this fraction of the grid's minimum mean squared LOCO error qualify


def value_centroids(
    z: np.ndarray, value_index: np.ndarray, labels: np.ndarray, roles: np.ndarray, role: str = FIT_ROLE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per value present in the rows of `role` (default: train): labels (V,), centroids (V, d) and clip counts (V,),
    in value-index order.

    z: (clips, d) features, standardized with the train scaler; only rows whose role is `role` are read.
    Raises ValueError if there are no such rows or one value's clips carry more than one label.
    """
    rows = np.asarray(roles) == role
    if not rows.any():
        raise ValueError(f"no rows with role {role!r}")
    z = np.asarray(z, dtype=np.float64)[rows]
    value_index, labels = np.asarray(value_index)[rows], np.asarray(labels)[rows]
    values, centroids, counts = [], [], []
    for i in np.unique(value_index):
        at = value_index == i
        label = np.unique(labels[at])
        if len(label) != 1:
            raise ValueError(f"value index {i}: {role} labels {label}")
        values.append(float(label[0]))
        centroids.append(z[at].mean(axis=0))
        counts.append(int(at.sum()))
    return np.array(values), np.stack(centroids), np.array(counts)


def centroid_noise(z: np.ndarray, value_index: np.ndarray, roles: np.ndarray, role: str = FIT_ROLE) -> np.ndarray:
    """(V,) expected squared norm of each centroid's sampling error, in value_centroids' order: the unbiased trace of
    the within-value covariance over the clip count, sum |z_i - c|² / (n (n - 1)).

    In expectation a left-out centroid's LOCO error² = this + the curve's own error², so it is the floor under every
    LOCO error. Raises ValueError if a value has fewer than 2 clips.
    """
    rows = np.asarray(roles) == role
    z = np.asarray(z, dtype=np.float64)[rows]
    value_index = np.asarray(value_index)[rows]
    out = []
    for i in np.unique(value_index):
        at = z[value_index == i]
        n = len(at)
        if n < 2:
            raise ValueError(f"value index {i}: {n} clip(s), need 2 for a noise estimate")
        out.append(float(((at - at.mean(axis=0)) ** 2).sum() / (n * (n - 1))))
    return np.array(out)


def centroid_pca(centroids: np.ndarray, k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """PCA of the centroids: mean (d,) and orthonormal axes (d, k'), k' = min(k, rank). k=None keeps every axis
    with non-zero variance (rank <= V - 1)."""
    c = np.asarray(centroids, dtype=np.float64)
    mean = c.mean(axis=0)
    _, s, vt = np.linalg.svd(c - mean, full_matrices=False)
    rank = int((s > RANK_TOLERANCE * s[0]).sum())
    return mean, vt[: rank if k is None else min(k, rank)].T


@dataclass(frozen=True)
class Curve:
    """A fitted curve in the standardized feature space: point(v) = mean + coords(v) @ axesᵀ."""

    kind: str  # "open" or "loop"
    smooth: float | int | str  # open: lam >= 0, LINE or EXACT; loop: harmonics H >= 1 or EXACT
    mean: np.ndarray  # (d,)
    axes: np.ndarray  # (d, k) orthonormal PCA axes
    coords: Callable[[np.ndarray], np.ndarray]  # (n,) label values -> (n, k) PCA coordinates

    def __call__(self, values: np.ndarray) -> np.ndarray:
        """(n,) label values -> (n, d) points on the curve."""
        v = np.atleast_1d(np.asarray(values, dtype=np.float64))
        return self.mean + self.coords(v) @ self.axes.T


def trig_design(degrees: np.ndarray, harmonics: int) -> np.ndarray:
    """(n, 2H + 1) columns: 1, cos(hθ) for h = 1...H, sin(hθ) for h = 1...H."""
    t = np.deg2rad(np.asarray(degrees, dtype=np.float64))[:, None] * np.arange(1, harmonics + 1)
    return np.hstack([np.ones((len(t), 1)), np.cos(t), np.sin(t)])


def open_coords(values: np.ndarray, coords: np.ndarray, smooth: float | str) -> Callable:
    """Open curve through (values, coords): LINE, EXACT (natural cubic) or a smoothing spline with lam = smooth."""
    if smooth == LINE:
        slope, intercept = np.polyfit(values, coords, 1)
        return lambda v: v[:, None] * slope + intercept
    if smooth == EXACT:
        return make_interp_spline(values, coords, k=3, bc_type="natural")
    if isinstance(smooth, str) or smooth < 0:
        raise ValueError(f"open curve: smooth must be {LINE!r}, {EXACT!r} or lam >= 0, got {smooth!r}")
    return make_smoothing_spline(values, coords, lam=float(smooth))


def loop_coords(values: np.ndarray, coords: np.ndarray, smooth: int | str) -> Callable:
    """Closed curve in degrees: EXACT (periodic cubic) or a least-squares trig polynomial with H = smooth harmonics."""
    if smooth == EXACT:
        x = np.append(values, values[0] + PERIOD)
        return CubicSpline(x, np.vstack([coords, coords[:1]]), bc_type="periodic")
    if isinstance(smooth, str) or int(smooth) != smooth or not 1 <= smooth or 2 * smooth + 1 > len(values):
        raise ValueError(f"loop curve: smooth must be {EXACT!r} or 1 <= H <= (V - 1) / 2, got {smooth!r}")
    beta = np.linalg.lstsq(trig_design(values, int(smooth)), coords, rcond=None)[0]
    return lambda v: trig_design(v, int(smooth)) @ beta


def check_values(kind: str, values: np.ndarray) -> np.ndarray:
    """values as float64; raises ValueError for an unknown kind or values not strictly increasing (loop: within one
    period)."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    values = np.asarray(values, dtype=np.float64)
    if np.any(np.diff(values) <= 0) or (kind == "loop" and values[-1] - values[0] >= PERIOD):
        raise ValueError("values must increase strictly (loop: within one period)")
    return values


def curve_from_pca(
    kind: str, values: np.ndarray, centroids: np.ndarray, mean: np.ndarray, axes: np.ndarray, smooth
) -> Curve:
    """The curve through the centroids' coordinates on given PCA axes (mean (d,), axes (d, k))."""
    coords = (np.asarray(centroids, dtype=np.float64) - mean) @ axes
    fit = open_coords if kind == "open" else loop_coords
    return Curve(kind, smooth, mean, axes, fit(values, coords, smooth))


def fit_curve(kind: str, values: np.ndarray, centroids: np.ndarray, k: int | None, smooth) -> Curve:
    """PCA of the centroids (k axes; None = all), then an open or loop curve through their coordinates."""
    values = check_values(kind, values)
    mean, axes = centroid_pca(centroids, k)
    return curve_from_pca(kind, values, centroids, mean, axes, smooth)


def scored_folds(kind: str, n_values: int) -> range:
    """Centroids scored by LOCO: all on a loop; open curves skip both ends (that would be extrapolation)."""
    return range(1, n_values - 1) if kind == "open" else range(n_values)


def loco_predictions(kind: str, values: np.ndarray, centroids: np.ndarray, k: int | None, smooth) -> np.ndarray:
    """(V, d): row j = the curve fitted without centroid j (PCA refit too), evaluated at value j; NaN rows are not
    scored."""
    values = check_values(kind, values)
    centroids = np.asarray(centroids, dtype=np.float64)
    out = np.full(centroids.shape, np.nan)
    for j in scored_folds(kind, len(values)):
        keep = np.arange(len(values)) != j
        out[j] = fit_curve(kind, values[keep], centroids[keep], k, smooth)(values[j])[0]
    return out


def loco_errors(kind: str, values: np.ndarray, centroids: np.ndarray, k: int | None, smooth) -> np.ndarray:
    """(V,) Euclidean distance (full standardized space) from each left-out centroid to its LOCO prediction;
    NaN where not scored."""
    pred = loco_predictions(kind, values, centroids, k, smooth)
    return np.linalg.norm(np.asarray(centroids, dtype=np.float64) - pred, axis=1)


def smoothing_grid(kind: str, values: np.ndarray) -> list:
    """Candidate smoothing settings, smoothest first. Open: LINE, then lam = factor x (value range)³ from the largest
    factor down. Loop: H = 1...12. EXACT is never a candidate (reported as a variant)."""
    if kind == "open":
        span = float(values[-1] - values[0])
        return [LINE] + [float(f * span**3) for f in LAM_FACTORS[::-1]]
    return list(HARMONICS)


def loco_grid(kind: str, values: np.ndarray, centroids: np.ndarray, ks: tuple, smooths: list) -> np.ndarray:
    """(len(ks), len(smooths), V) squared LOCO errors in the full standardized space; NaN where not scored.

    Each fold's PCA is computed once and truncated to every k (the same axes centroid_pca(fold, k) returns).
    """
    values = check_values(kind, values)
    centroids = np.asarray(centroids, dtype=np.float64)
    out = np.full((len(ks), len(smooths), len(values)), np.nan)
    for j in scored_folds(kind, len(values)):
        keep = np.arange(len(values)) != j
        v, c = values[keep], centroids[keep]
        mean, axes = centroid_pca(c)
        for a, k in enumerate(ks):
            for b, smooth in enumerate(smooths):
                point = curve_from_pca(kind, v, c, mean, axes if k is None else axes[:, :k], smooth)(values[j])[0]
                out[a, b, j] = float(((centroids[j] - point) ** 2).sum())
    return out


def select_setting(mse: np.ndarray) -> tuple[int, int]:
    """Indices (a, b) into (ks, smooths) of a (len(ks), len(smooths)) mean-squared-LOCO grid: among settings within
    SELECTION_TOLERANCE of the grid minimum, the smallest k (ks ascending), then the smoothest (smooths ordered
    smoothest first)."""
    qualify = mse <= np.nanmin(mse) * (1 + SELECTION_TOLERANCE)
    a = int(np.flatnonzero(qualify.any(axis=1))[0])
    return a, int(np.flatnonzero(qualify[a])[0])

def unit_tangents(curve: Curve, values: np.ndarray) -> np.ndarray:
    """(n, d) unit tangent vectors of the curve at `values`, by central differences with step 1e-4 x their range."""
    v = np.asarray(values, dtype=np.float64)
    h = 1e-4 * float(v.max() - v.min())
    t = curve(v + h) - curve(v - h)
    return t / np.linalg.norm(t, axis=1, keepdims=True)


def covariance_axis(values: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean of the centroids (d,) and the unit direction of their least-squares line in the label, as (d, 1) axes.

    With equal clip counts per value this is the train covariance direction B / |B|.
    """
    c = np.asarray(centroids, dtype=np.float64)
    slope = np.polyfit(np.asarray(values, dtype=np.float64), c, 1)[0]
    return c.mean(axis=0), (slope / np.linalg.norm(slope))[:, None]


def fit_spacing_line(values: np.ndarray, centroids: np.ndarray, smooth) -> Curve:
    """Free-spacing line c̄ + u p(v): u = the centroids' least-squares line direction, p = an open curve in v through
    their projections on u. smooth = LINE gives exactly the least-squares line (the covariance arm's line)."""
    values = check_values("open", values)
    mean, axis = covariance_axis(values, centroids)
    return curve_from_pca("open", values, centroids, mean, axis, smooth)


def loco_spacing_grid(values: np.ndarray, centroids: np.ndarray, smooths: list) -> np.ndarray:
    """(len(smooths), V) squared LOCO errors of the free-spacing line; u, the mean and p are all refit without the
    left-out centroid. NaN where not scored (the two ends)."""
    values = check_values("open", values)
    centroids = np.asarray(centroids, dtype=np.float64)
    out = np.full((len(smooths), len(values)), np.nan)
    for j in scored_folds("open", len(values)):
        keep = np.arange(len(values)) != j
        v, c = values[keep], centroids[keep]
        mean, axis = covariance_axis(v, c)
        for b, smooth in enumerate(smooths):
            point = curve_from_pca("open", v, c, mean, axis, smooth)(values[j])[0]
            out[b, j] = float(((centroids[j] - point) ** 2).sum())
    return out

def participation_ratio(points: np.ndarray) -> float:
    """Effective number of dimensions of the rows: (Σλ)² / Σλ² over their covariance eigenvalues."""
    p = np.asarray(points, dtype=np.float64)
    lam = np.linalg.svd(p - p.mean(axis=0), compute_uv=False) ** 2
    return float(lam.sum() ** 2 / (lam**2).sum())


def harmonic_coefficients(degrees: np.ndarray, centroids: np.ndarray, harmonics: int):
    """Least-squares trig polynomial of the centroids in the angle: constant (d,), cos (H, d) and sin (H, d)
    coefficient vectors (the full-space form of a loop curve with H harmonics)."""
    beta = np.linalg.lstsq(trig_design(degrees, harmonics), np.asarray(centroids, dtype=np.float64), rcond=None)[0]
    return beta[0], beta[1 : harmonics + 1], beta[harmonics + 1 :]


def harmonic_power(cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """(H,) variance each harmonic contributes over a full turn: (|a_h|² + |b_h|²) / 2."""
    return 0.5 * ((cos**2).sum(axis=1) + (sin**2).sum(axis=1))


def principal_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosines of the principal angles between the column spans of a (d, p) and b (d, q), largest first."""
    qa, _ = np.linalg.qr(np.asarray(a, dtype=np.float64))
    qb, _ = np.linalg.qr(np.asarray(b, dtype=np.float64))
    return np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), 0.0, 1.0)

def path_positions(points_at: Callable, values: np.ndarray, kind: str, dense: int = 4001) -> tuple[np.ndarray, float]:
    """Arc-length position of each value along a curve (cumulative Euclidean length over a dense grid in the label)
    and the total length (loop: once around, 0...360 degrees)."""
    v = np.asarray(values, dtype=np.float64)
    grid = np.linspace(0.0, PERIOD, dense + 1) if kind == "loop" else np.linspace(v.min(), v.max(), dense)
    p = points_at(grid)
    cumulative = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    return np.interp(np.mod(v, PERIOD) if kind == "loop" else v, grid, cumulative), float(cumulative[-1])


def geodesic_distances(positions: np.ndarray, total: float, kind: str) -> np.ndarray:
    """(n, n) distances along the curve between arc-length positions; loop: the shorter way round."""
    d = np.abs(positions[:, None] - positions[None, :])
    return np.minimum(d, total - d) if kind == "loop" else d