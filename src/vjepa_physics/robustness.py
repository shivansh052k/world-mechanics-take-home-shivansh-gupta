"""Robustness checks on stored outputs: direction's transfer between motion types, error breakdowns by clip flag and
per-tubelet motion, and the overlap of the variables' subspaces."""
import numpy as np
from scipy.linalg import subspace_angles
from scipy.stats import rankdata

from vjepa_physics.confounds import FPS, FRAMES, clip_distance, in_window, overlap_window
from vjepa_physics.flags import TUBELET
from vjepa_physics.geometry import PX_PER_M, distance_travelled, frame_times
from vjepa_physics.metrics import angles_from_sincos, circular_errors, percentile_interval
from vjepa_physics.nullspace import RANK_TOLERANCE, random_span_basis
MOTION_TYPES = ("velocity", "acceleration")  # direction set: constant speed, or from rest with constant acceleration


def motion_masks(table: dict) -> dict[str, np.ndarray]:
    """Direction table -> {motion type: mask}. Raises ValueError unless every clip has exactly one of the two types."""
    motion = np.asarray(table["motion"])
    masks = {m: motion == m for m in MOTION_TYPES}
    if not (masks["velocity"] ^ masks["acceleration"]).all():
        raise ValueError("a clip is neither or both motion types")
    return masks


def distance_overlap(table: dict) -> np.ndarray:
    """Mask of direction clips whose distance travelled lies in the range both motion types reach (ends included)."""
    d = clip_distance(table)
    window = overlap_window([d[m] for m in motion_masks(table).values()])
    return in_window(d, window)



def clip_errors(variable: str, labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """(clips,) per-clip error: circular degrees for direction ((sin, cos) predictions), else |prediction - label|."""
    if variable == "direction":
        return circular_errors(labels, angles_from_sincos(predictions))
    return np.abs(np.asarray(predictions, dtype=np.float64) - np.asarray(labels, dtype=np.float64))


def within_tubelet_px(table: dict) -> np.ndarray:
    """(clips, 8) disk displacement in px inside each tubelet (between its two frames), from the metadata's motion;
    the same computation as the flags table's within-tubelet columns."""
    moved = distance_travelled(np.asarray(table["speed_mps"])[:, None], np.asarray(table["acceleration_mps2"])[:, None],
                               frame_times(FPS, FRAMES)[None]) * PX_PER_M
    return moved[:, 1::TUBELET] - moved[:, 0::TUBELET]


def stratified_difference(errors: np.ndarray, flagged: np.ndarray, strata: np.ndarray, n_resamples: int,
                          seed: int) -> dict:
    """Flagged - unflagged mean error within the strata holding both kinds of clips, averaged with weights = flagged
    clips per stratum; 95% interval from resampling flagged and unflagged clips separately within each stratum."""
    errors, flagged, strata = np.asarray(errors, np.float64), np.asarray(flagged, bool), np.asarray(strata)
    eligible = [s for s in np.unique(strata) if flagged[strata == s].any() and (~flagged[strata == s]).any()]
    if not eligible:
        raise ValueError("no stratum holds both flagged and unflagged clips")
    weights = np.array([flagged[strata == s].sum() for s in eligible], dtype=np.float64)
    weights /= weights.sum()
    rng = np.random.default_rng(seed)
    point, samples = 0.0, np.zeros(n_resamples)
    for w, s in zip(weights, eligible):
        f, u = errors[(strata == s) & flagged], errors[(strata == s) & ~flagged]
        point += w * (f.mean() - u.mean())
        samples += w * (f[rng.integers(0, len(f), (n_resamples, len(f)))].mean(axis=1)
                        - u[rng.integers(0, len(u), (n_resamples, len(u)))].mean(axis=1))
    in_strata = np.isin(strata, eligible)
    return {"difference": float(point), "ci": [float(c) for c in percentile_interval(samples)],
            "strata": [str(s) for s in eligible], "n_flagged": int((in_strata & flagged).sum()),
            "n_unflagged": int((in_strata & ~flagged).sum())}


def _slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xc = x - x.mean(axis=-1, keepdims=True)
    return (xc * (y - y.mean(axis=-1, keepdims=True))).sum(axis=-1) / (xc**2).sum(axis=-1)


def _pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ac, bc = a - a.mean(axis=-1, keepdims=True), b - b.mean(axis=-1, keepdims=True)
    return (ac * bc).sum(axis=-1) / np.sqrt((ac**2).sum(axis=-1) * (bc**2).sum(axis=-1))


def within_range_trend(labels: np.ndarray, predictions: np.ndarray, indices: np.ndarray) -> dict:
    """Does a readout still order labels inside a range? Least-squares slope of prediction on label and Spearman
    correlation, with 95% intervals over the resampled clip indices (B, n)."""
    x, y = np.asarray(labels, np.float64), np.asarray(predictions, np.float64)
    xs, ys = x[indices], y[indices]
    return {
        "slope": float(_slopes(x, y)), "slope_ci": [float(c) for c in percentile_interval(_slopes(xs, ys))],
        "spearman": float(_pearson(rankdata(x), rankdata(y))),
        "spearman_ci": [float(c) for c in percentile_interval(_pearson(rankdata(xs, axis=1), rankdata(ys, axis=1)))],
    }
    
SUBSPACE_KINDS = ("weights", "patterns")
NULL_KEYS = ("mean_angle_degrees", "overlap_a_on_b", "overlap_b_on_a", "grassmann")


def orthonormal(columns: np.ndarray) -> np.ndarray:
    """(d, k) orthonormal basis of the columns' span (reduced QR); ValueError if the columns are rank-deficient."""
    c = np.asarray(columns, dtype=np.float64)
    s = np.linalg.svd(c, compute_uv=False)
    if s[-1] <= RANK_TOLERANCE * s[0]:
        raise ValueError("columns are numerically rank-deficient")
    q, _ = np.linalg.qr(c)
    return q


def rescale(basis: np.ndarray, own_scale: np.ndarray, target_scale: np.ndarray, kind: str) -> np.ndarray:
    """Columns written in one standardized space (own_scale) -> another (target_scale; ones = raw space), unnormalized.

    weights (probe weight directions, covectors): x · (w / own) = z_target · (w * target / own), so scale by target / own.
    patterns (feature-space directions, e.g. covariance directions): a shift Δz_own = Δx / own, so scale by own / target.
    """
    ratio = np.asarray(target_scale, dtype=np.float64) / np.asarray(own_scale, dtype=np.float64)
    if kind == "weights":
        return np.asarray(basis, dtype=np.float64) * ratio[:, None]
    if kind == "patterns":
        return np.asarray(basis, dtype=np.float64) / ratio[:, None]
    raise ValueError(f"kind must be one of {SUBSPACE_KINDS}, got {kind!r}")


def re_express(basis: np.ndarray, own_scale: np.ndarray, target_scale: np.ndarray, kind: str) -> np.ndarray:
    """rescale, then an orthonormal basis of the result (the span is what principal angles compare)."""
    return orthonormal(rescale(basis, own_scale, target_scale, kind))


def principal_angles(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    """(min(k_a, k_b),) principal angles in radians, ascending (scipy's method, accurate near 0)."""
    return np.sort(subspace_angles(qa, qb))


def subspace_metrics(qa: np.ndarray, qb: np.ndarray) -> dict:
    """The physics paper's App. C.4 metrics for two orthonormal bases: mean principal angle (degrees), projection overlap
    ||Q_Aᵀ Q_B||²_F / dim(B) (and / dim(A)), Grassmann distance sqrt(sum θ²) (radians); all angles in degrees."""
    angles = principal_angles(qa, qb)
    overlap = float(((qa.T @ qb) ** 2).sum())
    return {
        "mean_angle_degrees": float(np.degrees(angles).mean()),
        "overlap_a_on_b": overlap / qb.shape[1], "overlap_b_on_a": overlap / qa.shape[1],
        "grassmann": float(np.sqrt((angles**2).sum())), "angles_degrees": np.degrees(angles).tolist(),
    }


def null_metrics(qa: np.ndarray, span_b: np.ndarray, n_cols: int, transform, n_draws: int,
                 seed: int) -> dict[str, np.ndarray]:
    """Metrics of qa against n_draws random n_cols-dim subspaces drawn inside span_b (B's own standardized train span)
    and passed through `transform` (the same re-expression as the real B); draw i uses seed + i."""
    out = {key: np.empty(n_draws) for key in NULL_KEYS}
    for i in range(n_draws):
        m = subspace_metrics(qa, transform(random_span_basis(span_b, n_cols, seed + i)))
        for key in NULL_KEYS:
            out[key][i] = m[key]
    return out


def null_reading(overlap: float, null_overlaps: np.ndarray) -> str:
    """Reading of an observed overlap against its null's central 95% (rule fixed before any run)."""
    low, high = percentile_interval(null_overlaps)
    if overlap > high:
        return "aligned beyond chance"
    if overlap < low:
        return "less aligned than chance"
    return "at chance"