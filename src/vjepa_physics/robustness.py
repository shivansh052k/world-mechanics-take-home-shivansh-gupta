"""Robustness checks on stored outputs: direction's transfer between motion types, error breakdowns by clip flag and
per-tubelet motion, and the overlap of the variables' subspaces."""
import numpy as np
from scipy.stats import rankdata

from vjepa_physics.confounds import FPS, FRAMES, clip_distance, in_window, overlap_window
from vjepa_physics.flags import TUBELET
from vjepa_physics.geometry import PX_PER_M, distance_travelled, frame_times
from vjepa_physics.metrics import angles_from_sincos, circular_errors, percentile_interval

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