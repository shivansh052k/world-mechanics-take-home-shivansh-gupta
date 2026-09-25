"""Probe scores: R² and MAE for any target, circular error for directions (degrees, math convention)."""
import numpy as np


def checked(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Both arrays as float64. Raises ValueError on a shape mismatch, an empty input or a non-finite value."""
    t = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    if t.shape != p.shape or t.ndim == 0 or t.size == 0:
        raise ValueError(f"shapes {t.shape} and {p.shape}: expected equal and non-empty")
    if not (np.isfinite(t).all() and np.isfinite(p).all()):
        raise ValueError("non-finite value in the targets or predictions")
    return t, p


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """1 - SS_res / SS_tot, with SS_tot about the mean of y_true (as sklearn's r2_score).

    (n,) targets give one score; (n, k) targets give the mean of the k per-column scores (sklearn's
    "uniform_average"), used for (sin, cos). Raises ValueError if a column of y_true is constant.
    """
    t, p = checked(y_true, y_pred)
    t2, p2 = t.reshape(len(t), -1), p.reshape(len(p), -1)
    ss_res = ((t2 - p2) ** 2).sum(axis=0)
    ss_tot = ((t2 - t2.mean(axis=0)) ** 2).sum(axis=0)
    if (t2 == t2[:1]).all(axis=0).any():  # compare values: a constant column's mean can be off by rounding
        raise ValueError("a target column is constant; R² is undefined")
    return float((1 - ss_res / ss_tot).mean())


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error of (n,) targets, in the label's units."""
    t, p = checked(y_true, y_pred)
    if t.ndim != 1:
        raise ValueError(f"expected (n,) targets, got {t.shape}")
    return float(np.abs(t - p).mean())


def sincos_targets(theta_degrees: np.ndarray) -> np.ndarray:
    """(n,) angles in degrees -> (n, 2) probe targets, columns (sin, cos)."""
    rad = np.radians(np.asarray(theta_degrees, dtype=np.float64))
    return np.column_stack([np.sin(rad), np.cos(rad)])


def angles_from_sincos(pred: np.ndarray) -> np.ndarray:
    """(n, 2) predicted (sin, cos) -> (n,) angles in degrees in [0, 360).

    The pair need not have unit length; only its direction counts. Raises ValueError for a (0, 0) pair,
    whose angle is undefined (arctan2 would silently return 0).
    """
    p = np.asarray(pred, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError(f"expected (n, 2) predictions, got {p.shape}")
    if (np.hypot(p[:, 0], p[:, 1]) == 0).any():
        raise ValueError("a predicted (sin, cos) pair is (0, 0); its angle is undefined")
    angles = np.degrees(np.arctan2(p[:, 0], p[:, 1])) % 360.0
    return np.where(angles >= 360.0, angles - 360.0, angles)  # a tiny negative angle can round to exactly 360


def circular_errors(true_degrees: np.ndarray, pred_degrees: np.ndarray) -> np.ndarray:
    """(n,) absolute angular differences in degrees, in [0, 180]; inputs may lie outside [0, 360)."""
    t, p = checked(true_degrees, pred_degrees)
    d = np.abs(p - t) % 360.0
    return np.minimum(d, 360.0 - d)


def circular_mae(true_degrees: np.ndarray, pred_degrees: np.ndarray) -> float:
    """Mean circular error in degrees (DATA.md's direction metric); chance level for uniform angles is 90."""
    return float(circular_errors(true_degrees, pred_degrees).mean())



def bootstrap_indices(n: int, n_resamples: int, seed: int) -> np.ndarray:
    """(n_resamples, n) clip indices drawn with replacement from one generator seeded with `seed`.

    Reuse the same matrix for every layer and method scored on the same clips: their resampled scores are then
    paired, so a difference between two of them gets its own (usually much smaller) spread.
    """
    if n < 2 or n_resamples < 1:
        raise ValueError(f"need n >= 2 and n_resamples >= 1, got {n}, {n_resamples}")
    return np.random.default_rng(seed).integers(0, n, size=(n_resamples, n))


def checked_indices(indices: np.ndarray, n: int) -> np.ndarray:
    """Resample indices as an int array; raises ValueError unless 2-D with every entry in [0, n)."""
    idx = np.asarray(indices)
    if idx.ndim != 2 or idx.dtype.kind not in "iu" or idx.min() < 0 or idx.max() >= n:
        raise ValueError(f"indices {idx.shape} {idx.dtype}: expected 2-D integers in [0, {n})")
    return idx


def resampled_r2(y_true: np.ndarray, y_pred: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """(B,) R² on each resample's rows, with r2's definition (evaluation mean; (n, k) -> mean of column scores).

    Raises ValueError if a resample has a constant target column (R² undefined).
    """
    t, p = checked(y_true, y_pred)
    idx = checked_indices(indices, len(t))
    tb = t.reshape(len(t), -1)[idx]  # (B, n, k)
    pb = p.reshape(len(p), -1)[idx]
    ss_res = ((tb - pb) ** 2).sum(axis=1)
    ss_tot = ((tb - tb.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    if (tb == tb[:, :1]).all(axis=1).any():  # compare values: a constant column's mean can be off by rounding
        raise ValueError("a resample has a constant target column; R² is undefined")
    return (1 - ss_res / ss_tot).mean(axis=1)


def resampled_mean(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """(B,) mean of per-clip `values` (absolute or circular errors) over each resample's rows."""
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1 or v.size == 0 or not np.isfinite(v).all():
        raise ValueError(f"expected finite (n,) values, got {v.shape}")
    return v[checked_indices(indices, len(v))].mean(axis=1)


def percentile_interval(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval: the (1 - level) / 2 and (1 + level) / 2 quantiles of the resampled statistic."""
    s = np.asarray(samples, dtype=np.float64)
    if s.ndim != 1 or s.size == 0 or not np.isfinite(s).all() or not 0 < level < 1:
        raise ValueError("expected finite (B,) samples and 0 < level < 1")
    low, high = np.percentile(s, [50 * (1 - level), 50 * (1 + level)])
    return float(low), float(high)