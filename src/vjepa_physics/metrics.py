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
    if (ss_tot == 0).any():
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