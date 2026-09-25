"""Where a layer curve rises: fixed rules applied to scores over layer indices 0-24 (0 = embedding)."""
import numpy as np

N_INDICES = 25  # hidden_states indices 0-24; the final norm is not part of the curve (same depth as block 23)
DEPTH = 24  # depth fraction = index / DEPTH
TRANSITION_FRACTION = 0.5  # transition = first index reaching min + 0.5 x (max - min)
LATE_RISE_FRACTION = 0.8  # also reported: the 80% rise point
MIN_CLEAR_RISE = 0.2  # max - min below this (in R²) is recorded as "no clear transition"


def checked_curves(curves: np.ndarray) -> np.ndarray:
    """(..., 25) finite float64 scores. Raises ValueError otherwise."""
    c = np.asarray(curves, dtype=np.float64)
    if c.ndim not in (1, 2) or c.shape[-1] != N_INDICES or not np.isfinite(c).all():
        raise ValueError(f"expected finite (25,) or (B, 25) curves, got {c.shape}")
    return c


def rise_index(curves: np.ndarray, fraction: float) -> np.ndarray:
    """First index whose score reaches min + fraction x (max - min) of its own curve; (B,) for (B, 25), 0-d for (25,)."""
    c = checked_curves(curves)
    low, high = c.min(axis=-1, keepdims=True), c.max(axis=-1, keepdims=True)
    return np.argmax(c >= low + fraction * (high - low), axis=-1)


def largest_jump_index(curves: np.ndarray) -> np.ndarray:
    """Index i with the largest single-layer increase c[i] - c[i-1] (the first one on ties)."""
    return np.argmax(np.diff(checked_curves(curves), axis=-1), axis=-1) + 1


def transition_points(curve: np.ndarray) -> dict:
    """The fixed summaries of one (25,) curve: transition (50%), 80% rise, largest jump, and whether the rise is clear."""
    c = checked_curves(curve)
    if c.ndim != 1:
        raise ValueError("transition_points takes one curve; use rise_index for a stack")
    transition = int(rise_index(c, TRANSITION_FRACTION))
    rise = float(c.max() - c.min())
    return {
        "transition_index": transition,
        "transition_depth_fraction": transition / DEPTH,
        "rise_80_index": int(rise_index(c, LATE_RISE_FRACTION)),
        "largest_jump_index": int(largest_jump_index(c)),
        "largest_jump": float(np.diff(c).max()),
        "min": float(c.min()), "max": float(c.max()), "rise": rise,
        "clear_transition": rise >= MIN_CLEAR_RISE,
    }