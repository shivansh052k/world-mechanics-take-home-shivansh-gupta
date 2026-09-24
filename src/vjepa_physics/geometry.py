"""Where the disk is, from its metadata: straight-line motion with constant acceleration.

A clip's disk starts at start_position_xy_m and moves along its direction theta. After t seconds it
has travelled s(t) = v t + a t^2 / 2 metres and moves at v + a t. Frame k shows the disk at t = k / fps,
so frame 0 shows the start position.
"""
import numpy as np


def frame_times(fps: int, frames: int) -> np.ndarray:
    """Time of each frame in seconds: frame k at k / fps."""
    return np.arange(frames) / fps


def distance_travelled(
    speed: float | np.ndarray, acceleration: float | np.ndarray, t: float | np.ndarray
) -> np.ndarray:
    """Distance along the direction of motion after t seconds: v t + a t^2 / 2 (metres).

    Elementwise on arrays; always returns an ndarray (0-d for scalar inputs).
    """
    t = np.asarray(t, dtype=float)
    return np.asarray(np.asarray(speed) * t + 0.5 * np.asarray(acceleration) * t**2)


def speed_at(speed: float | np.ndarray, acceleration: float | np.ndarray, t: float | np.ndarray) -> np.ndarray:
    """Speed after t seconds: v + a t (metres per second). Elementwise; always returns an ndarray."""
    return np.asarray(np.asarray(speed) + np.asarray(acceleration) * np.asarray(t, dtype=float))