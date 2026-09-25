"""Reference scores for the probes: a physics fit to the tracked disk (ceiling)."""
import numpy as np

from vjepa_physics.geometry import frame_times, pixel_to_world
from vjepa_physics.metrics import angles_from_sincos

FPS = 24  # every clip (DATA.md; metadata check)
MOTION_MODELS = ("quadratic", "linear")


def physics_fit(
    centre: np.ndarray, visible: np.ndarray, model: str = "quadratic", fps: int = FPS
) -> dict[str, float]:
    """Least-squares fit of p(t) = p0 + v t + a t^2 / 2 to one clip's tracked disk centres, frame k at t = k / fps.

    `centre` (frames, 2) px (col, row); `visible` (frames,) bool selects the frames used (fully visible disk: a
    cut-off disk biases the centroid). model "linear" fits p0 + v t (a = 0). Returns speed = |v| at t = 0 (m/s),
    acceleration = |a| (m/s^2), theta = direction of the fitted displacement from the first to the last used
    frame (degrees in [0, 360), 0 = right, 90 = up). Raises ValueError for an unknown model, mismatched shapes,
    fewer used frames than parameters + 1, or a non-finite centre on a used frame.
    """
    if model not in MOTION_MODELS:
        raise ValueError(f"unknown model {model!r}, expected one of {MOTION_MODELS}")
    centre = np.asarray(centre, dtype=float)
    visible = np.asarray(visible, dtype=bool)
    if centre.ndim != 2 or centre.shape[1] != 2 or visible.shape != centre.shape[:1]:
        raise ValueError(f"centre {centre.shape} and visible {visible.shape}: expected (frames, 2) and (frames,)")

    t = frame_times(fps, len(centre))[visible]
    columns = [np.ones_like(t), t] + ([0.5 * t**2] if model == "quadratic" else [])
    design = np.column_stack(columns)
    if len(t) < design.shape[1] + 1:
        raise ValueError(f"{len(t)} used frames, need at least {design.shape[1] + 1} for a {model} fit")
    x, y = pixel_to_world(centre[visible, 0], centre[visible, 1])
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("non-finite disk centre on a used frame")

    coef, *_ = np.linalg.lstsq(design, np.column_stack([x, y]), rcond=None)  # rows p0, v (, a); columns x, y
    v = coef[1]
    a = coef[2] if model == "quadratic" else np.zeros(2)
    dx, dy = (design[-1] - design[0]) @ coef  # fitted displacement, first to last used frame
    theta = angles_from_sincos(np.array([[dy, dx]]))[0]
    return {"speed": float(np.hypot(*v)), "acceleration": float(np.hypot(*a)), "theta": float(theta)}