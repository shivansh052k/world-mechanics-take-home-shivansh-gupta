"""Per-clip flags for the data audit: disk exits, clipping, sub-patch motion and a frozen start.

The flags describe clips; what to do with them is decided when the splits are built.
"""
import numpy as np

from vjepa_physics.geometry import PX_PER_M, distance_travelled, frame_times

PATCH_PX = 16  # V-JEPA 2 patch size: tokens cover 16 x 16 pixel squares
TUBELET = 2  # frames per token time step
FROZEN_THROUGH_FRAME = 3  # frames 0-3 = the first two token time steps
STILL_TOLERANCE_PX = 0.5  # half the smallest rendered move (disk positions snap to whole pixels)


def predicted_motion_px(meta: dict) -> np.ndarray:
    """(frames,) distance travelled since frame 0, in pixels, from metadata."""
    t = frame_times(meta["fps"], meta["frames"])
    return distance_travelled(meta["speed_mps"], meta["acceleration_mps2"], t) * PX_PER_M


def still_frames(centre: np.ndarray) -> int:
    """Frames after frame 0 before the tracked centre first moves more than STILL_TOLERANCE_PX from it.

    A frame without a disk (NaN centre) counts as moved, so it can never extend a still start.
    """
    distance = np.linalg.norm(centre[1:] - centre[0], axis=1)
    moved = ~(distance <= STILL_TOLERANCE_PX)  # NaN <= x is False, so NaN counts as moved
    return int(np.argmax(moved)) if moved.any() else len(moved)


def clip_flags(meta: dict, centre: np.ndarray, area: np.ndarray, touches_border: np.ndarray) -> dict:
    """Flags and the per-clip values behind them, from a clip's metadata and its tracked disk.

    `centre`, `area`, `touches_border` are one clip's arrays from tracking.track_disk.
    - exit: at least one frame has no disk pixels;
    - clipped: in at least one frame, a disk pixel lies in the first or last row or column;
    - sub_patch_motion: total displacement predicted from metadata is under one patch (PATCH_PX).
      Sub-patch shifts still change patch contents, so this does not mean the tokens see no motion;
    - frozen_start: the tracked centre stays within STILL_TOLERANCE_PX of frame 0 through
      FROZEN_THROUGH_FRAME, i.e. over the first two token time steps.
    Values: frame counts, predicted and tracked total displacement (px; tracked = first to last
    fully visible frame), still frames, and the predicted displacement within each tubelet (px).
    """
    motion = predicted_motion_px(meta)
    visible = np.flatnonzero((area > 0) & ~touches_border)
    tracked = (
        float(np.linalg.norm(centre[visible[-1]] - centre[visible[0]])) if len(visible) >= 2 else float("nan")
    )
    still = still_frames(centre)
    return {
        "exit": bool((area == 0).any()),
        "clipped": bool(touches_border.any()),
        "sub_patch_motion": bool(motion[-1] < PATCH_PX),
        "frozen_start": still >= FROZEN_THROUGH_FRAME,
        "frames_without_disk": int((area == 0).sum()),
        "frames_touching_border": int(touches_border.sum()),
        "predicted_displacement_px": float(motion[-1]),
        "tracked_displacement_px": tracked,
        "still_frames": still,
        "within_tubelet_px": (motion[1::TUBELET] - motion[0::TUBELET]).tolist(),
    }