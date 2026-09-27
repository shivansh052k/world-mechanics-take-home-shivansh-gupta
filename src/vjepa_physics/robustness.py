"""Robustness checks on stored outputs: direction's transfer between motion types, error breakdowns by clip flag and
per-tubelet motion, and the overlap of the variables' subspaces."""
import numpy as np

from vjepa_physics.confounds import clip_distance, in_window, overlap_window

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