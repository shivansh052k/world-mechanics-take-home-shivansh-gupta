"""Find the disk in decoded frames: its mask, centre, size, and whether it touches the image border."""
import numpy as np
from scipy import ndimage

# Disk pixels: brightest channel above this. The disk core is ~(234, 114, 39) and the background
# (29, 32, 29), so the threshold separates them with a wide margin, whatever the channel order.
DISK_THRESHOLD = 128
# 8-connectivity: pixels touching diagonally belong to the same object.
CONNECTIVITY = np.ones((3, 3), dtype=bool)


def disk_mask(clip: np.ndarray) -> np.ndarray:
    """(T, H, W, 3) uint8 clip -> (T, H, W) bool: pixels whose brightest channel exceeds DISK_THRESHOLD."""
    return clip.max(axis=-1) > DISK_THRESHOLD


def count_objects(mask: np.ndarray) -> int:
    """Number of connected pieces (8-connectivity) in a 2-D boolean mask."""
    result = ndimage.label(mask, structure=CONNECTIVITY)
    if not isinstance(result, tuple):  # label returns (labels, count) unless an output array is passed
        raise TypeError(f"expected (labels, count) from ndimage.label, got {type(result)}")
    return int(result[1])


def track_disk(clip: np.ndarray) -> dict[str, np.ndarray]:
    """Per frame: disk centre, mask area, number of connected pieces, and whether it touches the border.

    Returns "centre" (T, 2) float, the (col, row) mask centroid in pixel-index coordinates, NaN where
    the frame has no disk pixels; "area" (T,) int; "objects" (T,) int; "touches_border" (T,) bool.
    A mask touching the border may be a partly visible disk, whose centroid is biased inwards.
    """
    masks = disk_mask(clip)
    frames = len(masks)
    centre = np.full((frames, 2), np.nan)
    area = np.zeros(frames, dtype=int)
    objects = np.zeros(frames, dtype=int)
    touches_border = np.zeros(frames, dtype=bool)
    for k, mask in enumerate(masks):
        area[k] = int(mask.sum())
        if area[k] == 0:
            continue
        rows, cols = np.nonzero(mask)
        centre[k] = cols.mean(), rows.mean()
        objects[k] = count_objects(mask)
        touches_border[k] = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
    return {"centre": centre, "area": area, "objects": objects, "touches_border": touches_border}