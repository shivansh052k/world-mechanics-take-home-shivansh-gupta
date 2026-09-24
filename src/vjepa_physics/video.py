"""Decode video clips into raw RGB frames with PyAV."""
from pathlib import Path

import av
import numpy as np

N_FRAMES = 16
SIZE = 256


def load_clip(path: str | Path, n_frames: int = N_FRAMES, size: int = SIZE) -> np.ndarray:
    """Decode every frame of a clip into a (T, H, W, 3) uint8 RGB array, in time order.

    Nothing is resized, cropped, padded or dropped. Raises ValueError if the clip does not
    decode to exactly `n_frames` frames of `size` x `size`, or if frame timestamps are
    missing or not strictly increasing.
    """
    with av.open(str(path)) as container:
        frames = list(container.decode(video=0))
        pts: list[int] = []
        for f in frames:
            if f.pts is None:
                raise ValueError(f"{path}: frame {len(pts)} has no timestamp")
            pts.append(f.pts)
        rgb = [f.to_ndarray(format="rgb24") for f in frames]

    if len(frames) != n_frames:
        raise ValueError(f"{path}: expected {n_frames} frames, decoded {len(frames)}")
    if any(b <= a for a, b in zip(pts, pts[1:])):
        raise ValueError(f"{path}: frame timestamps not strictly increasing: {pts}")

    clip = np.stack(rgb)
    if clip.shape != (n_frames, size, size, 3) or clip.dtype != np.uint8:
        raise ValueError(f"{path}: got {clip.shape} {clip.dtype}, expected ({n_frames}, {size}, {size}, 3) uint8")
    return clip