"""Decode video clips into raw RGB frames with PyAV."""
from fractions import Fraction
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



def probe_clip(path: str | Path) -> dict:
    """Container and stream facts for one clip; frames are decoded but not converted to RGB.

    Returns codec, pixel format and size from the stream, the frame count the stream reports and
    the number actually decoded, each decoded frame's size, the stream's average rate, and each
    frame's timestamp in seconds as an exact Fraction (pts x time_base; None if either is missing).
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = list(container.decode(video=0))
        return {
            "codec": stream.codec_context.name,
            "pix_fmt": stream.codec_context.pix_fmt,
            "size": (stream.codec_context.width, stream.codec_context.height),
            "reported_frames": stream.frames,
            "decoded_frames": len(frames),
            "frame_sizes": sorted({(f.width, f.height) for f in frames}),
            "average_rate": stream.average_rate,
            "times_s": [
                None if f.pts is None or f.time_base is None else Fraction(f.pts) * f.time_base
                for f in frames
            ],
        }