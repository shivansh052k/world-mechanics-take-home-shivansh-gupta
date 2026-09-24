"""Sanity checks for video decoding and the clip loader.

Run one check at a time from the repo root, with .venv active:
    python scripts/check_video_loader.py <check>

Each check prints its result and stores it under its own key in
results/video_loader/checks.json, so that file holds the evidence for every check.
"""
import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from matplotlib.figure import Figure
from scipy import ndimage

from vjepa_physics.evidence import save_result
from vjepa_physics.video import load_clip

REPO = Path(__file__).resolve().parents[1]
CLIP = REPO / "data/speed/videos/scene_1000/video.mp4"  # speed 2.69 m/s, theta 230.625 deg
OUT = REPO / "results/video_loader/checks.json"
FIGURE = REPO / "results/video_loader/frames.png"


def check_inspect() -> dict:
    """Raw stream facts from PyAV, independent of the loader.

    Passes if: 16 frames are decoded and the container also reports 16; every frame is
    256 x 256; timestamps (pts x time_base, as exact fractions) rise by exactly 1/24 s from
    each frame to the next. Container, codec and pixel format are recorded as observations.
    """
    with av.open(str(CLIP)) as c:
        s = c.streams.video[0]
        frames = list(c.decode(video=0))
        observed = {
            "container": c.format.name,
            "codec": s.codec_context.name,
            "pix_fmt": s.codec_context.pix_fmt,
            "average_rate": str(s.average_rate),
            "frame_formats": sorted({f.format.name for f in frames}),
        }
        stream_frames = s.frames
        stream_time_base = s.time_base
        pts = [f.pts for f in frames]
        frame_time_bases = {f.time_base for f in frames}
        times = [
            Fraction(f.pts) * f.time_base
            for f in frames
            if f.pts is not None and f.time_base is not None
        ]
        sizes = sorted({(f.width, f.height) for f in frames})

    steps = {b - a for a, b in zip(times, times[1:])}
    criteria = {
        "sixteen_frames_decoded_and_reported": len(frames) == 16 and stream_frames == 16,
        "every_frame_256x256": sizes == [(256, 256)],
        "step_exactly_1_24_s": len(times) == len(frames) and steps == {Fraction(1, 24)},
    }
    return {
        "clip": str(CLIP.relative_to(REPO)),
        "observed": observed,
        "size": [s.width, s.height],
        "stream_frames": stream_frames,
        "decoded_frames": len(frames),
        "frame_sizes": sizes,
        "time_base": str(stream_time_base),
        "frame_time_bases_equal_stream": frame_time_bases == {stream_time_base},
        "pts": pts,
        "times_s": [str(t) for t in times],
        "steps_s": sorted(str(step) for step in steps),
        "criteria": criteria,
        "passed": all(criteria.values()),
    }


def expect_value_error(**kwargs) -> dict:
    """Call load_clip with a wrong expectation; it must raise ValueError, not adapt silently."""
    try:
        load_clip(CLIP, **kwargs)
    except ValueError as e:
        return {"raised": True, "message": str(e).replace(f"{REPO}/", "")}
    return {"raised": False, "message": None}


def check_load() -> dict:
    """load_clip output contract, and its guards against a wrong frame count or size.

    Expected: (16, 256, 256, 3) uint8, C-contiguous; asking for 15 frames or 224 px
    must raise ValueError instead of padding, truncating or resizing.
    """
    clip = load_clip(CLIP)
    guard_frames = expect_value_error(n_frames=15)
    guard_size = expect_value_error(size=224)
    return {
        "clip": str(CLIP.relative_to(REPO)),
        "shape": list(clip.shape),
        "dtype": str(clip.dtype),
        "c_contiguous": bool(clip.flags.c_contiguous),
        "min": int(clip.min()),
        "max": int(clip.max()),
        "guard_n_frames_15": guard_frames,
        "guard_size_224": guard_size,
        "passed": bool(
            clip.shape == (16, 256, 256, 3)
            and clip.dtype == np.uint8
            and clip.flags.c_contiguous
            and guard_frames["raised"]
            and guard_size["raised"]
        ),
    }


def check_repeat() -> dict:
    """Decoding is deterministic: two independent loads are bit-identical.

    Also records a SHA-256 of the decoded pixels, so later runs (or other library
    versions) can be compared against this one.
    """
    first = load_clip(CLIP)
    second = load_clip(CLIP)
    identical = bool(np.array_equal(first, second))
    return {
        "clip": str(CLIP.relative_to(REPO)),
        "identical": identical,
        "pixels_sha256": hashlib.sha256(first.tobytes()).hexdigest(),
        "passed": identical,
    }


def disk_mask(clip: np.ndarray) -> np.ndarray:
    """(T, H, W) bool: pixels whose brightest channel exceeds 128 (channel-order agnostic)."""
    return clip.max(axis=-1) > 128


def distance_to_disk(disk_frame: np.ndarray) -> np.ndarray:
    """Per pixel, Euclidean distance in px to the nearest disk pixel (0 on the disk)."""
    dist = ndimage.distance_transform_edt(~disk_frame)
    if not isinstance(dist, np.ndarray):  # return type depends on flags; we ask for distances only
        raise TypeError(f"expected an ndarray of distances, got {type(dist)}")
    return dist


def check_colour() -> dict:
    """Channel order is RGB: the disk is orange (R > G > B) on a dark background.

    The disk mask uses each pixel's brightest channel (> 128), so it does not assume the
    channel order it is testing. Also reports where the clip's extreme values lie, as
    distance in px to the nearest disk pixel.
    Expected: disk core mean ~ (227, 113, 44), background mean ~ (29, 31, 28).
    """
    clip = load_clip(CLIP)
    disk = disk_mask(clip)
    core = ndimage.binary_erosion(disk, structure=np.ones((1, 3, 3)), iterations=2)
    dist = np.stack([distance_to_disk(d) for d in disk])  # (T, H, W) px to nearest disk pixel
    far = dist > 16  # two 8x8 codec blocks away from the disk

    def location(hit: np.ndarray) -> dict:
        d = dist[hit]
        return {
            "pixels": int(hit.sum()),
            "dist_px_median": float(np.median(d)),
            "dist_px_max": float(d.max()),
        }

    lo, hi = int(clip.min()), int(clip.max())
    core_mean = clip[core].mean(axis=0)
    background_mean = clip[far].mean(axis=0)
    return {
        "clip": str(CLIP.relative_to(REPO)),
        "disk_pixels_per_frame": disk.sum(axis=(1, 2)).tolist(),
        "disk_core_pixels_per_frame": core.sum(axis=(1, 2)).tolist(),
        "disk_core_mean": [round(float(v), 1) for v in core_mean],
        "background_mean": [round(float(v), 1) for v in background_mean],
        "background_min_max": [int(clip[far].min()), int(clip[far].max())],
        "min_value": lo,
        "min_count_per_channel": [int((clip[..., c] == lo).sum()) for c in range(3)],
        "min_location": location((clip == lo).any(axis=-1)),
        "max_value": hi,
        "max_count_per_channel": [int((clip[..., c] == hi).sum()) for c in range(3)],
        "max_location": location((clip == hi).any(axis=-1)),
        "passed": bool(
            core_mean[0] > core_mean[1] > core_mean[2]
            and np.all(np.abs(core_mean - [227, 113, 44]) <= 10)
            and np.all(np.abs(background_mean - [29, 31, 28]) <= 10)
        ),
    }


def disk_centroids(clip: np.ndarray) -> np.ndarray:
    """(T, 2) disk centroid per frame as (col, row) in pixel-index coordinates."""
    centroids = []
    for frame_mask in disk_mask(clip):
        rows, cols = np.nonzero(frame_mask)
        centroids.append((cols.mean(), rows.mean()))
    return np.array(centroids)


def predicted_centroids(meta: dict, time_offset: int = 0, flip_y: bool = True) -> np.ndarray:
    """(T, 2) disk position from metadata as (col, row): 32 px/m, origin at pixel 128, t = k / fps."""
    t = (np.arange(meta["frames"]) + time_offset) / meta["fps"]
    theta = np.deg2rad(meta["theta_degrees"])
    travelled = meta["speed_mps"] * t + 0.5 * meta["acceleration_mps2"] * t**2
    x0, y0 = meta["start_position_xy_m"]
    x = x0 + travelled * np.cos(theta)
    y = y0 + travelled * np.sin(theta)
    col = 128 + 32 * x
    row = 128 - 32 * y if flip_y else 128 + 32 * y
    return np.stack([col, row], axis=1)


def check_order() -> dict:
    """Frame content is in time order and matches the metadata geometry.

    The documented mapping (t = k / fps, 32 px/m, y flipped) must fit within 1 px in every
    frame; reversed frames, a one-frame time shift and an unflipped y axis must all miss by > 2 px.
    """
    meta = json.loads((CLIP.parent / "metadata.json").read_text())
    measured = disk_centroids(load_clip(CLIP))

    def errors(pred: np.ndarray, meas: np.ndarray) -> np.ndarray:
        return np.linalg.norm(pred - meas, axis=1)

    documented = predicted_centroids(meta)
    residual = measured - documented
    alternatives = {
        "frames_reversed": errors(documented, measured[::-1]),
        "time_shift_plus_one_frame": errors(predicted_centroids(meta, time_offset=1), measured),
        "no_y_flip": errors(predicted_centroids(meta, flip_y=False), measured),
    }
    err = errors(documented, measured)
    return {
        "clip": str(CLIP.relative_to(REPO)),
        "metadata": {k: meta[k] for k in ("speed_mps", "acceleration_mps2", "theta_degrees", "start_position_xy_m", "fps", "frames")},
        "error_px_per_frame": [round(float(e), 3) for e in err],
        "error_px_max": round(float(err.max()), 3),
        "mean_residual_col_row_px": [round(float(v), 3) for v in residual.mean(axis=0)],
        "alternatives_error_px_max": {k: round(float(v.max()), 3) for k, v in alternatives.items()},
        "passed": bool(err.max() <= 1.0 and all(v.max() > 2.0 for v in alternatives.values())),
    }


def load_clip_opencv(path: Path) -> tuple[np.ndarray, str]:
    """Independent decode with OpenCV's FFmpeg backend: (T, H, W, 3) uint8 RGB, backend name."""
    # Imported here only: cv2 bundles its own FFmpeg, and loading it next to PyAV's
    # duplicates some Objective-C classes on macOS. Only the OpenCV checks need both.
    import cv2

    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {path} with the FFmpeg backend")
    backend = cap.getBackendName()
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames), backend


def check_opencv() -> dict:
    """PyAV (load_clip) and OpenCV decode the clip to pixel-identical RGB frames.

    OpenCV returns BGR, converted to RGB before comparing. Any difference is recorded with
    its size and its distance to the disk.
    """
    pyav = load_clip(CLIP)
    ocv, backend = load_clip_opencv(CLIP)
    same_shape = pyav.shape == ocv.shape
    result = {
        "clip": str(CLIP.relative_to(REPO)),
        "opencv_backend": backend,
        "pyav_shape": list(pyav.shape),
        "opencv_shape": list(ocv.shape),
        "opencv_dtype": str(ocv.dtype),
    }
    if not same_shape:
        return result | {"identical": False, "passed": False}

    diff = np.abs(pyav.astype(np.int16) - ocv.astype(np.int16))
    differs = (diff > 0).any(axis=-1)  # (T, H, W)
    dist = np.stack([distance_to_disk(d) for d in disk_mask(pyav)])
    identical = bool(np.array_equal(pyav, ocv))
    return result | {
        "identical": identical,
        "differing_pixels": int(differs.sum()),
        "max_abs_diff_per_channel": [int(diff[..., c].max()) for c in range(3)],
        "differing_dist_px_median_max": (
            [float(np.median(dist[differs])), float(dist[differs].max())] if differs.any() else None
        ),
        "passed": identical,
    }


def compare_rgb(a: np.ndarray, b: np.ndarray) -> dict:
    """Pixel-level comparison of two (T, H, W, 3) uint8 clips."""
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {
        "identical": bool(np.array_equal(a, b)),
        "differing_pixels": int((diff > 0).any(axis=-1).sum()),
        "max_abs_diff": int(diff.max()),
    }


def check_opencv_bicubic() -> dict:
    """Diagnose the PyAV/OpenCV mismatch by converting PyAV's frames with OpenCV's scaler flag.

    PyAV's to_ndarray converts YUV to RGB with swscale's BILINEAR flag by default; OpenCV
    uses BICUBIC. If PyAV with BICUBIC matches OpenCV exactly, the flag explains the whole
    difference. Diagnostic only: no pass/fail.
    """
    with av.open(str(CLIP)) as c:
        frames = list(c.decode(video=0))
        tags = sorted({(int(f.colorspace), int(f.color_range)) for f in frames})
        bilinear = np.stack([f.to_ndarray(format="rgb24") for f in frames])
        bicubic = np.stack([f.to_ndarray(format="rgb24", interpolation="BICUBIC") for f in frames])
    ocv, backend = load_clip_opencv(CLIP)
    return {
        "clip": str(CLIP.relative_to(REPO)),
        "opencv_backend": backend,
        "pyav_frame_colorspace_and_range": [list(t) for t in tags],
        "pyav_bilinear_vs_opencv": compare_rgb(bilinear, ocv),
        "pyav_bicubic_vs_opencv": compare_rgb(bicubic, ocv),
        "pyav_bilinear_vs_bicubic": compare_rgb(bilinear, bicubic),
    }


def check_opencv_diff_stats() -> dict:
    """Describe the PyAV - OpenCV pixel difference: signed histogram and mean by region.

    A constant offset suggests a colour-matrix or rounding-offset difference; a symmetric
    spread around 0 suggests rounding or dither noise. Diagnostic only: no pass/fail.
    """
    pyav = load_clip(CLIP)
    ocv, backend = load_clip_opencv(CLIP)
    diff = pyav.astype(np.int16) - ocv.astype(np.int16)  # signed, (T, H, W, 3)
    disk = disk_mask(pyav)
    far = np.stack([distance_to_disk(d) for d in disk]) > 16
    channels = ("R", "G", "B")
    return {
        "clip": str(CLIP.relative_to(REPO)),
        "opencv_backend": backend,
        "signed_diff_counts": {
            ch: {str(v): int((diff[..., c] == v).sum()) for v in range(-3, 4)}
            for c, ch in enumerate(channels)
        },
        "mean_signed_diff_all": [round(float(v), 3) for v in diff.reshape(-1, 3).mean(axis=0)],
        "mean_signed_diff_disk": [round(float(v), 3) for v in diff[disk].mean(axis=0)],
        "mean_signed_diff_far_background": [round(float(v), 3) for v in diff[far].mean(axis=0)],
        "mean_abs_diff_all": round(float(np.abs(diff).mean()), 3),
    }


def check_figure() -> dict:
    """Frames 0, 5, 10, 15 with the metadata-predicted (+) and measured (x) disk centre.

    Visual evidence that decoding, colour, frame order and the pixel mapping are right.
    The numeric version of this check is `order`.
    """
    clip = load_clip(CLIP)
    meta = json.loads((CLIP.parent / "metadata.json").read_text())
    predicted = predicted_centroids(meta)
    measured = disk_centroids(clip)
    shown = [0, 5, 10, 15]

    fig = Figure(figsize=(12, 3.6), layout="constrained")
    for ax, k in zip(fig.subplots(1, len(shown)), shown):
        ax.imshow(clip[k], interpolation="nearest")
        ax.plot(predicted[:, 0], predicted[:, 1], color="white", linewidth=0.8, alpha=0.5)
        ax.plot(*predicted[k], marker="+", color="cyan", markersize=12, markeredgewidth=1.5, linestyle="none")
        ax.plot(*measured[k], marker="x", color="black", markersize=7, markeredgewidth=1.5, linestyle="none")
        ax.set_title(f"frame {k}  (t = {k}/{meta['fps']} s)", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"{CLIP.relative_to(REPO)}: predicted (+) vs measured (x) disk centre; line = predicted path",
        fontsize=10,
    )
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=150)

    error = np.linalg.norm(predicted - measured, axis=1)
    return {
        "clip": str(CLIP.relative_to(REPO)),
        "figure": str(FIGURE.relative_to(REPO)),
        "frames_shown": shown,
        "error_px_shown_frames": [round(float(error[k]), 3) for k in shown],
    }


CHECKS = {
    "inspect": check_inspect,
    "load": check_load,
    "repeat": check_repeat,
    "colour": check_colour,
    "order": check_order,
    "opencv": check_opencv,
    "opencv_bicubic": check_opencv_bicubic,
    "opencv_diff_stats": check_opencv_diff_stats,
    "figure": check_figure,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("check", choices=sorted(CHECKS))
    name = parser.parse_args().check
    result = CHECKS[name]()
    print(json.dumps(result, indent=2))
    save_result(OUT, name, result)
    print(f"saved '{name}' -> {OUT.relative_to(REPO)}")
    if result.get("passed") is False:
        raise SystemExit(f"check '{name}' FAILED")


if __name__ == "__main__":
    main()