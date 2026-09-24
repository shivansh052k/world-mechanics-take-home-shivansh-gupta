"""Checks for clip preprocessing: V-JEPA 2 video processor with resize and center-crop off.

Run one check at a time from the repo root, with .venv active:
    python scripts/check_preprocessing.py <check>

Each check prints its result and stores it under its own key in
results/preprocessing/checks.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from vjepa_physics.evidence import save_result
from vjepa_physics.preprocess import load_processor, preprocess_clip
from vjepa_physics.video import load_clip

REPO = Path(__file__).resolve().parents[1]
CLIP = REPO / "data/speed/videos/scene_1000/video.mp4"  # speed 2.69 m/s, theta 230.625 deg
OUT = REPO / "results/preprocessing/checks.json"
FIGURE = REPO / "results/preprocessing/default_vs_ours.png"

# Shipped default: shortest edge 256 -> 292 (bilinear), then center-crop 256.
SCALE = 292 / 256
CROP_OFFSET = (292 - 256) / 2  # resized pixels removed on each side

# Checkpoint video_preprocessor_config.json: ImageNet statistics, rescale 1/255.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def processor_settings(native: bool) -> dict:
    p = load_processor(native)
    return {
        "do_resize": p.do_resize,
        "shortest_edge": p.size.shortest_edge,
        "do_center_crop": p.do_center_crop,
        "crop_height_width": [p.crop_size.height, p.crop_size.width],
        "do_rescale": p.do_rescale,
        "rescale_factor": p.rescale_factor,
        "do_normalize": p.do_normalize,
        "image_mean": list(p.image_mean),
        "image_std": list(p.image_std),
    }


def check_config() -> dict:
    """Effective processor settings: ours (resize and crop off) vs the shipped default.

    Expected: ours = no resize, no crop, rescale 1/255, ImageNet normalization;
    default = resize shortest edge to 292, center-crop 256 x 256, same rescale and normalization.
    """
    native = processor_settings(native=True)
    default = processor_settings(native=False)
    same_normalization = all(native[k] == default[k] for k in ("do_rescale", "rescale_factor", "do_normalize", "image_mean", "image_std"))
    return {
        "native": native,
        "default": default,
        "passed": bool(
            native["do_resize"] is False
            and native["do_center_crop"] is False
            and native["do_rescale"] is True
            and native["rescale_factor"] == 1 / 255
            and native["do_normalize"] is True
            and tuple(native["image_mean"]) == IMAGENET_MEAN
            and tuple(native["image_std"]) == IMAGENET_STD
            and default["do_resize"] is True
            and default["shortest_edge"] == 292
            and default["crop_height_width"] == [256, 256]
            and same_normalization
        ),
    }


def check_manual() -> dict:
    """Processor output equals (x / 255 - mean) / std computed by hand, in float64.

    The processor fuses this as (x - 255 mean) / (255 std) in float32, so the two agree
    up to float32 rounding; the tolerance is torch's default fp32 atol (1e-5).
    """
    clip = load_clip(CLIP)
    ours = preprocess_clip(clip)
    manual = (clip.astype(np.float64) / 255 - IMAGENET_MEAN) / IMAGENET_STD  # (T, H, W, 3)
    manual = manual.transpose(0, 3, 1, 2)  # (T, 3, H, W)
    diff = np.abs(ours.numpy().astype(np.float64) - manual)
    return {
        "clip": str(CLIP.relative_to(REPO)),
        "shape": list(ours.shape),
        "dtype": str(ours.dtype),
        "device": str(ours.device),
        "value_min_max": [round(float(ours.min()), 4), round(float(ours.max()), 4)],
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "passed": bool(
            tuple(ours.shape) == (16, 3, 256, 256)
            and str(ours.dtype) == "torch.float32"
            and diff.max() <= 1e-5
        ),
    }


def check_identity() -> dict:
    """Undoing the normalization and rounding recovers the decoded clip bit-for-bit.

    Proves the processor changed nothing spatially (no resize, crop, shift or channel
    swap): only the per-channel affine map was applied.
    """
    clip = load_clip(CLIP)
    ours = preprocess_clip(clip).numpy().astype(np.float64)  # (T, 3, H, W)
    mean = np.array(IMAGENET_MEAN).reshape(1, 3, 1, 1)
    std = np.array(IMAGENET_STD).reshape(1, 3, 1, 1)
    levels = (ours * std + mean) * 255  # back to 0..255, still float
    recovered = np.rint(levels).astype(np.uint8).transpose(0, 2, 3, 1)  # (T, H, W, 3)
    return {
        "clip": str(CLIP.relative_to(REPO)),
        "max_distance_to_integer_level": float(np.abs(levels - np.rint(levels)).max()),
        "identical": bool(np.array_equal(recovered, clip)),
        "passed": bool(np.array_equal(recovered, clip)),
    }


def to_levels(x: torch.Tensor) -> np.ndarray:
    """(T, 3, H, W) normalized input -> (T, H, W, 3) float pixel levels on the 0..255 scale."""
    mean = np.array(IMAGENET_MEAN).reshape(1, 3, 1, 1)
    std = np.array(IMAGENET_STD).reshape(1, 3, 1, 1)
    return ((x.numpy().astype(np.float64) * std + mean) * 255).transpose(0, 2, 3, 1)


def disk_centroids(levels: np.ndarray) -> np.ndarray:
    """(T, 2) centroid (col, row) of pixels whose brightest channel exceeds 128."""
    centroids = []
    for frame_mask in levels.max(axis=-1) > 128:
        rows, cols = np.nonzero(frame_mask)
        centroids.append((cols.mean(), rows.mean()))
    return np.array(centroids)


def check_default() -> dict:
    """What the shipped default (resize 292 + center-crop 256) does to the clip.

    Predicted: coordinates map as col' = SCALE * (col + 0.5) - 0.5 - CROP_OFFSET (same for
    rows), i.e. 32 px/m becomes 36.5 px/m and only original pixels 15.78..240.22 survive.
    Passes if the measured disk centroids follow that map. Saves the comparison figure.
    """
    clip = load_clip(CLIP)
    ours = to_levels(preprocess_clip(clip))
    default = to_levels(preprocess_clip(clip, native=False))
    c_ours, c_default = disk_centroids(ours), disk_centroids(default)

    expected_offset = 0.5 * SCALE - 0.5 - CROP_OFFSET
    free_fit = [np.polyfit(c_ours[:, a], c_default[:, a], 1) for a in range(2)]  # [slope, intercept] per axis
    offset_at_scale = (c_default - SCALE * c_ours).mean(axis=0)  # intercept with the slope fixed to SCALE
    kept = [CROP_OFFSET / SCALE, (CROP_OFFSET + 256) / SCALE]  # original pixel edges that survive the crop

    fig = Figure(figsize=(8, 8.4), layout="constrained")
    axes = fig.subplots(2, 2)
    for col, k in enumerate([0, 15]):
        for row, (name, levels) in enumerate([("ours (no resize, no crop)", ours), ("shipped default (resize 292, crop 256)", default)]):
            ax = axes[row, col]
            ax.imshow(np.clip(np.rint(levels[k]), 0, 255).astype(np.uint8), interpolation="nearest")
            ax.set_title(f"{name}\nframe {k}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[0, col].add_patch(Rectangle(
            (kept[0] - 0.5, kept[0] - 0.5), kept[1] - kept[0], kept[1] - kept[0],
            fill=False, edgecolor="white", linestyle="--", linewidth=1,
        ))
    fig.suptitle(
        f"{CLIP.relative_to(REPO)}\n"
        f"dashed box = the part of our input the shipped default keeps, "
        f"scaled x{SCALE:.4f} (32 -> {32 * SCALE:.1f} px/m)",
        fontsize=9,
    )
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=150)

    return {
        "clip": str(CLIP.relative_to(REPO)),
        "figure": str(FIGURE.relative_to(REPO)),
        "default_shape": list(default.shape),
        "predicted_scale": SCALE,
        "predicted_offset_px": expected_offset,
        "fitted_slope_col_row": [round(float(f[0]), 4) for f in free_fit],
        "offset_at_predicted_scale_col_row": [round(float(v), 3) for v in offset_at_scale],
        "kept_original_pixels_edges": [round(v, 2) for v in kept],
        "px_per_metre_default": 32 * SCALE,
        "disk_pixels_frame0_ours_default": [int((ours[0].max(-1) > 128).sum()), int((default[0].max(-1) > 128).sum())],
        "passed": bool(
            default.shape == (16, 256, 256, 3)
            and all(abs(f[0] - SCALE) <= 0.02 for f in free_fit)
            and np.all(np.abs(offset_at_scale - expected_offset) <= 0.5)
        ),
    }



CHECKS = {
    "config": check_config,
    "manual": check_manual,
    "identity": check_identity,
    "default": check_default,
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