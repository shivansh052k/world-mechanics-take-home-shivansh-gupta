"""Checks for disk tracking: the disk is found in every frame, and its positions match the metadata.

Run one check at a time from the repo root, with .venv active:
    python scripts/check_tracking.py <check>

Each check prints its result and stores it under its own key in results/tracking/checks.json.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import ndimage

from vjepa_physics.data import DATASETS, load_dataset, read_manifest, resolve
from vjepa_physics.evidence import file_sha256, save_result
from vjepa_physics.geometry import disk_centres, distance_travelled, frame_times, world_to_pixel
from vjepa_physics.tracking import disk_mask, track_disk
from vjepa_physics.video import load_clip

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = REPO / "results/tracking/checks.json"

# Tracked disk per clip and frame, for later steps (regenerable, git-ignored).
ARTIFACT = REPO / "artifacts/tracking/tracked_disk.npz"

# DATA.md: 1,500 + 1,536 + 1,536 clips of 16 frames.
EXPECTED_CLIPS = 4572
FRAMES = 16
CORE_EROSION = 2  # disk core for colour: mask shrunk by 2 px, leaving out the blended edge
SAMPLE = 20  # problem lists are saved as a count plus the first few entries

# The video loader's order-check criterion (tracked vs predicted centre), now applied to every clip.
MAPPING_TOLERANCE_PX = 1.0
# The displacement-angle diagnostic only uses clips whose disk moves at least this far while fully visible.
MIN_DISPLACEMENT_PX = 5.0

def sample(items) -> dict:
    """Count and the first SAMPLE entries, so failures are diagnosable without huge JSON."""
    items = list(items)
    return {"count": len(items), "first": items[:SAMPLE]}


def core_colour(clip: np.ndarray, touches_border: np.ndarray) -> np.ndarray:
    """(3,) mean RGB of the disk core over frames whose disk does not touch the border; NaN if none."""
    core = ndimage.binary_erosion(
        disk_mask(clip), structure=np.ones((1, 3, 3), dtype=bool), iterations=CORE_EROSION
    )
    core[touches_border] = False
    return clip[core].mean(axis=0) if core.any() else np.full(3, np.nan)


def check_track() -> dict:
    """Track the disk in every frame of every clip, and save the positions for later steps.

    Passes if all 4,572 clips are tracked with 16 frames each, and every frame with disk pixels has
    exactly one connected piece (DATA.md: a single disk). Writes artifacts/tracking/tracked_disk.npz:
    per clip its dataset and id, per frame the centre (col, row; NaN without disk pixels), mask area,
    number of pieces and border contact, and per clip the mean disk-core colour (for the colour
    check). Areas, disk-less and border-touching frames are diagnostics.
    """
    datasets, ids, centres, areas, objects, borders, colours = [], [], [], [], [], [], []
    for dataset in DATASETS:
        root = DATA / dataset
        for row in read_manifest(root):
            clip = load_clip(resolve(root, row["video"]))
            tracked = track_disk(clip)
            datasets.append(dataset)
            ids.append(row["id"])
            centres.append(tracked["centre"])
            areas.append(tracked["area"])
            objects.append(tracked["objects"])
            borders.append(tracked["touches_border"])
            colours.append(core_colour(clip, tracked["touches_border"]))

    dataset_of = np.array(datasets)
    id_of = np.array(ids)
    centre, area, pieces, border = np.stack(centres), np.stack(areas), np.stack(objects), np.stack(borders)
    colour = np.stack(colours)

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ARTIFACT, dataset=dataset_of, id=id_of, centre=centre, area=area,
        objects=pieces, touches_border=border, core_colour=colour,
    )

    has_disk = area > 0
    visible = has_disk & ~border
    multi = [
        f"{dataset_of[i]}/{id_of[i]} frame {k}: {pieces[i, k]} pieces"
        for i, k in zip(*np.nonzero(has_disk & (pieces != 1)))
    ]
    per_dataset = {}
    for dataset in DATASETS:
        sel = dataset_of == dataset
        visible_per_clip = visible[sel].sum(axis=1)
        per_dataset[dataset] = {
            "clips": int(sel.sum()),
            "frames_without_disk": int((~has_disk[sel]).sum()),
            "frames_touching_border": int(border[sel].sum()),
            "clips_with_frames_touching_border": int(border[sel].any(axis=1).sum()),
            "fully_visible_frames": int(visible[sel].sum()),
            "fewest_fully_visible_frames_in_a_clip": int(visible_per_clip.min()),
            "clips_with_fewer_than_3_fully_visible_frames": sample(
                f"id {i}" for i, n in zip(id_of[sel], visible_per_clip) if n < 3
            ),
            "area_fully_visible_min_max": [int(area[sel][visible[sel]].min()), int(area[sel][visible[sel]].max())],
            "pieces_histogram": {str(k): v for k, v in sorted(Counter(pieces[sel][has_disk[sel]].tolist()).items())},
        }

    criteria = {
        "all_clips_tracked": centre.shape == (EXPECTED_CLIPS, FRAMES, 2),
        "single_object": not multi,
    }
    return {
        "criteria": criteria,
        "clips": len(ids),
        "frames": int(area.size),
        "frames_with_several_pieces": sample(multi),
        "per_dataset": per_dataset,
        "artifact": {"path": str(ARTIFACT.relative_to(REPO)), "sha256": file_sha256(ARTIFACT)},
        "passed": all(criteria.values()),
    }

def world_positions(meta: dict, time_offset_frames: int = 0) -> np.ndarray:
    """(frames, 2) disk centre in world metres (x, y) from metadata; frame k at (k + offset) / fps."""
    t = frame_times(meta["fps"], meta["frames"]) + time_offset_frames / meta["fps"]
    s = distance_travelled(meta["speed_mps"], meta["acceleration_mps2"], t)
    theta = np.deg2rad(meta["theta_degrees"])
    x0, y0 = meta["start_position_xy_m"]
    return np.stack([x0 + s * np.cos(theta), y0 + s * np.sin(theta)], axis=1)


def to_pixels(world: np.ndarray, flip_y: bool = True) -> np.ndarray:
    """(..., 2) world metres -> (..., 2) pixel (col, row); flip_y=False is the rejected alternative."""
    col, row = world_to_pixel(world[..., 0], world[..., 1] if flip_y else -world[..., 1])
    return np.stack([col, row], axis=-1)


def stats(values: np.ndarray) -> dict:
    return {
        "max": round(float(values.max()), 3),
        "p99": round(float(np.percentile(values, 99)), 3),
        "median": round(float(np.median(values)), 3),
    }


def check_mapping() -> dict:
    """Tracked disk centres match the metadata under the pixel mapping, in every fully visible frame.

    Reads the positions saved by `track` (its SHA-256 must match the one `track` recorded, and its rows
    must follow the metadata order). Prediction: geometry.disk_centres (32 px/m, origin at pixel 128,
    y flipped, frame k at t = k / fps). Passes if every fully visible frame (disk pixels, not touching
    the border) is within 1.0 px of the prediction, and in every clip frame 0 is fully visible and
    within 1.0 px of the start position. Diagnostics per dataset: error spread and mean residual; the
    alternatives rejected on the test clip (frames reversed, time shifted by one frame, y not flipped),
    with how many clips each alone could not reject; least-squares fits of tracked col vs world x and
    row vs world y (independent of the mapping constants); errors in border frames; and the angle of
    the tracked displacement vs theta.
    """
    recorded = json.loads(OUT.read_text())["track"]["result"]["artifact"]["sha256"]
    with np.load(ARTIFACT) as tracked:
        dataset_of, id_of = tracked["dataset"], tracked["id"]
        centre, area, border = tracked["centre"], tracked["area"], tracked["touches_border"]

    metas = [meta for dataset in DATASETS for meta in load_dataset(DATA, dataset)]
    keys = [(dataset, meta["id"]) for dataset in DATASETS for meta in load_dataset(DATA, dataset)]
    aligned = keys == list(zip(dataset_of.tolist(), id_of.tolist()))

    visible = (area > 0) & ~border
    world = np.stack([world_positions(m) for m in metas])  # (clips, frames, 2)
    documented = np.stack([disk_centres(m) for m in metas])
    alternatives = {
        "frames_reversed": documented[:, ::-1],
        "time_shift_plus_one_frame": to_pixels(np.stack([world_positions(m, 1) for m in metas])),
        "no_y_flip": to_pixels(world, flip_y=False),
    }
    error = np.linalg.norm(centre - documented, axis=-1)  # NaN where the frame has no disk pixels
    theta = np.array([m["theta_degrees"] for m in metas])

    per_dataset = {}
    for dataset in DATASETS:
        sel = dataset_of == dataset
        vis = visible[sel]
        e = error[sel][vis]
        residual = (centre - documented)[sel][vis]
        worst = np.argsort(np.where(vis, error[sel], -1), axis=None)[::-1][:5]
        alternative_stats = {}
        for name, alt in alternatives.items():
            alt_error = np.linalg.norm(centre[sel] - alt[sel], axis=-1)
            per_clip_max = np.where(vis, alt_error, -np.inf).max(axis=1)
            alternative_stats[name] = stats(alt_error[vis]) | {
                "clips_it_alone_cannot_reject": int((per_clip_max <= MAPPING_TOLERANCE_PX).sum())
            }
        fits = {
            axis: dict(zip(("slope", "intercept"), (round(float(v), 4) for v in np.polyfit(world[sel][..., a][vis], centre[sel][..., a][vis], 1))))
            for axis, a in (("col_vs_world_x", 0), ("row_vs_world_y", 1))
        }
        angle_errors = []
        for c, v, t in zip(centre[sel], vis, theta[sel]):
            frames = np.flatnonzero(v)
            d = c[frames[-1]] - c[frames[0]]
            if np.hypot(*d) >= MIN_DISPLACEMENT_PX:
                seen = np.degrees(np.arctan2(-d[1], d[0]))  # rows point down, so flip for world angle
                angle_errors.append(abs((seen - t + 180) % 360 - 180))
        border_error = error[sel][border[sel]]
        per_dataset[dataset] = {
            "fully_visible_frames": int(vis.sum()),
            "error_px": stats(e),
            "mean_residual_col_row_px": [round(float(v), 4) for v in residual.mean(axis=0)],
            "worst_frames": [
                f"id {id_of[sel][i]} frame {k}: {error[sel][i, k]:.3f} px"
                for i, k in zip(*np.unravel_index(worst, vis.shape))
            ],
            "alternatives": alternative_stats,
            "fits": fits,
            "border_frames_error_px": stats(border_error) if border_error.size else None,
            "displacement_angle_error_deg": (
                stats(np.array(angle_errors)) | {"clips": len(angle_errors)} if angle_errors else None
            ),
        }

    criteria = {
        "artifact_matches_track": file_sha256(ARTIFACT) == recorded,
        "metadata_aligned": aligned,
        "mapping_within_1px": bool(visible.any() and np.nanmax(np.where(visible, error, np.nan)) <= MAPPING_TOLERANCE_PX),
        "frame0_is_start": bool(visible[:, 0].all() and (error[:, 0] <= MAPPING_TOLERANCE_PX).all()),
    }
    return {
        "criteria": criteria,
        "tolerance_px": MAPPING_TOLERANCE_PX,
        "artifact": {"path": str(ARTIFACT.relative_to(REPO)), "sha256": recorded},
        "frame0_error_px": stats(error[:, 0]),
        "per_dataset": per_dataset,
        "passed": all(criteria.values()),
    }

def check_documented_colour() -> dict:
    """The disk is blue, as DATA.md states ("a single blue disk").

    Reads each clip's mean disk-core colour saved by `track` (its SHA-256 must match the one `track`
    recorded). Passes if every clip has a measured colour and, in every clip, blue is the disk core's
    brightest channel. The measured colour range, which channel is brightest, and how many clips are
    ordered R > G > B (orange) are diagnostics.
    """
    recorded = json.loads(OUT.read_text())["track"]["result"]["artifact"]["sha256"]
    with np.load(ARTIFACT) as tracked:
        dataset_of, colour = tracked["dataset"], tracked["core_colour"]

    measured = np.isfinite(colour).all(axis=1)
    brightest = colour.argmax(axis=1)  # 0 = R, 1 = G, 2 = B
    per_dataset = {}
    for dataset in DATASETS:
        sel = (dataset_of == dataset) & measured
        c = colour[sel]
        per_dataset[dataset] = {
            "clips_measured": int(sel.sum()),
            "core_colour_mean_rgb": [round(float(v), 1) for v in c.mean(axis=0)],
            "core_colour_min_rgb": [round(float(v), 1) for v in c.min(axis=0)],
            "core_colour_max_rgb": [round(float(v), 1) for v in c.max(axis=0)],
            "brightest_channel_counts": {
                name: int((brightest[sel] == i).sum()) for i, name in enumerate(("red", "green", "blue"))
            },
            "clips_ordered_r_gt_g_gt_b": int(((c[:, 0] > c[:, 1]) & (c[:, 1] > c[:, 2])).sum()),
        }

    criteria = {
        "artifact_matches_track": file_sha256(ARTIFACT) == recorded,
        "all_clips_measured": bool(measured.all() and len(colour) == EXPECTED_CLIPS),
        "disk_is_blue": bool(measured.all() and (brightest == 2).all()),
    }
    return {
        "criteria": criteria,
        "documented": "DATA.md: 'a single blue disk moving on a dark background'",
        "per_dataset": per_dataset,
        "passed": all(criteria.values()),
    }

CHECKS = {
    "track": check_track,
    "mapping": check_mapping,
    "documented_colour": check_documented_colour,
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