"""Checks for the video files: every clip decodes to the expected format, and no clip is a duplicate.

Run one check at a time from the repo root, with .venv active:
    python scripts/check_videos.py <check>

Each check prints its result and stores it under its own key in results/videos/checks.json.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from vjepa_physics.data import DATASETS, load_dataset, read_manifest, resolve
from vjepa_physics.evidence import file_sha256, save_result
from vjepa_physics.geometry import disk_centres, distance_outside_image
from vjepa_physics.video import load_clip, probe_clip

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = REPO / "results/videos/checks.json"

# DATA.md: 1,500 + 1,536 + 1,536 clips of 256 x 256 pixels at 24 frames per second.
EXPECTED_CLIPS = 4572
SIZE = (256, 256)
FPS = 24
STEP = Fraction(1, FPS)

DISK_THRESHOLD = 128  # disk pixels: brightest channel above this (as in the decoder cross-check)
MEDIAN_OUTLIER = 10  # diagnostic: a frame's median colour this far from its dataset's typical one
CRITERIA = ("decodes", "stream_format", "timestamps", "no_uniform_frames")
SAMPLE = 20  # problem lists are saved as a count plus the first few entries

# Background colour: every frame's median colour, in every clip, is exactly this (format diagnostics).
BACKGROUND = np.array([29, 32, 29], dtype=np.uint8)
# Disk radius in px: the disk mask covers ~350 px (349-351 per frame on the test clip), and area = pi r^2.
DISK_RADIUS_PX = float(np.sqrt(350 / np.pi))

# Per-clip decoded-pixel hashes, kept as a reference for later runs (regenerable, git-ignored).
ARTIFACT = REPO / "artifacts/videos/decoded_hashes.csv"
# Decoded-pixel SHA-256 of data/speed/videos/scene_1000, saved by the video loader's repeat check.
TEST_CLIP = ("speed", 1000)
TEST_CLIP_SHA256 = "03285f4cf9ffc6ecf90541b47e7f08c217eab4e4809d42f7042baf29ab9d95c0"


def sample(items) -> dict:
    """Count and the first SAMPLE entries, so failures are diagnosable without huge JSON."""
    items = list(items)
    return {"count": len(items), "first": items[:SAMPLE]}


def clip_paths(dataset: str) -> list[tuple[int, Path]]:
    """(id, video path) for every clip of a dataset, in manifest order."""
    root = DATA / dataset
    return [(row["id"], resolve(root, row["video"])) for row in read_manifest(root)]


def check_format() -> dict:
    """Every clip decodes to 16 frames of 256 x 256 at a constant 1/24 s step, with no blank frame.

    Passes if, for every clip: load_clip succeeds (16 frames, 256 x 256 x 3 uint8, strictly
    increasing timestamps); the stream and every frame are 256 x 256 and the average rate is 24;
    consecutive timestamps (pts x time_base, exact fractions) differ by exactly 1/24 s; and no
    frame has every pixel identical in every channel. And all 4,572 clips are checked. Decoder
    errors are recorded as failures, not raised. Diagnostics: codec, pixel format, reported frame
    count, whether the first frame is at t = 0, all-black frames, frame median colours and outliers,
    frames without disk pixels (expected where the disk leaves the frame), and clips with identical
    consecutive frames (expected where the disk does not move).
    """
    per_dataset, total = {}, 0
    for dataset in DATASETS:
        clips = clip_paths(dataset)
        total += len(clips)
        failed: dict[str, list[str]] = {name: [] for name in CRITERIA}
        codecs, pix_fmts, reported = Counter(), Counter(), Counter()
        not_at_zero, no_disk_clips, frozen_clips = [], [], []
        black_frames = no_disk_frames = 0
        medians: list[tuple[int, int, np.ndarray]] = []  # (clip id, frame, median RGB)

        for clip_id, path in clips:
            try:
                info = probe_clip(path)
            except Exception as e:  # any decoder error is a recorded failure, not a crash
                failed["stream_format"].append(f"id {clip_id}: {type(e).__name__}: {e}")
                failed["timestamps"].append(f"id {clip_id}: not probed")
            else:
                codecs[info["codec"]] += 1
                pix_fmts[info["pix_fmt"]] += 1
                reported[str(info["reported_frames"])] += 1
                if not (info["size"] == SIZE and info["frame_sizes"] == [SIZE] and info["average_rate"] == FPS):
                    failed["stream_format"].append(
                        f"id {clip_id}: size {info['size']}, frame sizes {info['frame_sizes']}, rate {info['average_rate']}"
                    )
                times = info["times_s"]
                steps = None if None in times else {b - a for a, b in zip(times, times[1:])}
                if steps != {STEP}:
                    shown = "missing timestamps" if steps is None else sorted(map(str, steps))
                    failed["timestamps"].append(f"id {clip_id}: steps {shown}")
                if not times or times[0] != 0:
                    not_at_zero.append(f"id {clip_id}: first frame at {times[0] if times else None}")

            try:
                clip = load_clip(path)
            except Exception as e:
                failed["decodes"].append(f"id {clip_id}: {type(e).__name__}: {e}")
                failed["no_uniform_frames"].append(f"id {clip_id}: not decoded")
                continue
            uniform = [k for k, frame in enumerate(clip) if (frame == frame[0, 0]).all()]
            if uniform:
                failed["no_uniform_frames"].append(f"id {clip_id}: frames {uniform}")
            black_frames += int((clip.reshape(len(clip), -1).max(axis=1) == 0).sum())
            has_disk = (clip.max(axis=-1) > DISK_THRESHOLD).any(axis=(1, 2))
            if not has_disk.all():
                no_disk_frames += int((~has_disk).sum())
                no_disk_clips.append(clip_id)
            if any(np.array_equal(a, b) for a, b in zip(clip, clip[1:])):
                frozen_clips.append(clip_id)
            for k, median in enumerate(np.median(clip.reshape(len(clip), -1, 3), axis=1)):
                medians.append((clip_id, k, median))

        colours = np.array([m for _, _, m in medians]) if medians else np.empty((0, 3))
        typical = np.median(colours, axis=0) if len(colours) else None
        outliers = [] if typical is None else [
            f"id {i} frame {k}: {np.round(m, 1).tolist()}"
            for i, k, m in medians
            if np.abs(m - typical).max() > MEDIAN_OUTLIER
        ]
        per_dataset[dataset] = {
            "criteria": {name: not failed[name] for name in CRITERIA},
            "clips": len(clips),
            "problems": {name: sample(reasons) for name, reasons in failed.items()},
            "diagnostics": {
                "codecs": dict(codecs),
                "pixel_formats": dict(pix_fmts),
                "reported_frame_counts": dict(reported),
                "first_frame_not_at_zero": sample(not_at_zero),
                "all_black_frames": black_frames,
                "frame_median_colour_typical": None if typical is None else typical.tolist(),
                "frame_median_colour_min_max": (
                    [colours.min(axis=0).tolist(), colours.max(axis=0).tolist()] if len(colours) else None
                ),
                "frames_with_median_colour_outlier": sample(outliers),
                "frames_without_disk_pixels": no_disk_frames,
                "clips_with_frames_without_disk": sample(no_disk_clips),
                "clips_with_identical_consecutive_frames": sample(frozen_clips),
            },
        }

    criteria = {"all_clips_checked": total == EXPECTED_CLIPS}
    return {
        "criteria": criteria,
        "clips": total,
        "per_dataset": per_dataset,
        "passed": all(criteria.values())
        and all(all(r["criteria"].values()) for r in per_dataset.values()),
    }

def sci(value: float) -> float:
    """Round to 4 significant digits, for readable JSON."""
    return float(f"{value:.4g}")


def check_uniform_frames() -> dict:
    """Diagnostic: are the uniform frames found by `format` frames that the disk has fully left?

    For every frame of every clip: whether it is uniform (every pixel identical in every channel),
    its disk pixels (brightest channel > 128), its pixels that differ from the background colour, and
    how far the disk centre predicted from metadata lies outside the image. The explanation holds if
    every uniform frame (a) equals the background colour exactly, (b) has no disk pixels, and (c) has
    its predicted centre more than one disk radius outside the image. Also recorded: frames whose disk
    is predicted fully outside but that are not uniform; frames without disk pixels whose disk is
    predicted at least partly inside; frames with disk pixels whose disk is predicted fully outside
    (would contradict the pixel mapping); whether uniform frames always form the end of their clip.
    No pass/fail: this tests an explanation of the `format` failure, which stays on record.
    """
    per_dataset, all_hold = {}, True
    for dataset in DATASETS:
        uniform_frames, uniform_clips, not_at_end = [], set(), []
        broken: dict[str, list[str]] = {"not_background_colour": [], "has_disk_pixels": [], "disk_not_fully_outside": []}
        outside_not_uniform, no_disk_but_inside, disk_but_outside = [], [], []
        max_off_background = max_outside = 0.0

        for (clip_id, path), meta in zip(clip_paths(dataset), load_dataset(DATA, dataset), strict=True):
            if meta["id"] != clip_id:
                raise RuntimeError(f"{dataset}: manifest id {clip_id} but metadata id {meta['id']}")
            clip = load_clip(path)
            centres = disk_centres(meta)
            outside = distance_outside_image(centres[:, 0], centres[:, 1])
            fully_outside = outside > DISK_RADIUS_PX
            uniform = np.array([(frame == frame[0, 0]).all() for frame in clip])
            disk_pixels = (clip.max(axis=-1) > DISK_THRESHOLD).sum(axis=(1, 2))
            off_background = (clip != BACKGROUND).any(axis=-1).sum(axis=(1, 2))

            for k in range(len(clip)):
                where = f"id {clip_id} frame {k}"
                if uniform[k]:
                    uniform_frames.append(where)
                    uniform_clips.add(clip_id)
                    if not np.array_equal(clip[k, 0, 0], BACKGROUND):
                        broken["not_background_colour"].append(f"{where}: colour {clip[k, 0, 0].tolist()}")
                    if disk_pixels[k]:
                        broken["has_disk_pixels"].append(f"{where}: {int(disk_pixels[k])} disk px")
                    if not fully_outside[k]:
                        broken["disk_not_fully_outside"].append(f"{where}: centre {outside[k]:.2f} px outside")
                elif fully_outside[k]:
                    outside_not_uniform.append(
                        f"{where}: centre {outside[k]:.2f} px outside, "
                        f"{int(off_background[k])} px off background, {int(disk_pixels[k])} disk px"
                    )
                    max_off_background = max(max_off_background, float(off_background[k]))
                    max_outside = max(max_outside, float(outside[k]))
                if disk_pixels[k] == 0 and not fully_outside[k]:
                    no_disk_but_inside.append(f"{where}: centre {outside[k]:.2f} px outside")
                if disk_pixels[k] > 0 and fully_outside[k]:
                    disk_but_outside.append(f"{where}: {int(disk_pixels[k])} disk px, centre {outside[k]:.2f} px outside")

            if uniform.any() and not uniform[int(np.argmax(uniform)):].all():
                not_at_end.append(f"id {clip_id}: uniform frames {np.flatnonzero(uniform).tolist()}")

        holds = not any(broken.values())
        all_hold = all_hold and holds
        per_dataset[dataset] = {
            "explanation_holds": holds,
            "uniform_frames": sample(uniform_frames),
            "clips_with_uniform_frames": len(uniform_clips),
            "explanation_broken_by": {name: sample(items) for name, items in broken.items()},
            "uniform_frames_not_at_clip_end": sample(not_at_end),
            "fully_outside_but_not_uniform": sample(outside_not_uniform),
            "fully_outside_but_not_uniform_max_off_background_px": int(max_off_background),
            "fully_outside_but_not_uniform_max_centre_distance_px": sci(max_outside),
            "no_disk_pixels_but_disk_predicted_inside": sample(no_disk_but_inside),
            "disk_pixels_but_disk_predicted_fully_outside": sample(disk_but_outside),
        }

    return {
        "background_colour": BACKGROUND.tolist(),
        "disk_radius_px": sci(DISK_RADIUS_PX),
        "explanation_holds": all_hold,
        "per_dataset": per_dataset,
    }

def check_duplicates() -> dict:
    """No two clips decode to identical pixels, within or across the three datasets.

    Hashes each clip's decoded pixels (load_clip: all 16 frames, as bytes) with SHA-256. Passes if
    all 4,572 clips are hashed, all clip hashes are distinct, and the test clip's hash equals the one
    saved by the video loader's repeat check (same decoding, same hashing). Writes one row per clip
    (dataset, id, clip hash, 16 frame hashes) to artifacts/videos/decoded_hashes.csv as a reference
    for later runs. Frame hashes shared between clips are a diagnostic: frames repeat legitimately
    (uniform background frames after the disk leaves, frozen frames).
    """
    rows = []
    for dataset in DATASETS:
        for clip_id, path in clip_paths(dataset):
            clip = load_clip(path)
            rows.append({
                "clip": f"{dataset}/{clip_id}",
                "key": (dataset, clip_id),
                "clip_sha256": hashlib.sha256(clip.tobytes()).hexdigest(),
                "frame_sha256": [hashlib.sha256(frame.tobytes()).hexdigest() for frame in clip],
            })

    clips_by_hash, frame_owners = defaultdict(list), defaultdict(set)
    for row in rows:
        clips_by_hash[row["clip_sha256"]].append(row["clip"])
        for digest in row["frame_sha256"]:
            frame_owners[digest].add(row["clip"])
    duplicate_groups = sorted(sorted(group) for group in clips_by_hash.values() if len(group) > 1)
    shared = {digest: owners for digest, owners in frame_owners.items() if len(owners) > 1}
    background_sha256 = hashlib.sha256(np.broadcast_to(BACKGROUND, (256, 256, 3)).tobytes()).hexdigest()
    other_shared = [sorted(owners) for digest, owners in shared.items() if digest != background_sha256]
    test_hash = next((row["clip_sha256"] for row in rows if row["key"] == TEST_CLIP), None)

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    with ARTIFACT.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "id", "clip_sha256", *(f"frame_{k:02d}_sha256" for k in range(16))])
        for row in rows:
            writer.writerow([*row["key"], row["clip_sha256"], *row["frame_sha256"]])

    criteria = {
        "all_clips_hashed": len(rows) == EXPECTED_CLIPS,
        "no_duplicate_clips": not duplicate_groups,
        "test_clip_hash_matches_repeat_check": test_hash == TEST_CLIP_SHA256,
    }
    return {
        "criteria": criteria,
        "clips_hashed": len(rows),
        "distinct_clip_hashes": len(clips_by_hash),
        "duplicate_clip_groups": sample(duplicate_groups),
        "test_clip": {"clip": "/".join(map(str, TEST_CLIP)), "sha256": test_hash, "expected": TEST_CLIP_SHA256},
        "artifact": {
            "path": str(ARTIFACT.relative_to(REPO)),
            "rows": len(rows),
            "sha256": file_sha256(ARTIFACT),
        },
        "diagnostics": {
            "frames_hashed": sum(len(row["frame_sha256"]) for row in rows),
            "distinct_frame_hashes": len(frame_owners),
            "frame_hashes_shared_by_several_clips": len(shared),
            "clips_sharing_the_background_frame": len(shared.get(background_sha256, ())),
            "other_frames_shared_between_clips": sample(sorted(other_shared)),
        },
        "passed": all(criteria.values()),
    }
    

CHECKS = {
    "format": check_format,
    "uniform_frames": check_uniform_frames,
    "duplicates": check_duplicates,
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