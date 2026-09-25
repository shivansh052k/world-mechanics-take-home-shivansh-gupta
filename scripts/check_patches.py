"""Checks for the per-patch activations of the direction clips (local probes and spatial generalization).

Run one check at a time from the repo root, with .venv active:
    python scripts/check_patches.py <check>

Each check prints its result and stores it under its own key in results/patches/checks.json.
"""
import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from vjepa_physics.data import read_manifest, resolve
from vjepa_physics.evidence import file_sha256, save_result, verified_artifact
from vjepa_physics.extraction import PATCHES, PATCH_SITES, patch_activations
from vjepa_physics.joined import load_joined
from vjepa_physics.model import load_model, weights_fingerprint
from vjepa_physics.preprocess import preprocess_clip
from vjepa_physics.reproducibility import SEED, set_seeds
from vjepa_physics.video import load_clip

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = REPO / "results/patches/checks.json"
PATCH_DIR = REPO / "artifacts/patches"  # regenerable, git-ignored
DEVICE = "mps"
HIDDEN = 1024
VARIABLE = "direction"  # the local-to-global test uses the direction clips only
EXPECTED_CLIPS = 1500  # DATA.md

# In-memory weights fingerprint right after a clean load (results/model/checks.json, key "load").
REFERENCE_FINGERPRINT = "c865f524c1376e4452943b208d7d50ba588be9490604f235a3d9c9dc80804ede"
MARGIN_BYTES = 10 * 10**9  # free disk that must remain after the array is written: 10 GB
MEAN_TOLERANCE = 1e-6  # mean over the 256 patch vectors vs the stored all-token mean, relative
PROGRESS_EVERY = 100
SPOT_CLIPS = 16  # clips re-extracted live and compared bit for bit (also the reproducibility gate)


def array_sha256(a: np.ndarray) -> str:
    """SHA-256 of an array's bytes in C order."""
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def clip_paths(ids: np.ndarray) -> list[Path]:
    """Video path of every clip id, in the given order (manifest paths, resolved inside data/direction/)."""
    root = DATA / VARIABLE
    by_id = {row["id"]: resolve(root, row["video"]) for row in read_manifest(root)}
    return [by_id[int(i)] for i in ids]


def mean_mismatch(patch_row: np.ndarray, stored_row: np.ndarray) -> float:
    """max |patch mean - all-token mean| / max |all-token mean| over indices 0-24, both in float64.

    `patch_row` (25, 256, 1024): per-patch time averages; `stored_row` (26, 8, 1024): the stored per-time-step
    means of the same clip, whose mean over the 8 steps is the all-token mean.
    """
    a = patch_row.astype(np.float64).mean(axis=1)
    b = np.asarray(stored_row[:len(PATCH_SITES)], dtype=np.float64).mean(axis=1)
    return float(np.abs(a - b).max() / np.abs(b).max())


def check_extract_direction() -> dict:
    """Per-patch time-averaged activations of every direction clip, saved as one array.

    Clips in the joined table's id order (= the stored activations' order). Refuses to start unless the array's
    size plus 10 GB of disk are free. Batch size 1, MPS, fp32. Writes artifacts/patches/direction.npy, shape
    (1500, 25, 256, 1024) float32 (first to direction.partial.npy, renamed after the last clip), and
    direction_ids.npy. Passes if: the disk check held; all clips extracted, ids distinct; every stored row read
    back equals the computed one (hash per clip); all finite, no all-zero clip, all clips distinct; each clip's
    mean over the 256 patches equals its stored all-token mean within MEAN_TOLERANCE; weights unchanged.
    """
    table = load_joined(VARIABLE)
    ids, stored = table["id"], table["activations"]
    shape = (len(ids), len(PATCH_SITES), PATCHES, HIDDEN)
    needed = int(np.prod(shape)) * 4 + MARGIN_BYTES
    PATCH_DIR.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(PATCH_DIR).free
    if free < needed:
        return {"criteria": {"enough_free_disk": False}, "free_bytes": free, "needed_bytes": needed, "passed": False}

    set_seeds()
    model, _ = load_model(DEVICE)
    final = PATCH_DIR / f"{VARIABLE}.npy"
    partial = PATCH_DIR / f"{VARIABLE}.partial.npy"
    ids_path = PATCH_DIR / f"{VARIABLE}_ids.npy"

    array = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=shape)
    row_hashes, mismatch, seconds = [], [], []
    start = time.perf_counter()
    for i, path in enumerate(clip_paths(ids)):
        t = time.perf_counter()
        x = preprocess_clip(load_clip(path)).unsqueeze(0).to(DEVICE)
        row = patch_activations(model, x)[0].numpy()
        array[i] = row
        row_hashes.append(array_sha256(row))
        mismatch.append(mean_mismatch(row, stored[i]))
        seconds.append(time.perf_counter() - t)
        if (i + 1) % PROGRESS_EVERY == 0 or i + 1 == len(ids):
            print(f"{VARIABLE}: {i + 1}/{len(ids)} clips, {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    array.flush()
    del array
    os.replace(partial, final)
    np.save(ids_path, ids)

    saved = np.load(final, mmap_mode="r")
    reread, finite, nonzero = [], True, True
    for i in range(len(saved)):  # one pass over the 39 GB file
        row = np.asarray(saved[i])
        reread.append(array_sha256(row))
        finite = finite and bool(np.isfinite(row).all())
        nonzero = nonzero and bool(row.any())
    del saved
    fingerprint = weights_fingerprint(model)

    criteria = {
        "enough_free_disk": True,
        "all_clips_extracted": len(ids) == EXPECTED_CLIPS and len(set(ids.tolist())) == len(ids),
        "stored_equals_computed": reread == row_hashes,
        "all_finite": finite,
        "no_all_zero_clip": nonzero,
        "clips_distinct": len(set(row_hashes)) == len(row_hashes),
        "patch_mean_equals_all_token_mean": max(mismatch) <= MEAN_TOLERANCE,
        "weights_unchanged": fingerprint == REFERENCE_FINGERPRINT,
    }
    return {
        "criteria": criteria,
        "device": DEVICE,
        "batch_size": 1,
        "sites": list(PATCH_SITES),
        "free_bytes_before": free,
        "max_mean_mismatch": max(mismatch),
        "minutes_total": round((time.perf_counter() - start) / 60, 1),
        "seconds_per_clip_median": round(float(np.median(seconds)), 3),
        "seconds_per_clip_max": round(float(np.max(seconds)), 3),
        "artifact": {
            "path": str(final.relative_to(REPO)), "shape": list(shape), "dtype": "float32",
            "bytes": final.stat().st_size, "sha256": file_sha256(final),
        },
        "ids": {"path": str(ids_path.relative_to(REPO)), "sha256": file_sha256(ids_path)},
        "passed": all(criteria.values()),
    }


def check_verify() -> dict:
    """The stored per-patch array is intact, aligned with the joined table, and reproducible live.

    Reads the array and its ids through their recorded hashes. Passes if: shape (1500, 25, 256, 1024) float32;
    ids equal the joined table's ids in order; SPOT_CLIPS clips drawn without replacement by a generator seeded
    with SEED, re-extracted live, equal their stored rows bit for bit; weights unchanged at the end.
    """
    table = load_joined(VARIABLE)
    array = np.load(verified_artifact(OUT, "extract_direction"), mmap_mode="r")
    ids = np.load(verified_artifact(OUT, "extract_direction", "ids"))
    rng = np.random.default_rng(SEED)
    set_seeds()
    model, _ = load_model(DEVICE)
    paths = clip_paths(ids)
    picks = np.sort(rng.choice(len(ids), size=SPOT_CLIPS, replace=False)).tolist()
    mismatched = []
    for i in picks:
        x = preprocess_clip(load_clip(paths[i])).unsqueeze(0).to(DEVICE)
        if not np.array_equal(patch_activations(model, x)[0].numpy(), array[i]):
            mismatched.append(int(ids[i]))
    criteria = {
        "shape_and_dtype": array.shape == (EXPECTED_CLIPS, len(PATCH_SITES), PATCHES, HIDDEN)
        and array.dtype == np.float32,
        "ids_match_joined_table": bool(np.array_equal(ids, table["id"])),
        "spot_clips_bit_identical": not mismatched,
        "weights_unchanged": weights_fingerprint(model) == REFERENCE_FINGERPRINT,
    }
    return {
        "criteria": criteria,
        "device": DEVICE,
        "seed": SEED,
        "spot_clip_ids": [int(ids[i]) for i in picks],
        "mismatched_clip_ids": mismatched,
        "passed": all(criteria.values()),
    }


CHECKS = {
    "extract_direction": check_extract_direction,
    "verify": check_verify,
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