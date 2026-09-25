"""Checks for extracting pooled encoder activations for every clip.

Run one check at a time from the repo root, with .venv active:
    python scripts/check_extraction.py <check>

Each check prints its result and stores it under its own key in results/extraction/checks.json.
"""
import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from scipy.spatial.distance import pdist

from vjepa_physics.activations import capture_encoder
from vjepa_physics.data import DATASETS, read_manifest, resolve
from vjepa_physics.evidence import file_sha256, save_result, verified_artifact
from vjepa_physics.extraction import SITES, TIME_STEPS, TOKENS_PER_STEP, all_token_mean, pooled_activations
from vjepa_physics.model import load_model, weights_fingerprint
from vjepa_physics.preprocess import preprocess_clip
from vjepa_physics.reproducibility import SEED, set_seeds
from vjepa_physics.splits import read_splits
from vjepa_physics.video import load_clip

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = REPO / "results/extraction/checks.json"
ACTIVATIONS = REPO / "artifacts/activations"  # regenerable, git-ignored
DEVICE = "mps"
HIDDEN = 1024

# In-memory weights fingerprint right after a clean load (results/model/checks.json, key "load").
REFERENCE_FINGERPRINT = "c865f524c1376e4452943b208d7d50ba588be9490604f235a3d9c9dc80804ede"

TEST_CLIP = ("speed", 1000)  # speed 2.69 m/s, theta 230.625 deg
EXPECTED_CLIPS = {"direction": 1500, "speed": 1536, "acceleration": 1536}  # DATA.md
MIN_FREE_BYTES = 10 * 10**9  # free disk required before extracting: 10 GB
# fp32 pooling vs a float64 pool of the same tokens, relative: generous for fp32 rounding of 256-token means.
POOL_TOLERANCE = 1e-6
PROGRESS_EVERY = 100

SPLITS_CHECKS = REPO / "results/splits/checks.json"  # the split file's hash is recorded under key "build"
SPOT_CLIPS = 16  # clips per dataset re-extracted live and compared bit for bit (also the reproducibility gate)
DISTANCE_SITES = ("embedding", "block_23")  # sites for the nearest-pair diagnostic


def sci(value: float) -> float:
    """Round to 4 significant digits, for readable JSON."""
    return float(f"{value:.4g}")


def rel_error(a: torch.Tensor, reference: torch.Tensor) -> float:
    return ((a - reference).norm() / reference.norm()).item()


def array_sha256(a: np.ndarray) -> str:
    """SHA-256 of an array's bytes in C order."""
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def clip_input(dataset: str, clip_id: int) -> torch.Tensor:
    """(1, 16, 3, 256, 256) encoder input for one clip, on the CPU."""
    root = DATA / dataset
    row = next(r for r in read_manifest(root) if r["id"] == clip_id)
    return preprocess_clip(load_clip(resolve(root, row["video"]))).unsqueeze(0)


def check_pipeline() -> dict:
    """The extraction code on the test clip: right shape, deterministic, numerically right, model unchanged.

    Passes if: pooled_activations gives (1, 26, 8, 1024) float32; a repeat is bit-exact; it is within
    POOL_TOLERANCE (relative) of a float64 per-time-step pool of the same captured tokens, and so is the
    derived all-token mean vs a float64 mean over all tokens; final_norm differs from block_23; the weights
    fingerprint equals the clean-load reference. Records the SHA-256 of the pooled array, which the
    extraction of the test clip's dataset must reproduce.
    """
    set_seeds()
    model, _ = load_model(DEVICE)
    x = clip_input(*TEST_CLIP).to(DEVICE)
    pooled = pooled_activations(model, x)
    repeat = pooled_activations(model, x)

    with capture_encoder(model) as acts, torch.inference_mode():
        out = model(pixel_values_videos=x, skip_predictor=True)
        # Move first, then widen: a combined device + dtype move from MPS once gave silent zeros.
        tokens = {s: t.to("cpu").to(torch.float64) for s, t in (dict(acts) | {"final_norm": out.last_hidden_state}).items()}
    reference = torch.stack([tokens[s].reshape(1, TIME_STEPS, TOKENS_PER_STEP, HIDDEN).mean(dim=2) for s in SITES], dim=1)
    full = torch.stack([tokens[s].mean(dim=1) for s in SITES], dim=1)
    pool_error = rel_error(pooled.double(), reference)
    mean_error = rel_error(all_token_mean(pooled).double(), full)
    fingerprint = weights_fingerprint(model)

    criteria = {
        "shape_and_dtype": tuple(pooled.shape) == (1, len(SITES), TIME_STEPS, HIDDEN) and pooled.dtype == torch.float32,
        "repeat_bit_exact": bool(torch.equal(pooled, repeat)),
        "pool_matches_float64": pool_error <= POOL_TOLERANCE,
        "derived_mean_matches_float64": mean_error <= POOL_TOLERANCE,
        "final_norm_differs_from_block_23": not torch.equal(pooled[0, -1], pooled[0, -2]),
        "weights_unchanged": fingerprint == REFERENCE_FINGERPRINT,
    }
    return {
        "clip": f"{TEST_CLIP[0]}/{TEST_CLIP[1]}",
        "device": DEVICE,
        "sites": list(SITES),
        "criteria": criteria,
        "tolerance": POOL_TOLERANCE,
        "pool_rel_error_vs_float64": sci(pool_error),
        "derived_mean_rel_error_vs_float64": sci(mean_error),
        "pooled_sha256": array_sha256(pooled[0].numpy()),
        "passed": all(criteria.values()),
    }


def extract(dataset: str) -> dict:
    """Pooled activations of every clip of one dataset, in manifest order, saved as one array.

    Refuses to start unless the test-clip `pipeline` check has passed and at least 10 GB of disk are free.
    Batch size 1, MPS, fp32. Writes artifacts/activations/<dataset>.npy, shape (clips, 26, 8, 1024) float32
    (first to <dataset>.partial.npy, renamed only after the last clip, so a crash never leaves a complete-
    looking file), and <dataset>_ids.npy (clip ids in the same order). Passes if: the disk check held;
    every manifest clip was extracted (count as DATA.md, ids distinct); every stored row, read back from
    disk, equals the array computed in memory (hash per clip); every value is finite and no clip's array
    is all zero; no two clips have identical activations; the weights fingerprint is unchanged; and, for
    the test clip's dataset, the test clip's row equals the one `pipeline` recorded bit for bit.
    """
    pipeline = json.loads(OUT.read_text()).get("pipeline", {}).get("result", {}) if OUT.exists() else {}
    if pipeline.get("passed") is not True:
        raise RuntimeError("run and pass `pipeline` first")

    ACTIVATIONS.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(ACTIVATIONS).free
    if free < MIN_FREE_BYTES:
        return {"criteria": {"free_disk_at_least_10_gb": False}, "free_bytes": free, "passed": False}

    set_seeds()
    model, _ = load_model(DEVICE)
    root = DATA / dataset
    rows = read_manifest(root)
    final = ACTIVATIONS / f"{dataset}.npy"
    partial = ACTIVATIONS / f"{dataset}.partial.npy"
    ids_path = ACTIVATIONS / f"{dataset}_ids.npy"
    shape = (len(rows), len(SITES), TIME_STEPS, HIDDEN)

    array = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=shape)
    row_hashes, seconds = [], []
    start = time.perf_counter()
    for i, row in enumerate(rows):
        t = time.perf_counter()
        x = preprocess_clip(load_clip(resolve(root, row["video"]))).unsqueeze(0).to(DEVICE)
        pooled = pooled_activations(model, x)[0].numpy()
        array[i] = pooled
        row_hashes.append(array_sha256(pooled))
        seconds.append(time.perf_counter() - t)
        if (i + 1) % PROGRESS_EVERY == 0 or i + 1 == len(rows):
            print(f"{dataset}: {i + 1}/{len(rows)} clips, {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    array.flush()
    del array
    os.replace(partial, final)
    ids = np.array([row["id"] for row in rows], dtype=np.int64)
    np.save(ids_path, ids)

    stored = np.load(final, mmap_mode="r")
    reread = [array_sha256(stored[i]) for i in range(len(stored))]
    finite = all(bool(np.isfinite(stored[i]).all()) for i in range(len(stored)))
    no_zero_clip = all(bool(stored[i].any()) for i in range(len(stored)))
    del stored
    fingerprint = weights_fingerprint(model)

    criteria = {
        "free_disk_at_least_10_gb": True,
        "all_clips_extracted": len(rows) == EXPECTED_CLIPS[dataset] and len(set(ids.tolist())) == len(ids),
        "stored_equals_computed": reread == row_hashes,
        "all_finite": finite,
        "no_all_zero_clip": no_zero_clip,
        "clips_distinct": len(set(row_hashes)) == len(row_hashes),
        "weights_unchanged": fingerprint == REFERENCE_FINGERPRINT,
    }
    if dataset == TEST_CLIP[0]:
        criteria["test_clip_matches_pipeline"] = row_hashes[ids.tolist().index(TEST_CLIP[1])] == pipeline["pooled_sha256"]

    return {
        "criteria": criteria,
        "device": DEVICE,
        "batch_size": 1,
        "sites": list(SITES),
        "free_bytes_before": free,
        "minutes_total": sci((time.perf_counter() - start) / 60),
        "seconds_per_clip_median": sci(float(np.median(seconds))),
        "seconds_per_clip_max": sci(float(np.max(seconds))),
        "artifact": {
            "path": str(final.relative_to(REPO)), "shape": list(shape), "dtype": "float32",
            "bytes": final.stat().st_size, "sha256": file_sha256(final),
        },
        "ids": {"path": str(ids_path.relative_to(REPO)), "sha256": file_sha256(ids_path)},
        "passed": all(criteria.values()),
    }


def check_verify() -> dict:
    """The stored activations are intact, aligned with the splits, and reproducible live.

    Reads every activation array, id file and the split file through their recorded hashes. Passes if, per
    dataset: the array is (clips, 26, 8, 1024) float32 with DATA.md's clip count; its ids equal the split file's
    ids for that dataset, in order; and SPOT_CLIPS clips drawn without replacement by one generator seeded with
    SEED (datasets in DATASETS order), re-extracted live, equal their stored rows bit for bit. And the weights
    fingerprint is unchanged at the end. Diagnostics per dataset: per-site mean and std of the all-token means;
    at DISTANCE_SITES, the smallest Euclidean distance between two clips' all-token means, absolute and relative
    to the median distance (a near-duplicate guard).
    """
    splits = read_splits(verified_artifact(SPLITS_CHECKS, "build"))
    rng = np.random.default_rng(SEED)
    set_seeds()
    model, _ = load_model(DEVICE)

    per_dataset, dataset_criteria = {}, {}
    for dataset in DATASETS:
        key = f"extract_{dataset}"
        array = np.load(verified_artifact(OUT, key), mmap_mode="r")
        ids = np.load(verified_artifact(OUT, key, "ids"))
        split_ids = [r["id"] for r in splits if r["dataset"] == dataset]

        root = DATA / dataset
        paths = {row["id"]: resolve(root, row["video"]) for row in read_manifest(root)}
        picks = np.sort(rng.choice(len(ids), size=SPOT_CLIPS, replace=False)).tolist()
        mismatched = []
        for i in picks:
            x = preprocess_clip(load_clip(paths[int(ids[i])])).unsqueeze(0).to(DEVICE)
            if not np.array_equal(pooled_activations(model, x)[0].numpy(), array[i]):
                mismatched.append(int(ids[i]))

        means = array.mean(axis=2, dtype=np.float64)  # (clips, 26, 1024): all-token mean per site
        nearest = {}
        for site in DISTANCE_SITES:
            distances = pdist(means[:, SITES.index(site)])
            nearest[site] = {
                "min": sci(float(distances.min())),
                "min_over_median": sci(float(distances.min() / np.median(distances))),
            }

        criteria = {
            "shape_and_dtype": array.shape == (EXPECTED_CLIPS[dataset], len(SITES), TIME_STEPS, HIDDEN)
            and array.dtype == np.float32,
            "ids_match_splits": ids.tolist() == split_ids,
            "spot_clips_bit_identical": not mismatched,
        }
        dataset_criteria[dataset] = criteria
        per_dataset[dataset] = {
            "criteria": criteria,
            "spot_clip_ids": [int(ids[i]) for i in picks],
            "mismatched_clip_ids": mismatched,
            "site_mean": [sci(float(v)) for v in means.mean(axis=(0, 2))],
            "site_std": [sci(float(v)) for v in means.std(axis=(0, 2))],
            "nearest_pair_distance": nearest,
        }
        del array, means

    fingerprint_ok = weights_fingerprint(model) == REFERENCE_FINGERPRINT
    return {
        "criteria": {"weights_unchanged": fingerprint_ok},
        "device": DEVICE,
        "seed": SEED,
        "spot_clips_per_dataset": SPOT_CLIPS,
        "sites": list(SITES),
        "per_dataset": per_dataset,
        "passed": fingerprint_ok and all(all(c.values()) for c in dataset_criteria.values()),
    }


CHECKS = {
    "pipeline": check_pipeline,
    "extract_direction": lambda: extract("direction"),
    "extract_speed": lambda: extract("speed"),
    "extract_acceleration": lambda: extract("acceleration"),
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
