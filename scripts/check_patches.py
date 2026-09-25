"""Checks for the per-patch activations of the direction clips (local probes and spatial generalization).

Run one check at a time from the repo root, with .venv active:
    python scripts/check_patches.py <check>

Each check prints its result and stores it under its own key in results/patches/checks.json.
"""
import argparse
import hashlib
from collections import Counter
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from vjepa_physics.data import read_manifest, resolve
from vjepa_physics.curves import LATE_RISE_FRACTION, TRANSITION_FRACTION, rise_index, transition_points
from vjepa_physics.evidence import code_changes, file_sha256, repo_root, save_result, verified_artifact
from vjepa_physics.extraction import PATCHES, PATCH_SITES, patch_activations
from vjepa_physics.geometry import DISK_RADIUS_PX, PATCH_GRID, PATCH_PX, distance_to_patches, pixel_to_world
from vjepa_physics.joined import load_joined
from vjepa_physics.metrics import (
    angles_from_sincos, bootstrap_indices, circular_errors, percentile_interval, r2, resampled_r2,
)
from vjepa_physics.model import load_model, weights_fingerprint
from vjepa_physics.preprocess import preprocess_clip
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from vjepa_physics.probes import (
    ALPHAS, alpha_verdict, clip_folds, fit_probe, grouped_cv_ridge, probe_scores, probe_targets,
)
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

EVAL_ROLES = ("val_seen", "val_unseen")  # scored here; test is scored once, after the findings are recorded
EXPECTED_FIT = 813  # direction train clips (split decision)
PATCH_PROBES = PATCH_DIR / "patch_probes.npz"  # fitted probes + validation predictions (regenerable, git-ignored)
PROBES_CHECKS = REPO / "results/probes/checks.json"  # key "layer_curves": mean-pooled scores, printed for comparison

MIN_CATEGORY_CLIPS = 20  # a patch is reported for a category only with at least this many val_seen clips in it
OFF_PATH_MARGIN_PX = PATCH_PX  # off-path: the disk centre never came within radius + one patch of the patch
BREAKDOWN = PATCH_DIR / "patch_breakdown.npz"
N_RESAMPLES = 10_000  # clip resamples (bootstrap settings decision)
LEVEL = 0.95

HALF_NAMES = ("left", "right", "top", "bottom")
OPPOSITE = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
N_FOLDS = 5  # clip-grouped CV folds for the shared probe's alpha
SAMPLES_PER_CLIP = 128  # patches in one half
SUMS_TOLERANCE = 1e-10  # R² from per-clip sums vs directly from the samples
SPATIAL = PATCH_DIR / "spatial_generalization.npz"

POSITION_THRESHOLD = 0.5  # linear position-only val_seen R² at or above this goes to the planning chat
SHOWN_OFF_PATH = (0, 1, 6, 13, 24)  # off-path curve indices printed next to the baseline

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


def check_patch_probes() -> dict:
    """One ridge probe per patch position and layer index (25 x 256), fit on train, scored on validation clips.

    Features: each clip's time-averaged vector at that patch and index (the per-patch array, read through its
    recorded hash, one index at a time). Probes as for the mean-pooled curves (z-score on train, RidgeCV
    leave-one-out, alpha rule). Writes artifacts/patches/patch_probes.npz: every probe's scaler mean and scale,
    weights and intercept, alpha, and the validation predictions (NaN on every other row), so later analyses and
    the one-time test scoring need no refit. Curve per index: mean and median of the per-patch val_seen R²; the
    transition rule is applied to the mean curve (reported, not selected on). Passes if: the code is committed;
    ids align with the joined table; every probe saw exactly the 813 train clips; no alpha verdict is "failure";
    all scores finite; only validation rows predicted; one seeded patch per index refits bit-identically; the
    saved file equals what was computed.
    """
    committed = not code_changes(repo_root())
    table = load_joined(VARIABLE)
    roles, labels = table["role"], table["label"]
    y = probe_targets(VARIABLE, labels)
    validation = np.isin(roles, EVAL_ROLES)
    patches = np.load(verified_artifact(OUT, "extract_direction"), mmap_mode="r")
    ids = np.load(verified_artifact(OUT, "extract_direction", "ids"))
    n_index, n_patch = len(PATCH_SITES), PATCHES

    grid = (n_index, n_patch)
    alpha = np.zeros(grid)
    verdict = np.empty(grid, dtype="U9")
    n_fit = np.zeros(grid, dtype=np.int64)
    scores = {f"{role}_{m}": np.zeros(grid) for role in EVAL_ROLES for m in ("r2", "circular_mae")}
    scaler_mean = np.zeros((*grid, HIDDEN))
    scaler_scale = np.zeros((*grid, HIDDEN))
    coef = np.zeros((*grid, 2, HIDDEN))
    intercept = np.zeros((*grid, 2))
    predictions = np.full((len(y), *grid, 2), np.nan)
    spot = np.random.default_rng(SEED).integers(n_patch, size=n_index)
    refit_ok = []

    start = time.perf_counter()
    for i in range(n_index):
        block = np.asarray(patches[:, i])  # (clips, 256, 1024) float32: one index, read once
        for p in range(n_patch):
            x = block[:, p].astype(np.float64)
            probe = fit_probe(x, y, roles)
            predictions[validation, i, p] = probe.predict(x[validation])
            for role in EVAL_ROLES:
                s = probe_scores(VARIABLE, labels[roles == role], predictions[roles == role, i, p])
                scores[f"{role}_r2"][i, p] = s["r2"]
                scores[f"{role}_circular_mae"][i, p] = s["circular_mae"]
            alpha[i, p] = probe.alpha
            verdict[i, p] = alpha_verdict(probe.alpha_edge, scores["val_seen_r2"][i, p])
            n_fit[i, p] = probe.n_fit
            scaler_mean[i, p], scaler_scale[i, p] = probe.scaler.mean_, probe.scaler.scale_
            coef[i, p], intercept[i, p] = probe.ridge.coef_, probe.ridge.intercept_
            if p == spot[i]:
                again = fit_probe(x, y, roles)
                refit_ok.append(again.alpha == probe.alpha and np.array_equal(again.ridge.coef_, probe.ridge.coef_))
        del block
        print(f"index {i:2d} ({PATCH_SITES[i]}): mean val_seen R2 {scores['val_seen_r2'][i].mean():.3f},"
              f" {(time.perf_counter() - start) / 60:.1f} min", flush=True)

    arrays = {
        "sites": np.array(PATCH_SITES), "ids": table["id"], "roles": roles, "alpha": alpha, "alpha_verdict": verdict,
        "n_fit": n_fit, "scaler_mean": scaler_mean, "scaler_scale": scaler_scale, "coef": coef,
        "intercept": intercept, "validation_predictions": predictions, **scores,
    }
    np.savez_compressed(PATCH_PROBES, **arrays)
    with np.load(PATCH_PROBES) as saved:
        saved_ok = set(saved.files) == set(arrays) and all(
            np.array_equal(saved[key], value, equal_nan=value.dtype.kind == "f") for key, value in arrays.items()
        )

    mean_curve = scores["val_seen_r2"].mean(axis=1)
    curve = [
        {"index": i, "site": PATCH_SITES[i],
         "val_seen_r2_mean": float(mean_curve[i]), "val_seen_r2_median": float(np.median(scores["val_seen_r2"][i])),
         "val_seen_r2_max": float(scores["val_seen_r2"][i].max()),
         "val_unseen_r2_mean": float(scores["val_unseen_r2"][i].mean()),
         "val_seen_circular_mae_mean": float(scores["val_seen_circular_mae"][i].mean()),
         "alpha_verdicts": dict(Counter(verdict[i].tolist()))}
        for i in range(n_index)
    ]
    criteria = {
        "code_committed": committed,
        "ids_aligned": bool(np.array_equal(ids, table["id"])),
        "n_fit_equals_train_count": bool((n_fit == EXPECTED_FIT).all()),
        "no_alpha_failure": bool((verdict != "failure").all()),
        "scores_finite": all(bool(np.isfinite(v).all()) for v in scores.values()),
        "only_validation_rows_predicted": bool(np.isfinite(predictions[validation]).all()
                                               and np.isnan(predictions[~validation]).all()),
        "spot_refit_identical": len(refit_ok) == n_index and all(refit_ok),
        "saved_equals_computed": saved_ok,
    }
    return {
        "criteria": criteria,
        "curve": curve,
        "transition_on_mean_curve": transition_points(mean_curve),
        "minutes_total": round((time.perf_counter() - start) / 60, 1),
        "artifact": {"path": str(PATCH_PROBES.relative_to(REPO)), "sha256": file_sha256(PATCH_PROBES)},
        "passed": all(criteria.values()),
    }

TIE_TOLERANCE = 1e-12  # relative spread of the mean leave-one-out error across the alpha grid that counts as a tie


def check_patch_alpha_diagnostic() -> dict:
    """Diagnostic of patch_probes' alpha failures: are they ties on constant features?

    For every probe whose alpha verdict was "failure", refits the train scaler and RidgeCV on exactly the same
    train features with store_cv_results, and records the mean leave-one-out error per alpha. Rule fixed before
    running: the failures are explained (the grid edge carries no information) only if, for every failing probe,
    the mean leave-one-out error is identical across the whole grid within TIE_TOLERANCE relative (sklearn then
    keeps the first, smallest alpha). Also records per probe: position, edge, distinct train feature vectors,
    max train feature std, and val_seen clips whose vector differs from the train one. Diagnostic: no pass/fail
    of the data, only whether the explanation holds.
    """
    table = load_joined(VARIABLE)
    roles, labels = table["role"], table["label"]
    y = probe_targets(VARIABLE, labels)
    train, seen = roles == "train", roles == "val_seen"
    patches = np.load(verified_artifact(OUT, "extract_direction"), mmap_mode="r")
    with np.load(verified_artifact(OUT, "patch_probes")) as saved:
        verdict, alpha = saved["alpha_verdict"], saved["alpha"]
    failing = [(int(i), int(p)) for i, p in zip(*np.nonzero(verdict == "failure"))]

    probes, ties = [], []
    for i, p in failing:
        x = np.asarray(patches[:, i, p], dtype=np.float64)
        xs = StandardScaler().fit(x[train]).transform(x[train])
        ridge = RidgeCV(alphas=ALPHAS, store_cv_results=True).fit(xs, y[train])
        loo = ridge.cv_results_.mean(axis=tuple(range(ridge.cv_results_.ndim - 1)))  # mean error per alpha
        spread = float((loo.max() - loo.min()) / loo.mean())
        reference = x[train][0]
        ties.append(spread <= TIE_TOLERANCE)
        probes.append({
            "index": i, "site": PATCH_SITES[i], "patch": p, "row": p // 16, "col": p % 16,
            "alpha": float(alpha[i, p]),
            "edge": "lower" if alpha[i, p] == ALPHAS[0] else "upper" if alpha[i, p] == ALPHAS[-1] else None,
            "loo_relative_spread_over_grid": spread,
            "distinct_train_vectors": int(len(np.unique(x[train], axis=0))),
            "max_train_feature_std": float(x[train].std(axis=0).max()),
            "val_seen_clips_differing_from_train": int((~(x[seen] == reference).all(axis=1)).sum()),
        })
    return {
        "diagnostic": "patch_probes alpha failures",
        "rule": "explained only if every failing probe's mean LOO error is identical across the grid within 1e-12 relative",
        "failing_probes": len(failing),
        "explanation_holds": bool(failing) and all(ties),
        "probes": probes,
    }

def path_categories(table: dict) -> tuple[np.ndarray, np.ndarray]:
    """(clips, 256) on-path and off-path masks from the tracked disk, over frames with disk pixels.

    On-path: the disk overlapped the patch in at least one frame (nearest centre-to-square distance <= radius).
    Off-path: the disk centre never came within radius + one patch. Clips in between are in neither.
    """
    centre, present = table["centre"], table["area"] > 0
    distance = distance_to_patches(centre[..., 0], centre[..., 1])  # (clips, 16, 256); NaN where no disk
    nearest = np.where(present[..., None], distance, np.inf).min(axis=1)
    return nearest <= DISK_RADIUS_PX, nearest > DISK_RADIUS_PX + OFF_PATH_MARGIN_PX


def check_patch_breakdown() -> dict:
    """Per-patch val_seen scores split by on-path / off-path clips (no new fits), from the saved probe predictions.

    Writes artifacts/patches/patch_breakdown.npz (masks, val_seen counts per patch, R² and circular MAE per index,
    patch and category; NaN where fewer than MIN_CATEGORY_CLIPS clips). Summary per index: mean and median over
    the reported patches and their number; the transition rule on the off-path mean curve. Passes if: the code is
    committed; the saved predictions align with the joined table; on and off never overlap; every clip is on-path
    somewhere; both categories have reported patches; reported scores are finite; the saved file equals what was
    computed.
    """
    committed = not code_changes(repo_root())
    table = load_joined(VARIABLE)
    roles, labels = table["role"], table["label"]
    seen = roles == "val_seen"
    with np.load(verified_artifact(OUT, "patch_probes")) as saved:
        predictions, ids = saved["validation_predictions"], saved["ids"]
    on, off = path_categories(table)
    masks = {"on_path": on, "off_path": off}
    grid = (len(PATCH_SITES), PATCHES)

    arrays: dict[str, np.ndarray] = {"on_path_mask": on, "off_path_mask": off}
    for category, mask in masks.items():
        counts = (seen[:, None] & mask).sum(axis=0)
        r2_grid, mae_grid = np.full(grid, np.nan), np.full(grid, np.nan)
        for p in np.flatnonzero(counts >= MIN_CATEGORY_CLIPS):
            rows = seen & mask[:, p]
            for i in range(len(PATCH_SITES)):
                s = probe_scores(VARIABLE, labels[rows], predictions[rows, i, p])
                r2_grid[i, p], mae_grid[i, p] = s["r2"], s["circular_mae"]
        arrays |= {f"{category}_val_seen_counts": counts, f"{category}_r2": r2_grid,
                   f"{category}_circular_mae": mae_grid}

    summary = []
    for i in range(len(PATCH_SITES)):
        row = {"index": i, "site": PATCH_SITES[i]}
        for category in masks:
            values = arrays[f"{category}_r2"][i]
            reported = values[np.isfinite(values)]
            row[category] = {"patches": int(reported.size),
                             "r2_mean": float(reported.mean()) if reported.size else None,
                             "r2_median": float(np.median(reported)) if reported.size else None}
        summary.append(row)
    off_curve = np.array([r["off_path"]["r2_mean"] for r in summary], dtype=float)

    np.savez_compressed(BREAKDOWN, **arrays)
    with np.load(BREAKDOWN) as saved:
        saved_ok = set(saved.files) == set(arrays) and all(
            np.array_equal(saved[k], v, equal_nan=v.dtype.kind == "f") for k, v in arrays.items())
    reported_finite = all(bool(np.isfinite(arrays[f"{c}_circular_mae"][np.isfinite(arrays[f"{c}_r2"])]).all())
                          for c in masks)
    criteria = {
        "code_committed": committed,
        "ids_aligned": bool(np.array_equal(ids, table["id"])),
        "categories_disjoint": bool(not (on & off).any()),
        "every_clip_on_path_somewhere": bool(on.any(axis=1).all()),
        "both_categories_reported": all(summary[0][c]["patches"] > 0 for c in masks),
        "reported_scores_finite": reported_finite,
        "saved_equals_computed": saved_ok,
    }
    return {
        "criteria": criteria,
        "disk_radius_px": DISK_RADIUS_PX, "off_path_margin_px": OFF_PATH_MARGIN_PX,
        "min_category_clips": MIN_CATEGORY_CLIPS,
        "curve": summary,
        "transition_on_off_path_mean_curve":
            transition_points(off_curve) if np.isfinite(off_curve).all() else None,
        "artifact": {"path": str(BREAKDOWN.relative_to(REPO)), "sha256": file_sha256(BREAKDOWN)},
        "passed": all(criteria.values()),
    }


def check_patch_bootstrap() -> dict:
    """Headline intervals: mean per-patch val_seen R² per index, its transition, and its gap to the mean-pooled probe.

    Clip bootstrap (N_RESAMPLES, seed SEED, percentile LEVEL) over the val_seen clips, one index matrix shared by
    all 256 patches and the mean-pooled probe (so the gap is paired; same n and seed as the layer-curve bootstrap).
    Per resample, the mean over patches of per-patch R² gives a curve (indices 0-24); the transition rule is applied
    to every resampled curve. Passes if: the code is committed; both prediction files align with the joined table;
    every point estimate equals the saved score (per patch, and the mean-pooled one).
    """
    committed = not code_changes(repo_root())
    table = load_joined(VARIABLE)
    roles, labels = table["role"], table["label"]
    seen = roles == "val_seen"
    y = probe_targets(VARIABLE, labels)[seen]
    with np.load(verified_artifact(OUT, "patch_probes")) as saved:
        patch_pred, patch_ids, saved_r2 = saved["validation_predictions"][seen], saved["ids"], saved["val_seen_r2"]
    with np.load(verified_artifact(PROBES_CHECKS, "layer_curves")) as saved:
        pooled_pred = saved[f"{VARIABLE}_predictions"][seen][:, :len(PATCH_SITES)]
        pooled_ids = saved[f"{VARIABLE}_ids"]
    saved_pooled = json.loads(PROBES_CHECKS.read_text())["layer_curves"]["result"][VARIABLE]["sites"]

    idx = bootstrap_indices(int(seen.sum()), N_RESAMPLES, SEED)
    n_index = len(PATCH_SITES)
    point = np.zeros((n_index, PATCHES))
    patch_curves = np.zeros((N_RESAMPLES, n_index))
    pooled_point = np.zeros(n_index)
    pooled_curves = np.zeros((N_RESAMPLES, n_index))
    for i in range(n_index):
        for p in range(PATCHES):
            point[i, p] = r2(y, patch_pred[:, i, p])
            patch_curves[:, i] += resampled_r2(y, patch_pred[:, i, p], idx)
        patch_curves[:, i] /= PATCHES
        pooled_point[i] = r2(y, pooled_pred[:, i])
        pooled_curves[:, i] = resampled_r2(y, pooled_pred[:, i], idx)

    mean_point = point.mean(axis=1)
    curve = [
        {"index": i, "site": PATCH_SITES[i],
         "per_patch_mean_r2": {"point": float(mean_point[i]),
                               "ci": list(percentile_interval(patch_curves[:, i], LEVEL))},
         "gap_mean_pooled_minus_per_patch": {"point": float(pooled_point[i] - mean_point[i]),
                                             "ci": list(percentile_interval(pooled_curves[:, i] - patch_curves[:, i], LEVEL))}}
        for i in range(n_index)
    ]
    criteria = {
        "code_committed": committed,
        "inputs_aligned": bool(np.array_equal(patch_ids, table["id"]) and np.array_equal(pooled_ids, table["id"])),
        "per_patch_points_equal_saved": bool(np.array_equal(point, saved_r2)),
        "mean_pooled_points_equal_saved": all(
            pooled_point[i] == saved_pooled[PATCH_SITES[i]]["val_seen"]["r2"] for i in range(n_index)),
    }
    return {
        "criteria": criteria, "n_resamples": N_RESAMPLES, "level": LEVEL, "seed": SEED,
        "curve": curve,
        "transition_headline": transition_points(mean_point),
        "bootstrap_transition_index_counts":
            np.bincount(rise_index(patch_curves, TRANSITION_FRACTION), minlength=n_index).tolist(),
        "bootstrap_rise_80_index_counts":
            np.bincount(rise_index(patch_curves, LATE_RISE_FRACTION), minlength=n_index).tolist(),
        "passed": all(criteria.values()),
    }


def half_patches(half: str) -> np.ndarray:
    """The 128 patch indices (row * 16 + col) of one half of the frame."""
    row, col = np.divmod(np.arange(PATCHES), PATCH_GRID)
    mask = {"left": col < 8, "right": col >= 8, "top": row < 8, "bottom": row >= 8}[half]
    return np.flatnonzero(mask)


def half_samples(block: np.ndarray, clips: np.ndarray, patches: np.ndarray) -> np.ndarray:
    """(clips * 128, 1024) float64 samples from one index's (clips, 256, 1024) block, clip-major."""
    return block[np.ix_(clips, patches)].reshape(-1, block.shape[-1]).astype(np.float64)


def clip_sums(labels: np.ndarray, y: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per clip: squared residual sums per output (clips, 2) and circular-error sums in degrees (clips,)."""
    n = len(labels)
    residual = (np.repeat(y, SAMPLES_PER_CLIP, axis=0) - pred) ** 2
    errors = circular_errors(np.repeat(labels, SAMPLES_PER_CLIP), angles_from_sincos(pred))
    return residual.reshape(n, SAMPLES_PER_CLIP, -1).sum(axis=1), errors.reshape(n, SAMPLES_PER_CLIP).sum(axis=1)


def pooled_scores(y: np.ndarray, sse: np.ndarray, error_sum: np.ndarray,
                  idx: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """R² (mean over sin, cos) and circular MAE over all samples of the chosen clips, from per-clip sums.

    idx (B, n) resamples clips (each brings its 128 samples); None = every clip once. Returns (B,) arrays.
    """
    idx = np.arange(len(y))[None] if idx is None else idx
    count = idx.shape[1] * SAMPLES_PER_CLIP
    yb = y[idx]  # (B, n, 2); a clip's target is the same for all its samples
    ss_tot = (yb**2).sum(axis=1) * SAMPLES_PER_CLIP - (yb.sum(axis=1) * SAMPLES_PER_CLIP) ** 2 / count
    r2_b = (1 - sse[idx].sum(axis=1) / ss_tot).mean(axis=-1)
    return r2_b, error_sum[idx].sum(axis=1) / count


def check_spatial_generalization() -> dict:
    """One shared probe per layer index and frame half, scored on the same and the opposite half of val_seen clips.

    Samples = (clip, patch) pairs of one half (128 patches); train samples z-scored with the half's train scaler;
    alpha by 5-fold CV grouped by clip (seeded). Scores pool all samples: R² on (sin, cos), circular MAE; gap =
    same - across. Headline, averaged over the 4 halves, with a clip bootstrap (all 128 samples of a clip move
    together; one index matrix, so same, across and the gap are paired). val_unseen: point scores. Writes
    artifacts/patches/spatial_generalization.npz (fitted probes, CV curves, per-clip sums). Passes if: the code is
    committed; ids align; no clip in two folds; every fit used exactly the train clips' samples; scored clips are
    val_seen only; no alpha failure (alpha rule on the same-half val_seen R²); R² from per-clip sums equals R²
    from the samples within 1e-10; scores finite; the saved file equals what was computed.
    """
    committed = not code_changes(repo_root())
    table = load_joined(VARIABLE)
    roles, labels = table["role"], table["label"]
    y = probe_targets(VARIABLE, labels)
    train = np.flatnonzero(roles == "train")
    seen = np.flatnonzero(roles == "val_seen")
    unseen = np.flatnonzero(roles == "val_unseen")
    patches = np.load(verified_artifact(OUT, "extract_direction"), mmap_mode="r")
    ids = np.load(verified_artifact(OUT, "extract_direction", "ids"))
    halves = {h: half_patches(h) for h in HALF_NAMES}

    groups = np.repeat(table["id"][train], SAMPLES_PER_CLIP)
    folds = clip_folds(groups, N_FOLDS, SEED)
    first_fold = folds[::SAMPLES_PER_CLIP]  # fold of each train clip's first sample
    folds_grouped = bool((folds.reshape(-1, SAMPLES_PER_CLIP) == first_fold[:, None]).all())
    y_train = np.repeat(y[train], SAMPLES_PER_CLIP, axis=0)

    n_index, n_half = len(PATCH_SITES), len(HALF_NAMES)
    scaler_mean = np.zeros((n_index, n_half, HIDDEN))
    scaler_scale = np.zeros((n_index, n_half, HIDDEN))
    coef = np.zeros((n_index, n_half, 2, HIDDEN))
    intercept = np.zeros((n_index, n_half, 2))
    alpha = np.zeros((n_index, n_half))
    cv_mse = np.zeros((n_index, n_half, len(ALPHAS)))
    verdict = np.empty((n_index, n_half), dtype="U9")
    n_fit = np.zeros((n_index, n_half), dtype=np.int64)
    sse = np.zeros((n_index, n_half, 2, len(seen), 2))  # axis 2: 0 = same half, 1 = opposite half
    error_sum = np.zeros((n_index, n_half, 2, len(seen)))
    unseen_scores = np.zeros((n_index, n_half, 2, 2))  # last axis: R², circular MAE
    sums_ok = []

    start = time.perf_counter()
    for i in range(n_index):
        block = np.asarray(patches[:, i])  # (clips, 256, 1024) float32, one index read once
        for h, half in enumerate(HALF_NAMES):
            x = half_samples(block, train, halves[half])
            scaler = StandardScaler().fit(x)
            x = scaler.transform(x, copy=False)
            fit = grouped_cv_ridge(x, y_train, folds)
            del x
            scaler_mean[i, h], scaler_scale[i, h] = scaler.mean_, scaler.scale_
            coef[i, h], intercept[i, h], alpha[i, h] = fit.coef, fit.intercept, fit.alpha
            cv_mse[i, h], n_fit[i, h] = fit.cv_mse, fit.n_fit
            for w, target in enumerate((half, OPPOSITE[half])):
                pred = fit.predict(scaler.transform(half_samples(block, seen, halves[target]), copy=False))
                sse[i, h, w], error_sum[i, h, w] = clip_sums(labels[seen], y[seen], pred)
                from_sums = pooled_scores(y[seen], sse[i, h, w], error_sum[i, h, w])[0][0]
                sums_ok.append(abs(from_sums - r2(np.repeat(y[seen], SAMPLES_PER_CLIP, axis=0), pred)) <= SUMS_TOLERANCE)
                pu = fit.predict(scaler.transform(half_samples(block, unseen, halves[target]), copy=False))
                s = probe_scores(VARIABLE, np.repeat(labels[unseen], SAMPLES_PER_CLIP), pu)
                unseen_scores[i, h, w] = s["r2"], s["circular_mae"]
            same_r2 = pooled_scores(y[seen], sse[i, h, 0], error_sum[i, h, 0])[0][0]
            verdict[i, h] = alpha_verdict(fit.alpha_edge, same_r2)
        del block
        print(f"index {i:2d} ({PATCH_SITES[i]}): {(time.perf_counter() - start) / 60:.1f} min", flush=True)

    idx = bootstrap_indices(len(seen), N_RESAMPLES, SEED)
    curve, across_curves = [], np.zeros((N_RESAMPLES, n_index))
    for i in range(n_index):
        stats = {}
        for w, name in enumerate(("same", "across")):
            per_half = [pooled_scores(y[seen], sse[i, h, w], error_sum[i, h, w]) for h in range(n_half)]
            per_half_b = [pooled_scores(y[seen], sse[i, h, w], error_sum[i, h, w], idx) for h in range(n_half)]
            stats[name] = {
                "point": float(np.mean([p[0][0] for p in per_half])),
                "samples": np.mean([p[0] for p in per_half_b], axis=0),
                "circular_mae": float(np.mean([p[1][0] for p in per_half])),
                "per_half_r2": {HALF_NAMES[h]: float(per_half[h][0][0]) for h in range(n_half)},
            }
        across_curves[:, i] = stats["across"]["samples"]
        gap = stats["same"]["samples"] - stats["across"]["samples"]
        curve.append({
            "index": i, "site": PATCH_SITES[i],
            **{f"{name}_r2": {"point": stats[name]["point"], "ci": list(percentile_interval(stats[name]["samples"], LEVEL)),
                              "circular_mae": stats[name]["circular_mae"], "per_half": stats[name]["per_half_r2"]}
               for name in ("same", "across")},
            "gap_r2": {"point": stats["same"]["point"] - stats["across"]["point"],
                       "ci": list(percentile_interval(gap, LEVEL))},
            "val_unseen_mean": {"same_r2": float(unseen_scores[i, :, 0, 0].mean()),
                                "across_r2": float(unseen_scores[i, :, 1, 0].mean())},
            "alpha_verdicts": dict(Counter(verdict[i].tolist())),
        })
    across_point = np.array([row["across_r2"]["point"] for row in curve])

    arrays = {
        "sites": np.array(PATCH_SITES), "halves": np.array(HALF_NAMES), "val_seen_ids": table["id"][seen],
        "scaler_mean": scaler_mean, "scaler_scale": scaler_scale, "coef": coef, "intercept": intercept,
        "alpha": alpha, "alpha_verdict": verdict, "cv_mse": cv_mse, "n_fit": n_fit,
        "val_seen_sse": sse, "val_seen_circular_error_sum": error_sum, "val_unseen_scores": unseen_scores,
    }
    np.savez_compressed(SPATIAL, **arrays)
    with np.load(SPATIAL) as saved:
        saved_ok = set(saved.files) == set(arrays) and all(
            np.array_equal(saved[k], v, equal_nan=v.dtype.kind == "f") for k, v in arrays.items())
    criteria = {
        "code_committed": committed,
        "ids_aligned": bool(np.array_equal(ids, table["id"])),
        "no_clip_in_two_folds": folds_grouped,
        "fits_used_exactly_train_samples": bool((n_fit == len(train) * SAMPLES_PER_CLIP).all()),
        "scored_clips_val_seen_only": bool(set(seen).isdisjoint(train) and (roles[seen] == "val_seen").all()),
        "no_alpha_failure": bool((verdict != "failure").all()),
        "sums_match_direct_r2": len(sums_ok) == n_index * n_half * 2 and all(sums_ok),
        "scores_finite": bool(np.isfinite(sse).all() and np.isfinite(error_sum).all() and np.isfinite(unseen_scores).all()),
        "saved_equals_computed": saved_ok,
    }
    return {
        "criteria": criteria, "n_resamples": N_RESAMPLES, "level": LEVEL, "seed": SEED, "folds": N_FOLDS,
        "curve": curve,
        "transition_on_across_curve": transition_points(across_point),
        "bootstrap_across_transition_index_counts":
            np.bincount(rise_index(across_curves, TRANSITION_FRACTION), minlength=n_index).tolist(),
        "minutes_total": round((time.perf_counter() - start) / 60, 1),
        "artifact": {"path": str(SPATIAL.relative_to(REPO)), "sha256": file_sha256(SPATIAL)},
        "passed": all(criteria.values()),
    }

def mean_disk_position(table: dict) -> np.ndarray:
    """(clips, 2) tracked disk position (x, y) in metres, averaged over the frames with disk pixels."""
    x, y = pixel_to_world(table["centre"][..., 0], table["centre"][..., 1])  # NaN where no disk
    world = np.where((table["area"] > 0)[..., None], np.stack([x, y], axis=-1), np.nan)
    return np.nanmean(world, axis=1)


def cubic_terms(position: np.ndarray) -> np.ndarray:
    """(clips, 9) monomials of (x, y) up to degree 3."""
    x, y = position[:, 0], position[:, 1]
    return np.column_stack([x, y, x * x, x * y, y * y, x**3, x * x * y, x * y * y, y**3])


def check_position_baseline() -> dict:
    """How much of (sin, cos) the disk's mean position alone predicts: a bound on position-driven direction readout.

    Features: each clip's tracked mean disk position (linear), and its monomials up to degree 3 (cubic). Ordinary
    least squares with intercept (n = 813 >> 2 or 9 features, so no regularisation or alpha choice), fit on
    train, scored on val_seen and val_unseen with the probes' metrics; printed next to the off-path curve. Passes
    if: the code is committed; every clip has a finite position; fits used exactly the train clips; the linear
    baseline's val_seen R² < POSITION_THRESHOLD (otherwise the result goes to the planning chat).
    """
    committed = not code_changes(repo_root())
    table = load_joined(VARIABLE)
    roles, labels = table["role"], table["label"]
    y = probe_targets(VARIABLE, labels)
    train = roles == "train"
    position = mean_disk_position(table)
    features = {"linear": position, "cubic": cubic_terms(position)}

    baselines = {}
    for name, x in features.items():
        model = LinearRegression().fit(x[train], y[train])
        baselines[name] = {
            "features": int(x.shape[1]), "n_fit": int(train.sum()),
            **{role: probe_scores(VARIABLE, labels[roles == role], model.predict(x[roles == role]))
               for role in EVAL_ROLES},
        }
    off_path = json.loads(OUT.read_text())["patch_breakdown"]["result"]["curve"]
    criteria = {
        "code_committed": committed,
        "positions_finite": bool(np.isfinite(position).all()),
        "fit_on_train_only": all(b["n_fit"] == EXPECTED_FIT for b in baselines.values()),
        "linear_position_r2_below_threshold": baselines["linear"]["val_seen"]["r2"] < POSITION_THRESHOLD,
    }
    return {
        "criteria": criteria, "threshold": POSITION_THRESHOLD, "baselines": baselines,
        "off_path_val_seen_r2_mean": {PATCH_SITES[i]: off_path[i]["off_path"]["r2_mean"] for i in SHOWN_OFF_PATH},
        "passed": all(criteria.values()),
    }


CHECKS = {
    "extract_direction": check_extract_direction,
    "verify": check_verify,
    "patch_probes": check_patch_probes,
    "patch_alpha_diagnostic": check_patch_alpha_diagnostic,
    "patch_breakdown": check_patch_breakdown,
    "patch_bootstrap": check_patch_bootstrap,
    "spatial_generalization": check_spatial_generalization,
    "position_baseline": check_position_baseline,
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