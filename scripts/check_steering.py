"""Checks for multi-probe subspace steering: setup, activation cache, steering runs and their scores.

Run one check at a time from the repo root, with .venv active:
    python scripts/check_steering.py <check>

Each check prints its result and stores it under its own key in results/steering/checks.json.
"""
import argparse
from functools import partial
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from vjepa_physics.activations import capture_encoder
from vjepa_physics.data import DATASETS, read_manifest, resolve
from vjepa_physics.evidence import file_sha256, save_result, verified_artifact
from vjepa_physics.extraction import SITES, pool_time_steps
from vjepa_physics.intervention import edit_encoder, run_blocks
from vjepa_physics.joined import FLAG_NAMES, load_joined
from vjepa_physics.model import load_model, weights_fingerprint
from vjepa_physics.preprocess import preprocess_clip
from vjepa_physics.probes import Probe, fit_probe, probe_scores, probe_targets, site_features
from vjepa_physics.reproducibility import set_seeds
from vjepa_physics.steering import (
    N_CLIPS, QUARTILE_SIZE, RANDOM_SEEDS, STEERING_SITE, arm_shifts, arm_table, covariance_map, keyed_rng,
    load_probe_sequence, random_probe_counts, steered_features, steering_clips, steering_targets,
)
from vjepa_physics.video import load_clip

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/steering/checks.json"
ARTIFACTS = REPO / "artifacts/steering"  # regenerable, git-ignored

READOUT_SITES = (STEERING_SITE, "block_17")  # same-layer control (index 9) and primary readout (index 18)
VALIDATION_ROLES = ("val_seen", "val_unseen")  # readout probes are fit on these clips only
EXPECTED_READOUT_FIT = {"direction": 297, "speed": 304, "acceleration": 304}
MIN_READOUT_TRAIN_R2 = 0.9  # readout quality, scored on the train clips the readout never saw
MAP_TOLERANCE = 1e-10  # relative: raw-space readout map vs the fitted probe's own prediction
N_TARGETS, N_UNSEEN_TARGETS = 5, 3

DEVICE = "mps"
DATA = REPO / "data"
# In-memory weights fingerprint right after a clean load (results/model/checks.json, key "load").
REFERENCE_FINGERPRINT = "c865f524c1376e4452943b208d7d50ba588be9490604f235a3d9c9dc80804ede"
FIRST_BLOCK, READOUT_BLOCK = 9, 17  # partial forward: the blocks after the steering site, up to the readout site
CONTROL_RELATIVE_STD = 1e-2  # positive-control delta: per-element std as a fraction of the cached site's std
CONTROL_KEY = 0  # positive-control deltas: keyed_rng(variable index, clip id, CONTROL_KEY)
MIN_UNSTEERED_R2 = 0.9  # unsteered index-18 readout of the steered clips' own labels
BUDGET_MINUTES = 30  # a steering run projected above this goes back to the planning chat

HALVES = {"seen": "test_seen", "unseen": "test_unseen"}  # one saved run per variable and half
HIT_TOLERANCE = 1e-5  # probe arms: idx-9 steering probes vs target after the fp32 edit (target units)
SITE_TOLERANCE = 1e-4  # idx-9 features vs stored + shift after the fp32 edit (absolute)
LENGTH_TOLERANCE = 1e-12  # random arms: z-length vs the probes arm with the same n (relative)
FEATURE_ROWS = {STEERING_SITE: 0, f"block_{READOUT_BLOCK}": READOUT_BLOCK - FIRST_BLOCK + 1}  # rows of steered_features


def selection_groups(table: dict, rows: np.ndarray, variable: str, role: str) -> np.ndarray:
    """The group each clip was spread over when chosen: held-out value, angle octant, or value-index quartile."""
    if role == "test_unseen":
        return table["value_index"][rows]
    if variable == "direction":
        return table["octant"][rows]
    return table["value_index"][rows] // QUARTILE_SIZE


def readout_map(probe: Probe) -> tuple[np.ndarray, np.ndarray]:
    """Raw-space form of a fitted probe: prediction = x @ weights + offset, weights (d, m), offset (m,)."""
    d = len(probe.scaler.scale_)
    weights = np.reshape(probe.ridge.coef_, (-1, d)).T / probe.scaler.scale_[:, None]
    offset = np.atleast_1d(probe.ridge.intercept_) - probe.scaler.mean_ @ weights
    return weights, offset


def check_steering_setup() -> dict:
    """Clips to steer, target values and validation-fit readout probes, fixed before any model run.

    Per variable: 15 test-seen + 15 test-unseen clips (seeded; spread over value quartiles or octants, and over
    the held-out values) and 5 target values (3 test-unseen). Readout probes of all three variables at index 9 and
    18, fit on the validation clips only, scored on the train clips. Writes artifacts/steering/setup.npz. Passes
    if: clip counts, roles and spread as specified, ids distinct, selection repeatable; 5 targets with 3 unseen;
    every readout fit on exactly the validation clips, interior alpha, train R² >= MIN_READOUT_TRAIN_R2, raw-space
    map = the probe's prediction; saved file = computed. Test clips: only roles, labels and flags are read.
    """
    arrays: dict[str, np.ndarray] = {}
    result: dict = {}
    ok: dict[str, list[bool]] = {name: [] for name in (
        "clip_counts", "clip_roles", "clip_spread", "repeatable", "targets",
        "readout_fit", "readout_alpha", "readout_quality", "readout_map")}

    for variable in DATASETS:
        table = load_joined(variable)
        roles, labels = table["role"], table["label"]

        clips = steering_clips(table, variable)
        again = steering_clips(table, variable)
        ok["repeatable"].append(all(np.array_equal(clips[r], again[r]) for r in N_CLIPS))
        clip_record = {}
        for role, ids in clips.items():
            rows = np.flatnonzero(np.isin(table["id"], ids))
            groups = selection_groups(table, rows, variable, role)
            available = np.unique(selection_groups(table, np.flatnonzero(roles == role), variable, role))
            names, counts = np.unique(groups, return_counts=True)
            ok["clip_counts"].append(len(ids) == N_CLIPS[role] and len(np.unique(ids)) == len(ids) == len(rows))
            ok["clip_roles"].append(bool((roles[rows] == role).all()))
            ok["clip_spread"].append(bool(np.array_equal(names, available) and counts.max() - counts.min() <= 1))
            clip_record[role] = {
                "ids": ids.tolist(),
                "labels": labels[rows].tolist(),
                "groups": {str(g): int(c) for g, c in zip(names, counts)},
                "flags": {f: int(table[f][rows].sum()) for f in FLAG_NAMES},
            }
            arrays[f"{variable}_{role}_ids"] = ids

        targets = steering_targets(table, variable)
        ok["targets"].append(len(targets["label"]) == N_TARGETS and int(targets["unseen"].sum()) == N_UNSEEN_TARGETS)
        arrays |= {f"{variable}_target_{name}": value for name, value in targets.items()}

        seq = load_probe_sequence(variable)
        record = {
            "clips": clip_record,
            "targets": {name: value.tolist() for name, value in targets.items()},
            "k": seq.k, "headline_n": seq.k - 1, "random_probe_counts": list(random_probe_counts(seq.k)),
            "readouts": {},
        }

        # this variable's readout probes, fit on its own validation clips; later applied to every steered clip
        y = probe_targets(variable, labels)
        train = roles == "train"
        n_validation = int(np.isin(roles, VALIDATION_ROLES).sum())
        for site in READOUT_SITES:
            x = site_features(table["activations"], site)
            probe = fit_probe(x, y, roles, fit_roles=VALIDATION_ROLES)
            weights, offset = readout_map(probe)
            direct = probe.predict(x[train]).reshape(int(train.sum()), -1)
            map_diff = float(np.abs(x[train] @ weights + offset - direct).max() / np.abs(direct).max())
            score = probe_scores(variable, labels[train], direct[:, 0] if direct.shape[1] == 1 else direct)
            ok["readout_fit"].append(probe.n_fit == EXPECTED_READOUT_FIT[variable] == n_validation)
            ok["readout_alpha"].append(probe.alpha_edge is None)
            ok["readout_quality"].append(score["r2"] >= MIN_READOUT_TRAIN_R2)
            ok["readout_map"].append(map_diff <= MAP_TOLERANCE)
            arrays |= {f"readout_{variable}_{site}_weights": weights, f"readout_{variable}_{site}_offset": offset}
            record["readouts"][site] = {
                "n_fit": probe.n_fit, "alpha": probe.alpha, "alpha_edge": probe.alpha_edge,
                "map_max_relative_diff": map_diff, "train_scores": score,
            }
        result[variable] = record

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / "setup.npz"
    np.savez(path, **arrays)
    with np.load(path) as saved:
        saved_ok = set(saved.files) == set(arrays) and all(np.array_equal(saved[k], v) for k, v in arrays.items())

    criteria = {name: all(values) for name, values in ok.items()} | {"saved_equals_computed": saved_ok}
    return {
        "variables": result,
        "readout_sites": list(READOUT_SITES),
        "validation_roles": list(VALIDATION_ROLES),
        "min_readout_train_r2": MIN_READOUT_TRAIN_R2,
        "criteria": criteria,
        "artifact": {"path": str(path.relative_to(REPO)), "sha256": file_sha256(path)},
        "passed": all(criteria.values()),
    }

def count_forward_hooks(model: torch.nn.Module) -> int:
    return sum(len(module._forward_hooks) for module in model.modules())


def probe_feature(hidden: torch.Tensor) -> np.ndarray:
    """(1, 2048, D) tokens -> (D,) float64: the mean of the 8 per-time-step means, as the probes' features."""
    return np.asarray(pool_time_steps(hidden).to("cpu")[0], dtype=np.float64).mean(axis=0)


def row_hash(row: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()


def check_steering_cache() -> dict:
    """Cache the steering site's per-token output for every chosen clip, and prove the partial path exact on each.

    Per clip (MPS): one full forward pass. Its block_8 output (2048 x 1024 fp32) is written to
    artifacts/steering/cache_<variable>.npy (rows = test_seen then test_unseen ids, as in the setup). Passes if, on
    every clip: the recomputed pooled activations at block_8 and block_17 equal the stored ones; a zero edit through
    blocks 9-17 equals the full pass at each of those blocks; with a seeded random delta (same on every token), the
    partial path equals edit_encoder in a full pass at blocks 8-17 and changes block_17; the cache re-read equals what
    was computed; afterwards the forward-hook count and the weights fingerprint are unchanged; and per variable the
    unsteered index-18 readout of the clips' own labels has R² >= MIN_UNSTEERED_R2. Records the time per partial
    forward and each steering run's projected duration.
    """
    setup_path = verified_artifact(OUT, "steering_setup")
    setup_result = json.loads(OUT.read_text())["steering_setup"]["result"]
    with np.load(setup_path) as f:
        setup = {key: f[key] for key in f.files}
    readouts = {(v, s): (setup[f"readout_{v}_{s}_weights"], setup[f"readout_{v}_{s}_offset"])
                for v in DATASETS for s in READOUT_SITES}

    set_seeds()
    model, _ = load_model(DEVICE)
    hooks_before = count_forward_hooks(model)
    blocks = [f"block_{i}" for i in range(FIRST_BLOCK, READOUT_BLOCK + 1)]
    readout_block = f"block_{READOUT_BLOCK}"
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    result: dict = {"ids": {}, "unsteered": {}}
    failures: list[str] = []
    ok: dict[str, list[bool]] = {name: [] for name in (
        "stored_pooled", "zero_edit", "control_paths", "control_changes", "cache_saved", "unsteered_readout")}
    times: list[float] = []

    for variable in DATASETS:
        table = load_joined(variable)
        root = DATA / variable
        paths = {row["id"]: resolve(root, row["video"]) for row in read_manifest(root)}
        row_of = {int(i): r for r, i in enumerate(table["id"].tolist())}
        ids = np.concatenate([setup[f"{variable}_{role}_ids"] for role in N_CLIPS])
        result["ids"][variable] = {role: setup[f"{variable}_{role}_ids"].tolist() for role in N_CLIPS}

        path = ARTIFACTS / f"cache_{variable}.npy"
        cache = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(ids), 2048, 1024))
        written, features = [], {site: [] for site in READOUT_SITES}
        for c, clip_id in enumerate(ids.tolist()):
            row = row_of[clip_id]
            x = preprocess_clip(load_clip(paths[clip_id])).unsqueeze(0).to(DEVICE)
            with capture_encoder(model) as acts, torch.inference_mode():
                model(pixel_values_videos=x, skip_predictor=True)
            full = dict(acts)
            cached = full[STEERING_SITE]
            tag = f"{variable}/{clip_id}"

            stored = all(
                torch.equal(pool_time_steps(full[s]).to("cpu")[0],
                            torch.from_numpy(np.array(table["activations"][row, SITES.index(s)])))
                for s in READOUT_SITES
            )

            start = time.perf_counter()  # timed like a steering run: add, blocks 9-17, pooled means to the CPU
            zero = run_blocks(model, cached + torch.zeros_like(cached), FIRST_BLOCK, READOUT_BLOCK)
            torch.stack([zero[b].mean(dim=1) for b in blocks]).to("cpu")
            times.append(time.perf_counter() - start)
            zero_ok = all(torch.equal(zero[b], full[b]) for b in blocks)
            del zero

            rng = keyed_rng(DATASETS.index(variable), clip_id, CONTROL_KEY)
            scale = CONTROL_RELATIVE_STD * cached.std().item()
            delta = torch.from_numpy((rng.standard_normal(cached.shape[-1]) * scale).astype(np.float32)).to(DEVICE)
            partial = run_blocks(model, cached + delta, FIRST_BLOCK, READOUT_BLOCK)
            # The lambda is only called inside this iteration's context, so it always sees this delta.
            with capture_encoder(model) as acts, edit_encoder(model, STEERING_SITE, lambda h: h + delta), \
                    torch.inference_mode():
                model(pixel_values_videos=x, skip_predictor=True)
            edited = dict(acts)
            paths_ok = torch.equal(edited[STEERING_SITE], cached + delta) and all(
                torch.equal(partial[b], edited[b]) for b in blocks)
            changes = not torch.equal(partial[readout_block], full[readout_block])
            del partial, edited

            cache[c] = cached.to("cpu")[0].numpy()
            written.append(row_hash(cache[c]))
            for site in READOUT_SITES:
                features[site].append(probe_feature(full[site]))

            for name, passed in (("stored_pooled", stored), ("zero_edit", zero_ok),
                                 ("control_paths", paths_ok), ("control_changes", changes)):
                ok[name].append(passed)
                if not passed:
                    failures.append(f"{tag}/{name}")
            del full, cached

        cache.flush()
        del cache
        reread = np.load(path, mmap_mode="r")
        cache_ok = reread.shape == (len(ids), 2048, 1024) and [row_hash(r) for r in reread] == written
        ok["cache_saved"].append(cache_ok)
        del reread
        result[f"cache_{variable}"] = {"path": str(path.relative_to(REPO)), "sha256": file_sha256(path)}

        # unsteered readouts of each clip's own variable, per role and over all 30 clips
        labels = table["label"][[row_of[i] for i in ids.tolist()]]
        roles = np.repeat(list(N_CLIPS), [N_CLIPS[r] for r in N_CLIPS])
        record = {}
        for site in READOUT_SITES:
            weights, offset = readouts[(variable, site)]
            pred = np.stack(features[site]) @ weights + offset
            pred = pred[:, 0] if pred.shape[1] == 1 else pred
            record[site] = {"all": probe_scores(variable, labels, pred)} | {
                role: probe_scores(variable, labels[roles == role], pred[roles == role]) for role in N_CLIPS}
        ok["unsteered_readout"].append(record[readout_block]["all"]["r2"] >= MIN_UNSTEERED_R2)
        result["unsteered"][variable] = record

    median = float(np.median(times))
    projection = {}
    for variable in DATASETS:
        k = setup_result["variables"][variable]["k"]
        per_pair = k + 1 + len(random_probe_counts(k)) * RANDOM_SEEDS  # counts 1...K, covariance, random
        n_targets = len(setup[f"{variable}_target_label"])
        projection[variable] = {}
        for role in N_CLIPS:
            runs = N_CLIPS[role] * (n_targets * per_pair + 1)  # + one unedited run per clip
            minutes = runs * median / 60
            projection[variable][role] = {"runs": runs, "minutes": minutes, "within_budget": minutes <= BUDGET_MINUTES}

    hooks_after = count_forward_hooks(model)
    fingerprint = weights_fingerprint(model)
    criteria = {name: all(values) for name, values in ok.items()} | {
        "hooks_unchanged": hooks_before == hooks_after,
        "weights_unchanged": fingerprint == REFERENCE_FINGERPRINT,
    }
    return result | {
        "device": DEVICE,
        "partial_forward_blocks": [FIRST_BLOCK, READOUT_BLOCK],
        "control_relative_std": CONTROL_RELATIVE_STD,
        "min_unsteered_r2": MIN_UNSTEERED_R2,
        "partial_forward_seconds": {"median": median, "max": float(max(times)), "n": len(times)},
        "projection": projection,
        "budget_minutes": BUDGET_MINUTES,
        "failures": failures,
        "forward_hooks_before_and_after": [hooks_before, hooks_after],
        "weights_fingerprint": fingerprint,
        "criteria": criteria,
        "passed": all(criteria.values()),
    }

def steering_run(variable: str, half: str) -> dict:
    """Steer one variable's test-seen or test-unseen clips to every target with every arm, from the cache.

    Per clip: one unedited partial pass, then per target all arms of arm_table(K) (probe counts 1...K, covariance,
    keyed random directions). Saves per run the raw shift, the features at indices 9-18 (fp32) and the readouts of all
    three variables at index 9 and 18 (float64), plus the unedited ones, to artifacts/steering/runs_<variable>_<half>.npz.
    Passes if: setup and cache hash-verified with the same clip ids; every unedited pass = the stored features at
    index 9 and 18 exactly; after every edit the index-9 features = stored + shift within SITE_TOLERANCE and every
    probes arm's steering probes read the target within HIT_TOLERANCE; random lengths match within LENGTH_TOLERANCE;
    all outputs finite; hooks and weights unchanged; saved file = computed.
    """
    role = HALVES[half]
    setup_path = verified_artifact(OUT, "steering_setup")
    cache_path = verified_artifact(OUT, "steering_cache", f"cache_{variable}")
    saved_results = json.loads(OUT.read_text())
    with np.load(setup_path) as f:
        setup = {key: f[key] for key in f.files}
    ids = setup[f"{variable}_{role}_ids"]
    ids_ok = saved_results["steering_cache"]["result"]["ids"][variable][role] == ids.tolist()
    order = list(N_CLIPS)
    first_row = sum(N_CLIPS[r] for r in order[: order.index(role)])  # cache rows: test_seen, then test_unseen
    cache = np.load(cache_path, mmap_mode="r")
    readouts = {(v, s): (setup[f"readout_{v}_{s}_weights"], setup[f"readout_{v}_{s}_offset"])
                for v in DATASETS for s in READOUT_SITES}

    table = load_joined(variable)
    roles = table["role"]
    row_of = {int(i): r for r, i in enumerate(table["id"].tolist())}
    seq = load_probe_sequence(variable)
    x9 = site_features(table["activations"], STEERING_SITE)
    x18 = site_features(table["activations"], f"block_{READOUT_BLOCK}")
    z = (x9 - seq.mean) / seq.scale
    train = roles == "train"
    b = covariance_map(z[train], probe_targets(variable, table["label"])[train])
    span = train_span(z, roles)
    arms = arm_table(seq.k)
    target_labels = setup[f"{variable}_target_label"]
    n_clips, n_targets, n_arms, d = len(ids), len(target_labels), len(arms), x9.shape[1]
    n_rows = READOUT_BLOCK - FIRST_BLOCK + 2

    shifts = np.empty((n_clips, n_targets, n_arms, d))
    features = np.empty((n_clips, n_targets, n_arms, n_rows, d), dtype=np.float32)
    unedited = np.empty((n_clips, n_rows, d), dtype=np.float32)
    readout = {(v, s): np.empty((n_clips, n_targets, n_arms, len(readouts[(v, s)][1]))) for v, s in readouts}
    readout_unedited = {(v, s): np.empty((n_clips, len(readouts[(v, s)][1]))) for v, s in readouts}
    ok = {name: [] for name in ("unedited_equals_stored", "site", "hit", "length", "finite")}
    max_site, max_hit, max_length = 0.0, 0.0, 0.0
    probe_arms = [a for a, arm in enumerate(arms) if arm[0] == "probes"]

    set_seeds()
    model, _ = load_model(DEVICE)
    hooks_before = count_forward_hooks(model)
    start = time.perf_counter()
    for c, clip_id in enumerate(ids.tolist()):
        row = row_of[clip_id]
        cached = torch.from_numpy(np.array(cache[first_row + c]))[None].to(DEVICE)
        base = steered_features(model, cached, np.zeros((1, d)), FIRST_BLOCK, READOUT_BLOCK)[0]
        ok["unedited_equals_stored"].append(bool(np.array_equal(base[0], x9[row]) and np.array_equal(base[-1], x18[row])))
        unedited[c] = base
        for (v, s), (w, o) in readouts.items():
            readout_unedited[(v, s)][c] = base[FEATURE_ROWS[s]] @ w + o

        for t in range(n_targets):
            goal = probe_targets(variable, target_labels[t:t + 1]).reshape(-1)
            sh = arm_shifts(seq, b, span, variable, clip_id, t, x9[row], goal)
            feats = steered_features(model, cached, sh, FIRST_BLOCK, READOUT_BLOCK)
            site = float(np.abs(feats[:, 0] - (x9[row] + sh)).max())
            hit = max(float(np.abs(seq.outputs(feats[a, :1], arms[a][1]) - goal).max()) for a in probe_arms)
            lengths = {n: np.linalg.norm(sh[a] / seq.scale) for a in probe_arms for n in [arms[a][1]]}
            length = max(abs(np.linalg.norm(sh[a] / seq.scale) - lengths[arm[1]]) / lengths[arm[1]]
                         for a, arm in enumerate(arms) if arm[0] == "random")
            max_site, max_hit, max_length = max(max_site, site), max(max_hit, hit), max(max_length, length)
            ok["site"].append(site <= SITE_TOLERANCE)
            ok["hit"].append(hit <= HIT_TOLERANCE)
            ok["length"].append(length <= LENGTH_TOLERANCE)
            ok["finite"].append(bool(np.isfinite(feats).all()))
            shifts[c, t], features[c, t] = sh, feats
            for (v, s), (w, o) in readouts.items():
                readout[(v, s)][c, t] = feats[:, FEATURE_ROWS[s]] @ w + o
        print(f"{variable} {half}: clip {c + 1}/{n_clips} done, {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    seconds = time.perf_counter() - start
    hooks_after = count_forward_hooks(model)
    fingerprint = weights_fingerprint(model)

    arrays = {
        "ids": ids, "target_label": target_labels,
        "arm_kind": np.array([a[0] for a in arms]), "arm_n": np.array([a[1] for a in arms]),
        "arm_seed": np.array([a[2] for a in arms]),
        "shifts": shifts, "features": features, "unedited_features": unedited,
        **{f"readout_{v}_{s}": value for (v, s), value in readout.items()},
        **{f"readout_unedited_{v}_{s}": value for (v, s), value in readout_unedited.items()},
    }
    finite_readouts = all(np.isfinite(value).all() for value in arrays.values() if value.dtype.kind == "f")
    path = ARTIFACTS / f"runs_{variable}_{half}.npz"
    np.savez(path, **arrays)
    with np.load(path) as saved:
        saved_ok = set(saved.files) == set(arrays) and all(np.array_equal(saved[k], v) for k, v in arrays.items())

    criteria = {name: all(values) for name, values in ok.items()} | {
        "same_clip_ids": ids_ok,
        "arm_count": n_arms == seq.k + 1 + len(random_probe_counts(seq.k)) * RANDOM_SEEDS,
        "finite_outputs": finite_readouts,
        "hooks_unchanged": hooks_before == hooks_after,
        "weights_unchanged": fingerprint == REFERENCE_FINGERPRINT,
        "saved_equals_computed": saved_ok,
    }
    return {
        "variable": variable, "role": role, "ids": ids.tolist(), "k": seq.k,
        "targets": target_labels.tolist(), "arms": [list(a) for a in arms],
        "runs": n_clips * (n_targets * n_arms + 1), "seconds": seconds,
        "tolerances": {"hit": HIT_TOLERANCE, "site": SITE_TOLERANCE, "length": LENGTH_TOLERANCE},
        "max_site_diff": max_site, "max_hit_error": max_hit, "max_length_error": max_length,
        "forward_hooks_before_and_after": [hooks_before, hooks_after], "weights_fingerprint": fingerprint,
        "criteria": criteria,
        "artifact": {"path": str(path.relative_to(REPO)), "sha256": file_sha256(path)},
        "passed": all(criteria.values()),
    }

CHECKS = {
    "steering_setup": check_steering_setup,
    "steering_cache": check_steering_cache,
    **{f"steer_{v}_{h}": partial(steering_run, v, h) for v in DATASETS for h in HALVES},
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