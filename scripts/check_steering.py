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

import matplotlib

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler

from vjepa_physics.activations import capture_encoder
from vjepa_physics.baselines import FIT_ROLE as KERNEL_FIT_ROLE, rbf_kernel_ridge
from vjepa_physics.data import DATASETS, read_manifest, resolve
from vjepa_physics.evidence import file_sha256, require_clean_code, save_result, verified_artifact
from vjepa_physics.extraction import SITES, plot_index, pool_time_steps
from vjepa_physics.intervention import edit_encoder, run_blocks
from vjepa_physics.joined import FLAG_NAMES, load_joined
from vjepa_physics.metrics import bootstrap_indices, percentile_interval, resampled_mean
from vjepa_physics.model import load_model, weights_fingerprint
from vjepa_physics.nullspace import train_span
from vjepa_physics.plotting import AXIS, DATASET_COLOUR, GRID, INK, INK_MUTED, INK_SECONDARY, SURFACE, style_axes
from vjepa_physics.preprocess import preprocess_clip
from vjepa_physics.probes import Probe, fit_probe, probe_scores, probe_targets, site_features
from vjepa_physics.reproducibility import SEED, set_seeds
from vjepa_physics.steering import (
    N_CLIPS, QUARTILE_SIZE, RANDOM_SEEDS, STEERING_SITE, arm_shifts, arm_table, covariance_map, keyed_rng,
    label_difference, load_probe_sequence, random_probe_counts, readout_values, steered_features, steering_clips,
    steering_targets,
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

READOUT_SITE = f"block_{READOUT_BLOCK}"  # primary readout (index 18); STEERING_SITE = same-layer control (index 9)
READOUT_TOLERANCE = 1e-12  # relative: stored unedited readouts vs the setup maps on the stored features
BOOTSTRAP_RESAMPLES = 10_000

PROFILE_SITES = tuple(f"block_{i}" for i in range(8, READOUT_BLOCK + 1))  # indices 9-18 = rows 0-9 of the saved features
DECOMPOSITION_TOLERANCE = 1e-9  # relative: direct + block-update terms = total readout change
CONSISTENCY_TOLERANCE = 1e-5  # relative: readouts from the saved fp32 features vs the float64 readouts saved in the runs
LENGTH_EDGES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)  # bins of |shift_z| / median train clip distance; last bin open
OFF_DISTRIBUTION_RATIO = 1.0  # bins starting at or above this are labelled off-distribution (pre-stated)

CLIP_SECONDS = 15 / 24  # frame 0 to frame 15 at 24 fps
DISTANCE_PER_UNIT = {"speed": CLIP_SECONDS, "acceleration": CLIP_SECONDS**2 / 2}  # metres per m/s; per m/s² from rest
SPECIFICITY_PAIRS = (("speed", "acceleration"), ("acceleration", "speed"))  # (steered, read)

H12_LINEAR_GAIN = 0.9  # reading rule: the linear readout "says target" at this output-space gain or more
H12_ORIGINAL_FRACTION = 0.5  # reading rule: the kernel "still says original" if this fraction of runs is nearest it
H12_MEAN_FRACTION = 0.5  # reading rule: nearest the validation mean this often = off-distribution, uninformative
MIN_HAT_GAP = 1e-6  # kernel fits with a smaller min(1 - h_ii) are flagged: leave-one-out selection unreliable

FIGURE_REDUCTION = REPO / "results/steering/steering_reduction.png"
FIGURE_PROPAGATION = REPO / "results/steering/steering_propagation.png"
FIGURE_DPI = 200


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

def train_distance(z_train: np.ndarray) -> float:
    """Median Euclidean distance between two different train clips (standardized features)."""
    sq = (z_train**2).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2 * z_train @ z_train.T
    return float(np.median(np.sqrt(np.clip(d2[np.triu_indices(len(z_train), 1)], 0.0, None))))


def all_finite(obj) -> bool:
    if isinstance(obj, dict):
        return all(all_finite(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(all_finite(v) for v in obj)
    if isinstance(obj, float):
        return bool(np.isfinite(obj))
    return True


def check_steering_scores() -> dict:
    """Score the six steering runs (no model): readout errors per arm, same-layer control, specificity, shift sizes.

    Per steered variable, both halves combined (30 clips x 5 targets). For every arm group (probes n = 1...K,
    covariance, random at each probe count with its seeds pooled) and the unedited clip: the index-18 readout's error
    to the target and to the clip's own label (label units; circular degrees for direction), the error reduction vs
    the unedited clip (1 - mean error / mean unedited error) and the gain (least-squares slope of achieved on intended
    shift through the origin); the same at index 9 (same-layer control); the median standardized shift length and
    its ratio to the median train clip-to-clip distance. Breakdowns: all, test-seen / test-unseen clips, seen /
    unseen target values. Specificity: mean change of every variable's index-18 readout vs the unedited clip.
    Per-round shares of |shift|² at n = K - 1 and K. Clip-bootstrap 95% intervals of the index-18 error and
    reduction for probes K - 1 and K, covariance, random K - 1 and K, and the unedited error. Passes if: both
    halves come from the same clean commit and code; clip ids and targets = the setup's; the arm table is the same
    in both halves; stored unedited readouts = the setup maps on the stored features; all scores finite; bootstrap
    point estimates = the full-sample means.
    """
    saved = json.loads(OUT.read_text())
    with np.load(verified_artifact(OUT, "steering_setup")) as f:
        setup = {key: f[key] for key in f.files}
    n_clips = sum(N_CLIPS.values())
    boot = bootstrap_indices(n_clips, BOOTSTRAP_RESAMPLES, SEED)
    roles_order = list(HALVES.values())
    ok: dict[str, list[bool]] = {name: [] for name in (
        "same_code", "clean", "ids", "targets", "arms", "unedited_readouts", "finite", "bootstrap_point")}
    result: dict = {}

    for variable in DATASETS:
        runs, provenance = {}, {}
        for half, role in HALVES.items():
            key = f"steer_{variable}_{half}"
            provenance[role] = saved[key]["provenance"]
            with np.load(verified_artifact(OUT, key)) as f:
                runs[role] = {name: f[name] for name in f.files}
        first, second = (runs[r] for r in roles_order)
        p1, p2 = (provenance[r] for r in roles_order)
        targets = setup[f"{variable}_target_label"]
        unseen_target = setup[f"{variable}_target_unseen"]
        ok["same_code"].append(p1["git_commit"] == p2["git_commit"]
                               and p1["code"]["combined_sha256"] == p2["code"]["combined_sha256"])
        ok["clean"].append(not p1["git_dirty"] and not p2["git_dirty"])
        ok["ids"].append(all(np.array_equal(runs[r]["ids"], setup[f"{variable}_{r}_ids"]) for r in roles_order))
        ok["targets"].append(all(np.array_equal(runs[r]["target_label"], targets) for r in roles_order))
        ok["arms"].append(all(np.array_equal(first[k], second[k]) for k in ("arm_kind", "arm_n", "arm_seed")))

        def joined_array(name: str) -> np.ndarray:
            return np.concatenate([runs[r][name] for r in roles_order])

        ids = joined_array("ids")
        clip_role = np.repeat(roles_order, [len(runs[r]["ids"]) for r in roles_order])
        arms = list(zip(first["arm_kind"].tolist(), first["arm_n"].tolist(), first["arm_seed"].tolist()))
        groups: dict[str, list[int]] = {}
        for a, (kind, n, _) in enumerate(arms):
            groups.setdefault("covariance" if kind == "covariance" else f"{kind}_{n}", []).append(a)

        table = load_joined(variable)
        row_of = {int(i): r for r, i in enumerate(table["id"].tolist())}
        rows = np.array([row_of[int(i)] for i in ids])
        labels = table["label"][rows]
        seq = load_probe_sequence(variable)
        stored = {s: site_features(table["activations"], s) for s in READOUT_SITES}

        unedited_diff = 0.0
        for w in DATASETS:
            for s in READOUT_SITES:
                expected = stored[s][rows] @ setup[f"readout_{w}_{s}_weights"] + setup[f"readout_{w}_{s}_offset"]
                got = joined_array(f"readout_unedited_{w}_{s}")
                unedited_diff = max(unedited_diff, float(np.abs(got - expected).max() / np.abs(expected).max()))
        ok["unedited_readouts"].append(unedited_diff <= READOUT_TOLERANCE)

        # the steered variable's own readouts, in label units: (clips, targets, arms) and unedited (clips,)
        value = {s: readout_values(variable, joined_array(f"readout_{variable}_{s}")) for s in READOUT_SITES}
        base = {s: readout_values(variable, joined_array(f"readout_unedited_{variable}_{s}")) for s in READOUT_SITES}
        err = {s: np.abs(label_difference(variable, value[s], targets[None, :, None])) for s in READOUT_SITES}
        unsteered = {s: np.abs(label_difference(variable, base[s][:, None], targets[None, :])) for s in READOUT_SITES}
        intended = {s: label_difference(variable, targets[None, :], base[s][:, None]) for s in READOUT_SITES}
        achieved = {s: label_difference(variable, value[s], base[s][:, None, None]) for s in READOUT_SITES}
        error_original = np.abs(label_difference(variable, value[READOUT_SITE], labels[:, None, None]))
        unsteered_original = np.abs(label_difference(variable, base[READOUT_SITE], labels))

        shifts = joined_array("shifts")
        lengths = np.linalg.norm(shifts / seq.scale, axis=-1)  # standardized length, (clips, targets, arms)
        z_train = (stored[STEERING_SITE][table["role"] == "train"] - seq.mean) / seq.scale
        distance = train_distance(z_train)

        n_targets = len(targets)
        masks = {
            "all": np.ones((n_clips, n_targets), dtype=bool),
            "test_seen": np.repeat((clip_role == "test_seen")[:, None], n_targets, axis=1),
            "test_unseen": np.repeat((clip_role == "test_unseen")[:, None], n_targets, axis=1),
            "target_seen": np.repeat(~unseen_target[None, :], n_clips, axis=0),
            "target_unseen": np.repeat(unseen_target[None, :], n_clips, axis=0),
        }

        def summary(arm_rows: list[int] | None, mask: np.ndarray) -> dict:
            if arm_rows is None:  # the unedited clip
                return {
                    "error_target": float(unsteered[READOUT_SITE][mask].mean()),
                    "same_layer_error_target": float(unsteered[STEERING_SITE][mask].mean()),
                    "error_original": float(np.repeat(unsteered_original[:, None], n_targets, axis=1)[mask].mean()),
                }
            out = {}
            for s, prefix in ((READOUT_SITE, ""), (STEERING_SITE, "same_layer_")):
                e = err[s][:, :, arm_rows][mask]
                a = achieved[s][:, :, arm_rows][mask]
                i = np.broadcast_to(intended[s][mask][:, None], a.shape)
                out[f"{prefix}error_target"] = float(e.mean())
                out[f"{prefix}reduction"] = float(1 - e.mean() / unsteered[s][mask].mean())
                out[f"{prefix}gain"] = float((a * i).sum() / (i**2).sum())
            out["error_original"] = float(error_original[:, :, arm_rows][mask].mean())
            out["shift_length_median"] = float(np.median(lengths[:, :, arm_rows][mask]))
            out["shift_over_distance_median"] = out["shift_length_median"] / distance
            return out

        arm_scores = {name: {g: summary(idx, m) for g, m in masks.items()} for name, idx in groups.items()}
        arm_scores["unedited"] = {g: summary(None, m) for g, m in masks.items()}

        specificity = {}
        for w in DATASETS:
            if w == variable:
                change = np.abs(achieved[READOUT_SITE])
            else:
                v_w = readout_values(w, joined_array(f"readout_{w}_{READOUT_SITE}"))
                b_w = readout_values(w, joined_array(f"readout_unedited_{w}_{READOUT_SITE}"))
                change = np.abs(label_difference(w, v_w, b_w[:, None, None]))
            specificity[w] = {name: float(change[:, :, idx].mean()) for name, idx in groups.items()}

        m = seq.dims_per_round

        def round_shares(n: int) -> list[float]:
            dz = shifts[:, :, arms.index(("probes", n, -1))] / seq.scale
            parts = ((dz @ seq.basis[:, : n * m]).reshape(n_clips, n_targets, n, m) ** 2).sum(axis=-1)
            return np.median(parts / (dz**2).sum(axis=-1)[..., None], axis=(0, 1)).tolist()

        k = seq.k
        per_clip_unsteered = unsteered[READOUT_SITE].mean(axis=1)
        u = resampled_mean(per_clip_unsteered, boot)
        headline = {"unedited": {"error_target": float(per_clip_unsteered.mean()),
                                 "error_target_ci": percentile_interval(u)}}
        for name in (f"probes_{k - 1}", f"probes_{k}", "covariance", f"random_{k - 1}", f"random_{k}"):
            per_clip = err[READOUT_SITE][:, :, groups[name]].mean(axis=(1, 2))
            e = resampled_mean(per_clip, boot)
            point = arm_scores[name]["all"]
            ok["bootstrap_point"].append(bool(np.isclose(per_clip.mean(), point["error_target"], rtol=1e-12, atol=0)))
            headline[name] = {
                "error_target": point["error_target"], "error_target_ci": percentile_interval(e),
                "reduction": point["reduction"], "reduction_ci": percentile_interval(1 - e / u),
            }

        record = {
            "k": k, "headline_n": k - 1, "targets": targets.tolist(), "target_unseen": unseen_target.tolist(),
            "run_commit": p1["git_commit"], "train_clip_distance_median": distance,
            "unedited_readout_max_relative_diff": unedited_diff,
            "headline": headline, "arms": arm_scores, "specificity_idx18": specificity,
            "round_shares": {str(n): round_shares(n) for n in (k - 1, k)},
        }
        ok["finite"].append(all_finite(record))
        result[variable] = record

    criteria = {name: all(values) for name, values in ok.items()}
    return {
        "variables": result, "readout_site": READOUT_SITE, "same_layer_site": STEERING_SITE,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "criteria": criteria, "passed": all(criteria.values()),
    }


def check_steering_propagation() -> dict:
    """Post hoc, saved test outputs only (no model): how the edit propagates from index 9 to 18, and why.

    Per steered variable, both halves (30 clips x 5 targets): validation-fit ridge readouts (D-42 grid, LOO) of the
    steered variable at every index 9-18, applied to the saved steered and unedited features. Per index and arm group:
    the output-space gain (sum of achieved . intended / sum of |intended|², outputs = the readout's (sin, cos) or scalar,
    intended = target output - unedited output), split exactly into direct (readout map applied to the shift, which the
    skip connections carry unchanged) + block updates (map applied to the feature change minus the shift); and the error
    reduction in label units. At index 18: clip-bootstrap 95% intervals of the three gains for probes 1, K - 1, K and
    covariance; gain in bins of |shift_z| / median train clip distance (bins from 1.0 on labelled off-distribution),
    probes + covariance arms and random arms separately. Passes if: every profile readout has an interior alpha and
    train R² >= MIN_READOUT_TRAIN_R2; the refit index-9 and 18 maps = the setup's exactly; readouts from the saved
    features match the runs' saved readouts within CONSISTENCY_TOLERANCE; direct + blocks = total within
    DECOMPOSITION_TOLERANCE; everything finite.
    """
    with np.load(verified_artifact(OUT, "steering_setup")) as f:
        setup = {key: f[key] for key in f.files}
    n_clips = sum(N_CLIPS.values())
    boot = bootstrap_indices(n_clips, BOOTSTRAP_RESAMPLES, SEED)
    roles_order = list(HALVES.values())
    ok: dict[str, list[bool]] = {name: [] for name in (
        "readout_alpha", "readout_quality", "setup_maps", "consistent_with_runs", "decomposition", "finite")}
    result: dict = {}

    for variable in DATASETS:
        runs = {}
        for half, role in HALVES.items():
            with np.load(verified_artifact(OUT, f"steer_{variable}_{half}")) as f:
                runs[role] = {name: f[name] for name in f.files}

        def joined_array(name: str) -> np.ndarray:
            return np.concatenate([runs[r][name] for r in roles_order])

        first = runs[roles_order[0]]
        arms = list(zip(first["arm_kind"].tolist(), first["arm_n"].tolist(), first["arm_seed"].tolist()))
        groups: dict[str, list[int]] = {}
        for a, (kind, n, _) in enumerate(arms):
            groups.setdefault("covariance" if kind == "covariance" else f"{kind}_{n}", []).append(a)

        table = load_joined(variable)
        roles, all_labels = table["role"], table["label"]
        y = probe_targets(variable, all_labels)
        train = roles == "train"
        seq = load_probe_sequence(variable)
        k = seq.k
        targets = setup[f"{variable}_target_label"]
        target_out = probe_targets(variable, targets).reshape(len(targets), -1)  # (targets, m)

        features = joined_array("features")  # (clips, targets, arms, 10, d) fp32
        base_features = joined_array("unedited_features")  # (clips, 10, d) fp32
        shifts = joined_array("shifts")  # (clips, targets, arms, d) float64
        z_train = (site_features(table["activations"], STEERING_SITE)[train] - seq.mean) / seq.scale
        distance = train_distance(z_train)
        ratio = np.linalg.norm(shifts / seq.scale, axis=-1) / distance  # (clips, targets, arms)

        profile: dict = {}
        at_readout: dict = {}
        for j, site in enumerate(PROFILE_SITES):
            x = site_features(table["activations"], site)
            probe = fit_probe(x, y, roles, fit_roles=VALIDATION_ROLES)
            weights, offset = readout_map(probe)
            train_pred = probe.predict(x[train]).reshape(int(train.sum()), -1)
            score = probe_scores(variable, all_labels[train], train_pred[:, 0] if train_pred.shape[1] == 1 else train_pred)
            ok["readout_alpha"].append(probe.alpha_edge is None)
            ok["readout_quality"].append(score["r2"] >= MIN_READOUT_TRAIN_R2)
            if site in READOUT_SITES:
                ok["setup_maps"].append(bool(np.array_equal(weights, setup[f"readout_{variable}_{site}_weights"])
                                             and np.array_equal(offset, setup[f"readout_{variable}_{site}_offset"])))

            steered = features[:, :, :, j].astype(np.float64)
            base = base_features[:, j].astype(np.float64)
            out = steered @ weights + offset  # (clips, targets, arms, m)
            out0 = base @ weights + offset  # (clips, m)
            achieved = out - out0[:, None, None, :]
            direct = shifts @ weights
            blocks = (steered - base[:, None, None, :] - shifts) @ weights
            scale = max(float(np.abs(achieved).max()), float(np.abs(direct).max()))
            ok["decomposition"].append(float(np.abs(direct + blocks - achieved).max()) <= DECOMPOSITION_TOLERANCE * scale)
            if site in READOUT_SITES:
                saved_out = joined_array(f"readout_{variable}_{site}")
                ok["consistent_with_runs"].append(
                    float(np.abs(out - saved_out).max()) <= CONSISTENCY_TOLERANCE * float(np.abs(saved_out).max()))

            intended = target_out[None, :, :] - out0[:, None, :]  # (clips, targets, m)
            value = readout_values(variable, out)
            base_value = readout_values(variable, out0)
            err = np.abs(label_difference(variable, value, targets[None, :, None]))
            unsteered = np.abs(label_difference(variable, base_value[:, None], targets[None, :]))
            record = {"site": site, "alpha": probe.alpha, "train_scores": score, "arms": {}}
            for name, idx in groups.items():
                i = np.broadcast_to(intended[:, :, None, :], achieved[:, :, idx].shape)
                den = float((i**2).sum())
                record["arms"][name] = {
                    "gain": float((achieved[:, :, idx] * i).sum()) / den,
                    "gain_direct": float((direct[:, :, idx] * i).sum()) / den,
                    "gain_blocks": float((blocks[:, :, idx] * i).sum()) / den,
                    "reduction": float(1 - err[:, :, idx].mean() / unsteered.mean()),
                }
            profile[str(plot_index(site))] = record
            if site == READOUT_SITE:
                at_readout = {"total": achieved, "direct": direct, "blocks": blocks, "intended": intended}

        # index 18: bootstrap intervals of the three gains, paired over clips
        def boot_gain(part: np.ndarray, idx: list[int]) -> dict:
            i = np.broadcast_to(at_readout["intended"][:, :, None, :], part[:, :, idx].shape)
            num = (part[:, :, idx] * i).sum(axis=(1, 2, 3))
            den = (i**2).sum(axis=(1, 2, 3))
            return {"point": float(num.sum() / den.sum()),
                    "ci": percentile_interval(num[boot].sum(axis=1) / den[boot].sum(axis=1))}

        decomposition = {name: {part: boot_gain(at_readout[part], groups[name]) for part in ("total", "direct", "blocks")}
                         for name in ("probes_1", f"probes_{k - 1}", f"probes_{k}", "covariance")}

        # index 18: gain by shift size
        families = {"probes_and_covariance": [a for a, arm in enumerate(arms) if arm[0] != "random"],
                    "random": [a for a, arm in enumerate(arms) if arm[0] == "random"]}
        edges = [*LENGTH_EDGES, None]
        by_length = {}
        for family, idx in families.items():
            r = ratio[:, :, idx]
            a = at_readout["total"][:, :, idx]
            i = np.broadcast_to(at_readout["intended"][:, :, None, :], a.shape)
            bins = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                sel = (r >= lo) & (r < hi) if hi is not None else r >= lo
                n = int(sel.sum())
                gain = float((a[sel] * i[sel]).sum() / (i[sel] ** 2).sum()) if n else None
                bins.append({"from": lo, "to": hi, "runs": n, "gain": gain,
                             "off_distribution": lo >= OFF_DISTRIBUTION_RATIO})
            by_length[family] = bins

        record = {
            "k": k, "headline_n": k - 1, "train_clip_distance_median": distance,
            "profile": profile,
            "decomposition_idx18": decomposition,
            "gain_vs_length_idx18": by_length,
            "idx9_reduction_covariance_vs_headline": {
                "covariance": profile["9"]["arms"]["covariance"]["reduction"],
                f"probes_{k - 1}": profile["9"]["arms"][f"probes_{k - 1}"]["reduction"],
            },
        }
        ok["finite"].append(all_finite(record))
        result[variable] = record

    criteria = {name: all(values) for name, values in ok.items()}
    return {
        "post_hoc": True, "variables": result, "profile_sites": list(PROFILE_SITES),
        "gain_definition": "output space: sum(achieved . intended) / sum(|intended|^2)",
        "length_edges": list(LENGTH_EDGES), "off_distribution_ratio": OFF_DISTRIBUTION_RATIO,
        "criteria": criteria, "passed": all(criteria.values()),
    }

def load_runs(variable: str, extra: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
    """Readouts and ids of both halves of one variable's steering runs, read through their hashes and joined along the
    clip axis (test_seen first), plus any per-clip arrays named in `extra` (e.g. "features", "shifts"); target labels
    and the arm table from the first half."""
    def per_clip(name: str) -> bool:
        return name == "ids" or name.startswith("readout") or name in extra

    halves = []
    for half in HALVES:
        with np.load(verified_artifact(OUT, f"steer_{variable}_{half}")) as f:
            halves.append({name: f[name] for name in f.files if per_clip(name)}
                          | {name: f[name] for name in ("target_label", "arm_kind", "arm_n", "arm_seed")})
    joined = {name: np.concatenate([h[name] for h in halves]) for name in halves[0] if per_clip(name)}
    return joined | {name: halves[0][name] for name in ("target_label", "arm_kind", "arm_n", "arm_seed")}


def check_steering_specificity() -> dict:
    """Post hoc, saved test outputs only (no model): signed speed <-> acceleration cross-readout changes in metres.

    For each (steered, read) pair, readout site (index 9, 18) and arm group (probes 1, K/2, K - 1, K; covariance;
    random K - 1, K): the slope through the origin of the read variable's readout change on the steered variable's own
    readout change, both vs the unedited clip and in metres travelled over the clip (DISTANCE_PER_UNIT); also the slope
    on the intended shift. Prediction fixed before computing (planning chat): slope ≈ +1 under the shared-distance
    reading. Clip-bootstrap 95% intervals (paired resamples of the 30 clips). Passes if both halves of the steered
    variable come from one clean commit, everything is finite, and each bootstrap point equals the full-sample slope.
    """
    saved = json.loads(OUT.read_text())
    boot = bootstrap_indices(sum(N_CLIPS.values()), BOOTSTRAP_RESAMPLES, SEED)
    ok: dict[str, list[bool]] = {"clean": [], "finite": [], "bootstrap_point": []}
    result: dict = {}

    for steered, read in SPECIFICITY_PAIRS:
        provs = [saved[f"steer_{steered}_{h}"]["provenance"] for h in HALVES]
        ok["clean"].append(len({p["git_commit"] for p in provs}) == 1 and not any(p["git_dirty"] for p in provs))
        runs = load_runs(steered)
        arms = list(zip(runs["arm_kind"].tolist(), runs["arm_n"].tolist(), runs["arm_seed"].tolist()))
        groups: dict[str, list[int]] = {}
        for a, (kind, n, _) in enumerate(arms):
            groups.setdefault("covariance" if kind == "covariance" else f"{kind}_{n}", []).append(a)
        k = max(n for kind, n, _ in arms if kind == "probes")
        targets = runs["target_label"]

        record = {}
        for site in READOUT_SITES:
            base_own = readout_values(steered, runs[f"readout_unedited_{steered}_{site}"])  # (clips,)
            base_read = readout_values(read, runs[f"readout_unedited_{read}_{site}"])
            own = (readout_values(steered, runs[f"readout_{steered}_{site}"]) - base_own[:, None, None]) \
                * DISTANCE_PER_UNIT[steered]
            cross = (readout_values(read, runs[f"readout_{read}_{site}"]) - base_read[:, None, None]) \
                * DISTANCE_PER_UNIT[read]
            intended = (targets[None, :] - base_own[:, None]) * DISTANCE_PER_UNIT[steered]  # (clips, targets)

            site_record = {}
            for name in ("probes_1", f"probes_{k // 2}", f"probes_{k - 1}", f"probes_{k}", "covariance",
                         f"random_{k - 1}", f"random_{k}"):
                idx = groups[name]
                o, c = own[:, :, idx], cross[:, :, idx]
                i = np.broadcast_to(intended[:, :, None], o.shape)

                def slope(x: np.ndarray) -> dict:
                    num, den = (x * c).sum(axis=(1, 2)), (x**2).sum(axis=(1, 2))
                    point = float(num.sum() / den.sum())
                    ok["bootstrap_point"].append(bool(np.isclose(point, (x * c).sum() / (x**2).sum(), rtol=1e-12, atol=0)))
                    return {"point": point, "ci": percentile_interval(num[boot].sum(axis=1) / den[boot].sum(axis=1))}

                site_record[name] = {
                    "slope_on_own_change": slope(o),
                    "slope_on_intended": slope(i),
                    "mean_abs_own_change_m": float(np.abs(o).mean()),
                    "mean_abs_cross_change_m": float(np.abs(c).mean()),
                }
            record[site] = site_record
        result[f"{steered}_to_{read}"] = record

    ok["finite"].append(all_finite(result))
    criteria = {name: all(values) for name, values in ok.items()}
    return {
        "post_hoc": True, "pairs": result, "distance_per_unit": DISTANCE_PER_UNIT, "clip_seconds": CLIP_SECONDS,
        "predicted_slope": 1.0, "criteria": criteria, "passed": all(criteria.values()),
    }


def check_steering_kernel() -> dict:
    """Post hoc, saved test outputs only (no model): does a nonlinear readout still see the original value (H-12)?

    Per steered variable and site (index 9, where H-12 is read; index 18, consistency): an RBF kernel ridge readout
    (the nullspace checks' pipeline: median-heuristic gamma x factors, relative alpha grid, exact leave-one-out) fit on
    the validation clips' features z-scored with validation statistics, predicting the train clips (quality), the
    unedited steered clips and every saved steered point. Each steered kernel output is classed as nearest (output
    space) to the target, the clip's original label, or the validation mean. Per arm group and bin of
    |shift_z| / median train clip distance: those fractions, the kernel's output-space gain and the saved linear
    readout's gain at the same site. Reading rule (fixed before computing), index 9, probes K - 1, in-distribution
    shifts: linear gain >= H12_LINEAR_GAIN and nearest-original fraction >= H12_ORIGINAL_FRACTION -> "supports";
    nearest-mean fraction >= H12_MEAN_FRACTION -> "uninformative (off-distribution)"; else "not supported". Passes if
    every kernel fit has no lower-edge alpha and train-clip R² >= MIN_READOUT_TRAIN_R2, and all outputs are finite.
    """
    ok: dict[str, list[bool]] = {"no_lower_alpha": [], "kernel_quality": [], "finite": []}
    result: dict = {}
    for variable in DATASETS:
        runs = load_runs(variable, extra=("features", "unedited_features", "shifts"))
        arms = list(zip(runs["arm_kind"].tolist(), runs["arm_n"].tolist(), runs["arm_seed"].tolist()))
        groups: dict[str, list[int]] = {}
        for a, (kind, n, _) in enumerate(arms):
            groups.setdefault("covariance" if kind == "covariance" else f"{kind}_{n}", []).append(a)
        k = max(n for kind, n, _ in arms if kind == "probes")

        table = load_joined(variable)
        roles, labels = table["role"], table["label"]
        row_of = {int(i): r for r, i in enumerate(table["id"].tolist())}
        rows = np.array([row_of[int(i)] for i in runs["ids"]])
        y = probe_targets(variable, labels).reshape(len(labels), -1)
        val, train = np.isin(roles, VALIDATION_ROLES), roles == "train"
        targets = runs["target_label"]
        n_clips, n_targets, n_arms = len(rows), len(targets), len(arms)
        target_out = probe_targets(variable, targets).reshape(n_targets, -1)
        original_out = y[rows]
        mean_out = y[val].mean(axis=0)

        seq = load_probe_sequence(variable)
        z_train = (site_features(table["activations"], STEERING_SITE)[train] - seq.mean) / seq.scale
        distance = train_distance(z_train)
        ratio = np.linalg.norm(runs["shifts"] / seq.scale, axis=-1) / distance  # (clips, targets, arms)

        record: dict = {"train_clip_distance_median": distance}
        for site in READOUT_SITES:
            j = FEATURE_ROWS[site]
            x = site_features(table["activations"], site)
            steered = runs["features"][:, :, :, j].reshape(-1, x.shape[1]).astype(np.float64)
            unedited = runs["unedited_features"][:, j].astype(np.float64)
            points = np.vstack([x[val], x[train], unedited, steered])
            n_val, n_train, n_unedited = int(val.sum()), int(train.sum()), len(unedited)
            # the kernel readout is fit on the validation clips: they take the fit role, every other row is predicted
            fit_roles = np.array([KERNEL_FIT_ROLE] * n_val + ["predict"] * (len(points) - n_val))
            y_points = np.vstack([y[val], np.zeros((len(points) - n_val, y.shape[1]))])  # only fit rows' labels are used
            scaler = StandardScaler().fit(x[val])
            rbf = rbf_kernel_ridge(scaler.transform(points), y_points, fit_roles, fit_roles != KERNEL_FIT_ROLE)
            pred = rbf.fit.predictions.reshape(len(points), -1)
            p_train = pred[n_val:n_val + n_train]
            p_unedited = pred[n_val + n_train:n_val + n_train + n_unedited]
            p_steered = pred[n_val + n_train + n_unedited:].reshape(n_clips, n_targets, n_arms, -1)

            train_score = probe_scores(variable, labels[train], p_train[:, 0] if p_train.shape[1] == 1 else p_train)
            unedited_score = probe_scores(variable, labels[rows],
                                          p_unedited[:, 0] if p_unedited.shape[1] == 1 else p_unedited)
            ok["no_lower_alpha"].append(rbf.fit.alpha_edge != "lower")
            ok["kernel_quality"].append(train_score["r2"] >= MIN_READOUT_TRAIN_R2)
            ok["finite"].append(bool(np.isfinite(pred[n_val:]).all()))

            candidates = np.stack([
                np.broadcast_to(target_out[None, :, None, :], p_steered.shape),
                np.broadcast_to(original_out[:, None, None, :], p_steered.shape),
                np.broadcast_to(mean_out, p_steered.shape),
            ])
            nearest = np.linalg.norm(p_steered[None] - candidates, axis=-1).argmin(axis=0)  # 0 target, 1 original, 2 mean
            kernel_achieved = p_steered - p_unedited[:, None, None, :]
            kernel_intended = target_out[None, :, :] - p_unedited[:, None, :]
            linear = runs[f"readout_{variable}_{site}"]
            linear_base = runs[f"readout_unedited_{variable}_{site}"]
            linear_achieved = linear - linear_base[:, None, None, :]
            linear_intended = target_out[None, :, :] - linear_base[:, None, :]

            def gain(achieved: np.ndarray, intended: np.ndarray, idx: list[int], mask: np.ndarray) -> float | None:
                a = achieved[:, :, idx][mask]
                i = np.broadcast_to(intended[:, :, None, :], achieved[:, :, idx].shape)[mask]
                return float((a * i).sum() / (i**2).sum()) if len(a) else None

            edges = [*LENGTH_EDGES, None]
            arm_record = {}
            for name in ("probes_1", f"probes_{k // 2}", f"probes_{k - 1}", f"probes_{k}", "covariance",
                         f"random_{k - 1}", f"random_{k}"):
                idx = groups[name]
                r = ratio[:, :, idx]
                bins = {"all": np.ones_like(r, dtype=bool), "in_distribution": r < OFF_DISTRIBUTION_RATIO}
                bins |= {f"{lo}-{hi if hi is not None else 'inf'}": ((r >= lo) & (r < hi)) if hi is not None else r >= lo
                         for lo, hi in zip(edges[:-1], edges[1:])}
                arm_record[name] = {}
                for bin_name, mask in bins.items():
                    near = nearest[:, :, idx][mask]
                    arm_record[name][bin_name] = {
                        "runs": int(mask.sum()),
                        "nearest_target": float((near == 0).mean()) if len(near) else None,
                        "nearest_original": float((near == 1).mean()) if len(near) else None,
                        "nearest_mean": float((near == 2).mean()) if len(near) else None,
                        "kernel_gain": gain(kernel_achieved, kernel_intended, idx, mask),
                        "linear_gain": gain(linear_achieved, linear_intended, idx, mask),
                    }
            record[site] = {
                "gamma": rbf.gamma, "gamma_median": rbf.gamma_median, "gamma_edge": rbf.gamma_edge,
                "alpha": rbf.fit.alpha, "alpha_edge": rbf.fit.alpha_edge,
                "min_one_minus_hat": rbf.fit.min_one_minus_hat,
                "hat_gap_flag": bool(rbf.fit.min_one_minus_hat < MIN_HAT_GAP),
                "train_scores": train_score, "unedited_scores": unedited_score, "arms": arm_record,
            }

        head = record[STEERING_SITE]["arms"][f"probes_{k - 1}"]["in_distribution"]
        if head["linear_gain"] is not None and head["linear_gain"] >= H12_LINEAR_GAIN \
                and head["nearest_original"] >= H12_ORIGINAL_FRACTION:
            verdict = "supports"
        elif head["nearest_mean"] is not None and head["nearest_mean"] >= H12_MEAN_FRACTION:
            verdict = "uninformative (off-distribution)"
        else:
            verdict = "not supported"
        record["h12_reading_idx9"] = {"arm": f"probes_{k - 1}", "bin": "in_distribution", "verdict": verdict}
        ok["finite"].append(all_finite({key: value for key, value in record.items() if key != "h12_reading_idx9"}))
        result[variable] = record

    criteria = {name: all(values) for name, values in ok.items()}
    return {
        "post_hoc": True, "variables": result,
        "reading_rule": {"linear_gain": H12_LINEAR_GAIN, "nearest_original": H12_ORIGINAL_FRACTION,
                         "nearest_mean": H12_MEAN_FRACTION, "site": STEERING_SITE},
        "criteria": criteria, "passed": all(criteria.values()),
    }


def check_figure_steering() -> dict:
    """Two figures drawn from the saved steering_scores and steering_propagation results (nothing recomputed).

    steering_reduction.png: per variable, error reduction vs probes used n at the idx-18 readout (solid) and the
    independent idx-9 readout (dashed); random same-length edits (grey); covariance arm (diamonds, filled idx 18,
    hollow idx 9); 95% CIs at idx 18 for n = K - 1 (headline) and K. steering_propagation.png: top, output-space gain
    vs layer index 9-18 for probes K - 1, probes 1, covariance and random K - 1; bottom, the idx-18 gain split into
    direct and block-update parts with 95% CIs and the total (diamond), for probes 1, K - 1 and covariance (n = K
    omitted: off-distribution, scale). Passes if both PNGs are written and non-empty and every plotted value is finite.
    """
    saved = json.loads(OUT.read_text())
    scores = saved["steering_scores"]["result"]["variables"]
    propagation = saved["steering_propagation"]["result"]["variables"]
    plotted: list[float] = []

    # figure 1: error reduction vs probes used
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True, facecolor=SURFACE)
    for ax, variable in zip(axes, DATASETS):
        record, colour = scores[variable], DATASET_COLOUR[variable]
        k, arms = record["k"], record["arms"]
        ns = list(range(1, k + 1))
        for key, style, face in (("reduction", "-", colour), ("same_layer_reduction", "--", SURFACE)):
            ys = [arms[f"probes_{n}"]["all"][key] for n in ns]
            plotted += ys
            ax.plot(ns, ys, style, color=colour, linewidth=2, marker="o", markersize=6,
                    markerfacecolor=face, markeredgecolor=colour, zorder=3)
        random_ns = [n for n in ns if f"random_{n}" in arms]
        for key, style in (("reduction", "-"), ("same_layer_reduction", "--")):
            ys = [arms[f"random_{n}"]["all"][key] for n in random_ns]
            plotted += ys
            ax.plot(random_ns, ys, style, color=INK_MUTED, linewidth=1.2, marker="s", markersize=4, zorder=2)
        cov_x = k + 1.3
        cov = arms["covariance"]["all"]
        plotted += [cov["reduction"], cov["same_layer_reduction"]]
        ax.plot([cov_x], [cov["reduction"]], "D", color=colour, markersize=7, zorder=3)
        ax.plot([cov_x], [cov["same_layer_reduction"]], "D", color=colour, markerfacecolor=SURFACE, markersize=7,
                zorder=3)
        for n in (k - 1, k):
            head = record["headline"][f"probes_{n}"]
            low, high = head["reduction_ci"]
            ax.errorbar([n], [head["reduction"]], yerr=[[head["reduction"] - low], [high - head["reduction"]]],
                        color=INK_SECONDARY, capsize=3, linewidth=1, zorder=4)
        ax.axhline(0, color=AXIS, linewidth=1, zorder=1)
        ax.axvline(k - 1, color=GRID, linewidth=1.5, linestyle=":", zorder=0)
        ax.text(k - 1, 1.02, "headline", ha="center", va="bottom", fontsize=8, color=INK_MUTED)
        ax.set_xticks([*ns, cov_x])
        ax.set_xticklabels([*map(str, ns[:-1]), f"{k} (K)", "cov."])
        ax.set_ylim(-0.45, 1.1)
        ax.set_title(variable, fontsize=11)
        ax.set_xlabel("probes used (n)")
        style_axes(ax)
    axes[0].set_ylabel("error reduction vs unedited clip")
    handles = [
        Line2D([], [], color=INK_SECONDARY, linestyle="-", marker="o", label="probes: readout at idx 18 (9 blocks later)"),
        Line2D([], [], color=INK_SECONDARY, linestyle="--", marker="o", markerfacecolor=SURFACE,
               label="probes: independent readout at idx 9 (steering layer)"),
        Line2D([], [], color=INK_MUTED, marker="s", label="random edit, same length (solid idx 18, dashed idx 9)"),
        Line2D([], [], color=INK_SECONDARY, marker="D", linestyle="none",
               label="covariance direction(s) (filled idx 18, hollow idx 9)"),
        Line2D([], [], color=INK_SECONDARY, marker="|", markersize=10, linestyle="none", label="95% CI, idx 18"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)
    fig.suptitle("Steering at idx 9: the steering-layer readout follows, the readout 9 blocks later barely moves",
                 color=INK, fontsize=12)
    fig.text(0.5, 0.905, "n = K: shift 1–4 × the typical clip-to-clip distance (off-distribution)",
             ha="center", fontsize=8.5, color=INK_MUTED)
    fig.subplots_adjust(bottom=0.27, top=0.83, wspace=0.08)
    fig.savefig(FIGURE_REDUCTION, dpi=FIGURE_DPI, facecolor=SURFACE)
    plt.close(fig)

    # figure 2: propagation profile (top) and idx-18 decomposition (bottom)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.4), facecolor=SURFACE)
    indices = list(range(9, 19))
    for col, variable in enumerate(DATASETS):
        record, colour = propagation[variable], DATASET_COLOUR[variable]
        k = record["k"]
        ax = axes[0, col]
        for name, style, line_colour, width, marker in (
            (f"probes_{k - 1}", "-", colour, 2.2, "o"), ("probes_1", "--", colour, 1.4, None),
            ("covariance", ":", colour, 1.8, None), (f"random_{k - 1}", "-", INK_MUTED, 1.2, None),
        ):
            ys = [record["profile"][str(i)]["arms"][name]["gain"] for i in indices]
            plotted += ys
            ax.plot(indices, ys, style, color=line_colour, linewidth=width, marker=marker, markersize=4, zorder=3)
        ax.axhline(0, color=AXIS, linewidth=1, zorder=1)
        ax.set_ylim(-0.1, 1.08)
        ax.set_xticks(indices)
        ax.set_title(variable, fontsize=11)
        ax.set_xlabel("layer index (edit at 9)")
        style_axes(ax)
        if col:
            ax.set_yticklabels([])

        ax = axes[1, col]
        names = ["probes_1", f"probes_{k - 1}", "covariance"]
        decomposition = record["decomposition_idx18"]
        xs = np.arange(len(names))
        for offset, part, hatch in ((-0.2, "direct", None), (0.2, "blocks", "////")):
            values = [decomposition[n][part]["point"] for n in names]
            intervals = [decomposition[n][part]["ci"] for n in names]
            plotted += values
            ax.bar(xs + offset, values, width=0.36, color=colour if hatch is None else SURFACE, edgecolor=colour,
                   hatch=hatch, linewidth=1.2, zorder=2)
            ax.errorbar(xs + offset, values,
                        yerr=[[v - c[0] for v, c in zip(values, intervals)], [c[1] - v for v, c in zip(values, intervals)]],
                        fmt="none", ecolor=INK_SECONDARY, capsize=3, linewidth=1, zorder=3)
        totals = [decomposition[n]["total"]["point"] for n in names]
        plotted += totals
        ax.plot(xs, totals, "D", color=INK, markersize=6, linestyle="none", zorder=4)
        ax.axhline(0, color=AXIS, linewidth=1, zorder=1)
        ax.set_xticks(xs)
        ax.set_xticklabels(["1 probe", f"{k - 1} probes (K−1)", "covariance"])
        ax.set_ylim(-0.45, 0.6)
        style_axes(ax)
        if col:
            ax.set_yticklabels([])
    axes[0, 0].set_ylabel("gain (output space)")
    axes[1, 0].set_ylabel("gain at idx 18 (output space)")
    axes[0, 2].legend(handles=[
        Line2D([], [], color=INK_SECONDARY, linestyle="-", marker="o", label="probes, n = K−1 (headline)"),
        Line2D([], [], color=INK_SECONDARY, linestyle="--", label="1 probe"),
        Line2D([], [], color=INK_SECONDARY, linestyle=":", label="covariance direction(s)"),
        Line2D([], [], color=INK_MUTED, label="random, same length as K−1"),
    ], loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)
    fig.legend(handles=[
        Patch(facecolor=INK_SECONDARY, edgecolor=INK_SECONDARY, label="direct: the edit carried by skip connections"),
        Patch(facecolor=SURFACE, edgecolor=INK_SECONDARY, hatch="////", label="block updates (blocks 9–17)"),
        Line2D([], [], color=INK, marker="D", linestyle="none", label="total change (= direct + blocks)"),
    ], loc="lower center", ncol=3, frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)
    fig.suptitle("Where the edit is lost: gain by depth (top) and the idx-18 change split into its parts (bottom)",
                 color=INK, fontsize=12)
    fig.text(0.5, 0.925, "gain = readout change / intended change, in the readout's output space ((sin, cos) for "
             "direction); 95% clip-bootstrap CIs; n = K omitted (off-distribution)",
             ha="center", fontsize=8.5, color=INK_MUTED)
    fig.subplots_adjust(bottom=0.1, top=0.87, hspace=0.42, wspace=0.08)
    fig.savefig(FIGURE_PROPAGATION, dpi=FIGURE_DPI, facecolor=SURFACE)
    plt.close(fig)

    paths = {"reduction": FIGURE_REDUCTION, "propagation": FIGURE_PROPAGATION}
    criteria = {
        "written": all(p.exists() and p.stat().st_size > 0 for p in paths.values()),
        "finite": bool(np.isfinite(np.asarray(plotted, dtype=float)).all()),
    }
    return {
        "figures": {name: {"path": str(p.relative_to(REPO)), "sha256": file_sha256(p)} for name, p in paths.items()},
        "sources": {key: saved[key]["provenance"]["git_commit"] for key in ("steering_scores", "steering_propagation")},
        "values_plotted": len(plotted),
        "criteria": criteria, "passed": all(criteria.values()),
    }
    

CHECKS = {
    "steering_setup": check_steering_setup,
    "steering_cache": check_steering_cache,
    **{f"steer_{v}_{h}": partial(steering_run, v, h) for v in DATASETS for h in HALVES},
    "steering_scores": check_steering_scores,
    "steering_propagation": check_steering_propagation,
    "steering_specificity": check_steering_specificity,
    "steering_kernel": check_steering_kernel,
    "figure_steering": check_figure_steering,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("check", choices=sorted(CHECKS))
    name = parser.parse_args().check
    require_clean_code()  # after parsing, so --help still works with uncommitted code
    result = CHECKS[name]()
    print(json.dumps(result, indent=2))
    save_result(OUT, name, result)
    print(f"saved '{name}' -> {OUT.relative_to(REPO)}")
    if result.get("passed") is False:
        raise SystemExit(f"check '{name}' FAILED")


if __name__ == "__main__":
    main()