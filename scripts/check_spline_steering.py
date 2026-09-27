"""Spline steering (Part 2, Q3 / Q4): edits along the index-9 activation curves (spline paths, chords, the
covariance line, a free-spacing line, and time-structured covariance edits) on Phase 5's steering clips and targets.

Usage: python scripts/check_spline_steering.py <check>
"""
import argparse
import json
import time
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from vjepa_physics.baselines import squared_distances
from vjepa_physics.behavior import N_BINS, behavior_curve, curve_grid, map_probabilities, nearest_on_curve
from vjepa_physics.data import DATASETS
from vjepa_physics.evidence import file_sha256, require_clean_code, save_result, verified_artifact
from vjepa_physics.extraction import SITES, plot_index, pool_time_steps
from vjepa_physics.joined import load_joined
from vjepa_physics.manifolds import PCA_DIMS, fit_curve, fit_spacing_line, smoothing_grid
from vjepa_physics.metrics import bootstrap_indices, percentile_interval
from vjepa_physics.model import load_model, weights_fingerprint
from vjepa_physics.plotting import DATASET_COLOUR, INK, INK_MUTED, INK_SECONDARY, SURFACE, sequential_cmap, style_axes
from vjepa_physics.probes import fit_probe, probe_scores, probe_targets, site_features
from vjepa_physics.reproducibility import SEED, set_seeds
from vjepa_physics.steering import (
    N_CLIPS, PROFILE_SITES, READOUT_ROLES, STEERING_SITE, chord_shifts, covariance_map, covariance_path_shifts,
    label_difference, load_probe_sequence, readout_values, ridge_readout_map, spline_arm_shifts, spline_arm_table,
    steered_features, time_covariance_maps,
)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/spline_steering/checks.json"
ARTIFACTS = REPO / "artifacts/spline_steering"  # regenerable, git-ignored
STEERING_CHECKS = REPO / "results/steering/checks.json"  # Phase 5: steering_setup, steering_cache, steer_* runs
MANIFOLD_CHECKS = REPO / "results/manifolds/checks.json"  # Q1: manifold_loco (curves.npz), manifold_ladder

HALVES = {"seen": "test_seen", "unseen": "test_unseen"}  # Phase 5's saved runs, one per half
EXACT_TOLERANCE = 1e-12  # relative: two computation paths of the same shift

DEVICE = "mps"
FIRST_BLOCK, READOUT_BLOCK = 9, 17  # partial forward: the blocks after the steering site, up to the readout site
# In-memory weights fingerprint right after a clean load (results/model/checks.json, key "load").
REFERENCE_FINGERPRINT = "c865f524c1376e4452943b208d7d50ba588be9490604f235a3d9c9dc80804ede"
SITE_TOLERANCE = 1e-4  # index-9 features vs stored + shift (absolute); fixed on validation clips before any test run
STEP_TOLERANCE = 3e-4  # timed arms: each time step's pooled mean vs stored + that step's shift; fixed likewise

BOOTSTRAP_RESAMPLES = 10_000
MIN_READOUT_TRAIN_R2 = 0.9  # profile readouts, scored on train clips (as Phase 5)
REPRODUCE_TOLERANCE = 1e-12  # relative: Phase 5 arms' gain / reduction vs steering_propagation's saved numbers
DOWNSTREAM = slice(1, len(PROFILE_SITES))  # profile rows of indices 10-18 (pre-declared downstream summary)
C11_MIN_DIFFERENCE = 0.05  # H-13 confirmation rule: interval excludes 0 and (e) - (c) >= 0.05
HEADLINE_PAIRS = {  # name: (arm, reference arm, profile rows)
    "spline_minus_covariance_idx9": ("spline_1", "covariance_1", slice(0, 1)),
    "spline_minus_covariance_idx18": ("spline_1", "covariance_1", slice(9, 10)),
    "time_minus_covariance_downstream": ("time_covariance_1", "covariance_1", DOWNSTREAM),
    "reversed_minus_covariance_downstream": ("time_reversed_1", "covariance_1", DOWNSTREAM),
    "time_minus_reversed_downstream": ("time_covariance_1", "time_reversed_1", DOWNSTREAM),
}

BEHAVIOR_CHECKS = REPO / "results/behavior/checks.json"  # key "behavior_readouts": readouts.npz
NATURALNESS_SITES = ("block_8", "block_17")  # indices 9 (same layer, labelled) and 18 (evidence)
BIMODAL_BINS, BIMODAL_RATIO = 4, 0.5  # pre-stated: two peaks >= 4 bins (90°) apart, the second >= half the first
FAR_DEGREES = 90.0  # direction targets farther than this from the unedited reading = "far"

TABLE = REPO / "results/spline_steering/comparison_table.md"
MANIFOLD_KEYS = ("manifold_loco", "manifold_ladder", "manifold_dimension", "direction_harmonics")
BEHAVIOR_KEYS = ("behavior_readouts", "isometry", "isometry_local")
SPLINE_KEYS = ("spline_setup", "spline_scores", "spline_naturalness", "spline_held_out")
EXPECTED_FAILURES = {"behavior_readouts": {"validation_clips_per_bin_at_least_10"}}  # kept on record (planning chat)

FIGURE_PATHS = REPO / "results/spline_steering/spline_paths.png"
FIGURE_DPI = 200
BIN_CENTRES = (np.arange(N_BINS) * 4 + 1.5) * 5.625  # direction: centre angle (degrees) of each 4-value bin
PATH_FRACTIONS_SHOWN = (0.0, 0.25, 0.5, 0.75, 1.0)

FIGURE_PROFILE = REPO / "results/spline_steering/spline_profile.png"

def kind_of(variable: str) -> str:
    return "loop" if variable == "direction" else "open"


def relative(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(np.asarray(b)).max())


def arm_name(kind: str, fraction: float) -> str:
    return f"{kind}_{fraction:g}"


def check_spline_setup() -> dict:
    """Every Phase 6 edit for Phase 5's steering clips and targets, fixed before any forward pass.

    Per variable: Phase 5's 15 test-seen + 15 test-unseen clips and 5 targets (steering_setup, read through its hash),
    Q1's selected index-9 curve and the free-spacing B line (manifold_loco / manifold_ladder), the covariance map and
    the per-time-step covariance maps from train clips. Per clip and target: all arms of spline_arm_table. Writes
    artifacts/spline_steering/setup.npz. Passes if: clip ids = Phase 5's runs; the spline endpoint reaches the curve's
    target point, the chord's f = 1 equals the spline endpoint and the time-structured shifts average to the
    covariance shift (all within EXACT_TOLERANCE); covariance f = 1 equals Phase 5's saved covariance shift exactly;
    the reversed shift is the flipped one; finite; saved = computed. Records clamped starts, shift lengths / median
    train clip distance, and the spline-vs-covariance endpoint difference (geometry only; no readout).
    """
    start_time = time.perf_counter()
    with np.load(verified_artifact(STEERING_CHECKS, "steering_setup")) as f:
        setup = {key: f[key] for key in f.files}
    with np.load(verified_artifact(MANIFOLD_CHECKS, "manifold_loco")) as f:
        curves = {key: f[key] for key in f.files}
    ladder = json.loads(MANIFOLD_CHECKS.read_text())["manifold_ladder"]["result"]["variables"]

    arrays: dict[str, np.ndarray] = {}
    result: dict = {}
    ok: dict[str, list[bool]] = {name: [] for name in (
        "clip_ids_match_phase5", "covariance_equals_phase5", "reversed_is_flipped", "finite")}
    worst = {"spline_end_hits_target": 0.0, "chord_end_equals_spline_end": 0.0, "time_mean_equals_covariance": 0.0}

    for variable in DATASETS:
        table = load_joined(variable)
        roles, kind = table["role"], kind_of(variable)
        train = roles == "train"
        seq = load_probe_sequence(variable)
        x = site_features(table["activations"], STEERING_SITE)
        z = (x - seq.mean) / seq.scale
        y = probe_targets(variable, table["label"])
        b = covariance_map(z[train], y[train])
        steps = np.asarray(table["activations"][:, SITES.index(STEERING_SITE)], dtype=np.float64)[train]
        maps = time_covariance_maps((steps - seq.mean) / seq.scale, y[train])
        del steps
        prefix = f"{variable}_{STEERING_SITE}"
        values, cents = curves[f"{prefix}_values"], curves[f"{prefix}_centroids"]
        a_i, b_i = (int(i) for i in curves[f"{prefix}_selected"])
        curve = fit_curve(kind, values, cents, PCA_DIMS[a_i], smoothing_grid(kind, values)[b_i])
        line = (fit_spacing_line(values, cents, ladder[variable]["rungs"]["covariance_spacing"]["smooth"])
                if kind == "open" else None)

        phase5 = {}
        for half, role in HALVES.items():
            with np.load(verified_artifact(STEERING_CHECKS, f"steer_{variable}_{half}")) as f:
                phase5[role] = {key: f[key] for key in ("ids", "shifts", "arm_kind")}
            ok["clip_ids_match_phase5"].append(bool(np.array_equal(phase5[role]["ids"], setup[f"{variable}_{role}_ids"])))

        ids = np.concatenate([setup[f"{variable}_{role}_ids"] for role in N_CLIPS])
        targets = setup[f"{variable}_target_label"]
        unseen_target = setup[f"{variable}_target_unseen"].astype(bool)
        uniform_arms, timed_arms = spline_arm_table(variable)
        row_of = {int(i): r for r, i in enumerate(table["id"].tolist())}
        d = x.shape[1]
        uniform = np.empty((len(ids), len(targets), len(uniform_arms), d))
        timed = np.empty((len(ids), len(targets), len(timed_arms), 8, d))
        starts, clamped = np.empty(len(ids)), np.zeros(len(ids), dtype=bool)
        end_difference = np.empty((len(ids), len(targets)))
        spline_end = uniform_arms.index(("spline", 1.0))

        for c, clip_id in enumerate(ids.tolist()):
            row = row_of[clip_id]
            role = "test_seen" if c < N_CLIPS["test_seen"] else "test_unseen"
            c5 = c if role == "test_seen" else c - N_CLIPS["test_seen"]
            covariance_arm = int(np.flatnonzero(phase5[role]["arm_kind"] == "covariance")[0])
            for t, target in enumerate(targets.tolist()):
                s = spline_arm_shifts(seq, b, maps, curve, line, variable, values, x[row], target)
                uniform[c, t], timed[c, t] = s["uniform"], s["timed"]
                starts[c], clamped[c] = s["start"], s["clamped"]
                goal = probe_targets(variable, np.array([target])).reshape(-1)
                end_z = s["uniform"][spline_end] / seq.scale
                target_point = curve(np.array([target]))[0]
                worst["spline_end_hits_target"] = max(
                    worst["spline_end_hits_target"], relative(curve(np.array([s["start"]]))[0] + end_z, target_point))
                worst["chord_end_equals_spline_end"] = max(
                    worst["chord_end_equals_spline_end"], relative(chord_shifts(curve, s["start"], target)[-1], end_z))
                covariance_full = phase5[role]["shifts"][c5, t, covariance_arm]
                ok["covariance_equals_phase5"].append(
                    bool(np.array_equal(covariance_path_shifts(seq, b, x[row], goal)[-1], covariance_full)))
                worst["time_mean_equals_covariance"] = max(
                    worst["time_mean_equals_covariance"],
                    relative(s["timed"][0].mean(axis=0), covariance_full), relative(s["timed"][1].mean(axis=0), covariance_full))
                ok["reversed_is_flipped"].append(bool(np.array_equal(s["timed"][1], s["timed"][0][::-1])))
                covariance_z = covariance_full / seq.scale
                end_difference[c, t] = np.linalg.norm(end_z - covariance_z) / np.linalg.norm(covariance_z)

        distance = float(np.median(np.sqrt(squared_distances(z[train])[np.triu_indices(int(train.sum()), 1)])))
        lengths = {arm_name(k, f): float(np.median(np.linalg.norm(uniform[:, :, a] / seq.scale, axis=-1))) / distance
                   for a, (k, f) in enumerate(uniform_arms)}
        lengths |= {arm_name(k, f) + "_per_step": float(np.median(np.linalg.norm(timed[:, :, a] / seq.scale, axis=-1))) / distance
                    for a, (k, f) in enumerate(timed_arms)}
        quantiles = lambda v: [float(q) for q in np.quantile(v, (0.25, 0.5, 0.75))]
        result[variable] = {
            "n_clips": len(ids), "targets": targets.tolist(), "uniform_arms": [arm_name(*a) for a in uniform_arms],
            "timed_arms": [arm_name(*a) for a in timed_arms],
            "clamped_starts": int(clamped.sum()), "median_train_clip_distance": distance,
            "median_shift_over_clip_distance": lengths,
            "spline_vs_covariance_endpoint_relative": {
                "all": quantiles(end_difference), "seen_targets": quantiles(end_difference[:, ~unseen_target]),
                "unseen_targets": quantiles(end_difference[:, unseen_target])},
        }
        ok["finite"].append(bool(np.isfinite(uniform).all() and np.isfinite(timed).all()
                                 and np.isfinite(end_difference).all()))
        arrays |= {
            f"{variable}_ids": ids, f"{variable}_target_label": targets,
            f"{variable}_uniform_arms": np.array([arm_name(*a) for a in uniform_arms]),
            f"{variable}_timed_arms": np.array([arm_name(*a) for a in timed_arms]),
            f"{variable}_uniform": uniform, f"{variable}_timed": timed,
            f"{variable}_starts": starts, f"{variable}_clamped": clamped,
        }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / "setup.npz"
    np.savez(path, **arrays)
    with np.load(path) as saved:
        saved_ok = set(saved.files) == set(arrays) and all(np.array_equal(saved[k], v) for k, v in arrays.items())

    criteria = {name: all(v) for name, v in ok.items()} | {
        name: value <= EXACT_TOLERANCE for name, value in worst.items()} | {"saved_equals_computed": saved_ok}
    return {
        "variables": result, "worst_relative": worst, "exact_tolerance": EXACT_TOLERANCE,
        "seconds": time.perf_counter() - start_time,
        "criteria": criteria,
        "artifact": {"path": str(path.relative_to(REPO)), "sha256": file_sha256(path)},
        "passed": all(criteria.values()),
    }

def count_forward_hooks(model: torch.nn.Module) -> int:
    return sum(len(module._forward_hooks) for module in model.modules())


def per_step_difference(cached: torch.Tensor, delta_steps: np.ndarray, stored_steps: np.ndarray) -> float:
    """Max |pooled per-step mean of the edited site - (stored per-step mean + that step's shift)|."""
    delta = torch.from_numpy(delta_steps.astype(np.float32)).to(cached.device)
    tokens = torch.repeat_interleave(delta, cached.shape[1] // delta.shape[0], dim=0)
    pooled = np.asarray(pool_time_steps(cached + tokens).to("cpu")[0], dtype=np.float64)
    return float(np.abs(pooled - (stored_steps + delta_steps)).max())


def spline_run(variable: str, half: str) -> dict:
    """Steer one variable's test-seen or test-unseen clips with every Phase 6 arm, from Phase 5's cache.

    Per clip: one unedited partial pass, then per target every arm of spline_arm_table (shifts from spline_setup) and
    one replication of Phase 5's covariance arm (f = 1, its saved shift). Saves fp32 features at indices 9-18 per arm
    and the unedited features to artifacts/spline_steering/runs_<variable>_<half>.npz. Passes if: setup, Phase 5's
    run and cache are hash-verified with the same clip ids; every unedited pass = the stored features at index 9 and
    18 and = Phase 5's saved unedited features, bit for bit; every covariance replication = Phase 5's saved steered
    features at indices 9-18, bit for bit; site difference <= SITE_TOLERANCE; per-step difference <= STEP_TOLERANCE;
    all outputs finite; hooks and weights unchanged; saved file = computed.
    """
    role = HALVES[half]
    with np.load(verified_artifact(OUT, "spline_setup")) as f:
        setup = {key: f[key] for key in f.files if key.startswith(f"{variable}_")}
    with np.load(verified_artifact(STEERING_CHECKS, f"steer_{variable}_{half}")) as f:
        phase5 = {key: f[key] for key in ("ids", "arm_kind", "shifts", "features", "unedited_features")}
    cache = np.load(verified_artifact(STEERING_CHECKS, "steering_cache", f"cache_{variable}"), mmap_mode="r")
    order = list(N_CLIPS)
    first_row = sum(N_CLIPS[r] for r in order[: order.index(role)])  # rows: test_seen, then test_unseen
    n = N_CLIPS[role]
    ids = setup[f"{variable}_ids"][first_row:first_row + n]
    uniform = setup[f"{variable}_uniform"][first_row:first_row + n]
    timed = setup[f"{variable}_timed"][first_row:first_row + n]
    targets = setup[f"{variable}_target_label"]
    covariance_arm = int(np.flatnonzero(phase5["arm_kind"] == "covariance")[0])

    table = load_joined(variable)
    row_of = {int(i): r for r, i in enumerate(table["id"].tolist())}
    x9 = site_features(table["activations"], STEERING_SITE)
    x18 = site_features(table["activations"], f"block_{READOUT_BLOCK}")
    steps = table["activations"][:, SITES.index(STEERING_SITE)]
    d, n_rows, n_targets = x9.shape[1], READOUT_BLOCK - FIRST_BLOCK + 2, len(targets)
    features_uniform = np.empty((n, n_targets, uniform.shape[2], n_rows, d), dtype=np.float32)
    features_timed = np.empty((n, n_targets, timed.shape[2], n_rows, d), dtype=np.float32)
    features_covariance = np.empty((n, n_targets, n_rows, d), dtype=np.float32)
    unedited = np.empty((n, n_rows, d), dtype=np.float32)
    ok: dict[str, list[bool]] = {name: [] for name in (
        "unedited_equals_stored", "unedited_equals_phase5", "covariance_replicates_phase5", "site", "per_step", "finite")}
    max_site, max_step = 0.0, 0.0

    set_seeds()
    model, _ = load_model(DEVICE)
    hooks_before = count_forward_hooks(model)
    start = time.perf_counter()
    for c, clip_id in enumerate(ids.tolist()):
        row = row_of[clip_id]
        cached = torch.from_numpy(np.array(cache[first_row + c]))[None].to(DEVICE)
        base = steered_features(model, cached, np.zeros((1, d)), FIRST_BLOCK, READOUT_BLOCK)[0]
        ok["unedited_equals_stored"].append(bool(np.array_equal(base[0], x9[row]) and np.array_equal(base[-1], x18[row])))
        ok["unedited_equals_phase5"].append(bool(np.array_equal(base.astype(np.float32), phase5["unedited_features"][c])))
        unedited[c] = base
        stored_steps = np.asarray(steps[row], dtype=np.float64)
        for t in range(n_targets):
            fu = steered_features(model, cached, uniform[c, t], FIRST_BLOCK, READOUT_BLOCK)
            ft = steered_features(model, cached, timed[c, t], FIRST_BLOCK, READOUT_BLOCK)
            fc = steered_features(model, cached, phase5["shifts"][c, t, covariance_arm][None], FIRST_BLOCK, READOUT_BLOCK)[0]
            ok["covariance_replicates_phase5"].append(
                bool(np.array_equal(fc.astype(np.float32), phase5["features"][c, t, covariance_arm])))
            site = max(float(np.abs(fu[:, 0] - (x9[row] + uniform[c, t])).max()),
                       float(np.abs(ft[:, 0] - (x9[row] + timed[c, t].mean(axis=1))).max()))
            step = max(per_step_difference(cached, delta, stored_steps) for delta in timed[c, t])
            max_site, max_step = max(max_site, site), max(max_step, step)
            ok["site"].append(site <= SITE_TOLERANCE)
            ok["per_step"].append(step <= STEP_TOLERANCE)
            ok["finite"].append(bool(np.isfinite(fu).all() and np.isfinite(ft).all() and np.isfinite(fc).all()))
            features_uniform[c, t], features_timed[c, t], features_covariance[c, t] = fu, ft, fc
        print(f"{variable} {half}: clip {c + 1}/{n} done, {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    seconds = time.perf_counter() - start
    hooks_after = count_forward_hooks(model)
    fingerprint = weights_fingerprint(model)

    arrays = {
        "ids": ids, "target_label": targets,
        "uniform_arms": setup[f"{variable}_uniform_arms"], "timed_arms": setup[f"{variable}_timed_arms"],
        "features_uniform": features_uniform, "features_timed": features_timed,
        "features_covariance_1": features_covariance, "unedited_features": unedited,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / f"runs_{variable}_{half}.npz"
    np.savez(path, **arrays)
    with np.load(path) as saved:
        saved_ok = set(saved.files) == set(arrays) and all(np.array_equal(saved[k], v) for k, v in arrays.items())

    criteria = {name: all(values) for name, values in ok.items()} | {
        "same_clip_ids": bool(np.array_equal(ids, phase5["ids"])),
        "hooks_unchanged": hooks_before == hooks_after,
        "weights_unchanged": fingerprint == REFERENCE_FINGERPRINT,
        "saved_equals_computed": saved_ok,
    }
    return {
        "variable": variable, "role": role, "ids": ids.tolist(), "targets": targets.tolist(),
        "uniform_arms": arrays["uniform_arms"].tolist(), "timed_arms": arrays["timed_arms"].tolist(),
        "runs": n * (n_targets * (uniform.shape[2] + timed.shape[2] + 1) + 1), "seconds": seconds,
        "tolerances": {"site": SITE_TOLERANCE, "per_step": STEP_TOLERANCE},
        "max_site_diff": max_site, "max_per_step_diff": max_step,
        "forward_hooks_before_and_after": [hooks_before, hooks_after], "weights_fingerprint": fingerprint,
        "criteria": criteria,
        "artifact": {"path": str(path.relative_to(REPO)), "sha256": file_sha256(path)},
        "passed": all(criteria.values()),
    }
    

def joined_runs(checks: Path, key_format: str, variable: str, names: tuple[str, ...]) -> list[dict]:
    """Both halves of one variable's runs, read through their hashes (test_seen first)."""
    halves = []
    for half in HALVES:
        with np.load(verified_artifact(checks, key_format.format(variable=variable, half=half))) as f:
            halves.append({name: f[name] for name in names})
    return halves


def check_spline_scores() -> dict:
    """Progress scores of every Phase 6 arm and of Phase 5's arms over indices 9-18 (no model; saved runs only).

    Readouts: Phase 5's validation-fit ridge readouts (D-42 grid, LOO) at every index 9-18. Per arm and index: output-
    space gain sum(achieved . intended) / sum(|intended|²), intended = f x (target output - unedited output); direction
    also angle gain (intended f x the shorter-arc angle to the target); error reduction for f = 1 arms. Summary per arm:
    gain at index 9 and 18, mean gain over 10-18 (pre-declared), seen / unseen targets. Headline pairs with paired
    clip-bootstrap intervals; C11 rule. Passes if: every readout has an interior alpha and train R² >=
    MIN_READOUT_TRAIN_R2; Phase 6 and Phase 5 runs hold the same clips; Phase 5 arms reproduce steering_propagation's
    saved gain and reduction within REPRODUCE_TOLERANCE; everything finite.
    """
    propagation = json.loads(STEERING_CHECKS.read_text())["steering_propagation"]["result"]["variables"]
    with np.load(verified_artifact(STEERING_CHECKS, "steering_setup")) as f:
        steering_setup = {key: f[key] for key in f.files}
    boot = bootstrap_indices(sum(N_CLIPS.values()), BOOTSTRAP_RESAMPLES, SEED)
    ok: dict[str, list[bool]] = {name: [] for name in ("readout_alpha", "readout_quality", "same_ids", "finite")}
    worst_reproduction = 0.0
    result: dict = {}

    for variable in DATASETS:
        table = load_joined(variable)
        roles, labels = table["role"], table["label"]
        train = roles == "train"
        y = probe_targets(variable, labels)
        six = joined_runs(OUT, "spline_{variable}_{half}", variable, (
            "ids", "target_label", "uniform_arms", "timed_arms", "features_uniform", "features_timed",
            "features_covariance_1", "unedited_features"))
        five = joined_runs(STEERING_CHECKS, "steer_{variable}_{half}", variable, (
            "ids", "arm_kind", "arm_n", "features", "unedited_features"))
        ids = np.concatenate([h["ids"] for h in six])
        ok["same_ids"].append(bool(np.array_equal(ids, np.concatenate([h["ids"] for h in five]))))
        targets = six[0]["target_label"]
        unseen = steering_setup[f"{variable}_target_unseen"].astype(bool)
        target_out = probe_targets(variable, targets).reshape(len(targets), -1)
        unedited = np.concatenate([h["unedited_features"] for h in six])  # (N, 10, d) fp32

        arms: dict[str, tuple[np.ndarray, float]] = {}  # name: ((N, T, S, 10, d) features, fraction)
        uniform = np.concatenate([h["features_uniform"] for h in six])
        for a, name in enumerate(six[0]["uniform_arms"].tolist()):
            arms[name] = (uniform[:, :, a:a + 1], float(name.rsplit("_", 1)[1]))
        timed = np.concatenate([h["features_timed"] for h in six])
        for a, name in enumerate(six[0]["timed_arms"].tolist()):
            arms[name] = (timed[:, :, a:a + 1], 1.0)
        arms["covariance_1"] = (np.concatenate([h["features_covariance_1"] for h in six])[:, :, None], 1.0)
        features5 = np.concatenate([h["features"] for h in five])
        groups: dict[str, list[int]] = {}
        for a, (kind, n) in enumerate(zip(five[0]["arm_kind"].tolist(), five[0]["arm_n"].tolist())):
            groups.setdefault("covariance" if kind == "covariance" else f"{kind}_{n}", []).append(a)
        for group, idx in groups.items():
            arms[f"phase5_{group}"] = (features5[:, :, idx], 1.0)

        n_clips, n_targets, n_sites = len(ids), len(targets), len(PROFILE_SITES)
        num = {name: np.empty((n_clips, n_targets, n_sites)) for name in arms}
        den = {name: np.empty((n_clips, n_targets, n_sites)) for name in arms}
        profile: dict = {}
        for j, site in enumerate(PROFILE_SITES):
            x = site_features(table["activations"], site)
            probe = fit_probe(x, y, roles, fit_roles=READOUT_ROLES)
            weights, offset = ridge_readout_map(probe)
            train_pred = probe.predict(x[train]).reshape(int(train.sum()), -1)
            score = probe_scores(variable, labels[train], train_pred[:, 0] if train_pred.shape[1] == 1 else train_pred)
            ok["readout_alpha"].append(probe.alpha_edge is None)
            ok["readout_quality"].append(score["r2"] >= MIN_READOUT_TRAIN_R2)

            out0 = unedited[:, j].astype(np.float64) @ weights + offset  # (N, m)
            intended = target_out[None, :, :] - out0[:, None, :]  # (N, T, m)
            value0 = readout_values(variable, out0)
            unsteered = np.abs(label_difference(variable, value0[:, None], targets[None, :]))
            record: dict = {}
            for name, (feats, fraction) in arms.items():
                out = feats[:, :, :, j].astype(np.float64) @ weights + offset  # (N, T, S, m)
                achieved = out - out0[:, None, None, :]
                i = np.broadcast_to(fraction * intended[:, :, None, :], achieved.shape)
                num[name][:, :, j] = (achieved * i).sum(axis=(2, 3))
                den[name][:, :, j] = (i**2).sum(axis=(2, 3))
                rec = {"gain": float(num[name][:, :, j].sum() / den[name][:, :, j].sum())}
                if fraction == 1.0:
                    err = np.abs(label_difference(variable, readout_values(variable, out), targets[None, :, None]))
                    rec["reduction"] = float(1 - err.mean() / unsteered.mean())
                if variable == "direction":
                    moved = label_difference(variable, readout_values(variable, out), value0[:, None, None])
                    aim = np.broadcast_to(
                        fraction * label_difference(variable, targets[None, :], value0[:, None])[:, :, None], moved.shape)
                    rec["angle_gain"] = float((moved * aim).sum() / (aim**2).sum())
                if name.startswith("phase5_"):
                    saved = propagation[variable]["profile"][str(plot_index(site))]["arms"][name.removeprefix("phase5_")]
                    for key in ("gain", "reduction"):
                        worst_reproduction = max(worst_reproduction, abs(rec[key] - saved[key]) / max(1.0, abs(saved[key])))
                record[name] = rec
            profile[str(plot_index(site))] = {"alpha": probe.alpha, "train_scores": score, "arms": record}

        def gains(name: str, mask: np.ndarray | None = None) -> np.ndarray:
            m = np.ones(n_targets, dtype=bool) if mask is None else mask
            return num[name][:, m].sum(axis=(0, 1)) / den[name][:, m].sum(axis=(0, 1))

        def resampled(name: str) -> np.ndarray:
            return num[name].sum(axis=1)[boot].sum(axis=1) / den[name].sum(axis=1)[boot].sum(axis=1)  # (R, sites)

        summary = {}
        for name in arms:
            g, g_seen, g_unseen = gains(name), gains(name, ~unseen), gains(name, unseen)
            summary[name] = {
                "idx9": float(g[0]), "idx18": float(g[-1]), "downstream_mean": float(g[DOWNSTREAM].mean()),
                "seen_targets": {"idx9": float(g_seen[0]), "idx18": float(g_seen[-1]),
                                 "downstream_mean": float(g_seen[DOWNSTREAM].mean())},
                "unseen_targets": {"idx9": float(g_unseen[0]), "idx18": float(g_unseen[-1]),
                                   "downstream_mean": float(g_unseen[DOWNSTREAM].mean())},
            }
        headline = {}
        for pair, (arm, reference, rows) in HEADLINE_PAIRS.items():
            point = float(gains(arm)[rows].mean() - gains(reference)[rows].mean())
            samples = resampled(arm)[:, rows].mean(axis=1) - resampled(reference)[:, rows].mean(axis=1)
            headline[pair] = {"arm": arm, "reference": reference, "point": point,
                              "ci": list(percentile_interval(samples))}
        h13 = headline["time_minus_covariance_downstream"]
        c11 = {"triggered": bool((h13["ci"][0] > 0 or h13["ci"][1] < 0) and h13["point"] >= C11_MIN_DIFFERENCE),
               "rule": "interval of (e) - (c) mean gain over indices 10-18 excludes 0 and the difference is >= 0.05"}
        record = {"n_clips": n_clips, "targets": targets.tolist(), "unseen_targets": unseen.tolist(),
                  "profile": profile, "summary": summary, "headline": headline, "c11": c11}
        ok["finite"].append("NaN" not in json.dumps(record) and "Infinity" not in json.dumps(record))
        result[variable] = record

    criteria = {name: all(v) for name, v in ok.items()} | {"phase5_reproduced": worst_reproduction <= REPRODUCE_TOLERANCE}
    return {
        "variables": result, "profile_sites": list(PROFILE_SITES), "readout_roles": list(READOUT_ROLES),
        "gain_definition": "sum(achieved . intended) / sum(|intended|^2), intended = f x (target output - unedited output)",
        "downstream_summary": "mean gain over indices 10-18", "worst_phase5_reproduction": worst_reproduction,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "criteria": criteria, "passed": all(criteria.values()),
    }
    

def run_arms(variable: str) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Features of every Phase 6 arm (+ covariance f = 1 replication) and of Phase 5's K - 1 probe, covariance and
    K - 1 random arms, both halves joined: {name: ((N, T, S, 10, d) fp32, fraction)}, ids, targets, unedited."""
    six = joined_runs(OUT, "spline_{variable}_{half}", variable, (
        "ids", "target_label", "uniform_arms", "timed_arms", "features_uniform", "features_timed",
        "features_covariance_1", "unedited_features"))
    five = joined_runs(STEERING_CHECKS, "steer_{variable}_{half}", variable, ("ids", "arm_kind", "arm_n", "features"))
    arms: dict[str, tuple[np.ndarray, float]] = {}
    uniform = np.concatenate([h["features_uniform"] for h in six])
    for a, name in enumerate(six[0]["uniform_arms"].tolist()):
        arms[name] = (uniform[:, :, a:a + 1], float(name.rsplit("_", 1)[1]))
    timed = np.concatenate([h["features_timed"] for h in six])
    for a, name in enumerate(six[0]["timed_arms"].tolist()):
        arms[name] = (timed[:, :, a:a + 1], 1.0)
    arms["covariance_1"] = (np.concatenate([h["features_covariance_1"] for h in six])[:, :, None], 1.0)
    k = load_probe_sequence(variable).k
    features5 = np.concatenate([h["features"] for h in five])
    kinds, counts = five[0]["arm_kind"], five[0]["arm_n"]
    for name, mask in ((f"phase5_probes_{k - 1}", (kinds == "probes") & (counts == k - 1)),
                       ("phase5_covariance", kinds == "covariance"),
                       (f"phase5_random_{k - 1}", (kinds == "random") & (counts == k - 1))):
        arms[name] = (features5[:, :, np.flatnonzero(mask)], 1.0)
    ids = np.concatenate([h["ids"] for h in six])
    same = np.array_equal(ids, np.concatenate([h["ids"] for h in five]))
    return arms, ids if same else None, six[0]["target_label"], np.concatenate([h["unedited_features"] for h in six])


def is_bimodal(p: np.ndarray) -> bool:
    """Pre-stated rule on a 16-bin distribution (circular): the highest peak and the largest peak >= BIMODAL_BINS bins
    from it; bimodal if that second peak is >= BIMODAL_RATIO x the highest."""
    peaks = np.flatnonzero((p >= np.roll(p, 1)) & (p > np.roll(p, -1)))
    if len(peaks) < 2:
        return False
    order = peaks[np.argsort(p[peaks])[::-1]]
    for q in order[1:]:
        if min((q - order[0]) % N_BINS, (order[0] - q) % N_BINS) >= BIMODAL_BINS:
            return bool(p[q] >= BIMODAL_RATIO * p[order[0]])
    return False


def check_spline_naturalness() -> dict:
    """Naturalness of every arm at indices 9 and 18 under the 16-bin behavior readout (A), paired with progress.

    Per arm: the edited clip's Hellinger distance to the behavior curve minus the same clip's unedited distance
    (excess), its percentile among natural train clips' distances, behavior progress (gain of the curve's nearest
    value toward f x (target - unedited nearest value)), mean entropy; direction also the bimodal fraction (pre-stated
    rule) and the ridge (sin, cos) readout length. Headline: midpoint excess spline f = 0.5 - chord f = 0.5 with a
    paired clip-bootstrap interval (direction also for near / far targets). Speed / acceleration carry the blurry-
    readout label. Passes if: the saved readouts and curves reproduce the saved natural train distances within
    EXACT_TOLERANCE; Phase 6 and Phase 5 runs hold the same clips; probabilities sum to 1; everything finite.
    """
    with np.load(verified_artifact(BEHAVIOR_CHECKS, "behavior_readouts")) as f:
        readouts = {key: f[key] for key in f.files}
    boot = bootstrap_indices(sum(N_CLIPS.values()), BOOTSTRAP_RESAMPLES, SEED)
    ok: dict[str, list[bool]] = {name: [] for name in ("same_ids", "sums_to_one", "finite")}
    worst_natural = 0.0
    result: dict = {}

    for variable in DATASETS:
        table = load_joined(variable)
        roles, labels, kind = table["role"], table["label"], kind_of(variable)
        train = roles == "train"
        arms, ids, targets, unedited = run_arms(variable)
        ok["same_ids"].append(ids is not None)
        record: dict = {"blurry_readout": variable != "direction", "sites": {}, "headline": {}}
        for site in NATURALNESS_SITES:
            j = PROFILE_SITES.index(site)
            weights, offset = readouts[f"{variable}_{site}_weights"], readouts[f"{variable}_{site}_offset"]
            values = readouts[f"{variable}_{site}_curve_values"]
            curve = behavior_curve(kind, values, readouts[f"{variable}_{site}_curve_sqrt_centroids"])
            grid = curve_grid(kind, values)
            saved_natural = readouts[f"{variable}_{site}_natural_train_distance"]
            natural_now, _ = nearest_on_curve(
                np.sqrt(map_probabilities(weights, offset, site_features(table["activations"], site)[train])), curve, grid)
            worst_natural = max(worst_natural, relative(natural_now, saved_natural))
            natural = np.sort(saved_natural)

            p0 = map_probabilities(weights, offset, unedited[:, j].astype(np.float64))
            d0, v0 = nearest_on_curve(np.sqrt(p0), curve, grid)
            to_target = label_difference(variable, targets[None, :], v0[:, None])  # (N, T)
            far = np.abs(to_target) > FAR_DEGREES if variable == "direction" else None
            site_record: dict = {"unedited": {
                "distance_median": float(np.median(d0)),
                "natural_percentile_median": float(np.median(np.searchsorted(natural, d0, side="right") / len(natural)))}}
            if variable == "direction":
                probe = fit_probe(site_features(table["activations"], site), probe_targets(variable, labels), roles,
                                  fit_roles=READOUT_ROLES)
                ridge_w, ridge_o = ridge_readout_map(probe)
                site_record["unedited"]["readout_length_mean"] = float(
                    np.linalg.norm(unedited[:, j].astype(np.float64) @ ridge_w + ridge_o, axis=-1).mean())

            per_clip = {}
            for name, (feats, fraction) in arms.items():
                f = feats[:, :, :, j].astype(np.float64)
                shape = f.shape[:3]
                p = map_probabilities(weights, offset, f.reshape(-1, f.shape[-1]))
                ok["sums_to_one"].append(float(np.abs(p.sum(axis=1) - 1.0).max()) <= 1e-10)
                dist, near = nearest_on_curve(np.sqrt(p), curve, grid)
                dist, near, p = dist.reshape(shape), near.reshape(shape), p.reshape(*shape, N_BINS)
                excess = dist - d0[:, None, None]
                percentile = np.searchsorted(natural, dist, side="right") / len(natural)
                moved = label_difference(variable, near, v0[:, None, None])
                aim = np.broadcast_to(fraction * to_target[:, :, None], moved.shape)
                entropy = -(p * np.log(np.clip(p, 1e-300, None))).sum(axis=-1)
                rec = {
                    "excess_hellinger_mean": float(excess.mean()), "excess_hellinger_median": float(np.median(excess)),
                    "natural_percentile_median": float(np.median(percentile)),
                    "share_above_natural_95": float((percentile > 0.95).mean()),
                    "behavior_gain": float((moved * aim).sum() / (aim**2).sum()),
                    "entropy_mean": float(entropy.mean()),
                }
                if variable == "direction":
                    bimodal = np.array([is_bimodal(q) for q in p.reshape(-1, N_BINS)]).reshape(shape)
                    rec["bimodal_share"] = float(bimodal.mean())
                    rec["bimodal_share_far_targets"] = float(bimodal[far].mean())
                    rec["readout_length_mean"] = float(np.linalg.norm(f @ ridge_w + ridge_o, axis=-1).mean())
                    rec["excess_far_targets"] = float(excess[far].mean())
                    rec["excess_near_targets"] = float(excess[~far].mean())
                site_record[name] = rec
                per_clip[name] = excess

            headline = {}
            subsets = {"all": np.ones_like(to_target, dtype=bool)}
            if far is not None:
                subsets |= {"far_targets": far, "near_targets": ~far}
            for subset, mask in subsets.items():
                diff = np.where(mask, (per_clip["spline_0.5"] - per_clip["chord_0.5"])[:, :, 0], 0.0)
                n = mask.sum(axis=1)
                point = float(diff.sum() / n.sum())
                samples = diff.sum(axis=1)[boot].sum(axis=1) / n[boot].sum(axis=1)
                headline[subset] = {"point": point, "ci": list(percentile_interval(samples))}
            record["sites"][str(plot_index(site))] = site_record
            record["headline"][f"midpoint_spline_minus_chord_idx{plot_index(site)}"] = headline
        ok["finite"].append("NaN" not in json.dumps(record) and "Infinity" not in json.dumps(record))
        result[variable] = record

    criteria = {name: all(v) for name, v in ok.items()} | {"natural_distances_reproduced": worst_natural <= EXACT_TOLERANCE}
    return {
        "variables": result, "sites": list(NATURALNESS_SITES), "worst_natural_reproduction": worst_natural,
        "bimodal_rule": f"two peaks >= {BIMODAL_BINS} bins apart, the second >= {BIMODAL_RATIO} x the first",
        "notes": ["excess = edited distance to the behavior curve - the same clip's unedited distance (Hellinger)",
                  "speed / acceleration: blurry readout (top-1 0.28 / 0.26, within one bin 0.72 / 0.65)",
                  "index 9 is the same-layer readout (labelled); index 18 is the evidence"],
        "criteria": criteria, "passed": all(criteria.values()),
    }


def held_out_arms(variable: str) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """The spline endpoint, the covariance line (f = 1) and Phase 5's K - 1 probes from run_arms."""
    arms, ids, targets, unedited = run_arms(variable)
    k = load_probe_sequence(variable).k
    keep = ("spline_1", "covariance_1", f"phase5_probes_{k - 1}")
    return {name: arms[name] for name in keep}, ids, targets, unedited


def progress_terms(variable: str, table: dict, arms: dict, unedited: np.ndarray, targets: np.ndarray):
    """Per arm: gain numerator and denominator per (clip, target, profile index), as in spline_scores."""
    roles, labels = table["role"], table["label"]
    y = probe_targets(variable, labels)
    target_out = probe_targets(variable, targets).reshape(len(targets), -1)
    shape = (len(unedited), len(targets), len(PROFILE_SITES))
    num = {name: np.empty(shape) for name in arms}
    den = {name: np.empty(shape) for name in arms}
    for j, site in enumerate(PROFILE_SITES):
        probe = fit_probe(site_features(table["activations"], site), y, roles, fit_roles=READOUT_ROLES)
        weights, offset = ridge_readout_map(probe)
        out0 = unedited[:, j].astype(np.float64) @ weights + offset
        intended = target_out[None, :, :] - out0[:, None, :]
        for name, (feats, fraction) in arms.items():
            achieved = feats[:, :, :, j].astype(np.float64) @ weights + offset - out0[:, None, None, :]
            i = np.broadcast_to(fraction * intended[:, :, None, :], achieved.shape)
            num[name][:, :, j] = (achieved * i).sum(axis=(2, 3))
            den[name][:, :, j] = (i**2).sum(axis=(2, 3))
    return num, den


def naturalness_terms(variable: str, readouts: dict, arms: dict, unedited: np.ndarray) -> dict:
    """Per site and arm: (clip, target) excess Hellinger (mean over seeds), as in spline_naturalness."""
    kind = kind_of(variable)
    out: dict = {}
    for site in NATURALNESS_SITES:
        j = PROFILE_SITES.index(site)
        weights, offset = readouts[f"{variable}_{site}_weights"], readouts[f"{variable}_{site}_offset"]
        values = readouts[f"{variable}_{site}_curve_values"]
        curve = behavior_curve(kind, values, readouts[f"{variable}_{site}_curve_sqrt_centroids"])
        grid = curve_grid(kind, values)
        d0, _ = nearest_on_curve(np.sqrt(map_probabilities(weights, offset, unedited[:, j].astype(np.float64))), curve, grid)
        out[site] = {}
        for name, (feats, _) in arms.items():
            f = feats[:, :, :, j].astype(np.float64)
            dist, _ = nearest_on_curve(np.sqrt(map_probabilities(weights, offset, f.reshape(-1, f.shape[-1]))), curve, grid)
            out[site][name] = (dist.reshape(f.shape[:3]) - d0[:, None, None]).mean(axis=2)
    return out


def check_spline_held_out() -> dict:
    """C7: progress and naturalness of the spline endpoint, the covariance line and K - 1 probes on held-out targets
    and clips (no model; saved runs only).

    Subsets of (clip, target): all; seen / test-unseen target values (unseen = no centroid in the curve fit);
    test-seen / test-unseen clips; unseen clip x unseen target. Per subset and arm: ridge gain at index 9, 18 and mean
    over 10-18 (spline_scores' readouts), excess Hellinger at index 9 and 18 (spline_naturalness' readout A). Paired
    clip-bootstrap intervals on the unseen targets: spline - covariance and spline - probes (gain at 18, downstream
    mean, excess at 9). Passes if: the all-target numbers reproduce spline_scores and spline_naturalness within
    REPRODUCE_TOLERANCE; same clip ids; everything finite.
    """
    saved = json.loads(OUT.read_text())
    scores = saved["spline_scores"]["result"]["variables"]
    natural = saved["spline_naturalness"]["result"]["variables"]
    with np.load(verified_artifact(BEHAVIOR_CHECKS, "behavior_readouts")) as f:
        readouts = {key: f[key] for key in f.files}
    with np.load(verified_artifact(STEERING_CHECKS, "steering_setup")) as f:
        steering_setup = {key: f[key] for key in f.files}
    boot = bootstrap_indices(sum(N_CLIPS.values()), BOOTSTRAP_RESAMPLES, SEED)
    ok: dict[str, list[bool]] = {"same_ids": [], "finite": []}
    worst = 0.0
    result: dict = {}

    for variable in DATASETS:
        table = load_joined(variable)
        arms, ids, targets, unedited = held_out_arms(variable)
        ok["same_ids"].append(ids is not None)
        n_clips, n_targets = len(unedited), len(targets)
        num, den = progress_terms(variable, table, arms, unedited, targets)
        excess = naturalness_terms(variable, readouts, arms, unedited)

        for name in arms:  # reproduction of the saved all-target numbers
            g = num[name].sum(axis=(0, 1)) / den[name].sum(axis=(0, 1))
            s = scores[variable]["summary"][name]
            for mine, theirs in ((g[0], s["idx9"]), (g[-1], s["idx18"]), (g[DOWNSTREAM].mean(), s["downstream_mean"])):
                worst = max(worst, abs(float(mine) - theirs) / max(1.0, abs(theirs)))
            for site in NATURALNESS_SITES:
                theirs = natural[variable]["sites"][str(plot_index(site))][name]["excess_hellinger_mean"]
                worst = max(worst, abs(float(excess[site][name].mean()) - theirs) / max(1.0, abs(theirs)))

        unseen_target = np.broadcast_to(steering_setup[f"{variable}_target_unseen"].astype(bool)[None, :], (n_clips, n_targets))
        unseen_clip = np.broadcast_to((np.arange(n_clips) >= N_CLIPS["test_seen"])[:, None], (n_clips, n_targets))
        subsets = {"all": np.ones((n_clips, n_targets), dtype=bool), "seen_targets": ~unseen_target,
                   "unseen_targets": unseen_target, "seen_clips": ~unseen_clip, "unseen_clips": unseen_clip,
                   "unseen_clip_and_target": unseen_clip & unseen_target}

        def gain(name: str, mask: np.ndarray) -> np.ndarray:
            return (np.where(mask[..., None], num[name], 0).sum(axis=(0, 1))
                    / np.where(mask[..., None], den[name], 0).sum(axis=(0, 1)))

        def mean_excess(site: str, name: str, mask: np.ndarray) -> float:
            return float(excess[site][name][mask].mean())

        record: dict = {"subsets": {}, "paired_unseen_targets": {}}
        for subset, mask in subsets.items():
            record["subsets"][subset] = {"n_pairs": int(mask.sum()), "arms": {name: {
                "gain_idx9": float(gain(name, mask)[0]), "gain_idx18": float(gain(name, mask)[-1]),
                "gain_downstream_mean": float(gain(name, mask)[DOWNSTREAM].mean()),
                "excess_idx9": mean_excess("block_8", name, mask), "excess_idx18": mean_excess("block_17", name, mask),
            } for name in arms}}

        mask = subsets["unseen_targets"]
        spline, references = "spline_1", [a for a in arms if a != "spline_1"]
        for reference in references:
            pair: dict = {}
            for label, rows in (("gain_idx18", slice(9, 10)), ("gain_downstream_mean", DOWNSTREAM)):
                def resampled(name: str) -> np.ndarray:
                    n_c = np.where(mask[..., None], num[name], 0).sum(axis=1)
                    d_c = np.where(mask[..., None], den[name], 0).sum(axis=1)
                    return (n_c[boot].sum(axis=1) / d_c[boot].sum(axis=1))[:, rows].mean(axis=1)
                point = float(gain(spline, mask)[rows].mean() - gain(reference, mask)[rows].mean())
                pair[label] = {"point": point, "ci": list(percentile_interval(resampled(spline) - resampled(reference)))}
            count = mask.sum(axis=1)
            e_c = {name: np.where(mask, excess["block_8"][name], 0).sum(axis=1) for name in (spline, reference)}
            samples = (e_c[spline] - e_c[reference])[boot].sum(axis=1) / count[boot].sum(axis=1)
            pair["excess_idx9"] = {"point": float((e_c[spline] - e_c[reference]).sum() / count.sum()),
                                   "ci": list(percentile_interval(samples))}
            record["paired_unseen_targets"][f"spline_1_minus_{reference}"] = pair
        ok["finite"].append("NaN" not in json.dumps(record) and "Infinity" not in json.dumps(record))
        result[variable] = record

    criteria = {name: all(v) for name, v in ok.items()} | {"reproduces_saved": worst <= REPRODUCE_TOLERANCE}
    return {"variables": result, "worst_reproduction": worst,
            "note": "unseen targets are values with no centroid in the curve fit (interpolation); index 9 = same layer",
            "criteria": criteria, "passed": all(criteria.values())}
    

def fmt(x: float, digits: int = 3) -> str:
    s = f"{x:.{digits}f}"
    return s.lstrip("-") if float(s) == 0 else s  # no "-0.000"


def fmt_ci(ci: list, digits: int = 3) -> str:
    return f"[{fmt(ci[0], digits)}, {fmt(ci[1], digits)}]"


def probe_shift_ratio(variable: str, distance: float) -> float:
    """Median |shift_z| / median train clip distance of Phase 5's K - 1 probe edits (both halves, saved shifts)."""
    seq = load_probe_sequence(variable)
    ratios = []
    for half in HALVES:
        with np.load(verified_artifact(STEERING_CHECKS, f"steer_{variable}_{half}")) as f:
            kinds, counts, shifts = f["arm_kind"], f["arm_n"], f["shifts"]
        a = int(np.flatnonzero((kinds == "probes") & (counts == seq.k - 1))[0])
        ratios.append(np.linalg.norm(shifts[:, :, a] / seq.scale, axis=-1).ravel())
    return float(np.median(np.concatenate(ratios))) / distance


def check_comparison_table() -> dict:
    """The 6.12-6.14 table (one Markdown table, three blocks) from saved results only.

    Representation (Q1 ladder, H-01 dimension, H-02 harmonics, Q2 isometry), steering (Q3: edit size, progress over
    indices 9-18, naturalness, held-out targets; random arm as floor) and token structure (Q4). Writes
    results/spline_steering/comparison_table.md. Passes if: every source key exists and was saved from clean code;
    no source failed except behavior_readouts' recorded bin-count criterion; computed values finite; saved = rendered.
    """
    sources: dict[str, dict] = {}
    ok: dict[str, list[bool]] = {"sources_clean": [], "sources_passed_or_recorded": []}
    for checks, keys in ((MANIFOLD_CHECKS, MANIFOLD_KEYS), (BEHAVIOR_CHECKS, BEHAVIOR_KEYS), (OUT, SPLINE_KEYS)):
        saved = json.loads(checks.read_text())
        for key in keys:
            entry = saved[key]
            sources[key] = {"result": entry["result"], "commit": entry["provenance"]["git_commit"]}
            ok["sources_clean"].append(entry["provenance"]["git_dirty"] is False)
            failed = {name for name, value in entry["result"]["criteria"].items() if value is not True}
            ok["sources_passed_or_recorded"].append(failed <= EXPECTED_FAILURES.get(key, set()))
    r = {key: value["result"] for key, value in sources.items()}
    loco, ladder, dim = (r["manifold_loco"]["variables"], r["manifold_ladder"]["variables"],
                         r["manifold_dimension"]["variables"])
    iso, local = r["isometry"]["variables"], r["isometry_local"]["variables"]
    setup, scores = r["spline_setup"]["variables"], r["spline_scores"]["variables"]
    natural, held = r["spline_naturalness"]["variables"], r["spline_held_out"]["variables"]
    harm = r["direction_harmonics"]["sites"][STEERING_SITE]["power_share"]
    k_minus_1 = {v: load_probe_sequence(v).k - 1 for v in DATASETS}
    probe_ratio = {v: probe_shift_ratio(v, setup[v]["median_train_clip_distance"]) for v in DATASETS}

    def row(block: str, quantity: str, cell, source: str) -> str:
        return f"| {block} | {quantity} | " + " | ".join(cell(v) for v in DATASETS) + f" | `{source}` |"

    def site9(v): return loco[v][STEERING_SITE]
    none = lambda v: "—"
    rows = [
        row("Q1", "LOCO MSE, straight line → curve (idx 9)", lambda v: f"{site9(v)['line']['mse']:.1f} → {site9(v)['selected']['mse']:.1f}", "manifold_loco"),
        row("Q1", "line − curve [95% CI]", lambda v: f"{site9(v)['gap_line_minus_selected']['mean']:.1f} [{site9(v)['gap_line_minus_selected']['ci'][0]:.1f}, {site9(v)['gap_line_minus_selected']['ci'][1]:.1f}]", "manifold_loco"),
        row("Q1", "uneven-spacing share, B line / PC1 line (curvature = 1 − share)", lambda v: none(v) if v == "direction" else f"{fmt(ladder[v]['spacing_share']['covariance_split']['share'], 2)} / {fmt(ladder[v]['spacing_share']['pc1_split']['share'], 2)}", "manifold_ladder"),
        row("H-01", "selected PCA dim k / Phase 4 K·m", lambda v: f"{dim[v][STEERING_SITE]['selected_k']} / {dim[v][STEERING_SITE]['K_dims']}", "manifold_dimension"),
        row("H-01", "participation ratio of the fitted curve", lambda v: fmt(dim[v][STEERING_SITE]['curve_participation_ratio'], 2), "manifold_dimension"),
        row("H-02", "harmonic power share h1 / h2 / h3+h4", lambda v: f"{fmt(harm[0], 2)} / {fmt(harm[1], 2)} / {fmt(harm[2] + harm[3], 2)}" if v == "direction" else "—", "direction_harmonics"),
        row("Q2", "all pairs: r(act. geodesic, beh. geodesic) / r(label, beh.)", lambda v: f"{fmt(iso[v]['conditions']['A_idx18']['point']['geodesic_r'])} / {fmt(iso[v]['conditions']['A_idx18']['point']['label_r'])}", "isometry"),
        row("Q2", "geodesic − label [95% CI]", lambda v: f"{fmt(iso[v]['conditions']['A_idx18']['point']['geodesic_minus_label'])} {fmt_ci(iso[v]['conditions']['A_idx18']['ci']['geodesic_minus_label'])}", "isometry"),
        row("Q2", "local speed r [95% CI] (split-half ceiling)", lambda v: f"{fmt(local[v]['conditions']['A_idx18']['point']['cross'], 2)} {fmt_ci(local[v]['conditions']['A_idx18']['ci']['cross'], 2)} ({fmt(local[v]['conditions']['A_idx18']['point']['ceiling'], 2)})", "isometry_local"),
        row("Q3", "edit size ÷ clip distance: spline / covariance / free spacing / K−1 probes", lambda v: " / ".join([
            fmt(setup[v]['median_shift_over_clip_distance']['spline_1'], 2),
            fmt(setup[v]['median_shift_over_clip_distance']['covariance_0.75'] / 0.75, 2),
            "—" if v == "direction" else fmt(setup[v]['median_shift_over_clip_distance']['spacing_line_1'], 2),
            fmt(probe_ratio[v], 2)]), "spline_setup; steer_*"),
        row("Q3", "spline vs covariance endpoint, ‖δ_spline − δ_cov‖ / ‖δ_cov‖ (median)", lambda v: fmt(setup[v]['spline_vs_covariance_endpoint_relative']['all'][1], 2), "spline_setup"),
    ]
    methods = lambda v: {"spline": "spline_1", "covariance line": "covariance_1", "free-spacing line": "spacing_line_1",
                         "K−1 probes": f"phase5_probes_{k_minus_1[v]}", "random (floor)": f"phase5_random_{k_minus_1[v]}"}
    for label in ("spline", "covariance line", "free-spacing line", "K−1 probes", "random (floor)"):
        rows.append(row("Q3", f"gain idx 9 / 18 / mean 10–18: {label}", lambda v, label=label: "—" if methods(v)[label] not in scores[v]["summary"] else " / ".join(
            fmt(scores[v]["summary"][methods(v)[label]][key]) for key in ("idx9", "idx18", "downstream_mean")), "spline_scores"))
    for label in ("spline", "covariance line", "K−1 probes", "random (floor)"):
        rows.append(row("Q3", f"excess naturalness idx 9 / 18: {label}", lambda v, label=label: " / ".join(
            fmt(natural[v]["sites"][site][methods(v)[label]]["excess_hellinger_mean"]) for site in ("9", "18")), "spline_naturalness"))
    rows += [
        row("Q3", "midpoint excess, spline − chord, idx 9 [CI]", lambda v: f"{fmt(natural[v]['headline']['midpoint_spline_minus_chord_idx9']['all']['point'])} {fmt_ci(natural[v]['headline']['midpoint_spline_minus_chord_idx9']['all']['ci'])}", "spline_naturalness"),
        row("Q3", "midpoint excess, spline − chord, idx 18 [CI]", lambda v: f"{fmt(natural[v]['headline']['midpoint_spline_minus_chord_idx18']['all']['point'])} {fmt_ci(natural[v]['headline']['midpoint_spline_minus_chord_idx18']['all']['ci'])}", "spline_naturalness"),
        row("Q3", "held-out targets: spline gain idx 9, seen / unseen", lambda v: f"{fmt(held[v]['subsets']['seen_targets']['arms']['spline_1']['gain_idx9'])} / {fmt(held[v]['subsets']['unseen_targets']['arms']['spline_1']['gain_idx9'])}", "spline_held_out"),
        row("Q4", "time-structured − uniform, mean gain 10–18 [CI]", lambda v: f"{fmt(scores[v]['headline']['time_minus_covariance_downstream']['point'])} {fmt_ci(scores[v]['headline']['time_minus_covariance_downstream']['ci'])}", "spline_scores"),
        row("Q4", "time-reversed − uniform [CI]", lambda v: f"{fmt(scores[v]['headline']['reversed_minus_covariance_downstream']['point'])} {fmt_ci(scores[v]['headline']['reversed_minus_covariance_downstream']['ci'])}", "spline_scores"),
        row("Q4", "confirmation run triggered (C11)", lambda v: "yes" if scores[v]["c11"]["triggered"] else "no", "spline_scores"),
    ]
    notes = [
        "Gain = Σ achieved·intended / Σ |intended|² in the readout's output space (direction: (sin, cos)); 1 = the "
        "readout reaches the target, 0 = no change. Readouts: validation-fit ridge probes at each index (D-16).",
        "Index 9 is the steering layer (same-layer readout, partly by construction); index 18 is the evidence.",
        "Excess naturalness = Hellinger distance of the edited clip's 16-bin readout to the behavior curve minus the same "
        "clip's unedited distance (negative = more natural than the unedited clip). Speed / acceleration: blurry readout "
        "(top-1 0.28 / 0.26, within one bin 0.72 / 0.65).",
        "Unseen targets: values with no centroid in the curve fit (interpolation). Q4 arms were designed after Phase 5's "
        "results; the pre-declared confirmation threshold (≥ 0.05) was not met, so Q4 is not confirmed on fresh clips.",
        "Isometry: split halves of the train clips; readout A at index 18. Local speed = arc length per unit label.",
    ]
    text = "\n".join([
        "# Spline vs multi-probe steering: comparison table (6.12-6.14)", "",
        "Generated by `scripts/check_spline_steering.py comparison_table` from saved results only.", "",
        "| Block | Quantity | Direction | Speed | Acceleration | Source key |", "|---|---|---|---|---|---|",
        *rows, "", "Notes:", "", *[f"- {n}" for n in notes], "",
        "Sources (commit): " + ", ".join(f"`{k}` {v['commit'][:7]}" for k, v in sources.items()), "",
    ])
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(text)
    criteria = {name: all(v) for name, v in ok.items()} | {
        "computed_finite": bool(np.isfinite(list(probe_ratio.values())).all()),
        "saved_equals_rendered": TABLE.read_text() == text,
    }
    return {
        "table": {"path": str(TABLE.relative_to(REPO)), "sha256": file_sha256(TABLE)},
        "k_minus_1_probe_shift_over_clip_distance": probe_ratio,
        "sources": {k: v["commit"] for k, v in sources.items()},
        "criteria": criteria, "passed": all(criteria.values()),
    }
    

def check_figure_spline_paths() -> dict:
    """Figure (slide 3, Q3): direction's path along the curve vs the straight chord, and midpoint naturalness.

    Rows 1-2: for one direction clip x target — chosen by a fixed rule, the largest start-to-target angle (shorter
    arc) in spline_setup — the 16-bin readout distribution at f = 0 (unedited), 0.25, 0.5, 0.75, 1 along the spline
    path (left) and the chord (right), at index 9 (row 1) and 18 (row 2). Row 3: midpoint excess naturalness spline -
    chord with 95% intervals for all three variables at index 9 and 18 (spline_naturalness). Saved results only.
    Passes if: clip ids = spline_setup's; every plotted distribution sums to 1; the figure is written.
    """
    with np.load(verified_artifact(BEHAVIOR_CHECKS, "behavior_readouts")) as f:
        readouts = {key: f[key] for key in f.files if key.startswith("direction_")}
    with np.load(verified_artifact(OUT, "spline_setup")) as f:
        setup = {key: f[key] for key in ("direction_ids", "direction_starts", "direction_target_label")}
    saved = json.loads(OUT.read_text())
    headline = {v: saved["spline_naturalness"]["result"]["variables"][v]["headline"] for v in DATASETS}
    six = joined_runs(OUT, "spline_{variable}_{half}", "direction", ("ids", "uniform_arms", "features_uniform",
                                                                     "unedited_features"))
    ids = np.concatenate([h["ids"] for h in six])
    arms = six[0]["uniform_arms"].tolist()
    features = np.concatenate([h["features_uniform"] for h in six])
    unedited = np.concatenate([h["unedited_features"] for h in six])
    starts, targets = setup["direction_starts"], setup["direction_target_label"]
    gap = np.abs(label_difference("direction", targets[None, :], starts[:, None]))
    c, t = (int(i) for i in np.unravel_index(np.argmax(gap), gap.shape))
    paths = {"along the curve (spline)": ["spline_0.25", "spline_0.5", "spline_0.75", "spline_1"],
             "straight chord": ["chord_0.25", "chord_0.5", "chord_0.75", "spline_1"]}  # chord f = 1 = spline endpoint

    fig = plt.figure(figsize=(14, 11.5), facecolor=SURFACE)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.95], hspace=0.55, wspace=0.12, top=0.9)
    colours = sequential_cmap()(np.linspace(0.3, 1.0, len(PATH_FRACTIONS_SHOWN)))
    sums_ok = []
    for r, site in enumerate(("block_8", "block_17")):
        j = PROFILE_SITES.index(site)
        weights, offset = readouts[f"direction_{site}_weights"], readouts[f"direction_{site}_offset"]
        first = None
        for col, (label, names) in enumerate(paths.items()):
            ax = fig.add_subplot(grid[r, col], sharey=first)
            first = first or ax
            rows = np.stack([unedited[c, j]] + [features[c, t, arms.index(n), j] for n in names]).astype(np.float64)
            p = map_probabilities(weights, offset, rows)
            sums_ok.append(float(np.abs(p.sum(axis=1) - 1.0).max()) <= 1e-10)
            for colour, fraction, prob in zip(colours, PATH_FRACTIONS_SHOWN, p):
                ax.plot(BIN_CENTRES, prob, color=colour, lw=2.0, marker="o", ms=4, zorder=3,
                        label={0.0: "f = 0 (unedited clip)", 1.0: "f = 1 (target)"}.get(fraction, f"f = {fraction:g}"))
            for angle in (starts[c], targets[t]):
                ax.axvline(angle, color=INK_SECONDARY, lw=1.0, ls=(0, (4, 3)), zorder=1)
            ax.set_xlim(0, 360)
            ax.set_xticks(range(0, 361, 90), [f"{a}°" for a in range(0, 361, 90)])
            ax.set_xlabel("readout bin (direction)", color=INK_SECONDARY)
            if col == 0:
                ax.set_ylabel("bin probability", color=INK_SECONDARY)
            ax.set_ylim(bottom=0)
            ax.set_title(f"Index {plot_index(site)}: {label}", color=INK, fontsize=11, loc="left")
            style_axes(ax)
            if r == 0 and col == 0:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

    ax = fig.add_subplot(grid[2, :])
    for i, variable in enumerate(DATASETS):
        for k, (site_key, filled) in enumerate((("midpoint_spline_minus_chord_idx9", True),
                                                 ("midpoint_spline_minus_chord_idx18", False))):
            h = headline[variable][site_key]["all"]
            x = i + (k - 0.5) * 0.25
            colour = DATASET_COLOUR[variable]
            ax.errorbar(x, h["point"], yerr=[[h["point"] - h["ci"][0]], [h["ci"][1] - h["point"]]], fmt="o", ms=8,
                        color=colour, mfc=colour if filled else SURFACE, mec=colour, mew=1.8, elinewidth=1.8,
                        capsize=4, zorder=3)
    ax.axhline(0, color=INK_MUTED, lw=1.0, zorder=1)
    ax.set_xticks(range(len(DATASETS)), [v.capitalize() for v in DATASETS])
    ax.set_xlim(-0.6, len(DATASETS) - 0.4)
    ax.set_ylabel("midpoint excess Hellinger,\nspline − chord (95% CI)", color=INK_SECONDARY)
    ax.set_title("Midpoint naturalness: below 0 = the spline midpoint looks more like a real clip than the chord's",
                 color=INK, fontsize=11, loc="left")
    ax.legend(handles=[Line2D([], [], color=INK, marker="o", ms=8, lw=0, label="index 9 (steering layer)"),
                       Line2D([], [], color=INK, marker="o", ms=8, lw=0, mfc=SURFACE, mew=1.8, label="index 18")],
              frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    style_axes(ax)
    fig.suptitle(f"Direction, clip {int(ids[c])}: start {starts[c]:.0f}° → target {targets[t]:.1f}° (dashed lines; "
                 f"largest start–target angle among the 150 pairs)", color=INK, fontsize=12, x=0.02, ha="left", y=0.985)
    fig.legend(legend_handles, legend_labels, loc="upper center", ncol=len(PATH_FRACTIONS_SHOWN), frameon=False,
               fontsize=9, labelcolor=INK, bbox_to_anchor=(0.5, 0.955))
    FIGURE_PATHS.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATHS, dpi=FIGURE_DPI, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    criteria = {"same_ids": bool(np.array_equal(ids, setup["direction_ids"])), "sums_to_one": all(sums_ok),
                "figure_written": FIGURE_PATHS.exists()}
    return {
        "figure": {"path": str(FIGURE_PATHS.relative_to(REPO)), "sha256": file_sha256(FIGURE_PATHS)},
        "example": {"rule": "largest shorter-arc angle between start and target (spline_setup)", "clip_id": int(ids[c]),
                    "target_index": t, "start_degrees": float(starts[c]), "target_degrees": float(targets[t]),
                    "angle": float(gap[c, t])},
        "sources": {key: saved[key]["provenance"]["git_commit"] for key in ("spline_setup", "spline_naturalness")},
        "criteria": criteria, "passed": all(criteria.values()),
    }

def check_figure_spline_profile() -> dict:
    """Figure (slide 4, Q3 + Q4): propagation of each edit from index 9 to 18, and the token-structure effect.

    Row 1 (one panel per variable): output-space gain at indices 9-18 of the spline endpoint, the covariance line,
    the time-structured covariance edit, Phase 5's K - 1 probes and the random floor (spline_scores' saved profile).
    Row 2: mean gain over indices 10-18 minus the uniform covariance edit, time-structured and time-reversed, with
    95% intervals and the pre-declared confirmation threshold (C11). Saved results only. Passes if: the covariance
    replication's profile equals Phase 5's covariance profile at every index; plotted values finite; figure written.
    """
    saved = json.loads(OUT.read_text())
    scores = saved["spline_scores"]["result"]["variables"]
    indices = [plot_index(s) for s in PROFILE_SITES]
    ok: dict[str, list[bool]] = {"covariance_equals_phase5": [], "finite": []}

    fig = plt.figure(figsize=(15, 9.5), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.15, 0.85], hspace=0.45, wspace=0.14, top=0.86)
    first = None
    for col, variable in enumerate(DATASETS):
        k1 = load_probe_sequence(variable).k - 1
        colour = DATASET_COLOUR[variable]
        profile = scores[variable]["profile"]

        def gains(arm: str) -> list[float]:
            return [profile[str(i)]["arms"][arm]["gain"] for i in indices]

        ok["covariance_equals_phase5"].append(gains("covariance_1") == gains("phase5_covariance"))
        styles = [
            (f"phase5_random_{k1}", {"color": INK_MUTED, "lw": 1.2}),
            (f"phase5_probes_{k1}", {"color": INK_SECONDARY, "lw": 1.6, "ls": (0, (4, 3)), "marker": "^", "ms": 5}),
            ("time_covariance_1", {"color": colour, "lw": 1.4, "ls": (0, (1, 2)), "marker": "s", "ms": 5}),
            ("covariance_1", {"color": colour, "lw": 2.0, "marker": "o", "ms": 5}),
            ("spline_1", {"color": colour, "lw": 0, "marker": "o", "ms": 10, "mfc": SURFACE, "mew": 1.6}),
        ]
        ax = fig.add_subplot(grid[0, col], sharey=first)
        first = first or ax
        for z, (arm, style) in enumerate(styles):
            values = gains(arm)
            ok["finite"].append(bool(np.isfinite(values).all()))
            ax.plot(indices, values, zorder=2 + z, **style)
        at18 = {name: profile["18"]["arms"][arm]["gain"] for name, arm in
                (("spline", "spline_1"), ("covariance", "covariance_1"), ("probes", f"phase5_probes_{k1}"))}
        ax.text(0.98, 0.95, "index 18: " + " · ".join(f"{n} {g:.3f}" for n, g in at18.items()), transform=ax.transAxes,
                ha="right", va="top", color=INK_SECONDARY, fontsize=8.5)
        ax.axhline(0, color=INK_MUTED, lw=0.8, zorder=1)
        ax.set_xticks(indices)
        ax.set_xlabel("layer index (steered at 9)", color=INK_SECONDARY)
        if col == 0:
            ax.set_ylabel("gain (1 = readout reaches the target)", color=INK_SECONDARY)
        ax.set_title(variable.capitalize() + (" (gain on (sin, cos))" if variable == "direction" else ""),
                     color=INK, fontsize=11, loc="left")
        style_axes(ax)

    ax = fig.add_subplot(grid[1, :])
    for i, variable in enumerate(DATASETS):
        colour = DATASET_COLOUR[variable]
        for k, (key, filled) in enumerate((("time_minus_covariance_downstream", True),
                                           ("reversed_minus_covariance_downstream", False))):
            h = scores[variable]["headline"][key]
            ok["finite"].append(bool(np.isfinite([h["point"], *h["ci"]]).all()))
            ax.errorbar(i + (k - 0.5) * 0.25, h["point"], yerr=[[h["point"] - h["ci"][0]], [h["ci"][1] - h["point"]]],
                        fmt="s", ms=8, color=colour, mfc=colour if filled else SURFACE, mec=colour, mew=1.8,
                        elinewidth=1.8, capsize=4, zorder=3)
    ax.axhline(0, color=INK_MUTED, lw=1.0, zorder=1)
    ax.axhline(C11_MIN_DIFFERENCE, color=INK_SECONDARY, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.annotate(f"pre-declared confirmation threshold (+{C11_MIN_DIFFERENCE:g}): not reached", (0.01, C11_MIN_DIFFERENCE),
                xycoords=("axes fraction", "data"), xytext=(0, 4), textcoords="offset points", color=INK_SECONDARY, fontsize=9)
    ax.set_xticks(range(len(DATASETS)), [v.capitalize() for v in DATASETS])
    ax.set_xlim(-0.6, len(DATASETS) - 0.4)
    ax.set_ylabel("mean gain over idx 10–18,\nminus uniform edit (95% CI)", color=INK_SECONDARY)
    ax.set_title("Token-structured edit: each time step gets its own covariance shift (reversed order = control); "
                 "95% CIs are narrower than the markers", color=INK, fontsize=11, loc="left")
    ax.legend(handles=[Line2D([], [], color=INK, marker="s", ms=8, lw=0, label="time-structured − uniform"),
                       Line2D([], [], color=INK, marker="s", ms=8, lw=0, mfc=SURFACE, mew=1.8,
                              label="time-reversed − uniform")],
              frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    style_axes(ax)

    handles = [
        Line2D([], [], color=INK, lw=0, marker="o", ms=9, mfc=SURFACE, mew=1.6, label="spline endpoint"),
        Line2D([], [], color=INK, lw=2.0, marker="o", ms=5, label="covariance line"),
        Line2D([], [], color=INK, lw=1.4, ls=(0, (1, 2)), marker="s", ms=5, label="time-structured covariance"),
        Line2D([], [], color=INK_SECONDARY, lw=1.6, ls=(0, (4, 3)), marker="^", ms=5, label="K − 1 probes (Phase 5)"),
        Line2D([], [], color=INK_MUTED, lw=1.2, label="random, same length (floor)"),
    ]
    fig.suptitle("Propagation of the edit from the steering layer (index 9) to index 18; colour = variable",
                 color=INK, fontsize=12, x=0.02, ha="left", y=0.985)
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=9, labelcolor=INK,
               bbox_to_anchor=(0.5, 0.945))
    FIGURE_PROFILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PROFILE, dpi=FIGURE_DPI, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    criteria = {name: all(v) for name, v in ok.items()} | {"figure_written": FIGURE_PROFILE.exists()}
    return {
        "figure": {"path": str(FIGURE_PROFILE.relative_to(REPO)), "sha256": file_sha256(FIGURE_PROFILE)},
        "sources": {"spline_scores": saved["spline_scores"]["provenance"]["git_commit"]},
        "criteria": criteria, "passed": all(criteria.values()),
    }
    
    
CHECKS = {
    "spline_setup": check_spline_setup,
    **{f"spline_{v}_{h}": partial(spline_run, v, h) for v in DATASETS for h in HALVES},
    "spline_scores": check_spline_scores,
    "spline_naturalness": check_spline_naturalness,
    "spline_held_out": check_spline_held_out,
    "comparison_table": check_comparison_table,
    "figure_spline_paths": check_figure_spline_paths,
    "figure_spline_profile": check_figure_spline_profile,
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