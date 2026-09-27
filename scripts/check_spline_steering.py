"""Spline steering (Part 2, Q3 / Q4): edits along the index-9 activation curves (spline paths, chords, the
covariance line, a free-spacing line, and time-structured covariance edits) on Phase 5's steering clips and targets.

Usage: python scripts/check_spline_steering.py <check>
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from vjepa_physics.baselines import squared_distances
from vjepa_physics.data import DATASETS
from vjepa_physics.evidence import file_sha256, require_clean_code, save_result, verified_artifact
from vjepa_physics.extraction import SITES
from vjepa_physics.joined import load_joined
from vjepa_physics.manifolds import PCA_DIMS, fit_curve, fit_spacing_line, smoothing_grid
from vjepa_physics.probes import probe_targets, site_features
from vjepa_physics.steering import (
    N_CLIPS, STEERING_SITE, chord_shifts, covariance_map, covariance_path_shifts, load_probe_sequence,
    spline_arm_shifts, spline_arm_table, time_covariance_maps,
)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/spline_steering/checks.json"
ARTIFACTS = REPO / "artifacts/spline_steering"  # regenerable, git-ignored
STEERING_CHECKS = REPO / "results/steering/checks.json"  # Phase 5: steering_setup, steering_cache, steer_* runs
MANIFOLD_CHECKS = REPO / "results/manifolds/checks.json"  # Q1: manifold_loco (curves.npz), manifold_ladder

HALVES = {"seen": "test_seen", "unseen": "test_unseen"}  # Phase 5's saved runs, one per half
EXACT_TOLERANCE = 1e-12  # relative: two computation paths of the same shift


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


CHECKS = {
    "spline_setup": check_spline_setup,
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