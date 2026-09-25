"""Checks for the probes' reference baselines: exact pixel Gram matrices for the pixel floor.

Run one check at a time from the repo root, with .venv active:
    python scripts/check_baselines.py <check>

Each check prints a summary and stores its full result under its own key in results/baselines/checks.json.
"""
import argparse
import csv
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np

from vjepa_physics.baselines import exact_gram, kernel_ridge, physics_fit, pixel_matrix
from vjepa_physics.data import DATASETS, load_dataset, read_manifest, resolve
from vjepa_physics.probes import alpha_verdict, probe_scores, probe_targets
from vjepa_physics.evidence import file_sha256, save_result, verified_artifact
from vjepa_physics.geometry import disk_centres
from vjepa_physics.joined import load_joined
from vjepa_physics.metrics import circular_errors, sincos_targets

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = REPO / "results/baselines/checks.json"
BASELINES = REPO / "artifacts/baselines"  # regenerable, git-ignored
VIDEOS_CHECKS = REPO / "results/videos/checks.json"  # key "duplicates": per-clip decoded-pixel SHA-256

EXPECTED_CLIPS = {"direction": 1500, "speed": 1536, "acceleration": 1536}  # DATA.md
FRAMES = 16
EXACT_LIMIT = 2**53  # float64 holds every integer below this exactly
# "full": raw RGB of all 16 frames (headline floor); "time_average": the frame sum = time-averaged frame x 16.
GRAM_KINDS = ("full", "time_average")

EXPECTED_FIT = {"direction": 813, "speed": 832, "acceleration": 832}  # train clips per variable (split decision)
EVAL_ROLES = ("val_seen", "val_unseen")  # scored here; test is scored once, after the layer choice is frozen
PROBES_CHECKS = REPO / "results/probes/checks.json"  # key "layer_curves": probe scores, printed for comparison

# Physics-fit ceiling: per variable, the headline model first (design-agnostic, like the probes), then the
# design-informed variant (constant velocity for speed, start from rest for acceleration), reported labelled.
CEILING_MODELS = {"direction": ("quadratic",), "speed": ("quadratic", "linear"), "acceleration": ("quadratic", "from_rest")}
CEILING_OUTPUT = {"direction": "theta", "speed": "speed", "acceleration": "acceleration"}  # physics_fit key read
EXACT_TOLERANCE = 1e-9  # a fit to metadata-predicted positions must return the label to within this


def clip_paths(variable: str, ids: np.ndarray) -> list[Path]:
    """Video path of every clip id, in the given order (manifest paths, resolved inside data/<variable>/)."""
    root = DATA / variable
    by_id = {row["id"]: resolve(root, row["video"]) for row in read_manifest(root)}
    return [by_id[int(i)] for i in ids]


def sum_of_squares(x: np.ndarray) -> np.ndarray:
    """Per row, the exact integer sum of squares (int64 arithmetic, row by row), as float64."""
    return np.array([float(np.dot(row.astype(np.int64), row.astype(np.int64))) for row in x])


def peak_rss_bytes() -> int:
    """Peak resident memory of this process so far (macOS reports ru_maxrss in bytes)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def check_pixel_grams() -> dict:
    """Build and save the exact pixel Gram matrices of every variable, full RGB and time-averaged.

    Clips are decoded in the joined table's id order. Writes artifacts/baselines/pixel_gram_<variable>_<kind>.npy
    (float64, clips x clips). Passes if, per variable: every decoded clip's SHA-256 equals the data audit's (same
    pixels, same order); and per Gram: shape = clips x clips; exactly symmetric; every entry an integer below
    2^53 (so exact); diagonal = each row's sum of squares in int64. Timings and peak memory are recorded, not judged.
    """
    with verified_artifact(VIDEOS_CHECKS, "duplicates").open(newline="") as f:
        audited = {(row["dataset"], int(row["id"])): row["clip_sha256"] for row in csv.DictReader(f)}
    BASELINES.mkdir(parents=True, exist_ok=True)

    result: dict = {}
    passed = True
    for variable in DATASETS:
        ids = load_joined(variable)["id"]
        start = time.perf_counter()
        pixels = pixel_matrix(clip_paths(variable, ids))
        decode_seconds = time.perf_counter() - start
        decoded_ok = len(pixels) == EXPECTED_CLIPS[variable] and all(
            hashlib.sha256(row.tobytes()).hexdigest() == audited[(variable, int(i))] for row, i in zip(pixels, ids)
        )
        features = {
            "full": pixels,
            # same as pixel_matrix(..., time_average=True), from the already decoded pixels
            "time_average": pixels.reshape(len(pixels), FRAMES, -1).sum(axis=1, dtype=np.uint16),
        }
        grams = {}
        for kind in GRAM_KINDS:
            x = features[kind]
            start = time.perf_counter()
            gram = exact_gram(x)
            seconds = time.perf_counter() - start
            path = BASELINES / f"pixel_gram_{variable}_{kind}.npy"
            np.save(path, gram)
            criteria = {
                "shape_is_clips_by_clips": gram.shape == (len(ids), len(ids)),
                "exactly_symmetric": bool(np.array_equal(gram, gram.T)),
                "integer_below_2_53": bool(np.array_equal(gram, np.round(gram)) and gram.max() < EXACT_LIMIT),
                "diagonal_is_sum_of_squares": bool(np.array_equal(np.diag(gram), sum_of_squares(x))),
            }
            passed = passed and all(criteria.values())
            grams[kind] = {"criteria": criteria, "features": int(x.shape[1]), "max_entry": float(gram.max()),
                           "gram_seconds": round(seconds, 1)}
            result[f"gram_{variable}_{kind}"] = {"path": str(path.relative_to(REPO)), "sha256": file_sha256(path)}
        passed = passed and decoded_ok
        result[variable] = {"clips": len(ids), "decoded_equals_audit_hashes": decoded_ok,
                            "decode_seconds": round(decode_seconds, 1), "grams": grams}
        del pixels, features, x

    result["peak_rss_bytes"] = peak_rss_bytes()
    result["passed"] = passed
    return result


def print_pixel_grams(result: dict) -> None:
    """Per variable: decode check and time; per Gram: criteria, time, file hash prefix."""
    for variable in DATASETS:
        r = result[variable]
        print(f"\n{variable}: {r['clips']} clips, decoded = audit {r['decoded_equals_audit_hashes']},"
              f" decode {r['decode_seconds']} s")
        for kind, g in r["grams"].items():
            sha = result[f"gram_{variable}_{kind}"]["sha256"][:12]
            print(f"  {kind:12s} {g['features']:>9,d} features  max {g['max_entry']:.4g}  {g['gram_seconds']} s"
                  f"  sha {sha}…  {g['criteria']}")
    print(f"\npeak RSS {result['peak_rss_bytes'] / 1e9:.2f} GB")


def check_pixel_floor() -> dict:
    """Ridge on raw pixels through the saved Gram matrices: the probes' floor, full RGB and time-averaged.

    Grams are read through their recorded hashes; rows follow the joined table, so roles and labels align.
    Same targets, metrics, train-only fit and alpha rule as the probes; alphas relative to the centred train
    Gram's mean diagonal, so the frame-sum scaling of the time-averaged Gram does not matter. Writes
    artifacts/baselines/floor_predictions.npz (NaN on every non-validation row). Passes if: every fit saw exactly
    the train clips; no alpha verdict is "failure"; every score is finite; only validation rows were predicted;
    the saved file equals what was computed.
    """
    result: dict = {}
    arrays: dict[str, np.ndarray] = {}
    n_fit_ok, verdict_ok, finite_ok, rows_ok = [], [], [], []
    for variable in DATASETS:
        table = load_joined(variable)
        roles, labels = table["role"], table["label"]
        y = probe_targets(variable, labels)
        evaluated = np.isin(roles, EVAL_ROLES)
        kinds = {}
        for kind in GRAM_KINDS:
            gram = np.load(verified_artifact(OUT, "pixel_grams", f"gram_{variable}_{kind}"))
            if gram.shape != (len(roles), len(roles)):
                raise RuntimeError(f"{variable} {kind}: Gram {gram.shape} does not match {len(roles)} clips")
            fit = kernel_ridge(gram, y, roles, evaluated)
            scores = {
                role: probe_scores(variable, labels[roles == role], fit.predictions[roles == role])
                for role in EVAL_ROLES
            }
            verdict = alpha_verdict(fit.alpha_edge, scores["val_seen"]["r2"])
            kinds[kind] = {"alpha": fit.alpha, "alpha_edge": fit.alpha_edge, "alpha_verdict": verdict,
                           "n_fit": fit.n_fit, **scores}
            n_fit_ok.append(fit.n_fit == EXPECTED_FIT[variable])
            verdict_ok.append(verdict != "failure")
            finite_ok.append(all(np.isfinite(v) for s in scores.values() for v in s.values()))
            rows_ok.append(bool(np.isfinite(fit.predictions[evaluated]).all()
                                and np.isnan(fit.predictions[~evaluated]).all()))
            arrays[f"{variable}_{kind}_predictions"] = fit.predictions
        arrays[f"{variable}_ids"] = table["id"]
        arrays[f"{variable}_roles"] = roles
        result[variable] = kinds

    path = BASELINES / "floor_predictions.npz"
    np.savez_compressed(path, **arrays)
    with np.load(path) as saved:
        saved_ok = set(saved.files) == set(arrays) and all(
            np.array_equal(saved[key], value, equal_nan=value.dtype.kind == "f") for key, value in arrays.items()
        )
    criteria = {
        "n_fit_equals_train_count": all(n_fit_ok),
        "no_alpha_failure": all(verdict_ok),
        "scores_finite": all(finite_ok),
        "only_validation_rows_predicted": all(rows_ok),
        "saved_equals_computed": saved_ok,
    }
    return {"criteria": criteria, **result,
            "artifact": {"path": str(path.relative_to(REPO)), "sha256": file_sha256(path)},
            "passed": all(criteria.values())}


def print_pixel_floor(result: dict) -> None:
    """Per variable and floor: alpha, verdict, val_seen / val_unseen scores; the probes' embedding and block_0 for scale."""
    probes = json.loads(PROBES_CHECKS.read_text())["layer_curves"]["result"]
    for variable in DATASETS:
        other = "circular_mae" if variable == "direction" else "mae"
        print(f"\n{variable}")
        for kind, r in result[variable].items():
            print(f"  floor {kind:12s} alpha {r['alpha']:10.4g} {str(r['alpha_edge']):5s} {r['alpha_verdict']:9s}"
                  f"  seen R2 {r['val_seen']['r2']:7.3f} {other} {r['val_seen'][other]:7.3f}"
                  f"  unseen R2 {r['val_unseen']['r2']:7.3f} {other} {r['val_unseen'][other]:7.3f}")
        for site in ("embedding", "block_0"):
            s = probes[variable]["sites"][site]["val_seen"]
            print(f"  probe {site:12s} seen R2 {s['r2']:7.3f} {other} {s[other]:7.3f}")
    print(json.dumps({"criteria": result["criteria"], "artifact": result["artifact"]}, indent=2))


def check_physics_ceiling() -> dict:
    """The probes' ceiling: a per-clip physics fit to the tracked disk (nothing is fitted across clips).

    Validation rows only (test is scored with the final numbers). Visible frames = disk present and not touching
    the border. Writes artifacts/baselines/ceiling_estimates.npz (per variable and model, the estimated label;
    NaN on non-validation rows). Passes if: every validation clip is fitted with a finite estimate and nothing
    else is estimated; the same fit on metadata-predicted positions returns every label to within
    EXACT_TOLERANCE (the fit is right); every score is finite; the saved file equals what was computed.
    """
    result: dict = {}
    arrays: dict[str, np.ndarray] = {}
    rows_ok, exact_ok, finite_ok = [], [], []
    for variable in DATASETS:
        table = load_joined(variable)
        roles, labels = table["role"], table["label"]
        evaluated = np.isin(roles, EVAL_ROLES)
        meta = {m["id"]: m for m in load_dataset(DATA, variable)}
        visible = (table["area"] > 0) & ~table["touches_border"]
        output = CEILING_OUTPUT[variable]
        models = {}
        for model in CEILING_MODELS[variable]:
            estimates = np.full(len(roles), np.nan)
            exact = np.full(len(roles), np.nan)
            for k in np.flatnonzero(evaluated):
                estimates[k] = physics_fit(table["centre"][k], visible[k], model)[output]
                all_frames = np.ones(len(visible[k]), dtype=bool)
                exact[k] = physics_fit(disk_centres(meta[int(table["id"][k])]), all_frames, model)[output]
            if variable == "direction":
                exact_error = float(circular_errors(labels[evaluated], exact[evaluated]).max())
                predictions = sincos_targets(estimates)
            else:
                exact_error = float(np.abs(exact[evaluated] - labels[evaluated]).max())
                predictions = estimates
            scores = {
                role: probe_scores(variable, labels[roles == role], predictions[roles == role])
                for role in EVAL_ROLES
            }
            models[model] = {"headline": model == CEILING_MODELS[variable][0],
                             "max_error_on_metadata_positions": exact_error, **scores}
            rows_ok.append(bool(np.isfinite(estimates[evaluated]).all() and np.isnan(estimates[~evaluated]).all()))
            exact_ok.append(exact_error <= EXACT_TOLERANCE)
            finite_ok.append(all(np.isfinite(v) for s in scores.values() for v in s.values()))
            arrays[f"{variable}_{model}_estimates"] = estimates
        arrays[f"{variable}_ids"] = table["id"]
        arrays[f"{variable}_roles"] = roles
        result[variable] = models

    path = BASELINES / "ceiling_estimates.npz"
    np.savez_compressed(path, **arrays)
    with np.load(path) as saved:
        saved_ok = set(saved.files) == set(arrays) and all(
            np.array_equal(saved[key], value, equal_nan=value.dtype.kind == "f") for key, value in arrays.items()
        )
    criteria = {
        "only_validation_rows_estimated": all(rows_ok),
        "exact_on_metadata_positions": all(exact_ok),
        "scores_finite": all(finite_ok),
        "saved_equals_computed": saved_ok,
    }
    return {"criteria": criteria, **result, "exact_tolerance": EXACT_TOLERANCE,
            "artifact": {"path": str(path.relative_to(REPO)), "sha256": file_sha256(path)},
            "passed": all(criteria.values())}


def print_physics_ceiling(result: dict) -> None:
    """Per variable and model: headline or design-informed, val_seen / val_unseen scores, exact-recovery error."""
    for variable in DATASETS:
        other = "circular_mae" if variable == "direction" else "mae"
        print(f"\n{variable}")
        for model, r in result[variable].items():
            label = "headline" if r["headline"] else "design-informed"
            print(f"  {model:10s} {label:15s} seen R2 {r['val_seen']['r2']:.4f} {other} {r['val_seen'][other]:.4f}"
                  f"  unseen R2 {r['val_unseen']['r2']:.4f} {other} {r['val_unseen'][other]:.4f}"
                  f"  exact err {r['max_error_on_metadata_positions']:.1e}")
    print(json.dumps({"criteria": result["criteria"], "artifact": result["artifact"]}, indent=2))


CHECKS = {
    "pixel_grams": (check_pixel_grams, print_pixel_grams),
    "pixel_floor": (check_pixel_floor, print_pixel_floor),
    "physics_ceiling": (check_physics_ceiling, print_physics_ceiling),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("check", choices=sorted(CHECKS))
    name = parser.parse_args().check
    run, show = CHECKS[name]
    result = run()
    show(result)
    print(json.dumps({"passed": result["passed"]}, indent=2))
    save_result(OUT, name, result)
    print(f"saved '{name}' -> {OUT.relative_to(REPO)}")
    if result.get("passed") is False:
        raise SystemExit(f"check '{name}' FAILED")


if __name__ == "__main__":
    main()