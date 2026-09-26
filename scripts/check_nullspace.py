"""Checks for iterative nullspace probing: fit a ridge probe, remove its weight directions, fit again.

Run one check at a time from the repo root, with .venv active:
    python scripts/check_nullspace.py <check>

Each check prints a summary and stores its full result under its own key in results/nullspace/checks.json.
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from vjepa_physics.data import DATASETS
from vjepa_physics.evidence import file_sha256, save_result, verified_artifact
from vjepa_physics.extraction import SITES, plot_index
from vjepa_physics.joined import load_joined
from vjepa_physics.metrics import mae
from vjepa_physics.nullspace import (
    NULL_R2, RANK_TOLERANCE, composite_maps, curve_summary, first_true, nullspace_alpha_verdict, redundancy_counts,
    round_scores, run_rounds, train_scaler,
)
from vjepa_physics.probes import ALPHAS, probe_targets, site_features

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/nullspace/checks.json"
ARTIFACTS = REPO / "artifacts/nullspace"  # regenerable, git-ignored
LAYER_CURVES = REPO / "results/probes/checks.json"  # key "layer_curves": round 1 must reproduce it

NULLSPACE_SITES = ("block_0", "block_8", "block_17")  # indices 1 (transition), 9 (headline), 18 (plateau)
N_ROUNDS = 150
REFIT_ROUNDS = 10  # a second run of this many rounds must give the same first rounds (the sequence is deterministic)
EXPECTED_FIT = {"direction": 813, "speed": 832, "acceleration": 832}  # train clips per variable
EVAL_ROLES = ("val_seen", "val_unseen")  # scored here; test is scored once, after K is recorded

THRESHOLDS = (0.3, 0.1, 0.05)  # R² crossings read off every curve (the paper's stopping thresholds)
PAPER_THRESHOLDS = {  # as the paper states them per variable (appendix text vs figure caption); none for acceleration
    "direction": {"appendix": 0.1, "figure_caption": 0.3},
    "speed": {"appendix": 0.05, "figure_caption": 0.1},
}
CHANCE_CIRCULAR_MAE = 80.0  # paper's secondary rule for direction: circular MAE above this (degrees)
BASELINE_MAE_FRACTION = 0.9  # paper's secondary rule for speed: MAE above this x the mean-prediction MAE

ROUND1_TOLERANCE = 1e-12  # relative, accepted only if round 1 is not bit-identical to the layer-curve probe
ORTHONORMAL_TOLERANCE = 1e-10
LEAK_TOLERANCE = 1e-8
COMPOSITE_TOLERANCE = 1e-9  # relative


def secondary_rule(variable: str, scores: list[dict], baseline_mae: float | None) -> dict:
    """First round where the paper's secondary rule says the variable is gone (val-seen)."""
    if variable == "direction":
        errors = np.array([s["circular_mae"] for s in scores])
        return {"rule": "circular MAE > 80 degrees", "source": "paper", "round": first_true(errors > CHANCE_CIRCULAR_MAE)}
    errors = np.array([s["mae"] for s in scores])
    return {
        "rule": "MAE > 0.9 x mean-prediction MAE", "source": "paper" if variable == "speed" else "extension",
        "baseline_mae": baseline_mae, "round": first_true(errors > BASELINE_MAE_FRACTION * baseline_mae),
    }


def check_nullspace_rounds() -> dict:
    """Iterative nullspace probing at three layers, all rounds, scored on validation; probe sequence saved.

    Per variable and site: one train scaler, then N_ROUNDS ridge probes, each fit on train with every earlier
    probe's weight directions projected out (standardized space). K = first round with val_seen R² < NULL_R2.
    Writes artifacts/nullspace/rounds.npz (scaler, basis, weights, intercepts, alphas, composite raw-space maps,
    validation predictions with NaN elsewhere). Passes if: n_fit exact; ids and round 1 equal the layer-curve
    probe; basis orthonormal; no leak into removed directions; every weight block full rank; composite maps =
    round-by-round predictions; a refit gives identical first rounds; only validation rows predicted; scores
    finite; no alpha failure; saved file = computed.
    """
    with np.load(verified_artifact(LAYER_CURVES, "layer_curves")) as f:
        curves = {key: f[key] for key in f.files}
    curve_alphas = json.loads(LAYER_CURVES.read_text())["layer_curves"]["result"]

    arrays: dict[str, np.ndarray] = {"sites": np.array(NULLSPACE_SITES)}
    result: dict = {}
    ok: dict[str, list[bool]] = {name: [] for name in (
        "n_fit", "ids", "round1", "orthonormal", "leak", "rank", "composite", "refit", "rows", "finite", "alpha")}

    for variable in DATASETS:
        table = load_joined(variable)
        acts, roles, labels = table["activations"], table["role"], table["label"]
        y = probe_targets(variable, labels)
        evaluated = np.isin(roles, EVAL_ROLES)
        eval_roles = roles[evaluated]
        ok["ids"].append(bool(np.array_equal(curves[f"{variable}_ids"], table["id"])))
        baseline = None
        if variable != "direction":
            train_mean = float(labels[roles == "train"].mean())
            seen = labels[roles == "val_seen"]
            baseline = mae(seen, np.full(len(seen), train_mean))

        sites = {}
        for site in NULLSPACE_SITES:
            x = site_features(acts, site)
            scaler = train_scaler(x, roles)
            z = scaler.transform(x)
            start = time.perf_counter()
            run = run_rounds(z, y, roles, evaluated, N_ROUNDS)
            seconds = time.perf_counter() - start
            again = run_rounds(z, y, roles, evaluated, REFIT_ROUNDS)
            maps, offsets = composite_maps(scaler, run)
            m = run.dims_per_round

            first = run.predictions[0][:, 0] if m == 1 else run.predictions[0]
            saved_first = curves[f"{variable}_predictions"][evaluated, SITES.index(site)]
            round1_bit_identical = bool(np.array_equal(first, saved_first))
            round1_diff = float(np.abs(first - saved_first).max() / np.abs(saved_first).max())
            saved_alpha = curve_alphas[variable]["sites"][site]["alpha"]
            q = run.basis
            orthonormal_error = float(np.abs(q.T @ q - np.eye(q.shape[1])).max())
            composite = x[evaluated] @ maps + offsets[:, None, :]  # (rounds, n_evaluated, m)
            composite_diff = float(np.abs(composite - run.predictions).max() / np.abs(run.predictions).max())

            scores = {role: round_scores(variable, labels[roles == role], run.predictions[:, eval_roles == role])
                      for role in EVAL_ROLES}
            seen_r2 = np.array([s["r2"] for s in scores["val_seen"]])
            unseen_r2 = np.array([s["r2"] for s in scores["val_unseen"]])
            verdicts = [nullspace_alpha_verdict(e, r) for e, r in zip(run.alpha_edges, seen_r2)]
            error_name = "circular_mae" if variable == "direction" else "mae"

            full: np.ndarray = np.full((len(y), N_ROUNDS, m), np.nan)
            full[evaluated] = run.predictions.transpose(1, 0, 2)
            prefix = f"{variable}_{site}"
            arrays |= {
                f"{prefix}_mean": np.asarray(scaler.mean_), f"{prefix}_scale": np.asarray(scaler.scale_),
                f"{prefix}_basis": q,
                f"{prefix}_weights": run.weights, f"{prefix}_intercepts": run.intercepts,
                f"{prefix}_alphas": run.alphas, f"{prefix}_maps": maps, f"{prefix}_offsets": offsets,
                f"{prefix}_predictions": full,
            }

            ok["n_fit"].append(run.n_fit == EXPECTED_FIT[variable])
            ok["round1"].append(bool(run.alphas[0] == saved_alpha and round1_diff <= ROUND1_TOLERANCE))
            ok["orthonormal"].append(orthonormal_error <= ORTHONORMAL_TOLERANCE)
            ok["leak"].append(float(run.leaks.max()) <= LEAK_TOLERANCE)
            ok["rank"].append(float(run.rank_ratios.min()) > RANK_TOLERANCE)
            ok["composite"].append(composite_diff <= COMPOSITE_TOLERANCE)
            ok["refit"].append(bool(
                np.array_equal(again.alphas, run.alphas[:REFIT_ROUNDS])
                and np.array_equal(again.weights, run.weights[:REFIT_ROUNDS])
                and np.array_equal(again.predictions, run.predictions[:REFIT_ROUNDS])
            ))
            ok["rows"].append(bool(np.isfinite(full[evaluated]).all() and np.isnan(full[~evaluated]).all()))
            ok["finite"].append(all(np.isfinite(v) for role in EVAL_ROLES for s in scores[role] for v in s.values()))
            ok["alpha"].append("failure" not in verdicts)

            sites[site] = {
                "plot_index": plot_index(site), "dims_per_round": m, "n_fit": run.n_fit, "seconds": seconds,
                "summary": curve_summary(seen_r2, m),
                "val_unseen_k": curve_summary(unseen_r2, m)["k"],
                "crossings": {str(t): {"val_seen": first_true(seen_r2 < t), "val_unseen": first_true(unseen_r2 < t)}
                              for t in THRESHOLDS},
                "secondary_rule": secondary_rule(variable, scores["val_seen"], baseline),
                "redundancy": redundancy_counts(seen_r2),
                "round1": {"alpha": float(run.alphas[0]), "layer_curves_alpha": saved_alpha,
                           "bit_identical": round1_bit_identical, "max_relative_diff": round1_diff},
                "orthonormal_error": orthonormal_error, "max_leak": float(run.leaks.max()),
                "min_rank_ratio": float(run.rank_ratios.min()), "composite_max_relative_diff": composite_diff,
                "val_seen_r2": seen_r2.tolist(), "val_unseen_r2": unseen_r2.tolist(),
                f"val_seen_{error_name}": [s[error_name] for s in scores["val_seen"]],
                f"val_unseen_{error_name}": [s[error_name] for s in scores["val_unseen"]],
                "alphas": run.alphas.tolist(), "alpha_edges": list(run.alpha_edges), "alpha_verdicts": verdicts,
            }
        arrays[f"{variable}_ids"] = table["id"]
        arrays[f"{variable}_roles"] = roles
        result[variable] = {"n_evaluated": int(evaluated.sum()), "paper_thresholds": PAPER_THRESHOLDS.get(variable),
                            "sites": sites}

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / "rounds.npz"
    np.savez(path, **arrays)
    with np.load(path) as saved:
        saved_ok = set(saved.files) == set(arrays) and all(
            np.array_equal(saved[key], value, equal_nan=value.dtype.kind == "f") for key, value in arrays.items()
        )

    criteria = {
        "n_fit_equals_train_count": all(ok["n_fit"]),
        "ids_match_layer_curves": all(ok["ids"]),
        "round1_equals_layer_curves": all(ok["round1"]),
        "basis_orthonormal": all(ok["orthonormal"]),
        "no_leak_into_removed_directions": all(ok["leak"]),
        "weight_blocks_full_rank": all(ok["rank"]),
        "composite_equals_round_by_round": all(ok["composite"]),
        "refit_identical": all(ok["refit"]),
        "only_validation_rows_predicted": all(ok["rows"]),
        "scores_finite": all(ok["finite"]),
        "no_alpha_failure": all(ok["alpha"]),
        "saved_equals_computed": saved_ok,
    }
    return {
        "criteria": criteria, "n_rounds": N_ROUNDS, "stop_r2": NULL_R2, "thresholds": list(THRESHOLDS), **result,
        "alpha_grid": [float(a) for a in ALPHAS],
        "artifact": {"path": str(path.relative_to(REPO)), "sha256": file_sha256(path)},
        "passed": all(criteria.values()),
    }


def print_nullspace_rounds(result: dict) -> None:
    """Per variable and site: K, the curve at chosen rounds, crossings, secondary rule, redundancy, alpha verdicts."""
    for variable in DATASETS:
        print(f"\n{variable}")
        for site, r in result[variable]["sites"].items():
            s = r["summary"]
            k = s["k"] if s["k"] is not None else f"> {result['n_rounds']}"
            c = r["val_seen_r2"]
            print(f"  {site:9s} idx {r['plot_index']:2d}  K {k}  dims before K {s['dims_before_k']}"
                  f"  last >= 0.1 at {s['last_round_at_or_above']}  max after K {s['max_r2_after_k']}"
                  f"  (unseen K {r['val_unseen_k']})  {r['seconds']:.0f} s")
            print("    val-seen R2  " + "  ".join(f"{n}:{c[n - 1]:.3f}" for n in (1, 2, 3, 5, 10, 20, 50, 100, 150)))
            print(f"    crossings {r['crossings']}")
            print(f"    secondary {r['secondary_rule']['round']}  redundancy "
                  + "  ".join(f"{f}: {v['consecutive']} / {v['total']}" for f, v in r["redundancy"].items())
                  + f"  verdicts {dict(Counter(r['alpha_verdicts']))}  round1 bit-identical {r['round1']['bit_identical']}")


CHECKS = {
    "nullspace_rounds": (check_nullspace_rounds, print_nullspace_rounds),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("check", choices=sorted(CHECKS))
    name = parser.parse_args().check
    run, show = CHECKS[name]
    result = run()
    show(result)
    print(json.dumps({k: result[k] for k in ("criteria", "artifact", "passed") if k in result}, indent=2))
    save_result(OUT, name, result)
    print(f"saved '{name}' -> {OUT.relative_to(REPO)}")
    if result.get("passed") is False:
        raise SystemExit(f"check '{name}' FAILED")


if __name__ == "__main__":
    main()