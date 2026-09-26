"""Checks for iterative nullspace probing: fit a ridge probe, remove its weight directions, fit again.

Run one check at a time from the repo root, with .venv active:
    python scripts/check_nullspace.py <check>

Each check prints a summary and stores its full result under its own key in results/nullspace/checks.json.
"""
import argparse
import json
import time
from collections import Counter
from sklearn.linear_model import Ridge
from pathlib import Path

import numpy as np

from vjepa_physics.data import DATASETS
from vjepa_physics.evidence import file_sha256, save_result, verified_artifact
from vjepa_physics.extraction import SITES, plot_index
from vjepa_physics.joined import load_joined
from vjepa_physics.metrics import mae
from vjepa_physics.nullspace import (
    EXHAUSTION_RATIO, NULL_R2, RANK_TOLERANCE, composite_maps, curve_summary, first_true, grid_edge,
    nullspace_alpha_verdict, project_out, random_span_basis, redundancy_counts, round_scores, run_rounds,
    train_ridge, train_scaler, train_span,
)
from vjepa_physics.probes import ALPHAS, nested_cv_predictions, probe_scores, probe_targets, site_features
from vjepa_physics.reproducibility import SEED

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

ERASURE_FOLDS = 5  # outer and inner folds of the fresh probe (nested, grouped by clip)
N_RANDOM_SEEDS = 5  # random-subspace arms: seeds SEED ... SEED + 4
CONSTANT_TOLERANCE = 1e-8  # covariance arm: train-fit prediction spread / label SD (Xᵀy_c = 0 ⇒ ridge weights 0)
REFIT_TOLERANCE = 1e-9  # relative: nullspace arm vs saved round K + 1; nested CV vs a sklearn refit

SWEEP_SITE = "block_8"  # headline layer (index 9)
SWEEP_ALPHAS = (1e-3, 1e-1, 1e1, 1e3, 1e5, 1e7)  # fixed ridge penalty per run (standardized space, same units as ALPHAS)


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

            full: np.ndarray = np.full((len(y), len(run.alphas), m), np.nan)  # rounds actually run (guard)
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
            ok["leak"].append(float(np.nanmax(run.leaks)) <= LEAK_TOLERANCE)  # NaN = the exhausted round
            ok["rank"].append(float(np.nanmin(run.rank_ratios)) > RANK_TOLERANCE)
            ok["composite"].append(composite_diff <= COMPOSITE_TOLERANCE)
            n_again = len(again.alphas)
            ok["refit"].append(bool(
                np.array_equal(again.alphas, run.alphas[:n_again])
                and np.array_equal(again.weights, run.weights[:n_again])
                and np.array_equal(again.predictions, run.predictions[:n_again])
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
                "exhausted_round": run.exhausted_round, "rounds_run": len(run.alphas),
                "orthonormal_error": orthonormal_error, "max_leak": float(np.nanmax(run.leaks)),
                "min_rank_ratio": float(np.nanmin(run.rank_ratios)), "composite_max_relative_diff": composite_diff,
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
            print("    val-seen R2  " + "  ".join(f"{n}:{c[n - 1]:.3f}" for n in (1, 2, 3, 5, 10, 20, 50, 100, 150)
                                           if n <= len(c)))
            print(f"    crossings {r['crossings']}")
            print(f"    secondary {r['secondary_rule']['round']}  redundancy "
                  + "  ".join(f"{f}: {v['consecutive']} / {v['total']}" for f, v in r["redundancy"].items())
                  + f"  verdicts {dict(Counter(r['alpha_verdicts']))}  round1 bit-identical {r['round1']['bit_identical']}")

def check_leak_diagnostic() -> dict:
    """Diagnostic of nullspace_rounds' leak failure, with its rule fixed before looking.

    Explanation tested: the leak comes only from rounds where the variable is gone (alpha at the upper edge,
    "no_signal"), where the ridge weights shrink until their part outside the removed subspace is at rounding level.
    Leaks are recomputed from the saved basis and weights (hash-guarded), not taken from the run. The explanation holds
    if (a) every round with leak > LEAK_TOLERANCE has alpha verdict "no_signal", and (b) every round with verdict "ok"
    has leak <= LEAK_TOLERANCE (so K and every probe before it are unaffected). Also recorded per round: leak, weight
    norm, norm of the leaked part.
    """
    saved = json.loads(OUT.read_text())["nullspace_rounds"]["result"]
    with np.load(verified_artifact(OUT, "nullspace_rounds")) as f:
        arrays = {key: f[key] for key in f.files}

    result: dict = {}
    holds_a, holds_b = [], []
    for variable in DATASETS:
        sites = {}
        for site in NULLSPACE_SITES:
            entry = saved[variable]["sites"][site]
            prefix = f"{variable}_{site}"
            basis, weights = arrays[f"{prefix}_basis"], arrays[f"{prefix}_weights"]
            m = entry["dims_per_round"]
            weight_norms = np.linalg.norm(weights, axis=(1, 2))
            leaked_norms = np.array([np.linalg.norm(basis[:, : k * m].T @ w) for k, w in enumerate(weights)])
            leaks = leaked_norms / weight_norms
            verdicts = np.array(entry["alpha_verdicts"])
            leaking = leaks > LEAK_TOLERANCE
            ok_rounds = verdicts == "ok"
            holds_a.append(bool((verdicts[leaking] == "no_signal").all()))
            holds_b.append(bool((leaks[ok_rounds] <= LEAK_TOLERANCE).all()))
            sites[site] = {
                "k": entry["summary"]["k"],
                "first_leaking_round": first_true(leaking),
                "n_leaking_rounds": int(leaking.sum()),
                "last_ok_round": int(np.flatnonzero(ok_rounds)[-1]) + 1 if ok_rounds.any() else None,
                "max_leak_ok_rounds": float(leaks[ok_rounds].max()) if ok_rounds.any() else None,
                "verdicts_of_leaking_rounds": dict(Counter(verdicts[leaking].tolist())),
                "weight_norm_round1": float(weight_norms[0]),
                "min_weight_norm": float(weight_norms.min()),
                "leaked_norm_range_leaking_rounds": (
                    [float(leaked_norms[leaking].min()), float(leaked_norms[leaking].max())] if leaking.any() else None
                ),
                "leaks": leaks.tolist(),
                "weight_norms": weight_norms.tolist(),
                "leaked_norms": leaked_norms.tolist(),
            }
        result[variable] = {"sites": sites}

    criteria = {
        "leaking_rounds_are_no_signal": all(holds_a),
        "ok_rounds_do_not_leak": all(holds_b),
    }
    return {
        "criteria": criteria, "leak_tolerance": LEAK_TOLERANCE,
        "source": {"key": "nullspace_rounds", "artifact_sha256": json.loads(OUT.read_text())["nullspace_rounds"]
                   ["result"]["artifact"]["sha256"]},
        **result,
        "explanation_holds": all(criteria.values()),
    }


def print_leak_diagnostic(result: dict) -> None:
    """Per variable and site: K vs the first leaking round, counts, and the size of weights and leaked parts."""
    for variable in DATASETS:
        print(f"\n{variable}")
        for site, r in result[variable]["sites"].items():
            print(f"  {site:9s} K {r['k']}  last ok round {r['last_ok_round']}  first leaking round "
                  f"{r['first_leaking_round']}  leaking {r['n_leaking_rounds']}  verdicts {r['verdicts_of_leaking_rounds']}"
                  f"  max leak (ok rounds) {r['max_leak_ok_rounds']:.2e}")
            print(f"    |W| round 1 {r['weight_norm_round1']:.3e}  min |W| {r['min_weight_norm']:.3e}"
                  f"  leaked part (leaking rounds) {r['leaked_norm_range_leaking_rounds']}")
    print(f"\nexplanation holds: {result['explanation_holds']}")
    
    
def check_covariance_exhaustion() -> dict:
    """Mechanism behind the leak: the train cross-covariance runs out where the probe weights collapse.

    Per variable, site and round k: |P_{k-1} Zᵀ y_c| on the train rows (Frobenius), relative to round 1, with the
    scaler rebuilt from the activations (must equal the saved one exactly) and P from the saved basis. y_c is
    centred, so this is the cross-covariance the round-k ridge fit sees (centring X does not change Xᵀy_c); a ridge
    fit's weights are exactly 0 when it is 0. Rule fixed before looking, at every site: (a) the covariance ratio is
    <= EXHAUSTION_RATIO at the first leaking round (leak_diagnostic); (b) the first round with |W_k| <=
    EXHAUSTION_RATIO x |W_1| equals the first leaking round. Guard check: the guarded run_rounds stops at that round
    and reproduces the saved alphas and weights of every earlier round bit for bit.
    """
    records = json.loads(OUT.read_text())
    saved_rounds = records["nullspace_rounds"]["result"]
    leak = records["leak_diagnostic"]["result"]
    with np.load(verified_artifact(OUT, "nullspace_rounds")) as f:
        arrays = {key: f[key] for key in f.files}

    result: dict = {}
    ok: dict[str, list[bool]] = {"scaler": [], "a": [], "b": [], "guard": []}
    for variable in DATASETS:
        table = load_joined(variable)
        roles = table["role"]
        y = probe_targets(variable, table["label"])
        train = roles == "train"
        y_centred = y[train] - y[train].mean(axis=0)
        evaluated = np.isin(roles, EVAL_ROLES)
        sites = {}
        for site in NULLSPACE_SITES:
            prefix = f"{variable}_{site}"
            x = site_features(table["activations"], site)
            scaler = train_scaler(x, roles)
            ok["scaler"].append(bool(np.array_equal(scaler.mean_, arrays[f"{prefix}_mean"])
                                     and np.array_equal(scaler.scale_, arrays[f"{prefix}_scale"])))
            z = scaler.transform(x)
            cross = z[train].T @ y_centred  # (d,) or (d, 2)
            cross_rows = cross.reshape(len(cross), -1).T  # (m, d): project_out works on rows
            basis, weights = arrays[f"{prefix}_basis"], arrays[f"{prefix}_weights"]
            m = weights.shape[2]
            norms = np.array([np.linalg.norm(project_out(cross_rows, basis, k * m)) for k in range(len(weights))])
            ratios = norms / norms[0]
            weight_norms = np.linalg.norm(weights, axis=(1, 2))
            exhausted = first_true(weight_norms <= EXHAUSTION_RATIO * weight_norms[0])
            first_leaking = leak[variable]["sites"][site]["first_leaking_round"]
            ok["a"].append(first_leaking is not None and bool(ratios[first_leaking - 1] <= EXHAUSTION_RATIO))
            ok["b"].append(exhausted == first_leaking)

            guarded = run_rounds(z, y, roles, evaluated, N_ROUNDS)
            n = len(guarded.alphas)
            ok["guard"].append(bool(
                guarded.exhausted_round == exhausted
                and np.array_equal(guarded.alphas[: n - 1], arrays[f"{prefix}_alphas"][: n - 1])
                and np.array_equal(guarded.weights[: n - 1], weights[: n - 1])
            ))
            last_ok = leak[variable]["sites"][site]["last_ok_round"]
            sites[site] = {
                "k": saved_rounds[variable]["sites"][site]["summary"]["k"],
                "last_ok_round": last_ok,
                "first_leaking_round": first_leaking,
                "weight_exhausted_round": exhausted,
                "guarded_run_exhausted_round": guarded.exhausted_round,
                "first_round_covariance_below_ratio": first_true(ratios <= EXHAUSTION_RATIO),
                "covariance_ratio_at_first_leaking": float(ratios[first_leaking - 1]) if first_leaking else None,
                "covariance_ratios_last_ok_to_first_leaking": (
                    ratios[last_ok - 1: first_leaking].tolist() if last_ok and first_leaking else None
                ),
                "covariance_ratios": ratios.tolist(),
            }
        result[variable] = {"sites": sites}

    criteria = {
        "scaler_rebuilt_equals_saved": all(ok["scaler"]),
        "covariance_exhausted_at_first_leaking_round": all(ok["a"]),
        "weight_collapse_equals_first_leaking_round": all(ok["b"]),
        "guarded_run_stops_there_and_reproduces_saved_rounds": all(ok["guard"]),
    }
    return {
        "criteria": criteria, "exhaustion_ratio": EXHAUSTION_RATIO,
        "source": {"key": "nullspace_rounds", "artifact_sha256": saved_rounds["artifact"]["sha256"]},
        **result,
        "explanation_holds": all(criteria.values()),
    }


def print_covariance_exhaustion(result: dict) -> None:
    """Per variable and site: K, last ok round, first leaking round, where weights and covariance run out."""
    for variable in DATASETS:
        print(f"\n{variable}")
        for site, r in result[variable]["sites"].items():
            print(f"  {site:9s} K {r['k']}  last ok {r['last_ok_round']}  first leaking {r['first_leaking_round']}"
                  f"  weights exhausted {r['weight_exhausted_round']}  guarded stop {r['guarded_run_exhausted_round']}"
                  f"  covariance < ratio from {r['first_round_covariance_below_ratio']}"
                  f"  ratio at first leaking {r['covariance_ratio_at_first_leaking']:.2e}")
            print("    covariance ratio, last ok -> first leaking: "
                  + "  ".join(f"{v:.1e}" for v in r["covariance_ratios_last_ok_to_first_leaking"]))
    print(f"\nexplanation holds: {result['explanation_holds']}")
    

def flat(p: np.ndarray) -> np.ndarray:
    """(n, 1) predictions -> (n,); (n, 2) unchanged."""
    return p[:, 0] if p.ndim == 2 and p.shape[1] == 1 else p


def validation_scores(variable: str, labels: np.ndarray, roles: np.ndarray, pred: np.ndarray) -> dict:
    """DATA.md scores on all given validation clips and per validation role."""
    p = flat(pred)
    scores = {"validation": probe_scores(variable, labels, p)}
    for role in EVAL_ROLES:
        rows = roles == role
        scores[role] = probe_scores(variable, labels[rows], p[rows])
    return scores


def relative_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max() / np.abs(b).max())


def check_fresh_probe_erasure() -> dict:
    """Is the variable's information removed, or only the train-fitted readout?

    Per variable and site (idx 1 / 9 / 18), for each removal arm (none; the nullspace basis at K, K·m dims; the train
    cross-covariance directions, m dims; random train-span subspaces of m and K·m dims, N_RANDOM_SEEDS seeds): the
    train-fit probe (RidgeCV leave-one-out on train, as in the nullspace rounds) scored on val_seen, and a fresh probe
    fit on the validation clips only, scored out of fold (nested grouped CV, D-42 grid). Passes if: outer folds
    partition the validation clips; the nested fits equal sklearn Ridge refits; the nullspace arm's train fit equals
    saved round K + 1; the covariance arm's train-fit predictions are constant (Xᵀy_c = 0 ⇒ weights 0); bases
    orthonormal; scores finite; no alpha failure (every fit).
    """
    saved_rounds = json.loads(OUT.read_text())["nullspace_rounds"]["result"]
    with np.load(verified_artifact(OUT, "nullspace_rounds")) as f:
        arrays = {key: f[key] for key in f.files}

    result: dict = {}
    ok: dict[str, list[bool]] = {name: [] for name in (
        "partition", "sklearn", "saved_round", "constant", "orthonormal", "finite", "alpha")}
    for variable in DATASETS:
        table = load_joined(variable)
        roles, labels, ids = table["role"], table["label"], table["id"]
        y = probe_targets(variable, labels)
        evaluated = np.isin(roles, EVAL_ROLES)
        ev_roles, ev_labels, ev_ids, ev_y = roles[evaluated], labels[evaluated], ids[evaluated], y[evaluated]
        seen = ev_roles == "val_seen"
        train = roles == "train"
        label_sd = float(y[train].std())
        sites = {}
        for site in NULLSPACE_SITES:
            start = time.perf_counter()
            prefix = f"{variable}_{site}"
            x = site_features(table["activations"], site)
            z = train_scaler(x, roles).transform(x)
            summary = saved_rounds[variable]["sites"][site]
            m, k = summary["dims_per_round"], summary["summary"]["k"]
            cross = z[train].T @ (y[train] - y[train].mean(axis=0))
            covariance_basis, _ = np.linalg.qr(cross.reshape(len(cross), -1))
            span = train_span(z, roles)
            randoms = [random_span_basis(span, k * m, SEED + s) for s in range(N_RANDOM_SEEDS)]
            for q in (covariance_basis, *randoms):
                ok["orthonormal"].append(float(np.abs(q.T @ q - np.eye(q.shape[1])).max()) <= ORTHONORMAL_TOLERANCE)

            arms: dict[str, tuple[np.ndarray, int]] = {
                "none": (np.empty((z.shape[1], 0)), 0),
                "nullspace_k": (arrays[f"{prefix}_basis"], k * m),
                "covariance": (covariance_basis, m),
            }
            arms |= {f"random_m_seed{s}": (b, m) for s, b in enumerate(randoms)}
            arms |= {f"random_km_seed{s}": (b, k * m) for s, b in enumerate(randoms)}

            entries = {}
            for arm, (basis, n_cols) in arms.items():
                zp = project_out(z, basis, n_cols)
                ridge = train_ridge(zp, y, roles)
                train_pred = np.reshape(ridge.predict(zp[evaluated]), (len(ev_y), -1))
                train_seen = probe_scores(variable, ev_labels[seen], flat(train_pred)[seen])
                train_verdict = nullspace_alpha_verdict(grid_edge(float(ridge.alpha_), ALPHAS), train_seen["r2"])
                fresh = nested_cv_predictions(zp[evaluated], ev_y, ev_ids, ERASURE_FOLDS, SEED)
                fresh_scores = validation_scores(variable, ev_labels, ev_roles, fresh.predictions)
                fresh_verdicts = [nullspace_alpha_verdict(e, fresh_scores["validation"]["r2"])
                                  for e in fresh.alpha_edges]

                ok["partition"].append(bool(np.isfinite(fresh.predictions).all()
                                            and len(np.unique(fresh.folds)) == ERASURE_FOLDS))
                ok["finite"].append(all(np.isfinite(v) for s in (train_seen, *fresh_scores.values()) for v in s.values()))
                ok["alpha"].append(train_verdict != "failure" and "failure" not in fresh_verdicts)
                entry = {
                    "dims_removed": n_cols,
                    "train_fit": {"alpha": float(ridge.alpha_), "alpha_verdict": train_verdict, "val_seen": train_seen},
                    "fresh": {"alphas": fresh.alphas.tolist(), "alpha_edges": list(fresh.alpha_edges),
                              "alpha_verdicts": fresh_verdicts, **fresh_scores},
                }
                if arm == "none":
                    diffs = []
                    for f in range(ERASURE_FOLDS):
                        held = fresh.folds == f
                        refit = Ridge(alpha=fresh.alphas[f], fit_intercept=True).fit(zp[evaluated][~held], ev_y[~held])
                        diffs.append(relative_diff(fresh.predictions[held], refit.predict(zp[evaluated][held])))
                    entry["sklearn_refit_max_relative_diff"] = max(diffs)
                    ok["sklearn"].append(max(diffs) <= REFIT_TOLERANCE)
                if arm == "nullspace_k":
                    saved = arrays[f"{prefix}_predictions"][evaluated, k]  # round K + 1: fit after K·m dims removed
                    diff = relative_diff(train_pred, saved)
                    entry["saved_round"] = {"round": k + 1, "max_relative_diff": diff,
                                            "bit_identical": bool(np.array_equal(train_pred, saved))}
                    ok["saved_round"].append(diff <= REFIT_TOLERANCE)
                if arm == "covariance":
                    spread = float((train_pred.max(axis=0) - train_pred.min(axis=0)).max() / label_sd)
                    entry["train_fit_spread_over_label_sd"] = spread
                    ok["constant"].append(spread <= CONSTANT_TOLERANCE)
                entries[arm] = entry

            def aggregate(prefix_: str) -> dict:
                picked = [e for a, e in entries.items() if a.startswith(prefix_)]
                fresh_r2 = [e["fresh"]["validation"]["r2"] for e in picked]
                train_r2 = [e["train_fit"]["val_seen"]["r2"] for e in picked]
                return {"dims_removed": picked[0]["dims_removed"], "fresh_validation_r2_mean": float(np.mean(fresh_r2)),
                        "fresh_validation_r2_min": min(fresh_r2), "fresh_validation_r2_max": max(fresh_r2),
                        "train_fit_val_seen_r2_mean": float(np.mean(train_r2))}

            sites[site] = {
                "plot_index": plot_index(site), "k": k, "dims_per_round": m, "seconds": time.perf_counter() - start,
                "arms": entries, "random_m": aggregate("random_m_"), "random_km": aggregate("random_km_"),
            }
        result[variable] = {"n_validation": int(evaluated.sum()), "sites": sites}

    criteria = {
        "outer_folds_partition_clips": all(ok["partition"]),
        "nested_cv_matches_sklearn_refit": all(ok["sklearn"]),
        "nullspace_arm_equals_saved_round": all(ok["saved_round"]),
        "covariance_arm_train_fit_constant": all(ok["constant"]),
        "bases_orthonormal": all(ok["orthonormal"]),
        "scores_finite": all(ok["finite"]),
        "no_alpha_failure": all(ok["alpha"]),
    }
    return {
        "criteria": criteria, "folds": ERASURE_FOLDS, "seed": SEED, "n_random_seeds": N_RANDOM_SEEDS,
        "source": {"key": "nullspace_rounds", "artifact_sha256": saved_rounds["artifact"]["sha256"]},
        **result, "passed": all(criteria.values()),
    }


def print_fresh_probe_erasure(result: dict) -> None:
    """Per variable and site: train-fit val-seen R² vs fresh out-of-fold validation R² for each arm."""
    for variable in DATASETS:
        other = "circular_mae" if variable == "direction" else "mae"
        print(f"\n{variable}  (train-fit val-seen R2 | fresh CV R2 on validation | fresh {other})")
        for site, r in result[variable]["sites"].items():
            print(f"  {site} idx {r['plot_index']}  K {r['k']}  ({r['seconds']:.0f} s)")
            for arm in ("none", "nullspace_k", "covariance"):
                e = r["arms"][arm]
                print(f"    {arm:12s} dims {e['dims_removed']:3d}  {e['train_fit']['val_seen']['r2']:7.3f} | "
                      f"{e['fresh']['validation']['r2']:7.3f} | {e['fresh']['validation'][other]:7.3f}"
                      f"  verdicts {e['train_fit']['alpha_verdict']} / {dict(Counter(e['fresh']['alpha_verdicts']))}")
            for name in ("random_m", "random_km"):
                a = r[name]
                print(f"    {name:12s} dims {a['dims_removed']:3d}  {a['train_fit_val_seen_r2_mean']:7.3f} | "
                      f"{a['fresh_validation_r2_mean']:7.3f} (min {a['fresh_validation_r2_min']:.3f}, "
                      f"max {a['fresh_validation_r2_max']:.3f})")

def check_alpha_sweep() -> dict:
    """Does K depend on the probe's regularization? The nullspace rounds at the headline layer with a fixed alpha.

    Per variable and alpha in SWEEP_ALPHAS: run_rounds with a one-value alpha grid (guard on, cap N_ROUNDS), scored
    on val_seen; K = first round with R² < NULL_R2. The alpha edge rule does not apply (one-value grid); reaching the
    cap without exhaustion is expected at small alpha. Passes if: every run fits exactly the train clips; the basis
    is orthonormal; scores are finite; K <= the guard round wherever the guard fires (at exhaustion the prediction is
    the train mean, so val_seen R² <= 0). Reported: guard round vs K (exceptions to "guard > K" listed), max leak
    (observation), the leave-one-out run's K for reference.
    """
    saved_rounds = json.loads(OUT.read_text())["nullspace_rounds"]["result"]
    result: dict = {}
    ok: dict[str, list[bool]] = {"n_fit": [], "orthonormal": [], "finite": [], "k_guard": []}
    exceptions = []
    for variable in DATASETS:
        table = load_joined(variable)
        roles, labels = table["role"], table["label"]
        y = probe_targets(variable, labels)
        evaluated = np.isin(roles, EVAL_ROLES)
        seen = roles[evaluated] == "val_seen"
        x = site_features(table["activations"], SWEEP_SITE)
        z = train_scaler(x, roles).transform(x)
        error_name = "circular_mae" if variable == "direction" else "mae"
        runs = {}
        for alpha in SWEEP_ALPHAS:
            start = time.perf_counter()
            run = run_rounds(z, y, roles, evaluated, N_ROUNDS, alphas=np.array([alpha]))
            seconds = time.perf_counter() - start
            scores = round_scores(variable, labels[roles == "val_seen"], run.predictions[:, seen])
            seen_r2 = np.array([s["r2"] for s in scores])
            summary = curve_summary(seen_r2, run.dims_per_round)
            k, guard = summary["k"], run.exhausted_round
            q = run.basis
            ok["n_fit"].append(run.n_fit == EXPECTED_FIT[variable])
            ok["orthonormal"].append(float(np.abs(q.T @ q - np.eye(q.shape[1])).max()) <= ORTHONORMAL_TOLERANCE)
            ok["finite"].append(bool(np.isfinite(seen_r2).all()))
            ok["k_guard"].append(guard is None or (k is not None and k <= guard))
            if guard is not None and not (k is not None and k < guard):
                exceptions.append({"variable": variable, "alpha": alpha, "k": k, "guard_round": guard})
            runs[f"{alpha:g}"] = {
                "alpha": alpha, "seconds": seconds, "rounds_run": len(run.alphas), "exhausted_round": guard,
                "summary": summary, "round1_val_seen": scores[0],
                "max_leak_observation": float(np.nanmax(run.leaks)),
                "val_seen_r2": seen_r2.tolist(), f"val_seen_{error_name}": [s[error_name] for s in scores],
            }
        reference = saved_rounds[variable]["sites"][SWEEP_SITE]
        result[variable] = {
            "leave_one_out_reference": {"k": reference["summary"]["k"], "round1_alpha": reference["round1"]["alpha"],
                                        "round1_val_seen_r2": reference["val_seen_r2"][0]},
            "runs": runs,
        }

    criteria = {
        "n_fit_equals_train_count": all(ok["n_fit"]),
        "basis_orthonormal": all(ok["orthonormal"]),
        "scores_finite": all(ok["finite"]),
        "k_not_after_guard": all(ok["k_guard"]),
    }
    return {
        "criteria": criteria, "site": SWEEP_SITE, "plot_index": plot_index(SWEEP_SITE), "alphas": list(SWEEP_ALPHAS),
        "n_rounds": N_ROUNDS, "stop_r2": NULL_R2, "guard_not_after_k_exceptions": exceptions, **result,
        "passed": all(criteria.values()),
    }


def print_alpha_sweep(result: dict) -> None:
    """Per variable: the leave-one-out reference, then per fixed alpha round-1 R², K, guard round, rounds run."""
    for variable in DATASETS:
        ref = result[variable]["leave_one_out_reference"]
        print(f"\n{variable}  (leave-one-out: alpha {ref['round1_alpha']:.3g}, round-1 R2 {ref['round1_val_seen_r2']:.3f},"
              f" K {ref['k']})")
        for name, r in result[variable]["runs"].items():
            k = r["summary"]["k"] if r["summary"]["k"] is not None else f"> {result['n_rounds']}"
            print(f"  alpha {name:>6s}  round-1 R2 {r['round1_val_seen']['r2']:7.3f}  K {str(k):>6s}"
                  f"  guard {str(r['exhausted_round']):>5s}  rounds run {r['rounds_run']:3d}"
                  f"  max leak {r['max_leak_observation']:.1e}  {r['seconds']:.0f} s")
    print(f"\nguard not after K (exceptions): {result['guard_not_after_k_exceptions']}")            

CHECKS = {
    "nullspace_rounds": (check_nullspace_rounds, print_nullspace_rounds),
    "leak_diagnostic": (check_leak_diagnostic, print_leak_diagnostic),
    "covariance_exhaustion": (check_covariance_exhaustion, print_covariance_exhaustion),
    "fresh_probe_erasure": (check_fresh_probe_erasure, print_fresh_probe_erasure),
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