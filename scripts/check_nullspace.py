"""Checks for iterative nullspace probing: fit a ridge probe, remove its weight directions, fit again.

Run one check at a time from the repo root, with .venv active:
    python scripts/check_nullspace.py <check>

Each check prints a summary and stores its full result under its own key in results/nullspace/checks.json.
"""
import argparse
import json
import time
from collections import Counter
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics.pairwise import rbf_kernel
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from vjepa_physics.data import DATASETS
from vjepa_physics.evidence import code_changes, file_sha256, repo_root, save_result, verified_artifact
from vjepa_physics.extraction import SITES, plot_index
from vjepa_physics.joined import load_joined
from vjepa_physics.metrics import bootstrap_indices, mae, percentile_interval, resampled_r2
from vjepa_physics.nullspace import (
    EXHAUSTION_RATIO, NULL_R2, RANK_TOLERANCE, composite_maps, covariance_basis, curve_summary, first_true,
    grid_edge, nullspace_alpha_verdict, project_out, random_span_basis, redundancy_counts, round_scores, run_rounds,
    train_ridge, train_scaler, train_span,
)
from vjepa_physics.plotting import DATASET_COLOUR, INK, INK_MUTED, INK_SECONDARY, SURFACE, style_axes
from vjepa_physics.probes import (
    ALPHAS, clip_folds, grouped_cv_ridge, label_permutations, nested_cv_predictions, probe_scores, probe_targets,
    site_features,
)
from vjepa_physics.reproducibility import SEED
from vjepa_physics.baselines import FLOOR_ALPHAS, GAMMA_FACTORS, kernel_ridge, rbf_kernel_ridge, squared_distances

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
RBF_CHECK_ROWS = 200  # clips used to compare our RBF Gram with sklearn's

CONTROL_SITE = "block_8"  # kernel negative control at the headline layer (index 9)
KERNEL_PERMUTATIONS = 5  # shuffled train-label permutations (the first ones of the probe shuffled-label control)
SHUFFLED_MAX_R2 = 0.1  # every shuffled fit's val-seen R² must stay below this
SHUFFLED_MIN_MEAN_CIRCULAR_MAE = 80.0  # direction: circular MAE averaged over the permutations (chance 90)

EXTENDED_RELATIVE_ALPHAS = np.logspace(-9, 3, 49)  # the floor grid (0.25-decade steps) extended down to 1e-9
EXTENDED_GAMMA_FACTORS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
GRID_R2_TOLERANCE = 0.05  # the recovery finding does not depend on the grid if every refit stays within this
MIN_HAT_GAP = 1e-6  # numerical guard: flag a fit whose smallest 1 - h_ii falls below this

CONTROL_ROUNDS = N_ROUNDS  # control runs cover the same rounds as the real run
SPAN_TOLERANCE_CHECK = 1e-10  # control bases must lie in the train span to this (max abs residual)

PROFILE_INDICES = tuple(range(0, 25, 3))  # every third hidden_states index: 0, 3, ..., 24
PROFILE_STOP_R2 = 0.05  # the profile's curve is reported up to the first val-seen R² below this

TEST_ROLES = ("test_seen", "test_unseen")
LAYER_TESTS = REPO / "results/layer_curves/checks.json"  # key "test_scores": round 1 must match its probe scores
HEADLINE_SITE = "block_8"  # headline layer (index 9): bootstrap intervals here only
N_BOOTSTRAP = 10_000

FIGURE_ROUNDS = 20  # rounds shown: every real run is exhausted by round 19; the controls stay flat to 150 (saved)
FIGURE_PATH = REPO / "results/nullspace/nullspace_rounds.png"


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


def check_kernel_erasure() -> dict:
    """Does the variable survive linear erasure nonlinearly? RBF kernel ridge on the projected features.

    Per variable and site (idx 1 / 9 / 18) and arm (none; train cross-covariance directions, m dims; nullspace basis
    at K, K·m dims; random train-span, m dims, N_RANDOM_SEEDS seeds, the same bases as fresh_probe_erasure): RBF kernel
    ridge fit on train (gamma = factor x median-distance gamma, alpha from the floor's relative grid, both by exact
    LOO), scored on val_seen and val_unseen. Passes if: the RBF Gram equals sklearn's; kernel_ridge on a linear Gram
    equals RidgeCV (alpha, LOO error, predictions); every fit saw exactly the train clips; only validation rows were
    predicted; scores finite; no alpha failure. Gamma on a grid edge is reported, not judged.
    """
    records = json.loads(OUT.read_text())
    saved_rounds = records["nullspace_rounds"]["result"]
    linear = records["fresh_probe_erasure"]["result"]
    with np.load(verified_artifact(OUT, "nullspace_rounds")) as f:
        bases = {key: f[key] for key in f.files if key.endswith("_basis")}

    result: dict = {}
    ok: dict[str, list[bool]] = {name: [] for name in ("rbf", "linear", "n_fit", "rows", "finite", "alpha")}
    for variable in DATASETS:
        table = load_joined(variable)
        roles, labels = table["role"], table["label"]
        y = probe_targets(variable, labels)
        evaluated = np.isin(roles, EVAL_ROLES)
        train = roles == "train"
        error_name = "circular_mae" if variable == "direction" else "mae"
        sites = {}
        for site in NULLSPACE_SITES:
            start = time.perf_counter()
            x = site_features(table["activations"], site)
            z = train_scaler(x, roles).transform(x)
            summary = saved_rounds[variable]["sites"][site]
            m, k = summary["dims_per_round"], summary["summary"]["k"]
            span = train_span(z, roles)
            arms: dict[str, tuple[np.ndarray, int]] = {
                "none": (np.empty((z.shape[1], 0)), 0),
                "covariance": (covariance_basis(z, y, roles), m),
                "nullspace_k": (bases[f"{variable}_{site}_basis"], k * m),
            }
            arms |= {f"random_m_seed{s}": (random_span_basis(span, k * m, SEED + s)[:, :m], m)
                     for s in range(N_RANDOM_SEEDS)}

            entries = {}
            for arm, (basis, n_cols) in arms.items():
                zp = project_out(z, basis, n_cols)
                rbf = rbf_kernel_ridge(zp, y, roles, evaluated)
                pred = rbf.fit.predictions
                scores = {role: probe_scores(variable, labels[roles == role], pred[roles == role]) for role in EVAL_ROLES}
                verdict = nullspace_alpha_verdict(rbf.fit.alpha_edge, scores["val_seen"]["r2"])
                ok["n_fit"].append(rbf.fit.n_fit == EXPECTED_FIT[variable])
                ok["rows"].append(bool(np.isfinite(pred[evaluated]).all() and np.isnan(pred[~evaluated]).all()))
                ok["finite"].append(all(np.isfinite(v) for s in scores.values() for v in s.values()))
                ok["alpha"].append(verdict != "failure")
                entry = {
                    "dims_removed": n_cols, "gamma": rbf.gamma, "gamma_median": rbf.gamma_median,
                    "gamma_factor": rbf.gamma / rbf.gamma_median, "gamma_edge": rbf.gamma_edge,
                    "loo_mse_per_gamma": rbf.loo_mse.tolist(), "alpha": rbf.fit.alpha,
                    "alpha_edge": rbf.fit.alpha_edge, "alpha_verdict": verdict, **scores,
                    "linear_train_fit_val_seen_r2": linear[variable]["sites"][site]["arms"][arm]["train_fit"]["val_seen"]["r2"],
                }
                if arm == "none":
                    rows = slice(0, RBF_CHECK_ROWS)
                    ours = np.exp(-rbf.gamma * squared_distances(zp[rows]))
                    rbf_diff = float(np.abs(ours - rbf_kernel(zp[rows], gamma=rbf.gamma)).max())
                    lin = kernel_ridge(zp @ zp.T, y, roles, evaluated)
                    centred = zp[train] - zp[train].mean(axis=0)
                    ridge = RidgeCV(alphas=FLOOR_ALPHAS * (centred**2).sum() / train.sum(),
                                    store_cv_results=True).fit(zp[train], y[train])
                    ridge_loo = float(ridge.cv_results_.reshape(int(train.sum()), -1, len(FLOOR_ALPHAS))
                                      .mean(axis=(0, 1)).min())
                    checks = {
                        "rbf_gram_max_abs_diff": rbf_diff,
                        "linear_alpha_relative_diff": abs(lin.alpha - float(ridge.alpha_)) / float(ridge.alpha_),
                        "linear_loo_relative_diff": abs(lin.loo_mse - ridge_loo) / ridge_loo,
                        "linear_prediction_relative_diff": relative_diff(
                            lin.predictions[evaluated], np.reshape(ridge.predict(zp[evaluated]), lin.predictions[evaluated].shape)),
                    }
                    entry["implementation_checks"] = checks
                    ok["rbf"].append(rbf_diff <= 1e-10)
                    ok["linear"].append(checks["linear_alpha_relative_diff"] <= 1e-12
                                        and checks["linear_loo_relative_diff"] <= 1e-8
                                        and checks["linear_prediction_relative_diff"] <= 1e-8)
                entries[arm] = entry

            random_r2 = [entries[f"random_m_seed{s}"]["val_seen"]["r2"] for s in range(N_RANDOM_SEEDS)]
            sites[site] = {
                "plot_index": plot_index(site), "k": k, "dims_per_round": m, "seconds": time.perf_counter() - start,
                "arms": entries,
                "random_m": {"val_seen_r2_mean": float(np.mean(random_r2)), "val_seen_r2_min": min(random_r2),
                             "val_seen_r2_max": max(random_r2)},
            }
        result[variable] = {"error_metric": error_name, "sites": sites}

    criteria = {
        "rbf_gram_matches_sklearn": all(ok["rbf"]),
        "kernel_loo_matches_ridgecv": all(ok["linear"]),
        "n_fit_equals_train_count": all(ok["n_fit"]),
        "only_validation_rows_predicted": all(ok["rows"]),
        "scores_finite": all(ok["finite"]),
        "no_alpha_failure": all(ok["alpha"]),
    }
    return {
        "criteria": criteria, "gamma_factors": list(GAMMA_FACTORS), "relative_alphas": FLOOR_ALPHAS.tolist(),
        "source": {"key": "nullspace_rounds", "artifact_sha256": saved_rounds["artifact"]["sha256"]},
        **result, "passed": all(criteria.values()),
    }


def print_kernel_erasure(result: dict) -> None:
    """Per variable and site: linear train-fit val-seen R² vs RBF kernel val-seen R² per arm, gamma factor, verdict."""
    for variable in DATASETS:
        other = result[variable]["error_metric"]
        print(f"\n{variable}  (linear val-seen R2 | kernel val-seen R2 | kernel {other} | gamma factor | alpha verdict)")
        for site, r in result[variable]["sites"].items():
            print(f"  {site} idx {r['plot_index']}  K {r['k']}  ({r['seconds']:.0f} s)")
            for arm in ("none", "covariance", "nullspace_k", "random_m_seed0"):
                e = r["arms"][arm]
                print(f"    {arm:14s} dims {e['dims_removed']:3d}  {e['linear_train_fit_val_seen_r2']:7.3f} | "
                      f"{e['val_seen']['r2']:7.3f} | {e['val_seen'][other]:7.3f} | {e['gamma_factor']:4g}"
                      f"{' (edge)' if e['gamma_edge'] else ''} | {e['alpha_verdict']}")
            rm = r["random_m"]
            print(f"    random_m (5)   kernel val-seen R2 mean {rm['val_seen_r2_mean']:.3f} "
                  f"(min {rm['val_seen_r2_min']:.3f}, max {rm['val_seen_r2_max']:.3f})")
            print(f"    implementation checks {r['arms']['none']['implementation_checks']}")
            

def check_kernel_shuffled_labels() -> dict:
    """Negative control for the kernel probe after erasure: can the pipeline produce signal from shuffled labels?

    Per variable at CONTROL_SITE: the train cross-covariance direction(s) of the TRUE labels removed (as in the
    real covariance arm), then RBF kernel ridge with the same grids and train LOO selection fit on
    KERNEL_PERMUTATIONS permutations of the train labels (seed SEED; only train rows move), scored on val_seen with
    the true labels. Passes if: every shuffled fit's val-seen R² < SHUFFLED_MAX_R2; direction's circular MAE averaged
    over the permutations > SHUFFLED_MIN_MEAN_CIRCULAR_MAE; every fit saw exactly the train clips; no alpha failure
    (the alpha rule covers every fit). Chosen alphas and gammas recorded.
    """
    real = json.loads(OUT.read_text())["kernel_erasure"]["result"]
    result: dict = {}
    ok: dict[str, list[bool]] = {"r2": [], "circular": [], "n_fit": [], "alpha": []}
    for variable in DATASETS:
        table = load_joined(variable)
        roles, labels = table["role"], table["label"]
        y = probe_targets(variable, labels)
        evaluated = np.isin(roles, EVAL_ROLES)
        seen = roles == "val_seen"
        train = np.flatnonzero(roles == "train")
        x = site_features(table["activations"], CONTROL_SITE)
        z = train_scaler(x, roles).transform(x)
        basis = covariance_basis(z, y, roles)  # true labels, as in the real arm
        zp = project_out(z, basis, basis.shape[1])
        permutations = label_permutations(len(train), KERNEL_PERMUTATIONS, SEED)

        fits = []
        for permutation in permutations:
            y_shuffled = y.copy()
            y_shuffled[train] = y[train[permutation]]
            rbf = rbf_kernel_ridge(zp, y_shuffled, roles, evaluated)
            scores = probe_scores(variable, labels[seen], rbf.fit.predictions[seen])
            verdict = nullspace_alpha_verdict(rbf.fit.alpha_edge, scores["r2"])
            ok["n_fit"].append(rbf.fit.n_fit == EXPECTED_FIT[variable])
            ok["alpha"].append(verdict != "failure")
            fits.append({
                "labels_moved_fraction": float(np.mean(labels[train[permutation]] != labels[train])),
                "gamma_factor": rbf.gamma / rbf.gamma_median, "gamma_edge": rbf.gamma_edge,
                "alpha": rbf.fit.alpha, "alpha_edge": rbf.fit.alpha_edge, "alpha_verdict": verdict, **scores,
            })
        r2s = [f["r2"] for f in fits]
        ok["r2"].append(max(r2s) < SHUFFLED_MAX_R2)
        entry: dict = {"dims_removed": int(basis.shape[1]), "max_r2": max(r2s), "fits": fits,
                       "real_covariance_arm_val_seen_r2":
                           real[variable]["sites"][CONTROL_SITE]["arms"]["covariance"]["val_seen"]["r2"]}
        if variable == "direction":
            errors = [f["circular_mae"] for f in fits]
            entry |= {"mean_circular_mae": float(np.mean(errors)), "min_circular_mae": min(errors)}
            ok["circular"].append(entry["mean_circular_mae"] > SHUFFLED_MIN_MEAN_CIRCULAR_MAE)
        result[variable] = entry

    criteria = {
        "max_shuffled_r2_below_0_1": all(ok["r2"]),
        "direction_mean_circular_mae_above_80": bool(ok["circular"]) and all(ok["circular"]),
        "n_fit_equals_train_count": all(ok["n_fit"]),
        "no_alpha_failure": all(ok["alpha"]),
    }
    return {
        "criteria": criteria, "site": CONTROL_SITE, "plot_index": plot_index(CONTROL_SITE),
        "n_permutations": KERNEL_PERMUTATIONS, "seed": SEED, "gamma_factors": list(GAMMA_FACTORS),
        **result, "passed": all(criteria.values()),
    }


def print_kernel_shuffled_labels(result: dict) -> None:
    """Per variable: each shuffled fit's val-seen R², gamma factor and alpha verdict, next to the real arm's R²."""
    for variable in DATASETS:
        r = result[variable]
        extra = (f"  circular MAE mean {r['mean_circular_mae']:.2f} min {r['min_circular_mae']:.2f}"
                 if variable == "direction" else "")
        print(f"\n{variable}  (covariance removed, {r['dims_removed']} dims; real arm val-seen R2 "
              f"{r['real_covariance_arm_val_seen_r2']:.3f})  max shuffled R2 {r['max_r2']:.3f}{extra}")
        for i, f in enumerate(r["fits"]):
            print(f"  permutation {i}  R2 {f['r2']:7.3f}  gamma factor {f['gamma_factor']:g}"
                  f"{' (edge)' if f['gamma_edge'] else ''}  alpha {f['alpha']:.3g} {f['alpha_verdict']}"
                  f"  moved {f['labels_moved_fraction']:.3f}")


def check_kernel_grid_diagnostic() -> dict:
    """Diagnostic of kernel_erasure's alpha failure, with its rule fixed before looking.

    Refits only the arms whose alpha hit the lower grid edge (read from the saved kernel_erasure result): first with
    the original grids (must reproduce the saved val-seen R², integrity), then with alphas extended to 1e-9 relative
    and gamma factors 0.125-8. Rule: "the recovery finding does not depend on the grid" holds if every extended refit's
    val-seen R² is within GRID_R2_TOLERANCE of the saved one. An edge hit again is recorded, not failed. Numerical
    guard: min(1 - h_ii) at the chosen alpha, flagged below MIN_HAT_GAP.
    """
    records = json.loads(OUT.read_text())
    saved = records["kernel_erasure"]["result"]
    with np.load(verified_artifact(OUT, "nullspace_rounds")) as f:
        bases = {key: f[key] for key in f.files if key.endswith("_basis")}

    arms_out: list[dict] = []
    integrity, holds = [], []
    for variable in DATASETS:
        flagged = [(site, arm) for site, r in saved[variable]["sites"].items()
                   for arm, e in r["arms"].items() if e["alpha_edge"] == "lower"]
        if not flagged:
            continue
        table = load_joined(variable)
        roles, labels = table["role"], table["label"]
        y = probe_targets(variable, labels)
        evaluated = np.isin(roles, EVAL_ROLES)
        seen = roles == "val_seen"
        for site, arm in flagged:
            x = site_features(table["activations"], site)
            z = train_scaler(x, roles).transform(x)
            entry_saved = saved[variable]["sites"][site]["arms"][arm]
            if arm == "covariance":
                basis = covariance_basis(z, y, roles)
            elif arm == "nullspace_k":
                basis = bases[f"{variable}_{site}_basis"]
            else:
                raise RuntimeError(f"unexpected flagged arm {arm}")
            zp = project_out(z, basis, entry_saved["dims_removed"])

            original = rbf_kernel_ridge(zp, y, roles, evaluated)
            original_r2 = probe_scores(variable, labels[seen], original.fit.predictions[seen])["r2"]
            extended = rbf_kernel_ridge(zp, y, roles, evaluated, EXTENDED_GAMMA_FACTORS, EXTENDED_RELATIVE_ALPHAS)
            scores = probe_scores(variable, labels[seen], extended.fit.predictions[seen])
            diff = scores["r2"] - entry_saved["val_seen"]["r2"]
            integrity.append(abs(original_r2 - entry_saved["val_seen"]["r2"]) <= 1e-12)
            holds.append(abs(diff) <= GRID_R2_TOLERANCE)
            arms_out.append({
                "variable": variable, "site": site, "plot_index": plot_index(site), "arm": arm,
                "saved_val_seen_r2": entry_saved["val_seen"]["r2"], "original_grid_refit_r2": original_r2,
                "extended_val_seen": scores, "r2_change": diff,
                "extended_alpha": extended.fit.alpha, "extended_alpha_edge": extended.fit.alpha_edge,
                "extended_gamma_factor": extended.gamma / extended.gamma_median,
                "extended_gamma_edge": extended.gamma_edge, "extended_loo_mse_per_gamma": extended.loo_mse.tolist(),
                "min_one_minus_hat": extended.fit.min_one_minus_hat,
                "hat_gap_flag": bool(extended.fit.min_one_minus_hat < MIN_HAT_GAP),
            })

    criteria = {"original_grid_reproduces_saved": bool(integrity) and all(integrity)}
    return {
        "criteria": criteria, "n_flagged_arms": len(arms_out), "r2_tolerance": GRID_R2_TOLERANCE,
        "extended_relative_alphas": [float(EXTENDED_RELATIVE_ALPHAS[0]), float(EXTENDED_RELATIVE_ALPHAS[-1]),
                                     len(EXTENDED_RELATIVE_ALPHAS)],
        "extended_gamma_factors": list(EXTENDED_GAMMA_FACTORS), "min_hat_gap": MIN_HAT_GAP,
        "source": {"key": "kernel_erasure"}, "arms": arms_out,
        "explanation_holds": bool(holds) and all(holds),
    }


def print_kernel_grid_diagnostic(result: dict) -> None:
    """One line per flagged arm: saved vs extended R², change, new alpha/gamma edges, hat-gap guard."""
    for a in result["arms"]:
        print(f"{a['variable']:12s} {a['site']:9s} {a['arm']:11s}  saved {a['saved_val_seen_r2']:.3f}"
              f"  (orig refit {a['original_grid_refit_r2']:.3f})  extended {a['extended_val_seen']['r2']:.3f}"
              f"  change {a['r2_change']:+.3f}  alpha {a['extended_alpha']:.2e} edge {a['extended_alpha_edge']}"
              f"  gamma {a['extended_gamma_factor']:g} edge {a['extended_gamma_edge']}"
              f"  min(1-h) {a['min_one_minus_hat']:.2e}{'  FLAG' if a['hat_gap_flag'] else ''}")
    print(f"\ncriteria {result['criteria']}  explanation holds: {result['explanation_holds']}"
          f"  ({result['n_flagged_arms']} arms)")

def check_kernel_hat_gap() -> dict:
    """Observation: was each original kernel_erasure selection numerically sound?

    Refits all kernel_erasure fits with the original grids (integrity: each must reproduce its saved val-seen R²)
    and records min(1 - h_ii) at the chosen alpha. Quoting rule fixed before running: >= MIN_HAT_GAP -> the
    selection is numerically sound and the score is quoted as a point value (edge flag kept where it applies);
    below it or negative -> unreliable, quoted only as the range across the grids tried (original and, where it
    exists, the extended grid of kernel_grid_diagnostic).
    """
    records = json.loads(OUT.read_text())
    saved = records["kernel_erasure"]["result"]
    extended = {(a["variable"], a["site"], a["arm"]): a["extended_val_seen"]["r2"]
                for a in records["kernel_grid_diagnostic"]["result"]["arms"]}
    with np.load(verified_artifact(OUT, "nullspace_rounds")) as f:
        bases = {key: f[key] for key in f.files if key.endswith("_basis")}

    result: dict = {}
    reproduced = []
    for variable in DATASETS:
        table = load_joined(variable)
        roles, labels = table["role"], table["label"]
        y = probe_targets(variable, labels)
        evaluated = np.isin(roles, EVAL_ROLES)
        seen = roles == "val_seen"
        sites = {}
        for site, r in saved[variable]["sites"].items():
            x = site_features(table["activations"], site)
            z = train_scaler(x, roles).transform(x)
            span = train_span(z, roles)
            m, k = r["dims_per_round"], r["k"]
            arms = {}
            for arm, e in r["arms"].items():
                if arm == "none":
                    basis = np.empty((z.shape[1], 0))
                elif arm == "covariance":
                    basis = covariance_basis(z, y, roles)
                elif arm == "nullspace_k":
                    basis = bases[f"{variable}_{site}_basis"]
                else:  # random_m_seed<s>: the same bases as kernel_erasure
                    seed = int(arm.removeprefix("random_m_seed"))
                    basis = random_span_basis(span, k * m, SEED + seed)[:, :m]
                zp = project_out(z, basis, e["dims_removed"])
                rbf = rbf_kernel_ridge(zp, y, roles, evaluated)
                r2 = probe_scores(variable, labels[seen], rbf.fit.predictions[seen])["r2"]
                reproduced.append(abs(r2 - e["val_seen"]["r2"]) <= 1e-12)
                gap = rbf.fit.min_one_minus_hat
                sound = bool(gap >= MIN_HAT_GAP)
                grids = [e["val_seen"]["r2"]] + ([extended[(variable, site, arm)]]
                                                 if (variable, site, arm) in extended else [])
                arms[arm] = {
                    "val_seen_r2": e["val_seen"]["r2"], "refit_val_seen_r2": r2, "alpha_edge": e["alpha_edge"],
                    "min_one_minus_hat": gap, "selection_sound": sound,
                    "quote": "point" if sound else "range",
                    "cross_grid_range": [min(grids), max(grids)],
                }
            sites[site] = {"plot_index": plot_index(site), "arms": arms}
        result[variable] = {"sites": sites}

    all_arms = [a for v in result.values() for s in v["sites"].values() for a in s["arms"].values()]
    criteria = {"refits_reproduce_saved": bool(reproduced) and all(reproduced)}
    return {
        "criteria": criteria, "min_hat_gap": MIN_HAT_GAP, "n_fits": len(all_arms),
        "n_unreliable": sum(not a["selection_sound"] for a in all_arms),
        "source": {"keys": ["kernel_erasure", "kernel_grid_diagnostic"]},
        **result, "observation": True,
    }


def print_kernel_hat_gap(result: dict) -> None:
    """Per variable and site: min(1 - h_ii) and quoting mode for the main arms; random arms summarised."""
    for variable in DATASETS:
        print(f"\n{variable}")
        for site, r in result[variable]["sites"].items():
            parts = []
            for arm, a in r["arms"].items():
                if arm.startswith("random_m_seed") and arm != "random_m_seed0":
                    continue
                flag = " edge" if a["alpha_edge"] else ""
                parts.append(f"{arm} {a['min_one_minus_hat']:.1e} {a['quote']}{flag}")
            random_gaps = [a["min_one_minus_hat"] for name, a in r["arms"].items() if name.startswith("random_m_seed")]
            print(f"  {site:9s} " + " | ".join(parts) + f" | random min {min(random_gaps):.1e}")
    print(f"\ncriteria {result['criteria']}  unreliable {result['n_unreliable']} / {result['n_fits']}")



def control_run(kind: str) -> dict:
    """Shared body of the random-subspace and top-PC controls (validation only).

    Per variable and site (idx 1 / 9 / 18): the same train scaler as the real run; a fixed orthonormal basis inside
    the span of the standardized train rows -- kind "random": N_RANDOM_SEEDS seeded Gaussian rotations of the span
    (then QR); kind "pc": the top train principal axes; round k removes its first (k - 1)·m columns, so the removed
    size matches the real run round by round. Fresh RidgeCV each round (train rows only), scored on val_seen and
    val_unseen. Criteria: round 1 = the real round 1 (nothing removed yet, same fit); bases orthonormal and inside
    the train span; n_fit exact; scores finite; no alpha failure. The exhaustion guard only applies to the real
    run (it builds its basis from probe weights), so it never fires here.
    """
    records = json.loads(OUT.read_text())
    saved_rounds = records["nullspace_rounds"]["result"]
    with np.load(verified_artifact(OUT, "nullspace_rounds")) as f:
        real_predictions = {key: f[key] for key in f.files if key.endswith("_predictions")}

    arrays: dict[str, np.ndarray] = {"sites": np.array(NULLSPACE_SITES)}
    result: dict = {}
    ok: dict[str, list[bool]] = {name: [] for name in ("round1", "orthonormal", "span", "n_fit", "finite", "alpha")}
    seeds = list(range(N_RANDOM_SEEDS)) if kind == "random" else [0]
    for variable in DATASETS:
        table = load_joined(variable)
        roles, labels = table["role"], table["label"]
        y = probe_targets(variable, labels)
        evaluated = np.isin(roles, EVAL_ROLES)
        eval_roles = roles[evaluated]
        sites = {}
        for site in NULLSPACE_SITES:
            start = time.perf_counter()
            x = site_features(table["activations"], site)
            z = train_scaler(x, roles).transform(x)
            span = train_span(z, roles)
            real = saved_rounds[variable]["sites"][site]
            m, k = real["dims_per_round"], real["summary"]["k"]
            n_cols = (CONTROL_ROUNDS - 1) * m
            real_first = real_predictions[f"{variable}_{site}_predictions"][evaluated, 0]

            curves_seen, curves_unseen, per_seed = [], [], []
            for s in seeds:
                basis = random_span_basis(span, n_cols, SEED + s) if kind == "random" else span[:, :n_cols]
                ok["orthonormal"].append(float(np.abs(basis.T @ basis - np.eye(n_cols)).max()) <= ORTHONORMAL_TOLERANCE)
                ok["span"].append(float(np.abs(basis - span @ (span.T @ basis)).max()) <= SPAN_TOLERANCE_CHECK)
                run = run_rounds(z, y, roles, evaluated, CONTROL_ROUNDS, basis=basis)
                ok["round1"].append(bool(np.array_equal(run.predictions[0], real_first)))
                ok["n_fit"].append(run.n_fit == EXPECTED_FIT[variable])
                scores = {role: round_scores(variable, labels[roles == role], run.predictions[:, eval_roles == role])
                          for role in EVAL_ROLES}
                seen_r2 = np.array([sc["r2"] for sc in scores["val_seen"]])
                unseen_r2 = np.array([sc["r2"] for sc in scores["val_unseen"]])
                verdicts = [nullspace_alpha_verdict(e, r) for e, r in zip(run.alpha_edges, seen_r2)]
                ok["finite"].append(bool(np.isfinite(seen_r2).all() and np.isfinite(unseen_r2).all()))
                ok["alpha"].append("failure" not in verdicts)
                curves_seen.append(seen_r2)
                curves_unseen.append(unseen_r2)
                per_seed.append({
                    "seed": SEED + s if kind == "random" else None,
                    "k": curve_summary(seen_r2, m)["k"],
                    "val_seen_r2_at": {str(n): float(seen_r2[n - 1]) for n in (1, 2, 5, 10, k, k + 1, 50, 150)},
                    "alpha_verdicts": dict(Counter(verdicts)),
                })
            prefix = f"{variable}_{site}"
            arrays[f"{prefix}_val_seen_r2"] = np.array(curves_seen)  # (seeds, rounds)
            arrays[f"{prefix}_val_unseen_r2"] = np.array(curves_unseen)
            stacked = np.array(curves_seen)
            sites[site] = {
                "plot_index": plot_index(site), "dims_per_round": m, "real_k": k,
                "real_val_seen_r2_at_k": real["val_seen_r2"][k - 1],
                "control_val_seen_r2_at_real_k": {"mean": float(stacked[:, k - 1].mean()),
                                                  "min": float(stacked[:, k - 1].min()),
                                                  "max": float(stacked[:, k - 1].max())},
                "control_k": [p["k"] for p in per_seed], "per_seed": per_seed,
                "seconds": time.perf_counter() - start,
            }
        arrays[f"{variable}_roles"] = roles
        result[variable] = {"sites": sites}

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / f"{kind}_subspaces.npz"
    np.savez(path, **arrays)
    with np.load(path) as saved_file:
        saved_ok = set(saved_file.files) == set(arrays) and all(
            np.array_equal(saved_file[key], value) for key, value in arrays.items())
    criteria = {
        "round1_equals_real_round1": all(ok["round1"]),
        "bases_orthonormal": all(ok["orthonormal"]),
        "bases_in_train_span": all(ok["span"]),
        "n_fit_equals_train_count": all(ok["n_fit"]),
        "scores_finite": all(ok["finite"]),
        "no_alpha_failure": all(ok["alpha"]),
        "saved_equals_computed": saved_ok,
    }
    return {
        "criteria": criteria, "kind": kind, "n_rounds": CONTROL_ROUNDS, "seeds": [SEED + s for s in seeds],
        "guard_applies": False, "source": {"key": "nullspace_rounds", "artifact_sha256": saved_rounds["artifact"]["sha256"]},
        **result,
        "artifact": {"path": str(path.relative_to(REPO)), "sha256": file_sha256(path)},
        "passed": all(criteria.values()),
    }


def check_random_subspaces() -> dict:
    """Random-subspace control: remove random train-span directions of the real run's size each round."""
    return control_run("random")


def check_pc_subspaces() -> dict:
    """Top-principal-component control: remove the top train principal axes, the real run's size each round."""
    return control_run("pc")


def print_control_run(result: dict) -> None:
    """Per variable and site: real K and its R², the control's R² at that round (mean, min-max), control K per seed."""
    for variable in DATASETS:
        print(f"\n{variable}  ({result['kind']} control)")
        for site, r in result[variable]["sites"].items():
            c = r["control_val_seen_r2_at_real_k"]
            first = r["per_seed"][0]["val_seen_r2_at"]
            print(f"  {site:9s} idx {r['plot_index']:2d}  real K {r['real_k']} (R2 {r['real_val_seen_r2_at_k']:.3f})"
                  f"  control R2 at real K {c['mean']:.3f} [{c['min']:.3f}, {c['max']:.3f}]"
                  f"  control K {r['control_k']}  R2 at 10/50/150 {first['10']:.3f}/{first['50']:.3f}/{first['150']:.3f}"
                  f"  ({r['seconds']:.0f} s)")
            
            
def check_depth_profile() -> dict:
    """K versus depth: the real nullspace rounds at every third layer index (a procedure count, not a dimension).

    Per variable and index in PROFILE_INDICES: the same scaler and run_rounds as the full runs (leave-one-out alpha,
    exhaustion guard on, cap N_ROUNDS); curve reported up to the first val-seen R² < PROFILE_STOP_R2 (the guard stops
    the run a few rounds later; those rounds are not reported). Validation only, never used for selection. Passes if:
    at the indices of the full runs (9, 18) every round equals the saved run (alphas and validation predictions,
    bit-identical); n_fit exact; basis orthonormal; scores finite; no alpha failure; K <= the guard round wherever
    the guard fires (at exhaustion the prediction is the train mean, so val-seen R² <= 0). Reported: exceptions to
    "guard > K".
    """
    saved_rounds = json.loads(OUT.read_text())["nullspace_rounds"]["result"]
    with np.load(verified_artifact(OUT, "nullspace_rounds")) as f:
        saved = {key: f[key] for key in f.files if key.endswith(("_predictions", "_alphas"))}

    result: dict = {}
    ok: dict[str, list[bool]] = {name: [] for name in ("full", "n_fit", "orthonormal", "finite", "alpha", "guard")}
    exceptions = []
    for variable in DATASETS:
        table = load_joined(variable)
        roles, labels = table["role"], table["label"]
        y = probe_targets(variable, labels)
        evaluated = np.isin(roles, EVAL_ROLES)
        seen = roles[evaluated] == "val_seen"
        indices = {}
        for index in PROFILE_INDICES:
            site = SITES[index]
            if plot_index(site) != index:
                raise RuntimeError(f"{site} is not at index {index}")
            x = site_features(table["activations"], site)
            z = train_scaler(x, roles).transform(x)
            run = run_rounds(z, y, roles, evaluated, N_ROUNDS)
            scores = round_scores(variable, labels[roles == "val_seen"], run.predictions[:, seen])
            seen_r2 = np.array([s["r2"] for s in scores])
            m = run.dims_per_round
            k = curve_summary(seen_r2, m)["k"]
            k_stop = first_true(seen_r2 < PROFILE_STOP_R2)
            guard = run.exhausted_round
            verdicts = [nullspace_alpha_verdict(e, r) for e, r in zip(run.alpha_edges, seen_r2)]
            q = run.basis
            ok["n_fit"].append(run.n_fit == EXPECTED_FIT[variable])
            ok["orthonormal"].append(float(np.abs(q.T @ q - np.eye(q.shape[1])).max()) <= ORTHONORMAL_TOLERANCE)
            ok["finite"].append(bool(np.isfinite(seen_r2).all()))
            ok["alpha"].append("failure" not in verdicts)
            ok["guard"].append(guard is None or (k is not None and k <= guard and k_stop is not None and k_stop <= guard))
            if guard is not None and not (k is not None and k < guard):
                exceptions.append({"variable": variable, "index": index, "k": k, "guard_round": guard})

            full = None
            if site in NULLSPACE_SITES:
                n = len(run.alphas)
                prefix = f"{variable}_{site}"
                same = bool(np.array_equal(run.alphas, saved[f"{prefix}_alphas"][:n])
                            and np.array_equal(run.predictions, saved[f"{prefix}_predictions"][evaluated, :n].transpose(1, 0, 2)))
                ok["full"].append(same)
                full = {"rounds_compared": n, "identical": same,
                        "saved_k": saved_rounds[variable]["sites"][site]["summary"]["k"]}

            shown = len(seen_r2) if k_stop is None else k_stop
            indices[str(index)] = {
                "site": site, "depth_fraction": index / 24, "dims_per_round": m,
                "k": k, "dims_before_k": None if k is None else (k - 1) * m,
                "k_stop_0_05": k_stop, "dims_before_k_stop": None if k_stop is None else (k_stop - 1) * m,
                "round1_val_seen_r2": float(seen_r2[0]), "guard_round": guard, "rounds_run": len(run.alphas),
                "val_seen_r2": seen_r2[:shown].tolist(), "alpha_verdicts": dict(Counter(verdicts[:shown])),
                "matches_full_run": full,
            }
        result[variable] = {"indices": indices}

    criteria = {
        "full_run_indices_identical": bool(ok["full"]) and all(ok["full"]),
        "n_fit_equals_train_count": all(ok["n_fit"]),
        "basis_orthonormal": all(ok["orthonormal"]),
        "scores_finite": all(ok["finite"]),
        "no_alpha_failure": all(ok["alpha"]),
        "k_not_after_guard": all(ok["guard"]),
    }
    return {
        "criteria": criteria, "indices": list(PROFILE_INDICES), "stop_r2": PROFILE_STOP_R2, "k_r2": NULL_R2,
        "guard_not_after_k_exceptions": exceptions, **result, "passed": all(criteria.values()),
    }


def print_depth_profile(result: dict) -> None:
    """Per variable: one line per index with round-1 R², K (0.1) and K (0.05) with dims, and the guard round."""
    for variable in DATASETS:
        print(f"\n{variable}")
        for index, r in result[variable]["indices"].items():
            full = r["matches_full_run"]
            note = f"  = full run: {full['identical']}" if full else ""
            print(f"  idx {int(index):2d} ({r['site']:9s})  round-1 R2 {r['round1_val_seen_r2']:7.3f}"
                  f"  K {str(r['k']):>4s} (dims {str(r['dims_before_k']):>3s})"
                  f"  K0.05 {str(r['k_stop_0_05']):>4s} (dims {str(r['dims_before_k_stop']):>3s})"
                  f"  guard {r['guard_round']}{note}")
    print(f"\nguard not after K (exceptions): {result['guard_not_after_k_exceptions']}")
    
    
def check_nullspace_test_scores() -> dict:
    """One-time test of the frozen nullspace findings, from committed code (scope fixed before any test row was read).

    Per variable and site (idx 1 / 9 / 18): (1) nullspace curves from the saved composite maps, rounds up to the
    guard round; first the maps must reproduce the saved validation predictions (<= COMPOSITE_TOLERANCE relative, two
    computation paths) and round 1 must match the layer-curve test scores; then test_seen / test_unseen R² and error
    per round, K and crossings (reported, never selected on). (2) Erasure headline: train cross-covariance direction(s)
    removed (and "none"), fresh linear probe on the validation clips: the nested CV must reproduce the saved
    validation score, then one probe fit on all validation clips (alpha by grouped CV there) scores test. (3) Kernel
    arms none / covariance: original-grid refits (train only) must reproduce the saved val-seen R², then score test;
    quoting mode from kernel_hat_gap. Headline intervals (clip bootstrap) at idx 9 only. Passes if: code committed;
    the reproduce checks hold; round 1 matches; scores finite; no alpha failure; saved file = computed.
    """
    records = json.loads(OUT.read_text())
    saved_rounds = records["nullspace_rounds"]["result"]
    exhaustion = records["covariance_exhaustion"]["result"]
    fresh_saved = records["fresh_probe_erasure"]["result"]
    kernel_saved = records["kernel_erasure"]["result"]
    hat = records["kernel_hat_gap"]["result"]
    layer_tests = json.loads(LAYER_TESTS.read_text())["test_scores"]["result"]
    with np.load(verified_artifact(OUT, "nullspace_rounds")) as f:
        saved = {key: f[key] for key in f.files}
    committed = not code_changes(repo_root())

    arrays: dict[str, np.ndarray] = {"sites": np.array(NULLSPACE_SITES)}
    result: dict = {}
    ok: dict[str, list[bool]] = {name: [] for name in ("maps", "round1", "fresh", "kernel", "finite", "alpha")}
    for variable in DATASETS:
        table = load_joined(variable)
        roles, labels, ids = table["role"], table["label"], table["id"]
        y = probe_targets(variable, labels)
        evaluated = np.isin(roles, EVAL_ROLES)
        tested = np.isin(roles, TEST_ROLES)
        error_name = "circular_mae" if variable == "direction" else "mae"
        arrays[f"{variable}_test_ids"] = ids[tested]
        arrays[f"{variable}_test_roles"] = roles[tested]
        sites = {}
        for site in NULLSPACE_SITES:
            prefix = f"{variable}_{site}"
            x = site_features(table["activations"], site)
            guard = exhaustion[variable]["sites"][site]["guarded_run_exhausted_round"]

            # (1) nullspace curves from the saved composite maps
            composite = x @ saved[f"{prefix}_maps"][:guard] + saved[f"{prefix}_offsets"][:guard][:, None, :]
            maps_diff = relative_diff(composite[:, evaluated],
                                      saved[f"{prefix}_predictions"][evaluated, :guard].transpose(1, 0, 2))
            ok["maps"].append(maps_diff <= COMPOSITE_TOLERANCE)
            curves, round1 = {}, {}
            for role in TEST_ROLES:
                rows = roles == role
                scores = round_scores(variable, labels[rows], composite[:, rows])
                r2 = np.array([s["r2"] for s in scores])
                ok["finite"].append(bool(np.isfinite(r2).all()))
                layer_r2 = layer_tests[variable][role]["all"]["methods"][f"probe {site}"]["r2"]["point"]
                round1[role] = {"r2": float(r2[0]), "layer_test_r2": layer_r2, "diff": abs(float(r2[0]) - layer_r2)}
                ok["round1"].append(round1[role]["diff"] <= 1e-9)
                curves[role] = {
                    "k": first_true(r2 < NULL_R2), "crossings": {str(t): first_true(r2 < t) for t in THRESHOLDS},
                    "r2": r2.tolist(), error_name: [s[error_name] for s in scores],
                }
            arrays[f"{prefix}_nullspace_test_predictions"] = composite[:, tested]

            # (2) erasure headline: fresh linear probe fit on validation, scored on test
            z = train_scaler(x, roles).transform(x)
            basis = covariance_basis(z, y, roles)
            arms = {"none": (np.empty((z.shape[1], 0)), 0), "covariance": (basis, basis.shape[1])}
            fresh, kernel, predictions = {}, {}, {}
            for arm, (b, n_cols) in arms.items():
                zp = project_out(z, b, n_cols)
                nested = nested_cv_predictions(zp[evaluated], y[evaluated], ids[evaluated], ERASURE_FOLDS, SEED)
                cv_r2 = validation_scores(variable, labels[evaluated], roles[evaluated], nested.predictions)["validation"]["r2"]
                saved_cv = fresh_saved[variable]["sites"][site]["arms"][arm]["fresh"]["validation"]["r2"]
                ok["fresh"].append(abs(cv_r2 - saved_cv) <= 1e-12)
                final = grouped_cv_ridge(zp[evaluated], y[evaluated], clip_folds(ids[evaluated], ERASURE_FOLDS, SEED))
                verdict = nullspace_alpha_verdict(final.alpha_edge, cv_r2)
                ok["alpha"].append(verdict != "failure")
                fresh_pred = np.full((len(y), *y.shape[1:]), np.nan)
                fresh_pred[tested] = flat(final.predict(zp[tested]))
                fresh[arm] = {
                    "dims_removed": n_cols, "validation_cv_r2": cv_r2, "alpha": final.alpha,
                    "alpha_edge": final.alpha_edge, "alpha_verdict": verdict,
                    **{role: probe_scores(variable, labels[roles == role], fresh_pred[roles == role]) for role in TEST_ROLES},
                }

                # (3) kernel arm: original-grid refit on train, reproduce val-seen, then score test
                rbf = rbf_kernel_ridge(zp, y, roles, evaluated | tested)
                seen = roles == "val_seen"
                val_r2 = probe_scores(variable, labels[seen], rbf.fit.predictions[seen])["r2"]
                saved_kernel = kernel_saved[variable]["sites"][site]["arms"][arm]["val_seen"]["r2"]
                ok["kernel"].append(abs(val_r2 - saved_kernel) <= 1e-12)
                kernel[arm] = {
                    "dims_removed": n_cols, "val_seen_r2": val_r2,
                    "quote": hat[variable]["sites"][site]["arms"][arm]["quote"],
                    "alpha_edge": rbf.fit.alpha_edge,
                    **{role: probe_scores(variable, labels[roles == role], rbf.fit.predictions[roles == role])
                       for role in TEST_ROLES},
                }
                ok["finite"].append(all(np.isfinite(v) for role in TEST_ROLES
                                        for s in (fresh[arm][role], kernel[arm][role]) for v in s.values()))
                predictions[("fresh", arm)] = fresh_pred
                predictions[("kernel", arm)] = rbf.fit.predictions
                arrays[f"{prefix}_fresh_{arm}_test_predictions"] = fresh_pred[tested]
                arrays[f"{prefix}_kernel_{arm}_test_predictions"] = rbf.fit.predictions[tested]

            entry = {"plot_index": plot_index(site), "guard_round": guard,
                     "validation_k": saved_rounds[variable]["sites"][site]["summary"]["k"],
                     "round1": round1, "maps_max_relative_diff": maps_diff, "nullspace": curves,
                     "fresh_probe": fresh, "kernel": kernel}

            if site == HEADLINE_SITE:  # clip bootstrap for the headline numbers only
                intervals = {}
                for role in TEST_ROLES:
                    rows = np.flatnonzero(roles == role)
                    idx = bootstrap_indices(len(rows), N_BOOTSTRAP, SEED)
                    per_round = np.stack([resampled_r2(y[rows], flat(composite[r, rows]), idx) for r in range(guard)])
                    below = per_round < NULL_R2
                    k_samples = np.where(below.any(axis=0), below.argmax(axis=0) + 1, 0)  # 0 = never below
                    intervals[role] = {
                        "round1_r2": percentile_interval(per_round[0]),
                        "k_distribution": {str(k): int(c) for k, c in sorted(Counter(k_samples.tolist()).items())},
                        "fresh_covariance_r2": percentile_interval(
                            resampled_r2(y[rows], predictions[("fresh", "covariance")][rows], idx)),
                        "kernel_covariance_r2": percentile_interval(
                            resampled_r2(y[rows], predictions[("kernel", "covariance")][rows], idx)),
                    }
                entry["bootstrap"] = {"n_resamples": N_BOOTSTRAP, "seed": SEED, "level": 0.95, **intervals}
            sites[site] = entry
        result[variable] = {"error_metric": error_name, "sites": sites}

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / "test_predictions.npz"
    np.savez(path, **arrays)
    with np.load(path) as saved_file:
        saved_ok = set(saved_file.files) == set(arrays) and all(
            np.array_equal(saved_file[key], value, equal_nan=value.dtype.kind == "f") for key, value in arrays.items())
    criteria = {
        "code_committed": committed,
        "maps_reproduce_validation": all(ok["maps"]),
        "round1_matches_layer_test_scores": all(ok["round1"]),
        "fresh_probe_reproduces_saved": all(ok["fresh"]),
        "kernel_refits_reproduce_saved": all(ok["kernel"]),
        "scores_finite": all(ok["finite"]),
        "no_alpha_failure": all(ok["alpha"]),
        "saved_equals_computed": saved_ok,
    }
    return {
        "criteria": criteria, "roles": list(TEST_ROLES), "stop_r2": NULL_R2, **result,
        "artifact": {"path": str(path.relative_to(REPO)), "sha256": file_sha256(path)},
        "passed": all(criteria.values()),
    }


def print_nullspace_test_scores(result: dict) -> None:
    """Per variable and site: validation K vs test K, round-1 test R², fresh and kernel test R² (none / covariance)."""
    for variable in DATASETS:
        print(f"\n{variable}  (test_seen / test_unseen)")
        for site, r in result[variable]["sites"].items():
            n, fr, ke = r["nullspace"], r["fresh_probe"], r["kernel"]
            print(f"  {site:9s} idx {r['plot_index']:2d}  K val {r['validation_k']}  K test "
                  f"{n['test_seen']['k']} / {n['test_unseen']['k']}  round-1 R2 "
                  f"{r['round1']['test_seen']['r2']:.3f} / {r['round1']['test_unseen']['r2']:.3f}")
            for arm in ("none", "covariance"):
                print(f"    {arm:10s} fresh linear {fr[arm]['test_seen']['r2']:7.3f} / {fr[arm]['test_unseen']['r2']:7.3f}"
                      f"   kernel {ke[arm]['test_seen']['r2']:7.3f} / {ke[arm]['test_unseen']['r2']:7.3f} ({ke[arm]['quote']})")
            if "bootstrap" in r:
                for role in ("test_seen", "test_unseen"):
                    b = r["bootstrap"][role]
                    print(f"    {role:11s} round-1 R2 CI {np.round(b['round1_r2'], 3).tolist()}  K dist {b['k_distribution']}"
                          f"  fresh cov CI {np.round(b['fresh_covariance_r2'], 3).tolist()}"
                          f"  kernel cov CI {np.round(b['kernel_covariance_r2'], 3).tolist()}")
                    

def check_figure_nullspace() -> dict:
    """Figure: val-seen R² per round, frozen test curves and controls, per variable (rows) and layer (columns).

    Reads saved results only (hash-guarded artifacts): the real run up to its guard round (later rounds undefined),
    test-seen / test-unseen curves from the one-time test, the random-subspace band (min-max over seeds) and the
    top-PC control. Marks the stop line R² = 0.1, the paper's other thresholds, and K with the dims removed before it.
    """
    records = json.loads(OUT.read_text())
    rounds = records["nullspace_rounds"]["result"]
    exhaustion = records["covariance_exhaustion"]["result"]
    test = records["nullspace_test_scores"]["result"]
    with np.load(verified_artifact(OUT, "random_subspaces")) as f:
        random = {key: f[key] for key in f.files}
    with np.load(verified_artifact(OUT, "pc_subspaces")) as f:
        pcs = {key: f[key] for key in f.files}

    fig, axes = plt.subplots(3, 3, figsize=(12, 9.5), sharex=True, sharey=True, facecolor=SURFACE)
    x = np.arange(1, FIGURE_ROUNDS + 1)
    for row, variable in enumerate(DATASETS):
        colour = DATASET_COLOUR[variable]
        paper = sorted(set(PAPER_THRESHOLDS.get(variable, {}).values()) - {NULL_R2})
        for col, site in enumerate(NULLSPACE_SITES):
            ax = axes[row, col]
            style_axes(ax)
            prefix = f"{variable}_{site}"
            r = rounds[variable]["sites"][site]
            guard = exhaustion[variable]["sites"][site]["guarded_run_exhausted_round"]
            m, k = r["dims_per_round"], r["summary"]["k"]

            band = random[f"{prefix}_val_seen_r2"][:, :FIGURE_ROUNDS]
            ax.fill_between(x, band.min(axis=0), band.max(axis=0), color=INK_MUTED, alpha=0.35, linewidth=0)
            ax.plot(x, band.mean(axis=0), color=INK_MUTED, linewidth=1.4)
            ax.plot(x, pcs[f"{prefix}_val_seen_r2"][0, :FIGURE_ROUNDS], color=INK_SECONDARY, linewidth=1.2,
                    linestyle="-.")
            for role, style in (("test_seen", "--"), ("test_unseen", ":")):
                curve = test[variable]["sites"][site]["nullspace"][role]["r2"]
                ax.plot(np.arange(1, len(curve) + 1), curve, color=colour, linewidth=1.4, linestyle=style)
            ax.plot(np.arange(1, guard + 1), r["val_seen_r2"][:guard], color=colour, linewidth=2.0)

            ax.axhline(NULL_R2, color=INK, linewidth=0.8)
            for t in paper:
                ax.axhline(t, color=INK_MUTED, linewidth=0.8, linestyle=":")
            ax.axvline(k, color=INK_MUTED, linewidth=0.8)
            ax.text(k + 0.3, 0.55, f"K = {k} · {(k - 1) * m} dims", color=INK_SECONDARY, fontsize=9, va="center")

            if row == 0:
                note = " — headline" if site == "block_8" else ""
                ax.set_title(f"index {plot_index(site)} ({site.replace('_', ' ')}){note}", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{variable}\nR²")
            if row == 2:
                ax.set_xlabel("round")
    axes[0, 0].set_xlim(0.5, FIGURE_ROUNDS + 0.5)
    axes[0, 0].set_ylim(-0.1, 1.05)
    axes[0, 0].set_xticks([1, 5, 10, 15, 20])

    handles = [
        Line2D([], [], color=INK, linewidth=2.0, label="validation (val-seen)"),
        Line2D([], [], color=INK, linewidth=1.4, linestyle="--", label="test seen (frozen probes)"),
        Line2D([], [], color=INK, linewidth=1.4, linestyle=":", label="test unseen"),
        Line2D([], [], color=INK_MUTED, linewidth=1.4, label="random subspaces, 5 seeds (mean; band = min–max)"),
        Line2D([], [], color=INK_SECONDARY, linewidth=1.2, linestyle="-.", label="top principal components"),
        Line2D([], [], color=INK, linewidth=0.8, label="stop: R² = 0.1"),
        Line2D([], [], color=INK_MUTED, linewidth=0.8, linestyle=":",
               label="paper's other threshold (direction 0.3, speed 0.05)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.96))
    fig.suptitle("Iterative nullspace probing: readout per round (direction removes 2 dims per round, speed and "
                 "acceleration 1)", fontsize=11, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return {
        "figure": {"path": str(FIGURE_PATH.relative_to(REPO)), "sha256": file_sha256(FIGURE_PATH)},
        "rounds_shown": FIGURE_ROUNDS,
        "sources": {"random_subspaces": records["random_subspaces"]["result"]["artifact"]["sha256"],
                    "pc_subspaces": records["pc_subspaces"]["result"]["artifact"]["sha256"],
                    "keys": ["nullspace_rounds", "covariance_exhaustion", "nullspace_test_scores"]},
        "visual": True,
    }


def print_figure_nullspace(result: dict) -> None:
    print(f"wrote {result['figure']['path']}  sha256 {result['figure']['sha256'][:12]}…")
                  

CHECKS = {
    "nullspace_rounds": (check_nullspace_rounds, print_nullspace_rounds),
    "leak_diagnostic": (check_leak_diagnostic, print_leak_diagnostic),
    "covariance_exhaustion": (check_covariance_exhaustion, print_covariance_exhaustion),
    "fresh_probe_erasure": (check_fresh_probe_erasure, print_fresh_probe_erasure),
    "alpha_sweep": (check_alpha_sweep, print_alpha_sweep),
    "kernel_erasure": (check_kernel_erasure, print_kernel_erasure),
    "kernel_shuffled_labels": (check_kernel_shuffled_labels, print_kernel_shuffled_labels),
    "kernel_grid_diagnostic": (check_kernel_grid_diagnostic, print_kernel_grid_diagnostic),
    "kernel_hat_gap": (check_kernel_hat_gap, print_kernel_hat_gap),
    "random_subspaces": (check_random_subspaces, print_control_run),
    "pc_subspaces": (check_pc_subspaces, print_control_run),
    "depth_profile": (check_depth_profile, print_depth_profile),
    "nullspace_test_scores": (check_nullspace_test_scores, print_nullspace_test_scores),
    "figure_nullspace": (check_figure_nullspace, print_figure_nullspace),
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