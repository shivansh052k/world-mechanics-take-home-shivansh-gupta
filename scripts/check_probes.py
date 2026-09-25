"""Checks for the layer-wise linear probes: one ridge probe per site and variable, scored on validation clips.

Run one check at a time from the repo root, with .venv active:
    python scripts/check_probes.py <check>

Each check prints a summary and stores its full result under its own key in results/probes/checks.json.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from vjepa_physics.data import DATASETS
from vjepa_physics.evidence import file_sha256, save_result
from vjepa_physics.extraction import SITES, plot_index
from vjepa_physics.joined import load_joined
from vjepa_physics.reproducibility import SEED
from vjepa_physics.probes import ALPHAS, alpha_verdict, fit_probe, probe_scores, probe_targets, site_features

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/probes/checks.json"
PROBES = REPO / "artifacts/probes"  # regenerable, git-ignored

EXPECTED_FIT = {"direction": 813, "speed": 832, "acceleration": 832}  # train clips per variable (split decision)
EVAL_ROLES = ("val_seen", "val_unseen")  # scored here; test is scored once, after the layer choice is frozen

N_PERMUTATIONS = 20  # label permutations within train, per variable (the same ones at every site)
MAX_SHUFFLED_R2 = 0.1  # every shuffled fit's val_seen R² must stay below this
MIN_MEAN_SHUFFLED_CIRCULAR_MAE = 80.0  # per site, direction's circular MAE averaged over the permutations (chance 90)

def site_plot_index(site: str) -> int | None:
    """hidden_states index of a site (0 = embedding, 1-24 = blocks); None for the final norm, plotted separately."""
    return None if site == "final_norm" else plot_index(site)


def check_layer_curves() -> dict:
    """Fit one probe per site and variable on train; score it on val_seen and val_unseen; save the predictions.

    Writes artifacts/probes/layer_predictions.npz: per variable the clip ids, roles and predictions
    (clips, 26[, 2]), NaN on every row that is not a validation row. Passes if: every probe saw exactly the
    train clips (813 / 832); no alpha verdict is "failure" (alpha rule); every score is finite; only validation
    rows were predicted; a second fit gives an identical probe; the saved file equals what was computed.
    """
    result: dict = {}
    arrays: dict[str, np.ndarray] = {"sites": np.array(SITES)}
    n_fit_ok, verdict_ok, finite_ok, refit_ok, rows_ok = [], [], [], [], []

    for variable in DATASETS:
        table = load_joined(variable)
        acts, roles, labels = table["activations"], table["role"], table["label"]
        y = probe_targets(variable, labels)
        evaluated = np.isin(roles, EVAL_ROLES)
        predictions = np.full((len(y), len(SITES), *y.shape[1:]), np.nan)
        sites = {}
        for k, site in enumerate(SITES):
            x = site_features(acts, site)
            probe = fit_probe(x, y, roles)
            again = fit_probe(x, y, roles)
            predictions[evaluated, k] = probe.predict(x[evaluated])
            scores = {
                role: probe_scores(variable, labels[roles == role], predictions[roles == role, k])
                for role in EVAL_ROLES
            }
            verdict = alpha_verdict(probe.alpha_edge, scores["val_seen"]["r2"])
            sites[site] = {
                "plot_index": site_plot_index(site), "alpha": probe.alpha, "alpha_edge": probe.alpha_edge,
                "alpha_verdict": verdict, "n_fit": probe.n_fit, **scores,
            }
            n_fit_ok.append(probe.n_fit == EXPECTED_FIT[variable])
            verdict_ok.append(verdict != "failure")
            finite_ok.append(all(np.isfinite(v) for s in scores.values() for v in s.values()))
            refit_ok.append(
                again.alpha == probe.alpha
                and np.array_equal(again.ridge.coef_, probe.ridge.coef_)
                and np.array_equal(again.ridge.intercept_, probe.ridge.intercept_)
            )
        rows_ok.append(bool(np.isfinite(predictions[evaluated]).all() and np.isnan(predictions[~evaluated]).all()))
        arrays[f"{variable}_ids"] = table["id"]
        arrays[f"{variable}_roles"] = roles
        arrays[f"{variable}_predictions"] = predictions
        result[variable] = {"n_evaluated": int(evaluated.sum()), "sites": sites}

    PROBES.mkdir(parents=True, exist_ok=True)
    path = PROBES / "layer_predictions.npz"
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
        "refit_identical": all(refit_ok),
        "saved_equals_computed": saved_ok,
    }
    return {
        "criteria": criteria,
        **result,
        "alpha_grid": [float(a) for a in ALPHAS],
        "artifact": {"path": str(path.relative_to(REPO)), "sha256": file_sha256(path)},
        "passed": all(criteria.values()),
    }


def print_layer_curves(result: dict) -> None:
    """One line per site: plot index, alpha, verdict, val_seen and val_unseen scores."""
    for variable in DATASETS:
        print(f"\n{variable}")
        for site, r in result[variable]["sites"].items():
            other = "circular_mae" if variable == "direction" else "mae"
            print(f"  {site:10s} idx {str(r['plot_index']):>4s}  alpha {r['alpha']:9.3g}  {r['alpha_verdict']:9s}"
                  f"  seen R2 {r['val_seen']['r2']:7.3f} {other} {r['val_seen'][other]:7.3f}"
                  f"  unseen R2 {r['val_unseen']['r2']:7.3f} {other} {r['val_unseen'][other]:7.3f}")


def train_permutations(n_train: int, n: int, seed: int) -> np.ndarray:
    """(n, n_train) permutations of the train rows from one seeded generator. Raises if one is the identity."""
    rng = np.random.default_rng(seed)
    permutations = np.stack([rng.permutation(n_train) for _ in range(n)])
    if (permutations == np.arange(n_train)).all(axis=1).any():
        raise RuntimeError("a permutation is the identity")
    return permutations


def check_shuffled_labels() -> dict:
    """Probes fit on train with shuffled labels must score at chance on val_seen with the true labels.

    Per variable, N_PERMUTATIONS permutations of the train labels (seed SEED), the same at every site; only
    train rows' labels move, and only train rows are fitted. Passes if: every shuffled fit's val_seen R² <
    MAX_SHUFFLED_R2; for direction, the circular MAE averaged over the permutations > 80 degrees at every site;
    no alpha verdict is "failure" (the alpha rule covers controls); every fit saw exactly the train clips.
    """
    result: dict = {}
    fit_ok, verdict_ok, r2_ok, circular_ok = [], [], [], []
    for variable in DATASETS:
        table = load_joined(variable)
        acts, roles, labels = table["activations"], table["role"], table["label"]
        y = probe_targets(variable, labels)
        train = np.flatnonzero(roles == "train")
        seen = roles == "val_seen"
        permutations = train_permutations(len(train), N_PERMUTATIONS, SEED)
        shuffled = []
        for permutation in permutations:
            y_shuffled = y.copy()
            y_shuffled[train] = y[train[permutation]]
            shuffled.append(y_shuffled)
        moved = [float(np.mean(labels[train[p]] != labels[train])) for p in permutations]

        sites = {}
        for site in SITES:
            x = site_features(acts, site)
            fits = []
            for y_shuffled in shuffled:
                probe = fit_probe(x, y_shuffled, roles)
                scores = probe_scores(variable, labels[seen], probe.predict(x[seen]))
                fits.append((probe, scores, alpha_verdict(probe.alpha_edge, scores["r2"])))
            r2s = [scores["r2"] for _, scores, _ in fits]
            entry = {
                "plot_index": site_plot_index(site),
                "alphas": [probe.alpha for probe, _, _ in fits],
                "alpha_verdicts": dict(Counter(verdict for _, _, verdict in fits)),
                "r2": r2s,
                "max_r2": max(r2s),
            }
            if variable == "direction":
                errors = [scores["circular_mae"] for _, scores, _ in fits]
                entry |= {"circular_mae": errors, "mean_circular_mae": float(np.mean(errors)),
                          "min_circular_mae": min(errors)}
                circular_ok.append(entry["mean_circular_mae"] > MIN_MEAN_SHUFFLED_CIRCULAR_MAE)
            else:
                entry["mae"] = [scores["mae"] for _, scores, _ in fits]
            fit_ok.append(all(probe.n_fit == EXPECTED_FIT[variable] for probe, _, _ in fits))
            verdict_ok.append("failure" not in entry["alpha_verdicts"])
            r2_ok.append(entry["max_r2"] < MAX_SHUFFLED_R2)
            sites[site] = entry
        result[variable] = {"labels_moved_fraction_min": min(moved), "sites": sites}

    criteria = {
        "n_fit_equals_train_count": all(fit_ok),
        "no_alpha_failure": all(verdict_ok),
        "max_shuffled_r2_below_0_1": all(r2_ok),
        "direction_mean_circular_mae_above_80": bool(circular_ok) and all(circular_ok),
    }
    return {"criteria": criteria, "n_permutations": N_PERMUTATIONS, "seed": SEED, **result,
            "passed": all(criteria.values())}


def print_shuffled_labels(result: dict) -> None:
    """One line per site: alpha verdict counts, max shuffled R², and direction's mean / min circular MAE."""
    for variable in DATASETS:
        print(f"\n{variable}  (labels moved, min over permutations: {result[variable]['labels_moved_fraction_min']:.3f})")
        for site, r in result[variable]["sites"].items():
            extra = (f"  circular MAE mean {r['mean_circular_mae']:6.2f} min {r['min_circular_mae']:6.2f}"
                     if variable == "direction" else "")
            print(f"  {site:10s} max R2 {r['max_r2']:7.3f}  verdicts {r['alpha_verdicts']}{extra}")



CHECKS = {
    "layer_curves": (check_layer_curves, print_layer_curves),
    "shuffled_labels": (check_shuffled_labels, print_shuffled_labels),
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