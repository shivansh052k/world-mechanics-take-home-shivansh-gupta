"""Uncertainty of the layer curves and their baselines: a clip bootstrap on the saved validation predictions.

Run one check at a time from the repo root, with .venv active:
    python scripts/check_layer_curves.py <check>

Each check prints a summary and stores its full result under its own key in results/layer_curves/checks.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from vjepa_physics.curves import LATE_RISE_FRACTION, MIN_CLEAR_RISE, N_INDICES, TRANSITION_FRACTION, rise_index, transition_points
from vjepa_physics.data import DATASETS
from vjepa_physics.baselines import kernel_ridge, physics_fit
from vjepa_physics.data import load_dataset
from vjepa_physics.evidence import code_changes, file_sha256, repo_root, save_result, verified_artifact
from vjepa_physics.geometry import disk_centres
from vjepa_physics.extraction import SITES
from vjepa_physics.joined import load_joined
from vjepa_physics.metrics import (
    angles_from_sincos, bootstrap_indices, circular_errors, percentile_interval, r2, resampled_mean, resampled_r2,
    sincos_targets,
)
from vjepa_physics.plotting import AXIS, DATASET_COLOUR, INK, INK_MUTED, INK_SECONDARY, SURFACE, style_axes
from vjepa_physics.probes import fit_probe, probe_targets, site_features
from vjepa_physics.reproducibility import SEED

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/layer_curves/checks.json"
PROBES_CHECKS = REPO / "results/probes/checks.json"  # key "layer_curves": probe predictions and scores
BASELINES_CHECKS = REPO / "results/baselines/checks.json"  # keys "pixel_floor", "physics_ceiling"

N_RESAMPLES = 10_000  # clip resamples (bootstrap settings decision)
LEVEL = 0.95  # percentile interval
EVAL_ROLES = ("val_seen", "val_unseen")
SHOWN_INDICES = (0, 1, 4, 8, 12, 18, 24)  # layer indices printed in the terminal summary (all are saved)

FIGURE = REPO / "results/layer_curves/layer_curves.png"
ERROR_LABEL = {"direction": "circular MAE (°)", "speed": "MAE (m/s)", "acceleration": "MAE (m/s²)"}
# Reference lines (val-seen point estimates): name in the bootstrap result, label, line style.
REFERENCES = (
    ("floor full", "pixel floor, RGB", ":"),
    ("floor time_average", "pixel floor, time-averaged", (0, (1, 3))),
    ("ceiling quadratic", "physics-fit ceiling", "-."),
)

DATA = REPO / "data"
TEST_ROLES = ("test_seen", "test_unseen")
TEST_PREDICTIONS = REPO / "artifacts/probes/test_predictions.npz"  # regenerable, git-ignored
EXPECTED_FIT = {"direction": 813, "speed": 832, "acceleration": 832}  # train clips per variable (split decision)
CEILING_OUTPUT = {"direction": "theta", "speed": "speed", "acceleration": "acceleration"}  # physics_fit key read
EXACT_TOLERANCE = 1e-9  # physics fit on metadata-predicted positions must return the label within this
N_REFITS = 3 * 26 + 3 * 2  # probe sites + pixel floors, all variables

def error_name(variable: str) -> str:
    return "circular_mae" if variable == "direction" else "mae"


def load_npz(checks: Path, key: str) -> dict[str, np.ndarray]:
    """Every array of an artifact a check recorded under `key`, read through its recorded hash."""
    with np.load(verified_artifact(checks, key)) as f:
        return {name: f[name] for name in f.files}


def clip_errors(variable: str, labels: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """(n,) per-clip error: circular error in degrees for direction ((sin, cos) predictions), else absolute error."""
    if variable == "direction":
        return circular_errors(labels, angles_from_sincos(pred))
    return np.abs(labels - pred)


def resampled_scores(variable: str, labels: np.ndarray, pred: np.ndarray, idx: np.ndarray) -> dict:
    """{metric: (point estimate, (B,) resampled values)} for R² and the error metric, on the same resamples."""
    y = probe_targets(variable, labels)
    errors = clip_errors(variable, labels, pred)
    return {
        "r2": (r2(y, pred), resampled_r2(y, pred, idx)),
        error_name(variable): (float(errors.mean()), resampled_mean(errors, idx)),
    }


def summary(scores: dict) -> dict:
    """Point estimate and percentile interval per metric, plus 1 - R² (the error-term view of R²)."""
    out = {}
    for name, (point, samples) in scores.items():
        out[name] = {"point": point, "ci": list(percentile_interval(samples, LEVEL))}
    low, high = out["r2"]["ci"]
    out["one_minus_r2"] = {"point": 1 - out["r2"]["point"], "ci": [1 - high, 1 - low]}
    return out


def check_bootstrap() -> dict:
    """Clip bootstrap (10,000 resamples, seed SEED, percentile 95%) of the saved validation predictions.

    Per variable, role (val_seen, val_unseen) and clip subset (all; direction also without exit clips), one index
    matrix is shared by every method (26 probe sites, pixel floors, physics-fit ceilings), so all comparisons are
    paired. Records per method R², the error metric and 1 - R² with intervals; on all clips, the paired R²
    difference of each layer index 1-24 from the one before; on val_seen, the transition (the fixed rule on the
    point curve) and its distribution over resampled curves (indices 0-24, final norm excluded); on val_unseen,
    the error per held-out value. Unseen-split intervals are conditional on the held-out values. Passes if every
    input artifact lines up with its joined table (ids, roles) and every point estimate equals the score its own
    check saved (178 comparisons).
    """
    probes = load_npz(PROBES_CHECKS, "layer_curves")
    floors = load_npz(BASELINES_CHECKS, "pixel_floor")
    ceiling = load_npz(BASELINES_CHECKS, "physics_ceiling")
    saved_probes = json.loads(PROBES_CHECKS.read_text())["layer_curves"]["result"]
    saved_baselines = json.loads(BASELINES_CHECKS.read_text())
    saved_floor = saved_baselines["pixel_floor"]["result"]
    saved_ceiling = saved_baselines["physics_ceiling"]["result"]
    sites = [str(s) for s in probes["sites"]]
    curve_sites = sites[:N_INDICES]  # embedding, block_0 ... block_23

    result: dict = {"n_resamples": N_RESAMPLES, "level": LEVEL, "seed": SEED}
    aligned_ok, equal_ok = [], []
    for variable in DATASETS:
        table = load_joined(variable)
        labels, roles = table["label"], table["role"]
        for source in (probes, floors, ceiling):
            aligned_ok.append(np.array_equal(source[f"{variable}_ids"], table["id"])
                              and np.array_equal(source[f"{variable}_roles"], roles))
        floor_kinds = list(saved_floor[variable])
        ceiling_models = list(saved_ceiling[variable])
        methods = {f"probe {site}": probes[f"{variable}_predictions"][:, k] for k, site in enumerate(sites)}
        methods |= {f"floor {kind}": floors[f"{variable}_{kind}_predictions"] for kind in floor_kinds}
        for model in ceiling_models:
            estimates = ceiling[f"{variable}_{model}_estimates"]
            methods[f"ceiling {model}"] = sincos_targets(estimates) if variable == "direction" else estimates
        subsets = {"all": np.ones(len(roles), dtype=bool)}
        if variable == "direction":
            subsets["without_exit"] = ~table["exit"]

        per_role = {}
        for role in EVAL_ROLES:
            per_subset = {}
            for subset, keep in subsets.items():
                rows = (roles == role) & keep
                idx = bootstrap_indices(int(rows.sum()), N_RESAMPLES, SEED)
                scored = {name: resampled_scores(variable, labels[rows], pred[rows], idx) for name, pred in methods.items()}
                entry: dict = {"clips": int(rows.sum()), "methods": {name: summary(s) for name, s in scored.items()}}
                if subset == "all":
                    saved_points = [(f"probe {site}", saved_probes[variable]["sites"][site][role]) for site in sites]
                    saved_points += [(f"floor {kind}", saved_floor[variable][kind][role]) for kind in floor_kinds]
                    saved_points += [(f"ceiling {m}", saved_ceiling[variable][m][role]) for m in ceiling_models]
                    for name, saved in saved_points:
                        equal_ok.append(all(scored[name][metric][0] == saved[metric] for metric in scored[name]))

                    point = np.array([scored[f"probe {site}"]["r2"][0] for site in curve_sites])
                    curves = np.stack([scored[f"probe {site}"]["r2"][1] for site in curve_sites], axis=1)  # (B, 25)
                    entry["adjacent_r2_differences"] = [
                        {"index": i, "point": float(point[i] - point[i - 1]),
                         "ci": list(percentile_interval(curves[:, i] - curves[:, i - 1], LEVEL))}
                        for i in range(1, N_INDICES)
                    ]
                    if role == "val_seen":
                        entry["transition"] = {
                            "headline": transition_points(point),
                            "bootstrap_transition_index_counts":
                                np.bincount(rise_index(curves, TRANSITION_FRACTION), minlength=N_INDICES).tolist(),
                            "bootstrap_rise_80_index_counts":
                                np.bincount(rise_index(curves, LATE_RISE_FRACTION), minlength=N_INDICES).tolist(),
                            "bootstrap_clear_fraction":
                                float((curves.max(axis=1) - curves.min(axis=1) >= MIN_CLEAR_RISE).mean()),
                        }
                    if role == "val_unseen":
                        entry["per_value_error"] = {
                            f"{value:.6g}": {name: float(clip_errors(variable, labels[rows & (labels == value)],
                                                                     pred[rows & (labels == value)]).mean())
                                             for name, pred in methods.items()}
                            for value in np.unique(labels[rows])
                        }
                per_subset[subset] = entry
            per_role[role] = per_subset
        result[variable] = per_role

    criteria = {
        "inputs_aligned_with_tables": all(aligned_ok),
        "points_equal_saved_scores": len(equal_ok) == 178 and all(equal_ok),
    }
    return {"criteria": criteria, "consistency_comparisons": len(equal_ok), **result,
            "passed": all(criteria.values())}


def print_bootstrap(result: dict) -> None:
    """Transition and its distribution; selected layers, floors and ceilings with intervals; adjacent-difference signs."""
    for variable in DATASETS:
        other = error_name(variable)
        t = result[variable]["val_seen"]["all"]["transition"]
        counts = {i: c for i, c in enumerate(t["bootstrap_transition_index_counts"]) if c}
        counts80 = {i: c for i, c in enumerate(t["bootstrap_rise_80_index_counts"]) if c}
        print(f"\n{variable}: transition {t['headline']['transition_index']} (rise {t['headline']['rise']:.3f}),"
              f" bootstrap transition counts {counts}, 80% rise counts {counts80}, clear {t['bootstrap_clear_fraction']}")
        for role, per_subset in result[variable].items():
            for subset, entry in per_subset.items():
                print(f"  {role} / {subset} ({entry['clips']} clips)")
                shown = [name for name in entry["methods"] if name.startswith("probe")]
                shown = [shown[i] for i in SHOWN_INDICES] + ["probe final_norm"]
                shown += [m for m in entry["methods"] if not m.startswith("probe")]
                for name in shown:
                    m = entry["methods"][name]
                    print(f"    {name:22s} R2 {m['r2']['point']:7.4f} [{m['r2']['ci'][0]:7.4f}, {m['r2']['ci'][1]:7.4f}]"
                          f"  {other} {m[other]['point']:8.4f} [{m[other]['ci'][0]:8.4f}, {m[other]['ci'][1]:8.4f}]")
                if "adjacent_r2_differences" in entry:
                    up = [d["index"] for d in entry["adjacent_r2_differences"] if d["ci"][0] > 0]
                    down = [d["index"] for d in entry["adjacent_r2_differences"] if d["ci"][1] < 0]
                    print(f"    adjacent R2 change, CI above 0 at {up}; below 0 at {down}")
    print(json.dumps({"criteria": result["criteria"], "consistency_comparisons": result["consistency_comparisons"]}, indent=2))

def check_figure_layer_curves() -> dict:
    """Figure: probe score per layer index (0-24) for each variable, from the saved bootstrap result.

    Top row: R² on val_seen (line, 95% band) and val_unseen (dashed), pixel floors and physics-fit ceiling as
    labelled reference lines, the transition index as a vertical line. Bottom row: the error metric (MAE /
    circular MAE) on a log scale, same layout. The final norm is not drawn (same depth as block 23).
    Writes results/layer_curves/layer_curves.png. A visual check: records the file hash; reviewed by eye.
    """
    boot = json.loads(OUT.read_text())["bootstrap"]["result"]
    curve_sites = SITES[:N_INDICES]
    x = np.arange(N_INDICES)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), sharex=True, layout="constrained")
    fig.patch.set_facecolor(SURFACE)

    for col, variable in enumerate(DATASETS):
        colour = DATASET_COLOUR[variable]
        seen = boot[variable]["val_seen"]["all"]["methods"]
        transition = boot[variable]["val_seen"]["all"]["transition"]["headline"]["transition_index"]
        for row, metric in enumerate(("r2", error_name(variable))):
            ax = axes[row, col]
            style_axes(ax)
            for role, style in (("val_seen", "-"), ("val_unseen", "--")):
                methods = boot[variable][role]["all"]["methods"]
                point = np.array([methods[f"probe {s}"][metric]["point"] for s in curve_sites])
                if role == "val_seen":
                    ci = np.array([methods[f"probe {s}"][metric]["ci"] for s in curve_sites])
                    ax.fill_between(x, ci[:, 0], ci[:, 1], color=colour, alpha=0.22, linewidth=0)
                ax.plot(x, point, style, color=colour, linewidth=2,
                        marker="o" if role == "val_seen" else None, markersize=3.5)
            for name, text, line in REFERENCES:
                value = seen[name][metric]["point"]
                ax.axhline(value, color=INK_MUTED, linewidth=1, linestyle=line)
                at_left = name.startswith("ceiling") and row == 0  # the curve fills the top right
                ax.annotate(text, (transition + 0.4 if at_left else 24, value), xytext=(0, 3), textcoords="offset points",
                            ha="left" if at_left else "right", va="bottom", fontsize=8, color=INK_SECONDARY)
            ax.axvline(transition, color=INK_SECONDARY, linewidth=1)
            if row == 0:
                ax.set_title(variable, color=INK, fontsize=12)
                ax.set_ylim(-0.05, 1.09)
                ax.annotate(f"transition: index {transition}", (transition + 0.3, 0.75), fontsize=8,
                            color=INK_SECONDARY, va="center")
                top = ax.secondary_xaxis("top", functions=(lambda i: i / 24, lambda f: f * 24))
                top.set_xlabel("depth fraction (index / 24)", fontsize=9, color=INK_SECONDARY)
                top.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, labelsize=9)
                top.spines["top"].set_color(AXIS)
            else:
                ax.set_yscale("log")
                ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                ax.yaxis.set_minor_formatter(NullFormatter())
                ax.set_xlabel("layer index (0 = patch embedding, 24 = block 23)", fontsize=9)
            ax.set_ylabel("R²" if row == 0 else f"{ERROR_LABEL[variable]}, log scale", fontsize=9)
            ax.set_xlim(-0.5, 24.5)

    handles = [
        Line2D([], [], color=INK_SECONDARY, linewidth=2, marker="o", markersize=3.5, label="val-seen (95% band)"),
        Line2D([], [], color=INK_SECONDARY, linewidth=2, linestyle="--", label="val-unseen"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False, fontsize=9)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    written = FIGURE.exists() and FIGURE.stat().st_size > 0
    return {"figure": {"path": str(FIGURE.relative_to(REPO)), "sha256": file_sha256(FIGURE)},
            "source": "results/layer_curves/checks.json, key bootstrap", "passed": written}


def print_figure_layer_curves(result: dict) -> None:
    print(json.dumps(result["figure"], indent=2))


def check_test_scores() -> dict:
    """Final test scores, computed once after the layer choices were frozen, from committed code.

    Refits every probe (26 sites) and pixel floor on train exactly as before and requires the same alphas and
    bit-identical validation predictions as the saved checks, so the test scores come from the fits behind the
    curves; then predicts test_seen and test_unseen only. The physics-fit ceiling (nothing fitted) is estimated
    on test rows. Scored like the bootstrap (10,000 clip resamples, seed SEED, percentile 95%, one index matrix
    per variable x role x subset; direction also without exit clips; per-value errors on test_unseen). Writes
    artifacts/probes/test_predictions.npz (NaN on non-test rows). Passes if: the code is committed; all 84 refits
    match the saved alphas and validation predictions; every fit saw exactly the train clips; the ceiling fit is
    exact on metadata positions of the test clips; only test rows were predicted; the saved file equals what was
    computed.
    """
    committed = not code_changes(repo_root())
    saved_predictions = load_npz(PROBES_CHECKS, "layer_curves")
    saved_floor_predictions = load_npz(BASELINES_CHECKS, "pixel_floor")
    saved_probes = json.loads(PROBES_CHECKS.read_text())["layer_curves"]["result"]
    saved_baselines = json.loads(BASELINES_CHECKS.read_text())
    saved_floor = saved_baselines["pixel_floor"]["result"]
    saved_ceiling = saved_baselines["physics_ceiling"]["result"]
    sites = [str(s) for s in saved_predictions["sites"]]

    result: dict = {"n_resamples": N_RESAMPLES, "level": LEVEL, "seed": SEED}
    arrays: dict[str, np.ndarray] = {"sites": np.array(sites)}
    alpha_ok, validation_ok, fit_ok, exact_ok, rows_ok = [], [], [], [], []
    for variable in DATASETS:
        table = load_joined(variable)
        labels, roles = table["label"], table["role"]
        y = probe_targets(variable, labels)
        validation = np.isin(roles, EVAL_ROLES)
        test = np.isin(roles, TEST_ROLES)
        methods: dict[str, np.ndarray] = {}

        probe_test = np.full((len(y), len(sites), *y.shape[1:]), np.nan)
        for k, site in enumerate(sites):
            x = site_features(table["activations"], site)
            probe = fit_probe(x, y, roles)
            alpha_ok.append(probe.alpha == saved_probes[variable]["sites"][site]["alpha"])
            fit_ok.append(probe.n_fit == EXPECTED_FIT[variable])
            validation_ok.append(np.array_equal(probe.predict(x[validation]),
                                                saved_predictions[f"{variable}_predictions"][validation, k]))
            probe_test[test, k] = probe.predict(x[test])
            methods[f"probe {site}"] = probe_test[:, k]
        arrays[f"{variable}_probe_predictions"] = probe_test

        for kind in saved_floor[variable]:
            gram = np.load(verified_artifact(BASELINES_CHECKS, "pixel_grams", f"gram_{variable}_{kind}"))
            again = kernel_ridge(gram, y, roles, validation)  # the saved fit, reproduced
            validation_ok.append(np.array_equal(again.predictions[validation],
                                                saved_floor_predictions[f"{variable}_{kind}_predictions"][validation]))
            fit = kernel_ridge(gram, y, roles, test)
            alpha_ok.append(fit.alpha == again.alpha == saved_floor[variable][kind]["alpha"])
            fit_ok.append(fit.n_fit == EXPECTED_FIT[variable])
            methods[f"floor {kind}"] = fit.predictions
            arrays[f"{variable}_floor_{kind}_predictions"] = fit.predictions

        meta = {m["id"]: m for m in load_dataset(DATA, variable)}
        visible = (table["area"] > 0) & ~table["touches_border"]
        output = CEILING_OUTPUT[variable]
        for model in saved_ceiling[variable]:
            estimates = np.full(len(roles), np.nan)
            exact = np.full(len(roles), np.nan)
            for k in np.flatnonzero(test):
                estimates[k] = physics_fit(table["centre"][k], visible[k], model)[output]
                all_frames = np.ones(len(visible[k]), dtype=bool)
                exact[k] = physics_fit(disk_centres(meta[int(table["id"][k])]), all_frames, model)[output]
            if variable == "direction":
                exact_ok.append(float(circular_errors(labels[test], exact[test]).max()) <= EXACT_TOLERANCE)
                methods[f"ceiling {model}"] = sincos_targets(estimates)
            else:
                exact_ok.append(float(np.abs(exact[test] - labels[test]).max()) <= EXACT_TOLERANCE)
                methods[f"ceiling {model}"] = estimates
            arrays[f"{variable}_ceiling_{model}_estimates"] = estimates
        arrays[f"{variable}_ids"] = table["id"]
        arrays[f"{variable}_roles"] = roles
        rows_ok.append(all(np.isfinite(p[test]).all() and np.isnan(p[~test]).all() for p in methods.values()))

        subsets = {"all": np.ones(len(roles), dtype=bool)}
        if variable == "direction":
            subsets["without_exit"] = ~table["exit"]
        per_role = {}
        for role in TEST_ROLES:
            per_subset = {}
            for subset, keep in subsets.items():
                rows = (roles == role) & keep
                idx = bootstrap_indices(int(rows.sum()), N_RESAMPLES, SEED)
                scored = {name: resampled_scores(variable, labels[rows], pred[rows], idx) for name, pred in methods.items()}
                entry: dict = {"clips": int(rows.sum()), "methods": {name: summary(s) for name, s in scored.items()}}
                if role == "test_unseen" and subset == "all":
                    entry["per_value_error"] = {
                        f"{value:.6g}": {name: float(clip_errors(variable, labels[rows & (labels == value)],
                                                                 pred[rows & (labels == value)]).mean())
                                         for name, pred in methods.items()}
                        for value in np.unique(labels[rows])
                    }
                per_subset[subset] = entry
            per_role[role] = per_subset
        result[variable] = per_role

    np.savez_compressed(TEST_PREDICTIONS, **arrays)
    with np.load(TEST_PREDICTIONS) as saved:
        saved_ok = set(saved.files) == set(arrays) and all(
            np.array_equal(saved[key], value, equal_nan=value.dtype.kind == "f") for key, value in arrays.items()
        )
    criteria = {
        "code_committed": committed,
        "refit_alphas_equal_saved": len(alpha_ok) == N_REFITS and all(alpha_ok),
        "refit_validation_predictions_identical": len(validation_ok) == N_REFITS and all(validation_ok),
        "n_fit_equals_train_count": all(fit_ok),
        "ceiling_exact_on_metadata_positions": bool(exact_ok) and all(exact_ok),
        "only_test_rows_predicted": all(rows_ok),
        "saved_equals_computed": saved_ok,
    }
    return {"criteria": criteria, **result,
            "artifact": {"path": str(TEST_PREDICTIONS.relative_to(REPO)), "sha256": file_sha256(TEST_PREDICTIONS)},
            "passed": all(criteria.values())}


def print_test_scores(result: dict) -> None:
    """Per variable, role and subset: selected layers, final norm, floors and ceilings with intervals."""
    for variable in DATASETS:
        other = error_name(variable)
        print(f"\n{variable}")
        for role, per_subset in result[variable].items():
            for subset, entry in per_subset.items():
                print(f"  {role} / {subset} ({entry['clips']} clips)")
                probes = [name for name in entry["methods"] if name.startswith("probe")]
                shown = [probes[i] for i in SHOWN_INDICES] + ["probe final_norm"]
                shown += [m for m in entry["methods"] if not m.startswith("probe")]
                for name in shown:
                    m = entry["methods"][name]
                    print(f"    {name:22s} R2 {m['r2']['point']:7.4f} [{m['r2']['ci'][0]:7.4f}, {m['r2']['ci'][1]:7.4f}]"
                          f"  {other} {m[other]['point']:8.4f} [{m[other]['ci'][0]:8.4f}, {m[other]['ci'][1]:8.4f}]")
    print(json.dumps({"criteria": result["criteria"], "artifact": result["artifact"]}, indent=2))



CHECKS = {
    "bootstrap": (check_bootstrap, print_bootstrap),
    "figure_layer_curves": (check_figure_layer_curves, print_figure_layer_curves),
    "test_scores": (check_test_scores, print_test_scores),
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