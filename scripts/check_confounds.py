"""Confound controls for speed and acceleration: probes applied across the two sets, on a shared distance-travelled
scale. No model is run; stored pooled activations only (train rows fit, validation rows in the overlap window scored).

Usage: python scripts/check_confounds.py <check>
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from vjepa_physics.confounds import (
    PAIR, as_distance, clip_distance, clip_pairs, in_window, overlap_window, pair_weights, reading_verdict,
    shared_pair_rows, tau_fraction_from_slope, weighted_balanced_accuracy,
)
from vjepa_physics.evidence import file_sha256, require_clean_code, save_result, verified_artifact
from vjepa_physics.extraction import SITES, plot_index
from vjepa_physics.joined import load_joined
from vjepa_physics.metrics import bootstrap_indices, percentile_interval, r2
from vjepa_physics.plotting import DATASET_COLOUR, INK, INK_MUTED, INK_SECONDARY, SURFACE, style_axes
from vjepa_physics.probes import alpha_verdict, fit_probe, probe_scores, probe_targets, site_features
from vjepa_physics.reproducibility import SEED
from vjepa_physics.steering import STEERING_SITE

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/confounds/checks.json"
PROBE_CHECKS = REPO / "results/probes/checks.json"  # key "layer_curves": alphas and saved validation predictions

VALIDATION_ROLES = ("val_seen", "val_unseen")
HEADLINE_SITES = (STEERING_SITE, "block_17")  # indices 9 and 18: intervals and verdicts here only
BOOTSTRAP_RESAMPLES = 10_000
EXPECTED_FIT = {"speed": 832, "acceleration": 832}  # train clips per set
EXPECTED_SCORED = {"speed": 232, "acceleration": 288}  # validation clips inside the overlap window
POINT_TOLERANCE = 1e-12  # relative: identity resample vs the full-sample slope

EXPECTED_PAIRS = {"train": (32, 512, 512), "validation": (37, 208, 188)}  # pairs, speed clips, acceleration clips
MAX_CONTROL_ACCURACY = 0.6  # the distance-only classifier must stay at or below this (matching works)
SEPARABLE_ABOVE = 0.6  # reading: the balanced-accuracy CI's lower end above this and above the control
NO_SIGNAL_ACCURACY = 0.55  # an upper-edge alpha is expected only at or below this balanced accuracy

FIGURE = REPO / "results/confounds/confounds.png"
FIGURE_DPI = 200
FIGURE_SITE = "block_17"  # index 18
PROBE_LABEL = {"acceleration": "acceleration probe on speed clips", "speed": "speed probe on acceleration clips"}


def site_index(site: str) -> int | None:
    """hidden_states index of a site; None for the final norm."""
    return None if site == "final_norm" else plot_index(site)


def ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares slope and intercept of y on x."""
    xc = x - x.mean()
    slope = float((xc * (y - y.mean())).sum() / (xc**2).sum())
    return slope, float(y.mean() - slope * x.mean())


def resampled_slopes(x: np.ndarray, y: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """(B,) least-squares slopes of y on x, one per row of resampled clip indices (B, n)."""
    xs, ys = x[indices], y[indices]
    xc = xs - xs.mean(axis=1, keepdims=True)
    return (xc * (ys - ys.mean(axis=1, keepdims=True))).sum(axis=1) / (xc**2).sum(axis=1)


def all_finite(obj) -> bool:
    if isinstance(obj, dict):
        return all(all_finite(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(all_finite(v) for v in obj)
    if isinstance(obj, float):
        return bool(np.isfinite(obj))
    return True


def check_cross_applied_probes() -> dict:
    """Each set's probe (refit on its train clips as in the layer curves) applied to the other set's validation clips
    inside the overlap window, at every site; predictions and truth in metres travelled over the clip.

    Per site and probe: least-squares slope and intercept of cross-predicted on true distance, R² against the distance
    line, and the same for the probe's own validation clips in the window (reference). At indices 9 and 18: clip-bootstrap
    95% intervals; acceleration probe on speed clips -> reading_verdict (fixed rule); speed probe on acceleration clips
    -> tau / T = slope / 2 (no verdict). Passes if: n_fit exact; alpha and validation predictions = layer_curves exactly;
    no alpha failure; scored counts as expected; identity resample = full-sample slope; everything finite.
    """
    tables = {v: load_joined(v) for v in PAIR}
    distance = {v: clip_distance(tables[v]) for v in PAIR}
    window = overlap_window(distance.values())
    validation = {v: np.isin(tables[v]["role"], VALIDATION_ROLES) for v in PAIR}
    scored = {v: validation[v] & in_window(distance[v], window) for v in PAIR}
    boot = {v: bootstrap_indices(int(scored[v].sum()), BOOTSTRAP_RESAMPLES, SEED) for v in PAIR}
    saved_curves = json.loads(PROBE_CHECKS.read_text())["layer_curves"]["result"]
    with np.load(verified_artifact(PROBE_CHECKS, "layer_curves")) as f:
        saved_predictions = {v: f[f"{v}_predictions"] for v in PAIR}
        ids_ok = all(np.array_equal(f[f"{v}_ids"], tables[v]["id"]) for v in PAIR)

    ok: dict[str, list[bool]] = {name: [] for name in (
        "n_fit", "alpha_matches_layer_curves", "validation_reproduced", "no_alpha_failure", "bootstrap_point", "finite")}
    result: dict = {v: {} for v in PAIR}
    summary: dict = {}

    for k, site in enumerate(SITES):
        x = {v: site_features(tables[v]["activations"], site) for v in PAIR}
        for v in PAIR:
            other = PAIR[1 - PAIR.index(v)]
            t = tables[v]
            probe = fit_probe(x[v], probe_targets(v, t["label"]), t["role"])
            val_seen = t["role"] == "val_seen"
            own_r2 = probe_scores(v, t["label"][val_seen], probe.predict(x[v][val_seen]))["r2"]
            verdict = alpha_verdict(probe.alpha_edge, own_r2)
            ok["n_fit"].append(probe.n_fit == EXPECTED_FIT[v])
            ok["alpha_matches_layer_curves"].append(probe.alpha == saved_curves[v]["sites"][site]["alpha"])
            ok["validation_reproduced"].append(bool(np.array_equal(
                probe.predict(x[v][validation[v]]), saved_predictions[v][validation[v], k])))
            ok["no_alpha_failure"].append(verdict != "failure")

            true_d = distance[other][scored[other]]
            cross_d = as_distance(v, probe.predict(x[other][scored[other]]))
            own_true = distance[v][scored[v]]
            own_d = as_distance(v, probe.predict(x[v][scored[v]]))
            slope, intercept = ols(true_d, cross_d)
            own_slope, own_intercept = ols(own_true, own_d)
            record = {
                "plot_index": site_index(site), "alpha": probe.alpha, "alpha_verdict": verdict,
                "own_val_seen_r2": own_r2,
                "cross": {"applied_to": other, "n_clips": int(len(true_d)), "slope": slope, "intercept_m": intercept,
                          "r2_vs_distance": r2(true_d, cross_d)},
                "own_in_window": {"n_clips": int(len(own_true)), "slope": own_slope, "intercept_m": own_intercept,
                                  "r2_vs_distance": r2(own_true, own_d)},
            }
            if site in HEADLINE_SITES:
                ci = percentile_interval(resampled_slopes(true_d, cross_d, boot[other]))
                point = float(resampled_slopes(true_d, cross_d, np.arange(len(true_d))[None])[0])
                ok["bootstrap_point"].append(abs(point - slope) <= POINT_TOLERANCE * max(1.0, abs(slope)))
                record["cross"]["slope_ci"] = [float(c) for c in ci]
                record["own_in_window"]["slope_ci"] = [
                    float(c) for c in percentile_interval(resampled_slopes(own_true, own_d, boot[v]))]
                if v == "acceleration":
                    record["cross"]["verdict"] = reading_verdict(ci)
                else:
                    record["cross"]["tau_fraction"] = tau_fraction_from_slope(slope)
                    record["cross"]["tau_fraction_ci"] = [tau_fraction_from_slope(float(c)) for c in ci]
                summary.setdefault(f"{v}_probe_on_{other}_clips", {})[site_index(site)] = {
                    key: record["cross"][key] for key in record["cross"] if key not in ("applied_to", "n_clips")
                } | {"own_slope": own_slope, "own_slope_ci": record["own_in_window"]["slope_ci"]}
            ok["finite"].append(all_finite(record))
            result[v][site] = record

    criteria = {name: all(values) for name, values in ok.items()} | {
        "saved_ids_aligned": ids_ok,
        "scored_counts": all(int(scored[v].sum()) == EXPECTED_SCORED[v] for v in PAIR),
    }
    return {
        "window_m": list(window), "roles_scored": list(VALIDATION_ROLES),
        "note": "each probe is applied out of distribution: to clips of the other set, whose motion profile it never saw",
        "rule": {"acceleration_probe_on_speed_clips": "CI < 0.25 reads acceleration; CI > 0.5 reads a speed / distance "
                 "quantity; else mixed", "speed_probe_on_acceleration_clips": "tau / T = slope / 2, no verdict"},
        "summary": summary, "probes": result,
        "criteria": criteria, "passed": all(criteria.values()),
    }


def classifier_alpha_verdict(edge: str | None, accuracy: float) -> str:
    """The alpha rule for the set classifier: upper edge allowed only without signal; lower edge = failure."""
    if edge is None:
        return "ok"
    if edge == "upper" and accuracy <= NO_SIGNAL_ACCURACY:
        return "no_signal"
    return "failure"


def check_matched_distance_classifier() -> dict:
    """Speed-set vs acceleration-set classifier at matched distance travelled, at every site.

    Clips: the two sets' values inside the overlap window, paired one to one by nearest distance; train = pairs whose
    values both have train clips (16 + 16 each, balanced); validation = pairs with validation clips in both sets,
    weighted 1 / (clips of the pair in that class) so both classes share one distance distribution. Classifier: ridge
    on -1 (speed) / +1 (acceleration), train z-scoring, leave-one-out alpha. Scores: weighted balanced accuracy (sign of
    the output) and weighted AUC; clip-bootstrap 95% intervals at indices 9 and 18. Control: the same classifier on
    [distance, distance²] only. Passes if: pair and clip counts as expected; control balanced accuracy at or below
    MAX_CONTROL_ACCURACY; no alpha failure; everything finite. Reading at 9 / 18: separable if the interval's lower
    end is above SEPARABLE_ABOVE and above the control; index 0 is an observation.
    """
    tables = {v: load_joined(v) for v in PAIR}
    distance = {v: clip_distance(tables[v]) for v in PAIR}
    window = overlap_window(distance.values())
    ids = dict(zip(PAIR, clip_pairs(distance["speed"], distance["acceleration"], window)[:2]))
    masks = {}
    for name, roles in (("train", ("train",)), ("validation", VALIDATION_ROLES)):
        role_masks = [np.isin(tables[v]["role"], roles) for v in PAIR]
        masks[name] = dict(zip(PAIR, shared_pair_rows(ids["speed"], ids["acceleration"], *role_masks)))
    counts = {name: (len(np.unique(ids["speed"][m["speed"]])), int(m["speed"].sum()), int(m["acceleration"].sum()))
              for name, m in masks.items()}
    is_acceleration = {name: np.concatenate([np.zeros(masks[name]["speed"].sum(), bool),
                                             np.ones(masks[name]["acceleration"].sum(), bool)]) for name in masks}
    weights = np.concatenate([pair_weights(ids[v][masks["validation"][v]]) for v in PAIR])
    y_val = is_acceleration["validation"]
    boot = bootstrap_indices(len(y_val), BOOTSTRAP_RESAMPLES, SEED)

    def fit_and_score(features: dict[str, np.ndarray]):
        x_train, x_val = (np.concatenate([features[v][masks[name][v]] for v in PAIR]) for name in masks)
        target = np.where(is_acceleration["train"], 1.0, -1.0)
        probe = fit_probe(x_train, target, np.full(len(target), "train"))
        score = probe.predict(x_val)
        accuracy = weighted_balanced_accuracy(y_val, score > 0, weights)
        return probe, score, accuracy, float(roc_auc_score(y_val, score, sample_weight=weights))

    ok: dict[str, list[bool]] = {name: [] for name in ("n_fit", "no_alpha_failure", "finite")}
    control, _, control_accuracy, control_auc = fit_and_score(
        {v: np.stack([distance[v], distance[v] ** 2], axis=1) for v in PAIR})
    control_verdict = classifier_alpha_verdict(control.alpha_edge, control_accuracy)
    ok["no_alpha_failure"].append(control_verdict != "failure")

    sites, headline = {}, {}
    for site in SITES:
        x = {v: site_features(tables[v]["activations"], site) for v in PAIR}
        probe, score, accuracy, auc = fit_and_score(x)
        verdict = classifier_alpha_verdict(probe.alpha_edge, accuracy)
        record = {"plot_index": site_index(site), "alpha": probe.alpha, "alpha_edge": probe.alpha_edge,
                  "alpha_verdict": verdict, "balanced_accuracy": accuracy, "auc": auc}
        if site in HEADLINE_SITES:
            accuracy_ci = percentile_interval(np.array(
                [weighted_balanced_accuracy(y_val[i], score[i] > 0, weights[i]) for i in boot]))
            auc_ci = percentile_interval(np.array([roc_auc_score(y_val[i], score[i], sample_weight=weights[i])
                                                   for i in boot]))
            separable = accuracy_ci[0] > SEPARABLE_ABOVE and accuracy_ci[0] > control_accuracy
            record |= {"balanced_accuracy_ci": [float(c) for c in accuracy_ci], "auc_ci": [float(c) for c in auc_ci],
                       "reading": "motion profile linearly separable at matched distance" if separable
                       else "not separable by the pre-set rule"}
            headline[site_index(site)] = {key: record[key] for key in (
                "balanced_accuracy", "balanced_accuracy_ci", "auc", "auc_ci", "reading")}
        ok["n_fit"].append(probe.n_fit == sum(EXPECTED_PAIRS["train"][1:]))
        ok["no_alpha_failure"].append(verdict != "failure")
        ok["finite"].append(all_finite(record))
        sites[site] = record

    control_record = {"alpha": control.alpha, "alpha_verdict": control_verdict,
                      "balanced_accuracy": control_accuracy, "auc": control_auc}
    criteria = {name: all(values) for name, values in ok.items()} | {
        "counts_as_expected": all(counts[name] == EXPECTED_PAIRS[name] for name in masks),
        "control_at_chance": control_accuracy <= MAX_CONTROL_ACCURACY,
        "control_finite": all_finite(control_record),
    }
    summary = {
        "distance_only_control": control_record,
        "balanced_accuracy_by_index": {str(r["plot_index"]): round(r["balanced_accuracy"], 3) for r in sites.values()},
        "headline": headline,
    }
    return {
        "window_m": list(window), "counts": {name: dict(zip(("pairs", "speed_clips", "acceleration_clips"), c))
                                             for name, c in counts.items()},
        "weighting": "validation clips weighted 1 / (clips of their pair in their class)",
        "summary": summary, "sites": sites, "criteria": criteria, "passed": all(criteria.values()),
    }

def check_figure_confounds() -> dict:
    """Left: each probe applied to the other set's validation clips in the overlap window at index 18, reading vs true
    distance (m), with the slope-1 line; slopes and intervals from cross_applied_probes. Right: the matched-distance
    set classifier's balanced accuracy by layer index (0-24) with its index 9 / 18 intervals and the distance-only
    control, all from matched_distance_classifier. Passes if: the refit index-18 probes reproduce the saved validation
    predictions bit for bit and the saved slopes; the PNG is written and non-empty; every plotted value is finite.
    """
    saved = json.loads(OUT.read_text())
    cross = saved["cross_applied_probes"]["result"]["probes"]
    classifier = saved["matched_distance_classifier"]["result"]
    tables = {v: load_joined(v) for v in PAIR}
    distance = {v: clip_distance(tables[v]) for v in PAIR}
    window = overlap_window(distance.values())
    validation = {v: np.isin(tables[v]["role"], VALIDATION_ROLES) for v in PAIR}
    scored = {v: validation[v] & in_window(distance[v], window) for v in PAIR}
    k = SITES.index(FIGURE_SITE)
    with np.load(verified_artifact(PROBE_CHECKS, "layer_curves")) as f:
        saved_predictions = {v: f[f"{v}_predictions"] for v in PAIR}

    ok: dict[str, list[bool]] = {"validation_reproduced": [], "slope_matches_saved": []}
    plotted: list[float] = []
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor=SURFACE,
                                      gridspec_kw={"width_ratios": [1, 1.25]})

    top = 0.0
    for v in ("acceleration", "speed"):
        other = PAIR[1 - PAIR.index(v)]
        t = tables[v]
        x_own = site_features(t["activations"], FIGURE_SITE)
        probe = fit_probe(x_own, probe_targets(v, t["label"]), t["role"])
        ok["validation_reproduced"].append(bool(np.array_equal(
            probe.predict(x_own[validation[v]]), saved_predictions[v][validation[v], k])))
        true_d = distance[other][scored[other]]
        cross_d = as_distance(v, probe.predict(site_features(tables[other]["activations"], FIGURE_SITE)[scored[other]]))
        record = cross[v][FIGURE_SITE]["cross"]
        ok["slope_matches_saved"].append(
            abs(ols(true_d, cross_d)[0] - record["slope"]) <= POINT_TOLERANCE * max(1.0, abs(record["slope"])))
        low, high = record["slope_ci"]
        left.scatter(true_d, cross_d, s=14, color=DATASET_COLOUR[v], alpha=0.55, linewidths=0, zorder=2,
                     label=f"{PROBE_LABEL[v]}: slope {record['slope']:.2f} [{low:.2f}, {high:.2f}]")
        plotted += true_d.tolist() + cross_d.tolist()
        top = max(top, float(true_d.max()), float(cross_d.max()))
    edge = 1.05 * top
    left.plot([0, edge], [0, edge], "--", color=INK_MUTED, linewidth=1.2, zorder=1,
              label="slope 1: reads distance travelled")
    left.set_xlim(0, edge)
    left.set_ylim(0, edge)
    style_axes(left, grid_axis="both")
    left.set_title("Probes applied across sets read distance (index 18)", fontsize=11, loc="left", color=INK)
    left.set_xlabel("true distance travelled over the clip (m)", fontsize=9)
    left.set_ylabel("cross-applied probe reading (m)", fontsize=9)
    left.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=INK_SECONDARY)  # empty below the diagonal

    rows = sorted((r for r in classifier["sites"].values() if r["plot_index"] is not None),
                  key=lambda r: r["plot_index"])
    index = [r["plot_index"] for r in rows]
    accuracy = [r["balanced_accuracy"] for r in rows]
    plotted += accuracy
    right.plot(index, accuracy, "-", color=INK_SECONDARY, linewidth=2, marker="o", markersize=5, zorder=3)
    for r in rows:
        if "balanced_accuracy_ci" in r:
            low, high = r["balanced_accuracy_ci"]
            plotted += [low, high]
            right.plot([r["plot_index"]] * 2, [low, high], color=INK, linewidth=2, zorder=4)
    control = classifier["summary"]["distance_only_control"]["balanced_accuracy"]
    plotted.append(control)
    right.axhline(control, linestyle="--", color=INK_MUTED, linewidth=1.2, zorder=1)
    right.text(max(index), control + 0.012, f"distance-only control {control:.2f}", ha="right", va="bottom",
               fontsize=8.5, color=INK_MUTED)
    headline = classifier["summary"]["headline"]
    note = "\n".join(f"index {i}: {headline[str(i)]['balanced_accuracy']:.2f} "
                     f"[{headline[str(i)]['balanced_accuracy_ci'][0]:.2f}, "
                     f"{headline[str(i)]['balanced_accuracy_ci'][1]:.2f}]" for i in (9, 18))
    right.text(max(index), 0.62, note, ha="right", va="bottom", fontsize=8.5, color=INK_SECONDARY)
    right.set_ylim(0.45, 1.02)
    style_axes(right, grid_axis="y")
    right.set_title("Speed vs acceleration set, matched distance", fontsize=11, loc="left", color=INK)
    right.set_xlabel("layer index (0 = embedding)", fontsize=9)
    right.set_ylabel("balanced accuracy (validation)", fontsize=9)

    fig.suptitle("The probes read distance travelled, yet the two motion profiles stay separable",
                 fontsize=12, color=INK, x=0.06, ha="left")
    fig.subplots_adjust(bottom=0.13, top=0.84, wspace=0.25)
    fig.savefig(FIGURE, dpi=FIGURE_DPI, facecolor=SURFACE)
    plt.close(fig)

    criteria = {name: all(values) for name, values in ok.items()} | {
        "written": FIGURE.exists() and FIGURE.stat().st_size > 0,
        "finite": bool(np.isfinite(np.asarray(plotted, dtype=float)).all()),
    }
    return {
        "site": FIGURE_SITE, "summary": {"slopes_idx18": {v: cross[v][FIGURE_SITE]["cross"]["slope"] for v in PAIR},
                                         "control": control},
        "figure": {"path": str(FIGURE.relative_to(REPO)), "sha256": file_sha256(FIGURE)},
        "sources": {key: saved[key]["provenance"]["git_commit"]
                    for key in ("cross_applied_probes", "matched_distance_classifier")},
        "values_plotted": len(plotted), "criteria": criteria, "passed": all(criteria.values()),
    }

CHECKS = {
    "cross_applied_probes": check_cross_applied_probes,
    "matched_distance_classifier": check_matched_distance_classifier,
    "figure_confounds": check_figure_confounds,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("check", choices=sorted(CHECKS))
    name = parser.parse_args().check
    require_clean_code()  # after parsing, so --help still works with uncommitted code
    result = CHECKS[name]()
    save_result(OUT, name, result)
    print(json.dumps({"summary": result["summary"], "criteria": result["criteria"]}, indent=2))
    print(f"saved '{name}' -> {OUT.relative_to(REPO)}")
    if result.get("passed") is False:
        raise SystemExit(f"check '{name}' FAILED")


if __name__ == "__main__":
    main()