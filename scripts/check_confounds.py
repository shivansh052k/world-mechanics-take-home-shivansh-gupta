"""Confound controls for speed and acceleration: probes applied across the two sets, on a shared distance-travelled
scale. No model is run; stored pooled activations only (train rows fit, validation rows in the overlap window scored).

Usage: python scripts/check_confounds.py <check>
"""
import argparse
import json
from pathlib import Path

import numpy as np

from vjepa_physics.confounds import (
    PAIR, as_distance, clip_distance, in_window, overlap_window, reading_verdict, tau_fraction_from_slope,
)
from vjepa_physics.evidence import require_clean_code, save_result, verified_artifact
from vjepa_physics.extraction import SITES, plot_index
from vjepa_physics.joined import load_joined
from vjepa_physics.metrics import bootstrap_indices, percentile_interval, r2
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


CHECKS = {"cross_applied_probes": check_cross_applied_probes}


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