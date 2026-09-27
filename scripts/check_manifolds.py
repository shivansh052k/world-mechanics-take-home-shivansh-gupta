"""Activation manifolds (Part 2, geometry): per-value train centroids at indices 1 / 9 / 18, curves through them,
and leave-one-centroid-out (LOCO) selection of the PCA dimension and the smoothing. No model is run; stored pooled
activations only (train rows fit, val-unseen rows confirm).

Usage: python scripts/check_manifolds.py <check>
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from vjepa_physics.data import DATASETS
from vjepa_physics.evidence import file_sha256, require_clean_code, save_result
from vjepa_physics.extraction import plot_index
from vjepa_physics.joined import load_joined
from vjepa_physics.manifolds import (
    EXACT, PCA_DIMS, centroid_noise, fit_curve, loco_errors, loco_grid, select_setting, smoothing_grid,
    value_centroids,
)
from vjepa_physics.metrics import bootstrap_indices, percentile_interval, resampled_mean
from vjepa_physics.nullspace import train_scaler
from vjepa_physics.probes import probe_targets, site_features
from vjepa_physics.reproducibility import SEED
from vjepa_physics.steering import STEERING_SITE, covariance_map, load_probe_sequence

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/manifolds/checks.json"
ARTIFACTS = REPO / "artifacts/manifolds"  # regenerable, git-ignored

MANIFOLD_SITES = ("block_0", STEERING_SITE, "block_17")  # indices 1, 9 (selection, steering) and 18 (observations)
EXPECTED_VALUES = 52  # seen values per variable (D-38)
EXPECTED_COUNTS = {"speed": 16, "acceleration": 16}  # train clips per seen value; direction varies (F-79)
MIN_DIRECTION_COUNT = 11
LINE_TOLERANCE = 1e-10  # relative: centroid line vs Phase 5's covariance map (two computation paths)
EXACT_TOLERANCE = 1e-10  # relative: exact interpolation passes through every centroid
GRID_TOLERANCE = 1e-12  # relative: grid entry vs loco_errors² at one fold (two computation paths)
BOOTSTRAP_RESAMPLES = 10_000


def kind_of(variable: str) -> str:
    return "loop" if variable == "direction" else "open"


def relative(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(np.asarray(b)).max())


def standardized(table: dict, site: str):
    """(clips, d) features at `site`, z-scored with the train scaler (the computation behind Phase 4-5's space)."""
    x = site_features(table["activations"], site)
    scaler = train_scaler(x, table["role"])
    return (x - scaler.mean_) / scaler.scale_, scaler


def paired_gap(first_sq: np.ndarray, second_sq: np.ndarray) -> dict:
    """Mean over scored values of first - second (squared LOCO errors), with a value-bootstrap 95% interval."""
    d = (first_sq - second_sq)[np.isfinite(first_sq)]
    samples = resampled_mean(d, bootstrap_indices(len(d), BOOTSTRAP_RESAMPLES, SEED))
    return {"mean": float(d.mean()), "ci": list(percentile_interval(samples)), "n_values": int(len(d))}


def dim_name(k: int | None):
    return "all" if k is None else k


def check_manifold_loco() -> dict:
    """LOCO selection of each variable's activation curve at indices 1 / 9 / 18 (index 9 is the one steered).

    Per variable and site: train centroids per seen value in the train-standardized space; squared LOCO error of
    every (PCA dim, smoothing) on the grid; selection = smallest k, then smoothest, within 1% of the minimum mean;
    the best straight line (speed / acceleration: LINE; direction: H = 1) and its paired gap to the selection with a
    value-bootstrap interval; the centroid noise floor; exact interpolation as a variant; val-unseen centroids as
    confirmation. Writes artifacts/manifolds/curves.npz. Passes if: at index 9 the train scaler equals Phase 5's;
    52 values with the expected clip counts; speed / acceleration line = covariance map within LINE_TOLERANCE;
    exact curves pass through every centroid; a grid entry = loco_errors at one fold; val-unseen values are not
    train values; everything finite; saved file = computed.
    """
    start = time.perf_counter()
    arrays: dict[str, np.ndarray] = {}
    result: dict = {}
    ok: dict[str, list[bool]] = {name: [] for name in (
        "scaler_matches_steering", "value_counts", "line_equals_covariance", "exact_interpolates",
        "grid_matches_loco_errors", "val_unseen_disjoint", "finite")}

    for variable in DATASETS:
        table = load_joined(variable)
        roles, value_index, labels = table["role"], table["value_index"], table["label"]
        train = roles == "train"
        kind = kind_of(variable)
        seq = load_probe_sequence(variable)
        y = probe_targets(variable, labels).reshape(len(labels), -1)
        result[variable] = {}

        for site in MANIFOLD_SITES:
            z, scaler = standardized(table, site)
            if site == STEERING_SITE:
                ok["scaler_matches_steering"].append(
                    bool(np.array_equal(scaler.mean_, seq.mean) and np.array_equal(scaler.scale_, seq.scale)))
            values, cents, counts = value_centroids(z, value_index, labels, roles)
            noise = centroid_noise(z, value_index, roles)
            expected = EXPECTED_COUNTS.get(variable)
            ok["value_counts"].append(len(values) == EXPECTED_VALUES and (
                bool((counts == expected).all()) if expected else int(counts.min()) >= MIN_DIRECTION_COUNT))

            smooths = smoothing_grid(kind, values)
            grid = loco_grid(kind, values, cents, PCA_DIMS, smooths)  # (k, smoothing, V) squared errors
            mse = np.nanmean(grid, axis=2)
            a, b = select_setting(mse)
            k, smooth = PCA_DIMS[a], smooths[b]
            line_a = int(np.argmin(mse[:, 0]))  # best PCA dim for the straight line (grid column 0 is the smoothest)
            scored = np.isfinite(grid[0, 0])
            noise_sq = float(noise[scored].mean())

            j = len(values) // 2
            brute = float(loco_errors(kind, values, cents, k, smooth)[j] ** 2)
            ok["grid_matches_loco_errors"].append(abs(brute - grid[a, b, j]) <= GRID_TOLERANCE * brute)

            exact_all = fit_curve(kind, values, cents, None, EXACT)
            ok["exact_interpolates"].append(relative(exact_all(values), cents) <= EXACT_TOLERANCE)
            exact_sq = {"selected_k": loco_errors(kind, values, cents, k, EXACT) ** 2,
                        "all_axes": loco_errors(kind, values, cents, None, EXACT) ** 2}

            # the straight line (all axes) vs Phase 5's covariance map on the train clips at the same site
            b_map = covariance_map(z[train], y[train])
            targets = probe_targets(variable, values).reshape(len(values), -1)
            covariance_line = z[train].mean(axis=0) + (targets - y[train].mean(axis=0)) @ b_map
            line_all = fit_curve(kind, values, cents, None, smooths[0])
            line_vs_covariance = relative(line_all(values), covariance_line)
            if kind == "open":
                ok["line_equals_covariance"].append(line_vs_covariance <= LINE_TOLERANCE)

            # confirmation: val-unseen centroids against curves fit on all train centroids
            u_values, u_cents, u_counts = value_centroids(z, value_index, labels, roles, role="val_unseen")
            u_noise = centroid_noise(z, value_index, roles, role="val_unseen")
            ok["val_unseen_disjoint"].append(not bool(np.isin(u_values, values).any()))
            curves = {"selected": fit_curve(kind, values, cents, k, smooth),
                      "line": fit_curve(kind, values, cents, PCA_DIMS[line_a], smooths[0]),
                      "exact": exact_all}
            u_sq = {name: ((u_cents - c(u_values)) ** 2).sum(axis=1) for name, c in curves.items()}

            record = {
                "index": plot_index(site), "kind": kind, "n_values": len(values),
                "train_counts": [int(counts.min()), int(counts.max())],
                "pca_dims": [dim_name(d) for d in PCA_DIMS], "smoothing": smooths,
                "mse": mse.tolist(),
                "noise_mean_sq": noise_sq,
                "selected": {"k": dim_name(k), "smooth": smooth, "mse": float(mse[a, b]),
                             "excess_over_noise": float(mse[a, b] - noise_sq),
                             "k_at_grid_end": a in (0, len(PCA_DIMS) - 1), "smooth_at_grid_end": b in (0, len(smooths) - 1)},
                "line": {"k": dim_name(PCA_DIMS[line_a]), "mse": float(mse[line_a, 0]),
                         "excess_over_noise": float(mse[line_a, 0] - noise_sq)},
                "gap_line_minus_selected": paired_gap(grid[line_a, 0], grid[a, b]),
                "exact_mse": {name: float(np.nanmean(v)) for name, v in exact_sq.items()},
                "line_vs_covariance_relative": line_vs_covariance,
                "val_unseen": {"values": u_values.tolist(), "counts": u_counts.tolist(),
                               "noise_sq": u_noise.tolist(),
                               **{f"{name}_sq": v.tolist() for name, v in u_sq.items()}},
            }
            if kind == "open":  # observation: the smoothest spline on the grid vs the line
                record["largest_lam_vs_line_relative"] = relative(
                    fit_curve(kind, values, cents, None, smooths[1])(values), line_all(values))
            ok["finite"].append(bool(np.isfinite(mse).all() and np.isfinite(noise).all()
                                     and all(np.isfinite(v).all() for v in u_sq.values())))
            result[variable][site] = record

            prefix = f"{variable}_{site}"
            arrays |= {
                f"{prefix}_values": values, f"{prefix}_centroids": cents, f"{prefix}_counts": counts,
                f"{prefix}_noise": noise, f"{prefix}_loco_sq": grid,
                f"{prefix}_scaler_mean": scaler.mean_, f"{prefix}_scaler_scale": scaler.scale_,
                f"{prefix}_selected": np.array([a, b]), f"{prefix}_line_k_index": np.array(line_a),
                f"{prefix}_val_unseen_values": u_values, f"{prefix}_val_unseen_centroids": u_cents,
            }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / "curves.npz"
    np.savez(path, **arrays)
    with np.load(path) as saved:
        saved_ok = set(saved.files) == set(arrays) and all(
            np.array_equal(saved[key], value, equal_nan=True) for key, value in arrays.items())

    criteria = {name: all(values) for name, values in ok.items()} | {"saved_equals_computed": saved_ok}
    return {
        "variables": result,
        "sites": list(MANIFOLD_SITES), "selection_site": STEERING_SITE,
        "selection_rule": "mean squared LOCO error; smallest k, then smoothest, within 1% of the grid minimum",
        "seconds": time.perf_counter() - start,
        "criteria": criteria,
        "artifact": {"path": str(path.relative_to(REPO)), "sha256": file_sha256(path)},
        "passed": all(criteria.values()),
    }


CHECKS = {
    "manifold_loco": check_manifold_loco,
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