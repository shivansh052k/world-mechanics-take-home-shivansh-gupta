"""Activation manifolds (Part 2, geometry): per-value train centroids at indices 1 / 9 / 18, curves through them,
and leave-one-centroid-out (LOCO) selection of the PCA dimension and the smoothing. No model is run; stored pooled
activations only (train rows fit, val-unseen rows confirm).

Usage: python scripts/check_manifolds.py <check>
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler

from vjepa_physics.data import DATASETS
from vjepa_physics.evidence import file_sha256, require_clean_code, save_result, verified_artifact
from vjepa_physics.extraction import plot_index
from vjepa_physics.joined import load_joined
from vjepa_physics.manifolds import (
    EXACT, HARMONICS, LINE, PCA_DIMS, centroid_noise, fit_curve, fit_spacing_line, harmonic_coefficients,
    harmonic_power, loco_errors, loco_grid, loco_spacing_grid, participation_ratio, principal_cosines,
    select_setting, smoothing_grid, unit_tangents, value_centroids,
)
from vjepa_physics.metrics import bootstrap_indices, percentile_interval, resampled_mean
from vjepa_physics.nullspace import train_scaler
from vjepa_physics.plotting import DATASET_COLOUR, GRID, INK, INK_MUTED, INK_SECONDARY, SURFACE, style_axes
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

PAIR = ("speed", "acceleration")
CLIP_SECONDS = 15 / 24  # frame 0 to frame 15 at 24 fps
DISTANCE_PER_UNIT = {"speed": CLIP_SECONDS, "acceleration": CLIP_SECONDS**2 / 2}  # metres per m/s; per m/s² from rest
WINDOW_REFERENCE = (0.15625, 1.953125)  # F-65's overlap window of distances travelled (metres)
WINDOW_TOLERANCE = 1e-12
TANGENT_POINTS = 201  # matched distances for tangent angles (ends dropped)

MAX_HARMONIC = max(HARMONICS)  # spectrum up to the loop grid's largest H (12)
ROUND_HARMONICS = 6  # harmonic planes compared with each nullspace round block
PERMUTATIONS = 1_000  # angle-shuffle null for the harmonic spectrum
DENSE_POINTS = 2001  # curve samples for the curve's participation ratio
HARMONIC_TOLERANCE = 1e-10  # relative: full-space trig fit vs the loop curve with all PCA axes

FIGURE_MANIFOLDS = REPO / "results/manifolds/manifolds.png"
FIGURE_DPI = 200
REPRODUCE_TOLERANCE = 1e-12  # relative: figure's recomputed LOCO means vs the saved keys
LABEL_MARKS = {"direction": (0.0, 90.0, 180.0, 270.0), "speed": (0.25, 1.0, 2.0, 3.0, 4.0),
               "acceleration": (0.25, 2.5, 5.0, 7.5, 10.0)}  # values annotated along each curve
UNITS = {"direction": "°", "speed": " m/s", "acceleration": " m/s²"}

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


def angle_degrees(u: np.ndarray, v: np.ndarray) -> float:
    cos = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def check_speed_acceleration_manifold() -> dict:
    """Observation: do the speed and acceleration sets lie on one index-9 curve on a distance-travelled scale?

    Window = the overlap of both sets' distance ranges (F-65). One scaler fit on both sets' train clips in the window;
    per set, train centroids per seen value parameterized by distance (metres) and an open curve chosen by the LOCO
    rule. Each set's centroids are scored against the other set's curve at the same distance (raw, and after removing
    the mean offset) and compared with their own LOCO error (paired value-bootstrap intervals). Tangent angles at
    matched distances, straight-line slope angle, principal angles between the two curves' PCA axes, each curve's own
    tangent turn and arc length. Passes if: window = F-65's; 16 train clips per value; a grid entry = loco_errors at
    one fold for each set; everything finite.
    """
    tables = {v: load_joined(v) for v in PAIR}
    distance = {v: np.asarray(tables[v]["label"], dtype=np.float64) * DISTANCE_PER_UNIT[v] for v in PAIR}
    lo = max(float(distance[v].min()) for v in PAIR)
    hi = min(float(distance[v].max()) for v in PAIR)
    in_window = {v: (distance[v] >= lo) & (distance[v] <= hi) for v in PAIR}
    x = {v: site_features(tables[v]["activations"], STEERING_SITE) for v in PAIR}
    fit_rows = {v: in_window[v] & (tables[v]["role"] == "train") for v in PAIR}
    scaler = StandardScaler().fit(np.concatenate([x[v][fit_rows[v]] for v in PAIR]))

    ok: dict[str, list[bool]] = {name: [] for name in ("value_counts", "grid_matches_loco_errors")}
    sets, curves, per_set = {}, {}, {}
    for v in PAIR:
        rows = in_window[v]
        z = (x[v][rows] - scaler.mean_) / scaler.scale_
        values, cents, counts = value_centroids(
            z, tables[v]["value_index"][rows], distance[v][rows], tables[v]["role"][rows])
        ok["value_counts"].append(bool((counts == EXPECTED_COUNTS[v]).all()))
        smooths = smoothing_grid("open", values)
        grid = loco_grid("open", values, cents, PCA_DIMS, smooths)
        mse = np.nanmean(grid, axis=2)
        a, b = select_setting(mse)
        k, smooth = PCA_DIMS[a], smooths[b]
        j = len(values) // 2
        brute = float(loco_errors("open", values, cents, k, smooth)[j] ** 2)
        ok["grid_matches_loco_errors"].append(abs(brute - grid[a, b, j]) <= GRID_TOLERANCE * brute)
        line_a = int(np.argmin(mse[:, 0]))
        sets[v] = (values, cents, grid[a, b])
        curves[v] = fit_curve("open", values, cents, k, smooth)
        per_set[v] = {
            "n_values": len(values), "train_clips": int(fit_rows[v].sum()),
            "distance_range": [float(values[0]), float(values[-1])],
            "median_adjacent_spacing": float(np.median(np.linalg.norm(np.diff(cents, axis=0), axis=1))),
            "selected": {"k": dim_name(k), "smooth": smooth, "mse": float(mse[a, b])},
            "line": {"k": dim_name(PCA_DIMS[line_a]), "mse": float(mse[line_a, 0])},
            "gap_line_minus_selected": paired_gap(grid[line_a, 0], grid[a, b]),
        }

    cross = {}
    for v, other in (PAIR, PAIR[::-1]):
        values, cents, own_sq = sets[v]
        other_values = sets[other][0]
        inside = (values >= other_values[0]) & (values <= other_values[-1]) & np.isfinite(own_sq)
        residual = cents[inside] - curves[other](values[inside])
        offset = residual.mean(axis=0)
        cross_sq, aligned_sq = np.full(len(values), np.nan), np.full(len(values), np.nan)
        cross_sq[inside] = (residual**2).sum(axis=1)
        aligned_sq[inside] = ((residual - offset) ** 2).sum(axis=1)
        cross[f"{v}_centroids_on_{other}_curve"] = {
            "n_values": int(inside.sum()),
            "own_loco_mse": float(own_sq[inside].mean()),
            "cross_mse": float(np.nanmean(cross_sq)),
            "cross_aligned_mse": float(np.nanmean(aligned_sq)),
            "offset_norm": float(np.linalg.norm(offset)),
            "gap_cross_minus_own": paired_gap(cross_sq, own_sq),
            "gap_aligned_minus_own": paired_gap(aligned_sq, own_sq),
        }

    common_lo = max(float(sets[v][0][0]) for v in PAIR)
    common_hi = min(float(sets[v][0][-1]) for v in PAIR)
    matched = np.linspace(common_lo, common_hi, TANGENT_POINTS)[1:-1]
    tangents = {v: unit_tangents(curves[v], matched) for v in PAIR}
    cos = np.clip((tangents["speed"] * tangents["acceleration"]).sum(axis=1), -1.0, 1.0)
    tangent_angles = np.degrees(np.arccos(cos))
    lines = {v: fit_curve("open", sets[v][0], sets[v][1], None, LINE) for v in PAIR}
    ends = np.array([common_lo, common_hi])
    slopes = {v: np.diff(lines[v](ends), axis=0)[0] for v in PAIR}
    singular = np.linalg.svd(curves["speed"].axes.T @ curves["acceleration"].axes, compute_uv=False)
    dense = np.linspace(common_lo, common_hi, 2001)
    geometry = {
        "common_distance_range": [common_lo, common_hi],
        "tangent_angle_degrees": {"median": float(np.median(tangent_angles)), "min": float(tangent_angles.min()),
                                  "max": float(tangent_angles.max())},
        "line_slope_angle_degrees": angle_degrees(slopes["speed"], slopes["acceleration"]),
        "principal_angles_degrees": np.degrees(np.arccos(np.clip(singular, -1.0, 1.0))).tolist(),
        "own_tangent_turn_degrees": {v: angle_degrees(tangents[v][0], tangents[v][-1]) for v in PAIR},
        "arc_length": {v: float(np.linalg.norm(np.diff(curves[v](dense), axis=0), axis=1).sum()) for v in PAIR},
    }

    window_ok = abs(lo - WINDOW_REFERENCE[0]) <= WINDOW_TOLERANCE and abs(hi - WINDOW_REFERENCE[1]) <= WINDOW_TOLERANCE
    result = {"window": [lo, hi], "sets": per_set, "cross": cross, "geometry": geometry}
    criteria = {name: all(values) for name, values in ok.items()} | {
        "window_matches_reference": window_ok,
        "finite": bool(np.isfinite(json.dumps(result).count("NaN") == 0)),
    }
    return result | {
        "post_hoc": False, "observation": True, "site": STEERING_SITE,
        "criteria": criteria, "passed": all(criteria.values()),
    }


def ladder_share(first: np.ndarray, middle: np.ndarray, last: np.ndarray) -> dict:
    """Share of the first -> last gain in mean squared LOCO error reached at the middle rung, (first - middle) /
    (first - last), with a value-bootstrap interval (the same resamples for all three)."""
    keep = np.isfinite(first)
    f, m, l = first[keep], middle[keep], last[keep]
    idx = bootstrap_indices(len(f), BOOTSTRAP_RESAMPLES, SEED)
    rf, rm, rl = (resampled_mean(a, idx) for a in (f, m, l))
    return {"share": float((f.mean() - m.mean()) / (f.mean() - l.mean())),
            "ci": list(percentile_interval((rf - rm) / (rf - rl)))}


def check_manifold_ladder() -> dict:
    """Q1 ladder at index 9: which part of the curve's LOCO gain over the line is uneven spacing and which curvature.

    Speed / acceleration: line (linear in the label, best k) -> free-spacing lines -> selected curve. Two free-spacing
    rungs: the PC1 line (k = 1 row of the grid, its own LOCO-rule lam; best-fitting straight line -> the gain beyond it
    is the conservative curvature estimate) and the B line (covariance direction, refit per fold; nested with the
    covariance arm -> the steering-relevant split). Direction: ellipse (H = 1) -> selected curve. Per rung: mean
    squared LOCO, paired value-bootstrap gaps, spacing shares (line -> rung) / (line -> curve), and squared errors at
    the val-unseen centroids of curves fit on all train centroids. Passes if: the recomputed selection and k = 1 row
    equal manifold_loco's; the B line with linear p equals the covariance map within LINE_TOLERANCE; a spacing-grid
    entry equals a brute-force fold; everything finite.
    """
    saved = json.loads(OUT.read_text())["manifold_loco"]["result"]["variables"]
    ok: dict[str, list[bool]] = {name: [] for name in (
        "curve_matches_manifold_loco", "pc1_row_matches_manifold_loco", "spacing_line_equals_covariance",
        "spacing_grid_matches_brute_force", "finite")}
    result: dict = {}

    for variable in DATASETS:
        table = load_joined(variable)
        roles, value_index, labels = table["role"], table["value_index"], table["label"]
        train = roles == "train"
        kind = kind_of(variable)
        z, _ = standardized(table, STEERING_SITE)
        values, cents, _ = value_centroids(z, value_index, labels, roles)
        u_values, u_cents, _ = value_centroids(z, value_index, labels, roles, role="val_unseen")

        smooths = smoothing_grid(kind, values)
        grid = loco_grid(kind, values, cents, PCA_DIMS, smooths)
        mse = np.nanmean(grid, axis=2)
        a, b = select_setting(mse)
        line_a = int(np.argmin(mse[:, 0]))
        reference = saved[variable][STEERING_SITE]
        ok["curve_matches_manifold_loco"].append(
            dim_name(PCA_DIMS[a]) == reference["selected"]["k"] and smooths[b] == reference["selected"]["smooth"])
        ok["pc1_row_matches_manifold_loco"].append(relative(mse[0], np.array(reference["mse"][0])) <= GRID_TOLERANCE)

        rungs = {"line": (grid[line_a, 0], fit_curve(kind, values, cents, PCA_DIMS[line_a], smooths[0]))}
        if kind == "open":
            _, pc1_b = select_setting(mse[:1])
            rungs["pc1_spacing"] = (grid[0, pc1_b], fit_curve(kind, values, cents, 1, smooths[pc1_b]))
            spacing = loco_spacing_grid(values, cents, smooths)
            _, sb = select_setting(np.nanmean(spacing, axis=1)[None, :])
            rungs["covariance_spacing"] = (spacing[sb], fit_spacing_line(values, cents, smooths[sb]))

            y = probe_targets(variable, labels).reshape(len(labels), -1)
            b_map = covariance_map(z[train], y[train])
            covariance_line = z[train].mean(axis=0) + (values[:, None] - y[train].mean(axis=0)) @ b_map
            ok["spacing_line_equals_covariance"].append(
                relative(fit_spacing_line(values, cents, LINE)(values), covariance_line) <= LINE_TOLERANCE)
            j = len(values) // 2
            keep = np.arange(len(values)) != j
            brute = float(((cents[j] - fit_spacing_line(values[keep], cents[keep], smooths[sb])(values[j])[0]) ** 2).sum())
            ok["spacing_grid_matches_brute_force"].append(abs(brute - spacing[sb, j]) <= GRID_TOLERANCE * brute)
        rungs["curve"] = (grid[a, b], fit_curve(kind, values, cents, PCA_DIMS[a], smooths[b]))

        record: dict = {"rungs": {}, "gaps": {}, "val_unseen": {"values": u_values.tolist()}}
        for name, (sq, curve) in rungs.items():
            record["rungs"][name] = {"k": int(curve.axes.shape[1]), "smooth": curve.smooth, "mse": float(np.nanmean(sq))}
            record["val_unseen"][f"{name}_sq"] = ((u_cents - curve(u_values)) ** 2).sum(axis=1).tolist()
        pairs = [("line", "curve")]
        if kind == "open":
            pairs += [("line", "covariance_spacing"), ("covariance_spacing", "curve"),
                      ("line", "pc1_spacing"), ("pc1_spacing", "curve")]
            record["spacing_share"] = {
                "covariance_split": ladder_share(rungs["line"][0], rungs["covariance_spacing"][0], rungs["curve"][0]),
                "pc1_split": ladder_share(rungs["line"][0], rungs["pc1_spacing"][0], rungs["curve"][0]),
            }
        for first, second in pairs:
            record["gaps"][f"{first}_minus_{second}"] = paired_gap(rungs[first][0], rungs[second][0])
        ok["finite"].append("NaN" not in json.dumps(record) and "Infinity" not in json.dumps(record))
        result[variable] = record

    criteria = {name: all(values) for name, values in ok.items()}
    return {
        "site": STEERING_SITE, "variables": result,
        "share_definition": "(line - rung) / (line - curve), mean squared LOCO; curvature share = 1 - spacing share",
        "criteria": criteria, "passed": all(criteria.values()),
    }
    
def saved_curves() -> tuple[dict, dict]:
    """manifold_loco's saved arrays (read through their hash) and its result."""
    with np.load(verified_artifact(OUT, "manifold_loco")) as f:
        arrays = {key: f[key] for key in f.files}
    return arrays, json.loads(OUT.read_text())["manifold_loco"]["result"]["variables"]


def selected_curve(arrays: dict, variable: str, site: str):
    """values, centroids and the selected curve at one site, rebuilt from the saved grid indices."""
    prefix = f"{variable}_{site}"
    values, cents = arrays[f"{prefix}_values"], arrays[f"{prefix}_centroids"]
    a, b = (int(i) for i in arrays[f"{prefix}_selected"])
    kind = kind_of(variable)
    return values, cents, fit_curve(kind, values, cents, PCA_DIMS[a], smoothing_grid(kind, values)[b])


def dense_values(variable: str, values: np.ndarray) -> np.ndarray:
    if kind_of(variable) == "loop":
        return np.linspace(0.0, 360.0, DENSE_POINTS, endpoint=False)
    return np.linspace(values[0], values[-1], DENSE_POINTS)


def check_manifold_dimension() -> dict:
    """Observation (H-01): the manifold's dimension next to Phase 4's counts at indices 1 / 9 / 18.

    Per variable and site: m, K and K * m from the saved probe sequence; the selected PCA dimension k (rebuilt from
    manifold_loco's saved indices), the share of centroid variance in k axes, axes for 90% of it, the participation
    ratio of the raw centroids (sampling noise inflates it) and of the fitted curve sampled densely in the label.
    Passes if: the rebuilt selection equals manifold_loco's; the saved scaler equals the probe sequence's; finite.
    """
    arrays, saved = saved_curves()
    ok: dict[str, list[bool]] = {name: [] for name in ("selection_matches", "scaler_matches_nullspace", "finite")}
    result: dict = {}
    for variable in DATASETS:
        result[variable] = {}
        for site in MANIFOLD_SITES:
            values, cents, curve = selected_curve(arrays, variable, site)
            reference = saved[variable][site]["selected"]
            ok["selection_matches"].append(
                dim_name(PCA_DIMS[int(arrays[f"{variable}_{site}_selected"][0])]) == reference["k"]
                and curve.smooth == reference["smooth"])
            seq = load_probe_sequence(variable, site)
            ok["scaler_matches_nullspace"].append(
                bool(np.array_equal(arrays[f"{variable}_{site}_scaler_mean"], seq.mean)
                     and np.array_equal(arrays[f"{variable}_{site}_scaler_scale"], seq.scale)))
            lam = np.linalg.svd(cents - cents.mean(axis=0), compute_uv=False) ** 2
            share = np.cumsum(lam) / lam.sum()
            k = curve.axes.shape[1]
            record = {
                "index": plot_index(site), "m": seq.dims_per_round, "K": seq.k, "K_dims": seq.k * seq.dims_per_round,
                "selected_k": k, "centroid_variance_in_k": float(share[k - 1]),
                "axes_for_90_percent": int(np.searchsorted(share, 0.9) + 1),
                "centroid_participation_ratio": participation_ratio(cents),
                "curve_participation_ratio": participation_ratio(curve(dense_values(variable, values))),
            }
            ok["finite"].append("NaN" not in json.dumps(record) and "Infinity" not in json.dumps(record))
            result[variable][site] = record
    criteria = {name: all(v) for name, v in ok.items()}
    return {"observation": True, "variables": result, "criteria": criteria, "passed": all(criteria.values())}


def check_direction_harmonics() -> dict:
    """Observation (H-02): harmonic structure of the direction centroids and of Phase 4's nullspace rounds.

    Per site (indices 1 / 9 / 18), standardized train centroids: least-squares trig fit with 12 harmonics, power per
    harmonic as a share of centroid variance, and an angle-shuffle null (centroids permuted across angles, 95th
    percentile per harmonic). Per nullspace round 1...K (2-dim block of the saved basis, same standardized space):
    share of the block in each harmonic plane h = 1...6 (mean squared principal cosine), the closest harmonic, and
    the share inside the span of all harmonic coefficient vectors. Passes if: the saved scaler equals the probe
    sequence's; the 4-harmonic full-space fit equals the loop curve with all PCA axes; finite.
    """
    variable = "direction"
    arrays, _ = saved_curves()
    ok: dict[str, list[bool]] = {name: [] for name in ("scaler_matches_nullspace", "trig_fit_matches_curve", "finite")}
    result: dict = {}
    for site in MANIFOLD_SITES:
        prefix = f"{variable}_{site}"
        values, cents = arrays[f"{prefix}_values"], arrays[f"{prefix}_centroids"]
        seq = load_probe_sequence(variable, site)
        ok["scaler_matches_nullspace"].append(
            bool(np.array_equal(arrays[f"{prefix}_scaler_mean"], seq.mean)
                 and np.array_equal(arrays[f"{prefix}_scaler_scale"], seq.scale)))

        const, cos4, sin4 = harmonic_coefficients(values, cents, 4)
        h = np.arange(1, 5)
        t = np.deg2rad(values)[:, None] * h
        fit4 = const + np.cos(t) @ cos4 + np.sin(t) @ sin4
        ok["trig_fit_matches_curve"].append(
            relative(fit4, fit_curve("loop", values, cents, None, 4)(values)) <= HARMONIC_TOLERANCE)

        total = float(((cents - cents.mean(axis=0)) ** 2).sum(axis=1).mean())
        _, cos, sin = harmonic_coefficients(values, cents, MAX_HARMONIC)
        power = harmonic_power(cos, sin)
        rng = np.random.default_rng(np.random.SeedSequence([SEED, plot_index(site)]))
        null = np.stack([harmonic_power(*harmonic_coefficients(values, cents[rng.permutation(len(values))],
                                                               MAX_HARMONIC)[1:]) for _ in range(PERMUTATIONS)])
        null95 = np.percentile(null, 95, axis=0)

        planes = [np.stack([cos[i], sin[i]], axis=1) for i in range(ROUND_HARMONICS)]
        span = np.concatenate([cos, sin]).T  # (d, 2 * MAX_HARMONIC)
        m = seq.dims_per_round
        rounds = []
        for r in range(seq.k):
            block = seq.basis[:, r * m:(r + 1) * m]
            shares = [float((principal_cosines(block, p) ** 2).mean()) for p in planes]
            rounds.append({"round": r + 1, "harmonic_plane_share": shares,
                           "closest_harmonic": int(np.argmax(shares) + 1),
                           "share_in_harmonic_span": float((principal_cosines(block, span) ** 2).mean())})
        record = {
            "index": plot_index(site), "K": seq.k, "centroid_variance": total,
            "power_share": (power / total).tolist(), "null95_share": (null95 / total).tolist(),
            "above_null": [int(i + 1) for i in np.flatnonzero(power > null95)],
            "parseval_ratio": float(power.sum() / total),
            "rounds": rounds,
        }
        ok["finite"].append("NaN" not in json.dumps(record) and "Infinity" not in json.dumps(record))
        result[site] = record
    criteria = {name: all(v) for name, v in ok.items()}
    return {"observation": True, "sites": result, "permutations": PERMUTATIONS, "max_harmonic": MAX_HARMONIC,
            "criteria": criteria, "passed": all(criteria.values())}
    
def ladder_rungs(arrays: dict, ladder: dict, variable: str) -> dict[str, np.ndarray]:
    """Per-value squared LOCO errors of each curve family at index 9, from the saved grid (and a recomputed
    free-spacing B line and exact interpolation), in ladder order."""
    prefix = f"{variable}_{STEERING_SITE}"
    values, cents = arrays[f"{prefix}_values"], arrays[f"{prefix}_centroids"]
    grid = arrays[f"{prefix}_loco_sq"]
    a, b = (int(i) for i in arrays[f"{prefix}_selected"])
    line_a = int(arrays[f"{prefix}_line_k_index"])
    kind = kind_of(variable)
    smooths = smoothing_grid(kind, values)
    rec = ladder[variable]["rungs"]
    if kind == "open":
        rungs = {
            "straight\nline": grid[line_a, 0],
            "free\nspacing\n(PC 1)": grid[0, smooths.index(rec["pc1_spacing"]["smooth"])],
            "free\nspacing\n(B)": loco_spacing_grid(
                values, cents, [smooths[smooths.index(rec["covariance_spacing"]["smooth"])]])[0],
            "curve\n(selected)": grid[a, b],
        }
    else:
        rungs = {"ellipse\n(H = 1)": grid[line_a, 0], f"curve\n(H = {smooths[b]}, selected)": grid[a, b]}
    rungs["exact\ninterpolation"] = loco_errors(kind, values, cents, PCA_DIMS[a], EXACT) ** 2
    return rungs


def check_figure_manifolds() -> dict:
    """Figure (6.7): each variable's index-9 activation curve (2-D PCA view) and the Q1 ladder of held-out errors.

    Row 1: the selected curve, the straight reference (covariance line; direction: the H = 1 ellipse), train
    centroids and val-unseen centroids, projected on the first two PCs of the train centroids, with label values
    marked. Row 2: paired per-value squared LOCO errors across curve families with the mean. Reads only saved
    results (curves.npz through its hash, manifold_loco, manifold_ladder). Passes if: the recomputed rung means
    equal manifold_ladder's and the exact-interpolation mean equals manifold_loco's within REPRODUCE_TOLERANCE;
    the figure is written.
    """
    arrays, saved = saved_curves()
    all_results = json.loads(OUT.read_text())
    ladder = all_results["manifold_ladder"]["result"]["variables"]
    ok: dict[str, list[bool]] = {"ladder_reproduced": [], "exact_reproduced": []}

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 10), facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.42, "wspace": 0.28, "top": 0.92})
    for col, variable in enumerate(DATASETS):
        kind, colour = kind_of(variable), DATASET_COLOUR[variable]
        values, cents, curve = selected_curve(arrays, variable, STEERING_SITE)
        prefix = f"{variable}_{STEERING_SITE}"
        u_cents = arrays[f"{prefix}_val_unseen_centroids"]
        smooths = smoothing_grid(kind, values)
        straight = fit_curve(kind, values, cents, None, smooths[0])
        mean = cents.mean(axis=0)
        _, s, vt = np.linalg.svd(cents - mean, full_matrices=False)
        share = s**2 / (s**2).sum()

        def project(points: np.ndarray) -> np.ndarray:
            return (points - mean) @ vt[:2].T

        dense = dense_values(variable, values)
        if kind == "loop":
            dense = np.append(dense, 360.0)  # close the drawn loop
        ax = axes[0, col]
        ref, cur = project(straight(dense)), project(curve(dense))
        ax.plot(ref[:, 0], ref[:, 1], color=INK_SECONDARY, lw=1.2, ls=(0, (4, 3)), zorder=2)
        ax.plot(cur[:, 0], cur[:, 1], color=colour, lw=2.0, zorder=3)
        pc = project(cents)
        ax.scatter(pc[:, 0], pc[:, 1], s=14, color=INK_MUTED, linewidths=0, zorder=4)
        pu = project(u_cents)
        ax.scatter(pu[:, 0], pu[:, 1], s=64, facecolors=SURFACE, edgecolors=colour, linewidths=1.8, zorder=5)
        centre = cur.mean(axis=0)
        for v in LABEL_MARKS[variable]:
            p = project(curve(np.array([v])))[0]
            ax.scatter(p[0], p[1], s=22, color=INK, zorder=6, linewidths=0)
            out = (p - centre) / np.linalg.norm(p - centre)  # label outward, away from the curve's centre
            ax.annotate(f"{v:g}{UNITS[variable]}", p, xytext=tuple(14 * out), textcoords="offset points",
                        ha="left" if out[0] >= 0 else "right", va="center", color=INK_SECONDARY, fontsize=9,
                        zorder=7, bbox={"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": "none", "alpha": 0.85})
        a, b = (int(i) for i in arrays[f"{prefix}_selected"])
        setting = f"H {smooths[b]}" if kind == "loop" else f"λ {smooths[b]:.3g}"
        ax.set_title(f"{variable.capitalize()}: k {PCA_DIMS[a]}, {setting} (index 9)", color=INK, fontsize=11, loc="left")
        ax.set_xlabel(f"PC 1 ({share[0]:.0%} of centroid variance)", color=INK_SECONDARY)
        ax.set_ylabel(f"PC 2 ({share[1]:.0%})", color=INK_SECONDARY)
        ax.set_aspect("equal", adjustable="datalim")
        style_axes(ax, grid_axis=None)

        rungs = ladder_rungs(arrays, ladder, variable)
        rec = ladder[variable]["rungs"]
        names = {"open": ("line", "pc1_spacing", "covariance_spacing", "curve"), "loop": ("line", "curve")}[kind]
        for (label, errors), key in zip(rungs.items(), names):
            ok["ladder_reproduced"].append(
                abs(float(np.nanmean(errors)) - rec[key]["mse"]) <= REPRODUCE_TOLERANCE * rec[key]["mse"])
        exact_saved = saved[variable][STEERING_SITE]["exact_mse"]["selected_k"]
        ok["exact_reproduced"].append(
            abs(float(np.nanmean(rungs["exact\ninterpolation"])) - exact_saved) <= REPRODUCE_TOLERANCE * exact_saved)

        ax = axes[1, col]
        data = np.stack(list(rungs.values()))
        keep = np.isfinite(data).all(axis=0)
        xs = np.arange(len(rungs))
        for j in np.flatnonzero(keep):
            ax.plot(xs, data[:, j], color=INK_MUTED, lw=0.6, alpha=0.35, zorder=1)
        means = data[:, keep].mean(axis=1)
        ax.plot(xs, means, color=colour, lw=2.0, marker="o", ms=8, zorder=3)
        for x, m in zip(xs, means):
            ax.annotate(f"{m:.0f}", (x, m), xytext=(0, 9), textcoords="offset points", ha="center",
                        color=INK, fontsize=9, zorder=4)
        ax.set_xticks(xs, list(rungs), fontsize=8.5, color=INK_SECONDARY)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("squared LOCO error (standardized units²)", color=INK_SECONDARY)
        if kind == "open":
            shares = ladder[variable]["spacing_share"]["covariance_split"]
            note = f"uneven spacing {shares['share']:.0%} · curvature {1 - shares['share']:.0%} of the line → curve gain"
        else:
            gap = ladder[variable]["gaps"]["line_minus_curve"]
            note = f"ellipse − curve {gap['mean']:.0f} [{gap['ci'][0]:.0f}, {gap['ci'][1]:.0f}]"
        ax.set_title(note, color=INK_SECONDARY, fontsize=9.5, loc="left")
        style_axes(ax)

    handles = [
        Line2D([], [], color=INK, lw=2.0, label="selected curve (colour = variable)"),
        Line2D([], [], color=INK_SECONDARY, lw=1.2, ls=(0, (4, 3)), label="straight reference (covariance line; direction: H = 1 ellipse)"),
        Line2D([], [], color=INK_MUTED, marker="o", ms=4, lw=0, label="train centroid (seen value)"),
        Line2D([], [], color=INK, marker="o", ms=8, lw=0, markerfacecolor=SURFACE, label="held-out value (val-unseen centroid)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=9, labelcolor=INK,
               bbox_to_anchor=(0.5, 0.995))
    fig.text(0.5, 0.47, "Held-out centroid error by curve family (leave one centroid out; grey = one seen value)",
             ha="center", color=INK, fontsize=11)
    FIGURE_MANIFOLDS.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_MANIFOLDS, dpi=FIGURE_DPI, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    criteria = {name: all(v) for name, v in ok.items()} | {"figure_written": FIGURE_MANIFOLDS.exists()}
    return {
        "figure": {"path": str(FIGURE_MANIFOLDS.relative_to(REPO)), "sha256": file_sha256(FIGURE_MANIFOLDS)},
        "sources": {key: all_results[key]["provenance"]["git_commit"] for key in ("manifold_loco", "manifold_ladder")},
        "criteria": criteria, "passed": all(criteria.values()),
    }
    

CHECKS = {
    "manifold_loco": check_manifold_loco,
    "speed_acceleration_manifold": check_speed_acceleration_manifold,
    "manifold_ladder": check_manifold_ladder,
    "manifold_dimension": check_manifold_dimension,
    "direction_harmonics": check_direction_harmonics,
    "figure_manifolds": check_figure_manifolds,
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