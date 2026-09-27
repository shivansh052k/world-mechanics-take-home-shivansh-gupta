"""Behavior readouts (Part 2): 16-bin multinomial readouts of the label value fit on validation clips at indices
9-18, Goodfire-style behavior curves through train clips' mean square-root probabilities, natural clips' distances
to them, and the cross-fitted two-stage readout used for isometry. No model is run; stored pooled activations only.

Usage: python scripts/check_behavior.py <check>
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from vjepa_physics.baselines import squared_distances
from vjepa_physics.behavior import (
    N_BINS, READOUT_ROLES, behavior_curve, curve_grid, fit_bin_readout, map_probabilities, nearest_on_curve,
    readout_map, two_stage_features, unit, value_bins,
)
from vjepa_physics.data import DATASETS
from vjepa_physics.evidence import file_sha256, require_clean_code, save_result, verified_artifact
from vjepa_physics.extraction import plot_index
from vjepa_physics.joined import load_joined
from vjepa_physics.manifolds import (
    PCA_DIMS, PERIOD, fit_curve, geodesic_distances, path_positions, smoothing_grid, value_centroids,
)
from vjepa_physics.metrics import percentile_interval
from vjepa_physics.nullspace import train_scaler
from vjepa_physics.probes import probe_targets, site_features
from vjepa_physics.reproducibility import SEED
from vjepa_physics.steering import label_difference

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/behavior/checks.json"
ARTIFACTS = REPO / "artifacts/behavior"  # regenerable, git-ignored

PROFILE_SITES = tuple(f"block_{i}" for i in range(8, 18))  # indices 9-18
CURVE_SITES = ("block_8", "block_17")  # same-layer control (index 9) and readout layer (index 18)
TWO_STAGE_SITE = "block_17"  # the two-stage readout is used for isometry at the readout layer only
EXPECTED_FIT = {"direction": 297, "speed": 304, "acceleration": 304}
MIN_BIN_CLIPS = 10  # own criterion, kept as stated (planning chat)
BIN_FAILURE_CAUSE = ("bins holding test-unseen values have fewer validation clips by design (only their seen "
                     "values' clips; D-38)")
MAP_TOLERANCE = 1e-10  # probabilities: raw-space map vs sklearn, and row sums
CURVE_TOLERANCE = 1e-10  # the behavior curve passes through every normalized centroid
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)

MANIFOLD_CHECKS = REPO / "results/manifolds/checks.json"  # key "manifold_loco": curves.npz with Q1's selections
ACTIVATION_SITE = "block_8"  # index 9: the activation curve
ISOMETRY_CONDITIONS = {  # name: (probabilities key suffix in readouts.npz, label)
    "A_idx18": ("block_17_probabilities", "readout A, index 9 -> 18 (evidence)"),
    "A_idx9_same_layer": ("block_8_probabilities", "readout A, index 9 -> 9 (same-layer control, by construction)"),
    "B_idx18": ("block_17_two_stage_probabilities",
                "readout B, index 9 -> 18 (discriminability of the idx-18 ridge readout; close to label distance by construction)"),
}
ISOMETRY_RESAMPLES = 2_000


def kind_of(variable: str) -> str:
    return "loop" if variable == "direction" else "open"


def bin_quality(variable: str, p: np.ndarray, bins: np.ndarray) -> dict:
    """Top-1, within-one-bin (circular for direction), log-loss and per-bin top-1 of bin probabilities."""
    pred = p.argmax(axis=1)
    d = pred - bins
    near = np.isin(np.mod(d, N_BINS), (0, 1, N_BINS - 1)) if variable == "direction" else np.abs(d) <= 1
    return {
        "top1": float((pred == bins).mean()), "within_one_bin": float(near.mean()),
        "log_loss": float(-np.log(p[np.arange(len(p)), bins]).mean()),
        "top1_per_bin": [float((pred[bins == b] == b).mean()) for b in range(N_BINS)],
    }


def check_behavior_readouts() -> dict:
    """16-bin readouts at indices 9-18, behavior curves at 9 and 18, natural distances, two-stage readout at 18.

    Per variable: readout (A) = fit_bin_readout on the validation clips at every index 9-18, saved as a raw-space
    map; quality scored on the train clips (out of sample). Behavior curves at indices 9 and 18 through the train
    clips' mean √p per seen value; each train clip's Hellinger distance to its curve (natural reference) and the
    nearest point's value error. Readout (B) = cross-fitted two-stage readout at index 18 (isometry only). Writes
    artifacts/behavior/readouts.npz. Criteria: every readout fit on exactly the validation clips; C interior;
    converged; raw-space map = sklearn; rows sum to 1; curves pass through their centroids; >= MIN_BIN_CLIPS
    validation clips per bin (own criterion, expected to fail for direction, cause recorded); finite; saved =
    computed.
    """
    start = time.perf_counter()
    arrays: dict[str, np.ndarray] = {}
    result: dict = {}
    ok: dict[str, list[bool]] = {name: [] for name in (
        "fit_rows", "c_interior", "converged", "map_equals_sklearn", "sums_to_one", "curve_interpolates",
        "two_stage_c_interior", "two_stage_converged", "finite")}
    bins_ok: dict[str, bool] = {}

    for variable in DATASETS:
        table = load_joined(variable)
        roles, labels, value_index = table["role"], table["label"], table["value_index"]
        bins = value_bins(value_index)
        val, train = np.isin(roles, READOUT_ROLES), roles == "train"
        kind = kind_of(variable)
        val_counts = np.bincount(bins[val], minlength=N_BINS)
        bins_ok[variable] = int(val_counts.min()) >= MIN_BIN_CLIPS
        record: dict = {"validation_clips_per_bin": val_counts.tolist(), "readouts": {}, "curves": {}}

        probabilities = {}
        for site in PROFILE_SITES:
            x = site_features(table["activations"], site)
            readout = fit_bin_readout(x, bins, roles, SEED)
            weights, offset = readout_map(readout)
            p = map_probabilities(weights, offset, x)
            ok["fit_rows"].append(readout.n_fit == EXPECTED_FIT[variable] == int(val.sum()))
            ok["c_interior"].append(readout.c_edge is None)
            ok["converged"].append(readout.converged)
            ok["map_equals_sklearn"].append(
                float(np.abs(p[val] - readout.probabilities(x[val])).max()) <= MAP_TOLERANCE)
            ok["sums_to_one"].append(float(np.abs(p.sum(axis=1) - 1.0).max()) <= MAP_TOLERANCE)
            record["readouts"][str(plot_index(site))] = {
                "c": readout.c, "c_edge": readout.c_edge, "converged": readout.converged,
                "train": bin_quality(variable, p[train], bins[train]),
            }
            arrays |= {f"{variable}_{site}_weights": weights, f"{variable}_{site}_offset": offset}
            probabilities[site] = p

        for site in CURVE_SITES:
            values, sqrt_cents, counts = value_centroids(np.sqrt(probabilities[site]), value_index, labels, roles)
            curve = behavior_curve(kind, values, sqrt_cents)
            ok["curve_interpolates"].append(
                float(np.abs(curve.points(values) - unit(sqrt_cents)).max()) <= CURVE_TOLERANCE)
            distance, nearest = nearest_on_curve(np.sqrt(probabilities[site][train]), curve, curve_grid(kind, values))
            error = np.abs(label_difference(variable, nearest, labels[train]))
            record["curves"][str(plot_index(site))] = {
                "natural_train_distance_quantiles": dict(zip(map(str, QUANTILES), np.quantile(distance, QUANTILES).tolist())),
                "natural_train_distance_mean": float(distance.mean()),
                "nearest_value_abs_error_median": float(np.median(error)),
                "train_clips_per_value": [int(counts.min()), int(counts.max())],
            }
            arrays |= {
                f"{variable}_{site}_probabilities": probabilities[site],
                f"{variable}_{site}_curve_values": values, f"{variable}_{site}_curve_sqrt_centroids": sqrt_cents,
                f"{variable}_{site}_natural_train_distance": distance,
            }

        x = site_features(table["activations"], TWO_STAGE_SITE)
        features = two_stage_features(x, probe_targets(variable, labels), bins, roles, SEED)
        two_stage = fit_bin_readout(features, bins, roles, SEED)
        p2 = two_stage.probabilities(features)
        ok["two_stage_c_interior"].append(two_stage.c_edge is None)
        ok["two_stage_converged"].append(two_stage.converged)
        record["two_stage"] = {
            "site": TWO_STAGE_SITE, "use": "isometry only (naturalness degenerate by construction)",
            "n_features": int(features.shape[1]), "c": two_stage.c, "c_edge": two_stage.c_edge,
            "converged": two_stage.converged, "train": bin_quality(variable, p2[train], bins[train]),
        }
        arrays[f"{variable}_{TWO_STAGE_SITE}_two_stage_probabilities"] = p2
        ok["finite"].append("NaN" not in json.dumps(record) and "Infinity" not in json.dumps(record)
                            and all(np.isfinite(p).all() for p in probabilities.values()) and np.isfinite(p2).all())
        result[variable] = record

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / "readouts.npz"
    np.savez(path, **arrays)
    with np.load(path) as saved:
        saved_ok = set(saved.files) == set(arrays) and all(np.array_equal(saved[k], v) for k, v in arrays.items())

    criteria = {name: all(values) for name, values in ok.items()} | {
        "validation_clips_per_bin_at_least_10": all(bins_ok.values()),
        "saved_equals_computed": saved_ok,
    }
    return {
        "variables": result,
        "profile_sites": list(PROFILE_SITES), "curve_sites": list(CURVE_SITES), "fit_roles": list(READOUT_ROLES),
        "bin_criterion": {"per_variable": bins_ok, "minimum": MIN_BIN_CLIPS,
                          "failure_cause": None if all(bins_ok.values()) else BIN_FAILURE_CAUSE},
        "natural_reference_note": "train clips each contribute 1/n of their value's centroid (mildly in-sample)",
        "seconds": time.perf_counter() - start,
        "criteria": criteria,
        "artifact": {"path": str(path.relative_to(REPO)), "sha256": file_sha256(path)},
        "passed": all(criteria.values()),
    }


def split_halves(table: dict, variable: str) -> tuple[np.ndarray, np.ndarray]:
    """Two disjoint masks over the clips: per seen value, a seeded half of its train clips (floor) and the rest."""
    first = np.zeros(len(table["role"]), dtype=bool)
    second = np.zeros_like(first)
    train = table["role"] == "train"
    for v in np.unique(table["value_index"][train]):
        rows = np.flatnonzero(train & (table["value_index"] == v))
        rng = np.random.default_rng(np.random.SeedSequence([SEED, DATASETS.index(variable), int(v)]))
        perm = rng.permutation(rows)
        first[perm[: len(rows) // 2]] = True
        second[perm[len(rows) // 2:]] = True
    return first, second


def as_roles(mask: np.ndarray) -> np.ndarray:
    """A role array that value_centroids reads as train only where mask is True."""
    return np.where(mask, "train", "other")


def label_distances(variable: str, values: np.ndarray) -> np.ndarray:
    return np.abs(label_difference(variable, values[:, None], values[None, :]))


def correlation(a: np.ndarray, b: np.ndarray, rank: bool = False) -> float:
    if rank:
        a, b = rankdata(a), rankdata(b)
    return float(np.corrcoef(a, b)[0, 1])


def isometry_statistics(variable: str, kind: str, values, activation, behavior, code: int) -> dict:
    """Segment and all-pairs correlations between the activation curve and the behavior curve, with bootstraps."""
    n = len(values)
    act_pos, act_total = path_positions(activation, values, kind)
    beh_pos, beh_total = path_positions(lambda g: behavior.points(g) / np.sqrt(2.0), values, kind)
    g_act = geodesic_distances(act_pos, act_total, kind)
    g_beh = geodesic_distances(beh_pos, beh_total, kind)
    chord = np.sqrt(squared_distances(activation(values)))
    label = label_distances(variable, values)

    si = np.arange(n - 1) if kind == "open" else np.arange(n)
    sj = (si + 1) % n
    seg = {"activation": g_act[si, sj], "behavior": g_beh[si, sj], "label": label[si, sj]}
    i, j = np.triu_indices(n, 1)

    rng = np.random.default_rng(np.random.SeedSequence([SEED, DATASETS.index(variable), code]))
    boot = {name: [] for name in ("segment_r", "segment_label_r", "geodesic_r", "chord_r", "label_r",
                                  "geodesic_minus_label", "geodesic_minus_chord")}
    for _ in range(ISOMETRY_RESAMPLES):
        s = rng.integers(0, len(si), len(si))
        boot["segment_r"].append(correlation(seg["activation"][s], seg["behavior"][s]))
        boot["segment_label_r"].append(correlation(seg["label"][s], seg["behavior"][s]))
        idx = rng.integers(0, n, n)
        keep = idx[i] != idx[j]
        a, b = idx[i][keep], idx[j][keep]
        rg, rc, rl = (correlation(m[a, b], g_beh[a, b]) for m in (g_act, chord, label))
        boot["geodesic_r"].append(rg)
        boot["chord_r"].append(rc)
        boot["label_r"].append(rl)
        boot["geodesic_minus_label"].append(rg - rl)
        boot["geodesic_minus_chord"].append(rg - rc)
    ci = {name: list(percentile_interval(np.array(v))) for name, v in boot.items()}

    point = {
        "segment_r": correlation(seg["activation"], seg["behavior"]),
        "segment_r_spearman": correlation(seg["activation"], seg["behavior"], rank=True),
        "segment_label_r": correlation(seg["label"], seg["behavior"]),
        "geodesic_r": correlation(g_act[i, j], g_beh[i, j]),
        "geodesic_r_spearman": correlation(g_act[i, j], g_beh[i, j], rank=True),
        "chord_r": correlation(chord[i, j], g_beh[i, j]),
        "label_r": correlation(label[i, j], g_beh[i, j]),
    }
    point["geodesic_minus_label"] = point["geodesic_r"] - point["label_r"]
    point["geodesic_minus_chord"] = point["geodesic_r"] - point["chord_r"]
    return {"n_values": n, "n_segments": int(len(si)), "n_pairs": int(len(i)), "point": point, "ci": ci,
            "activation_length": act_total, "behavior_length_hellinger": beh_total,
            "behavior_max_pair_geodesic": float(g_beh.max())}


def check_isometry() -> dict:
    """Q2: activation-curve (index 9) vs behavior-curve geometry, on disjoint halves of the train clips.

    Per variable: split each seen value's train clips into two seeded halves; the first builds the activation curve
    at index 9 (Q1's selected PCA dimension and smoothing, read from manifold_loco's saved indices), the second the
    behavior curves (readout A at index 18 = evidence; A at index 9 = same-layer control; B at index 18 = two-stage
    readout). Statistics: correlation of adjacent-segment lengths (with a label-step null) and all-pairs correlations
    of the behavior geodesic with the activation geodesic, the activation chord and the label distance, with
    bootstrap intervals (segments / values). Passes if: the halves are disjoint and cover every train clip; both
    halves hold every seen value; the saved readouts and Q1's arrays are read through their hashes; the behavior
    curves pass through their centroids; everything finite.
    """
    with np.load(verified_artifact(OUT, "behavior_readouts")) as f:
        readouts = {key: f[key] for key in f.files}
    with np.load(verified_artifact(MANIFOLD_CHECKS, "manifold_loco")) as f:
        curves = {key: f[key] for key in f.files}

    ok: dict[str, list[bool]] = {name: [] for name in ("halves_disjoint_and_cover", "halves_hold_every_value",
                                                        "curve_interpolates", "finite")}
    result: dict = {}
    for code_v, variable in enumerate(DATASETS):
        table = load_joined(variable)
        roles, labels, value_index = table["role"], table["label"], table["value_index"]
        kind = kind_of(variable)
        first, second = split_halves(table, variable)
        train = roles == "train"
        ok["halves_disjoint_and_cover"].append(not bool((first & second).any()) and bool(((first | second) == train).all()))

        x = site_features(table["activations"], ACTIVATION_SITE)
        scaler = train_scaler(x, roles)
        z = (x - scaler.mean_) / scaler.scale_
        values, cents, counts_a = value_centroids(z, value_index, labels, as_roles(first))
        a, b = (int(k) for k in curves[f"{variable}_{ACTIVATION_SITE}_selected"])
        activation = fit_curve(kind, values, cents, PCA_DIMS[a], smoothing_grid(kind, values)[b])

        record = {"half_counts_min": {}, "conditions": {}}
        record["half_counts_min"]["activation"] = int(counts_a.min())
        for code_c, (name, (suffix, label)) in enumerate(ISOMETRY_CONDITIONS.items()):
            p = readouts[f"{variable}_{suffix}"]
            b_values, sqrt_cents, counts_b = value_centroids(np.sqrt(p), value_index, labels, as_roles(second))
            ok["halves_hold_every_value"].append(bool(np.array_equal(b_values, values)))
            record["half_counts_min"]["behavior"] = int(counts_b.min())
            behavior = behavior_curve(kind, b_values, sqrt_cents)
            ok["curve_interpolates"].append(
                float(np.abs(behavior.points(b_values) - unit(sqrt_cents)).max()) <= CURVE_TOLERANCE)
            stats = isometry_statistics(variable, kind, values, activation, behavior, 10 * code_v + code_c)
            record["conditions"][name] = {"label": label} | stats
        ok["finite"].append("NaN" not in json.dumps(record) and "Infinity" not in json.dumps(record))
        result[variable] = record

    criteria = {name: all(v) for name, v in ok.items()}
    return {
        "variables": result, "activation_site": ACTIVATION_SITE, "resamples": ISOMETRY_RESAMPLES,
        "notes": ["all-pairs geodesic correlations are inflated by additivity along a 1-D curve (d(a,c) = d(a,b) + "
                  "d(b,c)); segments are the informative test",
                  "Hellinger chords saturate for far pairs, so geodesic != chord there partly by construction",
                  "Goodfire's exclusion of pairs on a common geodesic leaves no pairs on a 1-D curve; split halves "
                  "are our guard against shared-noise inflation"],
        "criteria": criteria, "passed": all(criteria.values()),
    }
    

CHECKS = {
    "behavior_readouts": check_behavior_readouts,
    "isometry": check_isometry,
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