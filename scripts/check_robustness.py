"""Robustness checks on stored activations and saved outputs (no model is run): direction's transfer between motion
types, and probe errors broken down by clip flag and per-tubelet motion.

Usage: python scripts/check_robustness.py <check>
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from vjepa_physics.confounds import as_distance, clip_distance
from vjepa_physics.data import DATASETS
from vjepa_physics.evidence import file_sha256, require_clean_code, save_result, verified_artifact
from vjepa_physics.extraction import SITES, plot_index
from vjepa_physics.joined import load_joined
from vjepa_physics.metrics import (
    angles_from_sincos, bootstrap_indices, circular_errors, percentile_interval, r2, resampled_mean, resampled_r2,
)
from vjepa_physics.nullspace import covariance_basis, random_span_basis, train_scaler, train_span
from vjepa_physics.plotting import DATASET_COLOUR, INK, INK_MUTED, INK_SECONDARY, SURFACE, style_axes
from vjepa_physics.probes import alpha_verdict, fit_probe, probe_scores, probe_targets, site_features
from vjepa_physics.reproducibility import SEED
from vjepa_physics.robustness import (
    MOTION_TYPES, NULL_KEYS, clip_errors, distance_overlap, motion_masks, null_metrics, null_reading, re_express,
    rescale, stratified_difference, subspace_metrics, within_range_trend, within_tubelet_px,
)
from vjepa_physics.steering import STEERING_SITE, label_difference, load_probe_sequence, readout_values

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/robustness/checks.json"
ARTIFACTS = REPO / "artifacts/robustness"  # regenerable, git-ignored
PROBE_CHECKS = REPO / "results/probes/checks.json"  # key "layer_curves": all-train validation predictions

VALIDATION_ROLES = ("val_seen", "val_unseen")
TEST_ROLES = ("test_seen", "test_unseen")  # scored once, after the validation findings are fixed
HEADLINE_SITES = ("block_0", STEERING_SITE, "block_17")  # indices 1, 9, 18: intervals, readings, the one-time test
EXPECTED_FIT = {"velocity": 397, "acceleration": 416}  # train clips per motion type
BOOTSTRAP_RESAMPLES = 10_000
SHARED_R2 = 0.9  # reading bands on the across-type R² (planning chat)
TYPE_SPECIFIC_R2 = 0.5

FLAGS_TABLE = REPO / "results/tracking/clip_flags.csv"  # committed; within-tubelet px rounded to 4 decimals
CSV_ROUNDING_PX = 5.0001e-5  # half the table's last decimal, plus float slack
SCORE_TOLERANCE = 1e-12  # recomputed validation scores vs the ones layer_curves saved
FLAG_TOTALS = {  # all clips per dataset, from the data audit's flags check
    "direction": {"exit": 113, "clipped": 199, "sub_patch_motion": 150, "frozen_start": 92},
    "speed": {"exit": 0, "clipped": 0, "sub_patch_motion": 240, "frozen_start": 1},
    "acceleration": {"exit": 0, "clipped": 0, "sub_patch_motion": 360, "frozen_start": 267},
}
STRATIFIED_FLAGS = {  # (flag, stratum): flags that vary inside a stratum; speed's single frozen clip is not tested
    "direction": (("exit", "group"), ("clipped", "group"), ("frozen_start", "group")),
    "acceleration": (("frozen_start", "label"),),
}
TREND_VARIABLES = ("speed", "acceleration")  # sub_patch_motion is fixed by the label value in these sets

FIGURE_TUBELET = REPO / "results/robustness/tubelet_scatter.png"
FIGURE_DPI = 200
FIGURE_SITE = STEERING_SITE  # index 9
ERROR_LABEL = {"direction": "circular error (°)", "speed": "absolute error (m/s)",
               "acceleration": "absolute error (m/s²)"}

LAYER_CURVE_CHECKS = REPO / "results/layer_curves/checks.json"  # key "test_scores": one-time test predictions
STEERING_CHECKS = REPO / "results/steering/checks.json"  # Phase 5 steering runs (test clips by design)
STEERING_READOUT = "block_17"  # index 18, Phase 5's primary readout
MIN_STEERING_CLIPS = 5  # stratified steering difference only with at least this many flagged and unflagged clips

OVERLAP_PAIRS = (("speed", "acceleration"), ("speed", "direction"), ("acceleration", "direction"))
SUBSPACES = (("weights_1", "weights"), ("weights_k_minus_1", "weights"), ("patterns", "patterns"))  # (name, transform)
NULL_DRAWS = 1000
ALGEBRA_TOLERANCE = 1e-10  # orthonormality, re-expression invariance, direct vs transformed, span residual
ANGLE_TOLERANCE = 1e-7  # radians: scipy vs the SVD-cosine formula (arccos loses precision near 0)
VALIDITY_R2 = 0.5  # a readout counts as valid on the other set's clips at or above this (7.2's band edge)
MAX_VALIDITY_SPEED = 4.0  # m/s: direction velocity clips inside the speed set's range
VALIDITY_CHECKS = (("speed", "direction"), ("acceleration", "direction"), ("direction", "speed"),
                   ("direction", "acceleration"))  # (readout, clips it is applied to)
DIRECTION_SPECIFICITY = (("direction", "speed"), ("direction", "acceleration"), ("speed", "direction"),
                         ("acceleration", "direction"))  # (steered, read); speed <-> acceleration: steering_specificity


def site_index(site: str) -> int | None:
    """hidden_states index of a site; None for the final norm."""
    return None if site == "final_norm" else plot_index(site)


def all_finite(obj) -> bool:
    if isinstance(obj, dict):
        return all(all_finite(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(all_finite(v) for v in obj)
    if isinstance(obj, float):
        return bool(np.isfinite(obj))
    return True


def direction_scores(labels: np.ndarray, pred: np.ndarray) -> tuple[dict, np.ndarray]:
    """R² on (sin, cos) and circular MAE (degrees), plus the per-clip circular errors."""
    errors = circular_errors(labels, angles_from_sincos(pred))
    return {"r2": r2(probe_targets("direction", labels), pred), "circular_mae": float(errors.mean())}, errors


def transfer_reading(across_r2: float, gap_ci: list[float]) -> str:
    """Planning-chat bands on the across-type R², refined by whether the across - within error gap's CI includes 0."""
    if across_r2 < TYPE_SPECIFIC_R2:
        return "type-specific"
    if across_r2 < SHARED_R2:
        return "partly shared"
    return "shared" if gap_ci[0] <= 0.0 <= gap_ci[1] else "shared, partly type-specific"


def check_motion_type_transfer() -> dict:
    """Direction probes fit on the train clips of one motion type, scored on validation clips of the same type (within)
    and of the other type (across), at every site; all-train probes from the layer curves as reference.

    Subsets: all validation clips; clips in the distance range both types reach (distance_overlap); clips without exit.
    At indices 1, 9, 18: 95% clip-bootstrap intervals of the across R² and of the gap (across - within circular MAE;
    independent resamples of the two clip sets) and the reading; their validation predictions are saved for the one-time
    test. Passes if: each probe fit exactly its type's train clips; no alpha failure; refit identical (indices 1, 9, 18);
    all-train reference aligned; saved file = computed; everything finite.
    """
    table = load_joined("direction")
    labels, roles = table["label"], table["role"]
    y = probe_targets("direction", labels)
    masks, overlap = motion_masks(table), distance_overlap(table)
    validation = np.isin(roles, VALIDATION_ROLES)
    subsets = {"all": validation, "distance_overlap": validation & overlap, "without_exit": validation & ~table["exit"]}
    with np.load(verified_artifact(PROBE_CHECKS, "layer_curves")) as f:
        reference = f["direction_predictions"]
        ids_ok = bool(np.array_equal(f["direction_ids"], table["id"]))
    boot = {m: bootstrap_indices(int((validation & masks[m]).sum()), BOOTSTRAP_RESAMPLES, SEED + i)
            for i, m in enumerate(MOTION_TYPES)}

    ok: dict[str, list[bool]] = {name: [] for name in (
        "n_fit", "only_one_type_fitted", "no_alpha_failure", "refit_identical", "finite")}
    result, summary = {}, {}
    arrays: dict[str, np.ndarray] = {"ids": table["id"], "validation": validation}
    for k, site in enumerate(SITES):
        x = site_features(table["activations"], site)
        site_record: dict = {"plot_index": site_index(site)}
        for fit_type in MOTION_TYPES:
            other = MOTION_TYPES[1 - MOTION_TYPES.index(fit_type)]
            fit_roles = np.where(masks[fit_type], roles, "other_type")
            probe = fit_probe(x, y, fit_roles)
            ok["n_fit"].append(probe.n_fit == EXPECTED_FIT[fit_type])
            ok["only_one_type_fitted"].append(bool(masks[fit_type][fit_roles == "train"].all()))
            pred = np.full(y.shape, np.nan)
            pred[validation] = probe.predict(x[validation])
            within_seen = (roles == "val_seen") & masks[fit_type]
            verdict = alpha_verdict(probe.alpha_edge, r2(y[within_seen], pred[within_seen]))
            ok["no_alpha_failure"].append(verdict != "failure")

            record: dict = {"alpha": probe.alpha, "alpha_verdict": verdict}
            for subset, rows in subsets.items():
                record[subset] = {}
                for name, eval_type in (("within", fit_type), ("across", other)):
                    r = rows & masks[eval_type]
                    score, _ = direction_scores(labels[r], pred[r])
                    ref, _ = direction_scores(labels[r], reference[r, k])
                    record[subset][name] = score | {"n_clips": int(r.sum()), "all_train_reference": ref}

            if site in HEADLINE_SITES:
                again = fit_probe(x, y, fit_roles)
                ok["refit_identical"].append(again.alpha == probe.alpha
                                             and np.array_equal(again.ridge.coef_, probe.ridge.coef_)
                                             and np.array_equal(again.ridge.intercept_, probe.ridge.intercept_))
                rows = {"within": validation & masks[fit_type], "across": validation & masks[other]}
                errors = {name: direction_scores(labels[r], pred[r])[1] for name, r in rows.items()}
                gap = resampled_mean(errors["across"], boot[other]) - resampled_mean(errors["within"], boot[fit_type])
                gap_ci = [float(c) for c in percentile_interval(gap)]
                across_ci = [float(c) for c in percentile_interval(
                    resampled_r2(y[rows["across"]], pred[rows["across"]], boot[other]))]
                full = record["all"]
                record["headline"] = {
                    "across_r2_ci": across_ci,
                    "gap_circular_mae": full["across"]["circular_mae"] - full["within"]["circular_mae"],
                    "gap_ci": gap_ci, "reading": transfer_reading(full["across"]["r2"], gap_ci),
                }
                arrays[f"{site}_{fit_type}_predictions"] = pred
                summary.setdefault(str(site_index(site)), {})[f"{fit_type}_to_{other}"] = {
                    "within_r2": full["within"]["r2"], "across_r2": full["across"]["r2"], "across_r2_ci": across_ci,
                    "within_mae": full["within"]["circular_mae"], "across_mae": full["across"]["circular_mae"],
                    "gap_ci": gap_ci, "reading": record["headline"]["reading"],
                    "overlap_within_across_mae": [record["distance_overlap"][n]["circular_mae"]
                                                  for n in ("within", "across")],
                }
            ok["finite"].append(all_finite(record))
            site_record[fit_type] = record
        result[site] = site_record

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / "motion_type_predictions.npz"
    np.savez(path, **arrays)
    with np.load(path) as saved:
        saved_ok = set(saved.files) == set(arrays) and all(
            np.array_equal(saved[key], value, equal_nan=value.dtype.kind == "f") for key, value in arrays.items())

    criteria = {name: all(values) for name, values in ok.items()} | {
        "reference_aligned": ids_ok, "saved_equals_computed": saved_ok}
    return {
        "roles_scored": list(VALIDATION_ROLES), "subsets": list(subsets),
        "counts": {s: {m: int((rows & masks[m]).sum()) for m in MOTION_TYPES} for s, rows in subsets.items()},
        "reading_bands": {"shared": SHARED_R2, "type_specific_below": TYPE_SPECIFIC_R2},
        "summary": summary, "sites": result,
        "artifact": {"path": str(path.relative_to(REPO)), "sha256": file_sha256(path)},
        "criteria": criteria, "passed": all(criteria.values()),
    }


def check_motion_type_test() -> dict:
    """One-time test of the motion-type transfer at indices 1, 9, 18 (scope fixed before any test row was read).

    The six probes are refit as in motion_type_transfer and must reproduce its saved validation predictions bit for bit;
    then test-seen and test-unseen clips of each type are scored within / across (all, distance_overlap, without_exit),
    with the same metrics, bootstrap and reading bands on the combined test clips. Nothing is selected. Passes if:
    saved artifact hash-verified and aligned; validation reproduced; only test rows predicted; everything finite.
    """
    table = load_joined("direction")
    labels, roles = table["label"], table["role"]
    y = probe_targets("direction", labels)
    masks, overlap = motion_masks(table), distance_overlap(table)
    validation, test = np.isin(roles, VALIDATION_ROLES), np.isin(roles, TEST_ROLES)
    with np.load(verified_artifact(OUT, "motion_type_transfer")) as f:
        saved = {key: f[key] for key in f.files}
    aligned = bool(np.array_equal(saved["ids"], table["id"]) and np.array_equal(saved["validation"], validation))
    subsets = {"all": test, "distance_overlap": test & overlap, "without_exit": test & ~table["exit"]}
    groups = {"test": test, "test_seen": roles == "test_seen", "test_unseen": roles == "test_unseen"}
    boot = {m: bootstrap_indices(int((test & masks[m]).sum()), BOOTSTRAP_RESAMPLES, SEED + i)
            for i, m in enumerate(MOTION_TYPES)}

    ok: dict[str, list[bool]] = {name: [] for name in ("validation_reproduced", "only_test_rows_predicted", "finite")}
    result, summary = {}, {}
    for site in HEADLINE_SITES:
        x = site_features(table["activations"], site)
        site_record: dict = {"plot_index": site_index(site)}
        for fit_type in MOTION_TYPES:
            other = MOTION_TYPES[1 - MOTION_TYPES.index(fit_type)]
            probe = fit_probe(x, y, np.where(masks[fit_type], roles, "other_type"))
            val_pred = np.full(y.shape, np.nan)
            val_pred[validation] = probe.predict(x[validation])
            ok["validation_reproduced"].append(bool(np.array_equal(
                val_pred, saved[f"{site}_{fit_type}_predictions"], equal_nan=True)))
            pred = np.full(y.shape, np.nan)
            pred[test] = probe.predict(x[test])
            ok["only_test_rows_predicted"].append(bool(np.isnan(pred[~test]).all()))

            record: dict = {"alpha": probe.alpha}
            for group, g in groups.items():
                record[group] = {}
                for subset, rows in subsets.items():
                    record[group][subset] = {}
                    for name, eval_type in (("within", fit_type), ("across", other)):
                        r = rows & g & masks[eval_type]
                        score, _ = direction_scores(labels[r], pred[r])
                        record[group][subset][name] = score | {"n_clips": int(r.sum())}

            rows = {"within": test & masks[fit_type], "across": test & masks[other]}
            errors = {name: direction_scores(labels[r], pred[r])[1] for name, r in rows.items()}
            gap_ci = [float(c) for c in percentile_interval(
                resampled_mean(errors["across"], boot[other]) - resampled_mean(errors["within"], boot[fit_type]))]
            across_ci = [float(c) for c in percentile_interval(
                resampled_r2(y[rows["across"]], pred[rows["across"]], boot[other]))]
            full = record["test"]["all"]
            record["headline"] = {
                "across_r2_ci": across_ci,
                "gap_circular_mae": full["across"]["circular_mae"] - full["within"]["circular_mae"],
                "gap_ci": gap_ci, "reading": transfer_reading(full["across"]["r2"], gap_ci),
            }
            summary.setdefault(str(site_index(site)), {})[f"{fit_type}_to_{other}"] = {
                "within_r2": full["within"]["r2"], "across_r2": full["across"]["r2"], "across_r2_ci": across_ci,
                "within_mae": full["within"]["circular_mae"], "across_mae": full["across"]["circular_mae"],
                "gap_ci": gap_ci, "reading": record["headline"]["reading"],
                "across_mae_seen_unseen": [record[g]["all"]["across"]["circular_mae"] for g in TEST_ROLES],
                "overlap_within_across_mae": [record["test"]["distance_overlap"][n]["circular_mae"]
                                              for n in ("within", "across")],
            }
            ok["finite"].append(all_finite(record))
            site_record[fit_type] = record
        result[site] = site_record

    criteria = {name: all(values) for name, values in ok.items()} | {"saved_aligned": aligned}
    return {
        "roles_scored": list(TEST_ROLES), "subsets": list(subsets),
        "counts": {s: {m: int((rows & masks[m]).sum()) for m in MOTION_TYPES} for s, rows in subsets.items()},
        "reading_bands": {"shared": SHARED_R2, "type_specific_below": TYPE_SPECIFIC_R2},
        "summary": summary, "sites": result, "criteria": criteria, "passed": all(criteria.values()),
    }

def excess_reading(ci: list[float]) -> str:
    """Reading of a stratified flagged - unflagged error difference from its 95% interval."""
    if ci[0] > 0.0:
        return "excess error beyond stratum"
    if ci[1] < 0.0:
        return "lower error"
    return "no excess"


def trend_reading(trend: dict) -> str:
    """Continuous vs binary reading of a readout inside the sub-patch range (rules fixed before the run)."""
    if trend["slope_ci"][0] > 0.0 and trend["spearman_ci"][0] > 0.0:
        return "continuous inside the sub-patch range"
    if trend["slope_ci"][0] <= 0.0 <= trend["slope_ci"][1]:
        return "consistent with a binary detector"
    return "neither rule met"


def flags_table_tubelet_px(variable: str, ids: np.ndarray) -> np.ndarray:
    """(clips, 8) within-tubelet displacement from the committed flags table, in the order of `ids`."""
    with FLAGS_TABLE.open() as f:
        rows = {int(r["id"]): r for r in csv.DictReader(f) if r["dataset"] == variable}
    return np.array([[float(rows[int(i)][f"within_tubelet_{k}_px"]) for k in range(8)] for i in ids])


def check_flag_breakdown() -> dict:
    """Validation errors of the layer-curve probes at indices 1, 9, 18, broken down by clip flag (no fitting).

    (A) Stratified flagged - unflagged mean error (stratified_difference): direction exit / clipped / frozen_start
    within motion group, acceleration frozen_start within label value; raw difference and clips per stratum as
    observations. (B) Speed and acceleration clips with sub_patch_motion: slope and Spearman of prediction on label
    inside that range (within_range_trend), read continuous vs binary; full-range trend, mean error per value, relative
    error flagged vs unflagged, and acceleration's frozen_start trend as observations. Passes if: saved predictions
    hash-verified and aligned; flag totals = the data audit's; within-tubelet px = flags table; recomputed scores and
    per-clip error means = the saved layer-curve scores; everything finite.
    """
    with np.load(verified_artifact(PROBE_CHECKS, "layer_curves")) as f:
        saved = {key: f[key] for key in f.files}
    saved_scores = json.loads(PROBE_CHECKS.read_text())["layer_curves"]["result"]
    column = {site: SITES.index(site) for site in HEADLINE_SITES}

    ok: dict[str, list[bool]] = {name: [] for name in (
        "aligned", "flag_totals_match", "tubelet_px_match_flags_table", "scores_reproduced", "finite")}
    ok["aligned"].append([str(s) for s in saved["sites"]] == list(SITES))
    result, summary, counts = {}, {}, {}
    for variable in DATASETS:
        table = load_joined(variable)
        labels, roles = table["label"], table["role"]
        validation = np.isin(roles, VALIDATION_ROLES)
        ok["aligned"].append(bool(np.array_equal(saved[f"{variable}_ids"], table["id"])
                                  and np.array_equal(saved[f"{variable}_roles"], roles)))
        ok["flag_totals_match"].append({n: int(table[n].sum()) for n in FLAG_TOTALS[variable]} == FLAG_TOTALS[variable])
        px_gap = float(np.abs(within_tubelet_px(table) - flags_table_tubelet_px(variable, table["id"])).max())
        ok["tubelet_px_match_flags_table"].append(px_gap <= CSV_ROUNDING_PX)
        counts[variable] = {n: int((validation & table[n]).sum()) for n in FLAG_TOTALS[variable]}
        metric = "circular_mae" if variable == "direction" else "mae"

        variable_record: dict = {"tubelet_px_max_gap": px_gap}
        for site in HEADLINE_SITES:
            pred = saved[f"{variable}_predictions"][:, column[site]]
            errors = np.full(len(labels), np.nan)
            errors[validation] = clip_errors(variable, labels[validation], pred[validation])
            for role in VALIDATION_ROLES:
                r = roles == role
                ref = saved_scores[variable]["sites"][site][role]
                again = probe_scores(variable, labels[r], pred[r])
                ok["scores_reproduced"].append(
                    all(abs(again[key] - ref[key]) <= SCORE_TOLERANCE for key in ref)
                    and abs(float(errors[r].mean()) - ref[metric]) <= SCORE_TOLERANCE)

            record: dict = {"plot_index": site_index(site)}
            line: dict = {}
            for flag, stratum in STRATIFIED_FLAGS.get(variable, ()):
                flagged, strata = table[flag][validation], table[stratum][validation].astype(str)
                d = stratified_difference(errors[validation], flagged, strata, BOOTSTRAP_RESAMPLES, SEED)
                d["raw_difference"] = float(errors[validation & table[flag]].mean()
                                            - errors[validation & ~table[flag]].mean())
                d["clips_per_stratum"] = {s: [int(((strata == s) & flagged).sum()), int(((strata == s) & ~flagged).sum())]
                                          for s in d["strata"]}
                d["reading"] = excess_reading(d["ci"])
                record[flag] = d
                line[flag] = [round(d["difference"], 4), [round(c, 4) for c in d["ci"]], d["reading"]]

            if variable in TREND_VARIABLES:
                sub = validation & table["sub_patch_motion"]
                inside = within_range_trend(labels[sub], pred[sub],
                                            bootstrap_indices(int(sub.sum()), BOOTSTRAP_RESAMPLES, SEED))
                full = within_range_trend(labels[validation], pred[validation],
                                          bootstrap_indices(int(validation.sum()), BOOTSTRAP_RESAMPLES, SEED))
                relative = errors / labels
                record["sub_patch"] = {
                    "n_clips": int(sub.sum()), "values": [float(v) for v in np.unique(labels[sub])],
                    "inside": inside, "reading": trend_reading(inside), "full_range": full,
                    "mean_error_per_value": {f"{v:g}": float(errors[sub & (labels == v)].mean())
                                             for v in np.unique(labels[sub])},
                    "relative_error": {"flagged": float(relative[sub].mean()),
                                       "unflagged": float(relative[validation & ~table["sub_patch_motion"]].mean())},
                }
                if variable == "acceleration":
                    frozen = validation & table["frozen_start"]
                    record["frozen_start_trend"] = within_range_trend(
                        labels[frozen], pred[frozen], bootstrap_indices(int(frozen.sum()), BOOTSTRAP_RESAMPLES, SEED))
                line["sub_patch"] = [[round(c, 4) for c in inside["slope_ci"]],
                                     [round(c, 4) for c in inside["spearman_ci"]], record["sub_patch"]["reading"]]
            ok["finite"].append(all_finite(record))
            variable_record[site] = record
            summary.setdefault(variable, {})[str(site_index(site))] = line
        result[variable] = variable_record

    criteria = {name: all(values) for name, values in ok.items()}
    return {
        "roles_scored": list(VALIDATION_ROLES), "sites_scored": list(HEADLINE_SITES),
        "counts": counts, "stratified_flags": {v: [list(p) for p in pairs] for v, pairs in STRATIFIED_FLAGS.items()},
        "summary": summary, "variables": result, "criteria": criteria, "passed": all(criteria.values()),
    }

def check_figure_tubelet() -> dict:
    """Per-clip validation error of the index-9 layer-curve probe vs the clip's mean within-tubelet disk displacement
    (px), one panel per variable; direction exit clips hollow. Spearman of error on displacement as observations (all
    clips; direction also without exit; speed / acceleration also on relative error), confounded with label value.
    Passes if: saved predictions hash-verified and aligned; the PNG is written and non-empty; every plotted value finite.
    """
    with np.load(verified_artifact(PROBE_CHECKS, "layer_curves")) as f:
        saved = {key: f[key] for key in f.files}
    k = SITES.index(FIGURE_SITE)
    aligned, plotted, spearman, counts = [], [], {}, {}

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), facecolor=SURFACE)
    for ax, variable in zip(axes, DATASETS):
        table = load_joined(variable)
        labels = table["label"]
        validation = np.isin(table["role"], VALIDATION_ROLES)
        aligned.append(bool(np.array_equal(saved[f"{variable}_ids"], table["id"])))
        px = within_tubelet_px(table)[validation].mean(axis=1)
        errors = clip_errors(variable, labels[validation], saved[f"{variable}_predictions"][validation, k])
        exits = table["exit"][validation]
        colour = DATASET_COLOUR[variable]
        ax.scatter(px[~exits], errors[~exits], s=14, color=colour, alpha=0.55, linewidths=0, zorder=2)
        if exits.any():
            ax.scatter(px[exits], errors[exits], s=24, facecolors="none", edgecolors=colour, linewidths=1.2, zorder=3)
        plotted += px.tolist() + errors.tolist()
        top = float(errors.max())
        ax.set_ylim(-0.04 * top, 1.3 * top)  # headroom: the ρ note and legend sit above every point

        rho = {"all": float(spearmanr(px, errors).statistic)}
        if variable == "direction":
            rho["without_exit"] = float(spearmanr(px[~exits], errors[~exits]).statistic)
        else:
            rho["relative_error"] = float(spearmanr(px, errors / labels[validation]).statistic)
        spearman[variable] = rho
        counts[variable] = {"clips": int(validation.sum()), "exit": int(exits.sum())}

        style_axes(ax, grid_axis="both")
        ax.set_title(variable, fontsize=11, loc="left", color=INK)
        ax.set_xlabel("mean disk displacement within a tubelet (px)", fontsize=9)
        ax.set_ylabel(ERROR_LABEL[variable], fontsize=9)
        note = "   ".join(f"ρ {name.replace('_', ' ')} {value:+.2f}" for name, value in rho.items())
        ax.text(0.98, 0.97, note, transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK_SECONDARY)
        if variable == "direction":
            handles = [Line2D([], [], linestyle="none", marker="o", markersize=5, color=colour, alpha=0.55,
                              label="disk stays in frame"),
                       Line2D([], [], linestyle="none", marker="o", markersize=6, markerfacecolor="none",
                              markeredgecolor=colour, label="disk exits")]
            ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 0.9), fontsize=8.5, frameon=False,
                      labelcolor=INK_SECONDARY)

    fig.suptitle("Probe error vs per-tubelet motion (validation clips, index 9)", fontsize=12, color=INK, x=0.07,
                 ha="left")
    fig.text(0.07, 0.02, "Each dot = one clip. ρ = Spearman correlation of error with displacement; "
             "displacement grows with the label, so ρ is confounded with label value.",
             fontsize=8.5, color=INK_MUTED)
    fig.subplots_adjust(bottom=0.2, top=0.85, wspace=0.28)
    fig.savefig(FIGURE_TUBELET, dpi=FIGURE_DPI, facecolor=SURFACE)
    plt.close(fig)

    criteria = {
        "aligned": all(aligned),
        "written": FIGURE_TUBELET.exists() and FIGURE_TUBELET.stat().st_size > 0,
        "finite": bool(np.isfinite(np.asarray(plotted, dtype=float)).all()),
    }
    return {
        "site": FIGURE_SITE, "plot_index": site_index(FIGURE_SITE), "counts": counts, "summary": spearman,
        "figure": {"path": str(FIGURE_TUBELET.relative_to(REPO)), "sha256": file_sha256(FIGURE_TUBELET)},
        "values_plotted": len(plotted), "criteria": criteria, "passed": all(criteria.values()),
    }
    
def flag_records(variable: str, table: dict, errors: np.ndarray, predictions: np.ndarray, rows: np.ndarray) -> dict:
    """flag_breakdown's (A) and (B) for one site on the clips in `rows` (same rules, resamples and seed)."""
    labels = table["label"]
    record: dict = {}
    for flag, stratum in STRATIFIED_FLAGS.get(variable, ()):
        flagged, strata = table[flag][rows], table[stratum][rows].astype(str)
        d = stratified_difference(errors[rows], flagged, strata, BOOTSTRAP_RESAMPLES, SEED)
        d["raw_difference"] = float(errors[rows & table[flag]].mean() - errors[rows & ~table[flag]].mean())
        d["clips_per_stratum"] = {s: [int(((strata == s) & flagged).sum()), int(((strata == s) & ~flagged).sum())]
                                  for s in d["strata"]}
        d["reading"] = excess_reading(d["ci"])
        record[flag] = d
    if variable in TREND_VARIABLES:
        sub = rows & table["sub_patch_motion"]
        inside = within_range_trend(labels[sub], predictions[sub],
                                    bootstrap_indices(int(sub.sum()), BOOTSTRAP_RESAMPLES, SEED))
        full = within_range_trend(labels[rows], predictions[rows],
                                  bootstrap_indices(int(rows.sum()), BOOTSTRAP_RESAMPLES, SEED))
        relative = errors / labels
        record["sub_patch"] = {
            "n_clips": int(sub.sum()), "values": [float(v) for v in np.unique(labels[sub])],
            "inside": inside, "reading": trend_reading(inside), "full_range": full,
            "mean_error_per_value": {f"{v:g}": float(errors[sub & (labels == v)].mean())
                                     for v in np.unique(labels[sub])},
            "relative_error": {"flagged": float(relative[sub].mean()),
                               "unflagged": float(relative[rows & ~table["sub_patch_motion"]].mean())},
        }
        if variable == "acceleration":
            frozen = rows & table["frozen_start"]
            record["frozen_start_trend"] = within_range_trend(
                labels[frozen], predictions[frozen], bootstrap_indices(int(frozen.sum()), BOOTSTRAP_RESAMPLES, SEED))
    return record


def summary_line(record: dict) -> dict:
    """Printed digest of one site's flag_records: difference, CI and reading; trend CIs and reading."""
    line = {flag: [round(d["difference"], 4), [round(c, 4) for c in d["ci"]], d["reading"]]
            for flag, d in record.items() if isinstance(d, dict) and "reading" in d and "ci" in d}
    if "sub_patch" in record:
        inside = record["sub_patch"]["inside"]
        line["sub_patch"] = [[round(c, 4) for c in inside["slope_ci"]], [round(c, 4) for c in inside["spearman_ci"]],
                             record["sub_patch"]["reading"]]
    return line


def steering_by_flag(variable: str, headline: dict, k: int) -> tuple[dict, bool]:
    """Phase 5 index-18 steering (probes n = K - 1, both halves) broken down by flag; also whether the per-clip errors
    reproduce steering_scores' headline and unedited errors."""
    names = ("ids", "target_label", "arm_kind", "arm_n",
             f"readout_{variable}_{STEERING_READOUT}", f"readout_unedited_{variable}_{STEERING_READOUT}")
    runs = []
    for half in ("seen", "unseen"):  # test_seen, then test_unseen, as steering_scores joins them
        with np.load(verified_artifact(STEERING_CHECKS, f"steer_{variable}_{half}")) as f:
            runs.append({name: f[name] for name in names})
    ids = np.concatenate([r["ids"] for r in runs])
    targets = runs[0]["target_label"]
    arm = np.flatnonzero((runs[0]["arm_kind"] == "probes") & (runs[0]["arm_n"] == k - 1))
    value = readout_values(variable, np.concatenate([r[f"readout_{variable}_{STEERING_READOUT}"] for r in runs]))
    base = readout_values(variable, np.concatenate([r[f"readout_unedited_{variable}_{STEERING_READOUT}"] for r in runs]))
    steered = np.abs(label_difference(variable, value[:, :, arm], targets[None, :, None])).mean(axis=(1, 2))
    unedited = np.abs(label_difference(variable, base[:, None], targets[None, :])).mean(axis=1)
    reproduced = (len(arm) == 1
                  and bool(np.isclose(steered.mean(), headline[f"probes_{k - 1}"]["error_target"], rtol=1e-12, atol=0))
                  and bool(np.isclose(unedited.mean(), headline["unedited"]["error_target"], rtol=1e-12, atol=0)))

    table = load_joined(variable)
    row_of = {int(i): r for r, i in enumerate(table["id"].tolist())}
    rows = np.array([row_of[int(i)] for i in ids])
    drop = unedited - steered  # per clip, label units (degrees for direction)
    strata_of = dict(STRATIFIED_FLAGS.get(variable, ()))
    record: dict = {"k": k, "arm": f"probes_{k - 1}", "n_clips": int(len(ids)), "flags": {}}
    for flag in FLAG_TOTALS[variable]:
        f = table[flag][rows]
        entry: dict = {"n_flagged": int(f.sum()), "n_unflagged": int((~f).sum())}
        for name, m in (("flagged", f), ("unflagged", ~f)):
            if m.any():
                entry[name] = {"reduction": float(1 - steered[m].mean() / unedited[m].mean()),
                               "error_drop": float(drop[m].mean())}
        if flag in strata_of:
            strata = table[strata_of[flag]][rows].astype(str)
            eligible = [s for s in np.unique(strata) if f[strata == s].any() and (~f[strata == s]).any()]
            in_eligible = np.isin(strata, eligible)
            if (in_eligible & f).sum() >= MIN_STEERING_CLIPS and (in_eligible & ~f).sum() >= MIN_STEERING_CLIPS:
                entry["stratified_error_drop"] = stratified_difference(drop, f, strata, BOOTSTRAP_RESAMPLES, SEED)
            else:
                entry["stratified_error_drop"] = "too few to read"
        record["flags"][flag] = entry
    return record, reproduced


def check_flag_test() -> dict:
    """One-time test read for the flag breakdown (scope fixed before any test row was read for it).

    (E) flag_breakdown's (A) and (B) at indices 1, 9, 18 on the saved one-time test predictions (test_scores; test_seen
    + test_unseen), reported as a labelled confirmation, never selected on. (D) Phase 5 steering (index-18 readout,
    probes n = K - 1) broken down by flag: counts, reduction flagged / unflagged, stratified per-clip error drop only
    with >= MIN_STEERING_CLIPS flagged and unflagged clips in eligible strata; observation only. Passes if: artifacts
    hash-verified and aligned; flag_records on the validation rows reproduces the saved flag_breakdown exactly; only
    test rows predicted; recomputed test error means = test_scores' saved points; steering per-clip errors reproduce
    steering_scores; everything finite.
    """
    with np.load(verified_artifact(PROBE_CHECKS, "layer_curves")) as f:
        validation_saved = {key: f[key] for key in f.files}
    with np.load(verified_artifact(LAYER_CURVE_CHECKS, "test_scores")) as f:
        test_saved = {key: f[key] for key in f.files}
    test_scores = json.loads(LAYER_CURVE_CHECKS.read_text())["test_scores"]["result"]
    breakdown = json.loads(OUT.read_text())["flag_breakdown"]["result"]["variables"]
    steering = json.loads(STEERING_CHECKS.read_text())["steering_scores"]["result"]["variables"]
    column = {site: SITES.index(site) for site in HEADLINE_SITES}

    ok: dict[str, list[bool]] = {name: [] for name in (
        "aligned", "validation_reproduced", "only_test_rows_predicted", "test_scores_reproduced",
        "steering_reproduced", "finite")}
    ok["aligned"].append([str(s) for s in test_saved["sites"]] == list(SITES))
    result, summary, counts = {}, {}, {}
    for variable in DATASETS:
        table = load_joined(variable)
        labels, roles = table["label"], table["role"]
        validation, test = np.isin(roles, VALIDATION_ROLES), np.isin(roles, TEST_ROLES)
        for source in (validation_saved, test_saved):
            ok["aligned"].append(bool(np.array_equal(source[f"{variable}_ids"], table["id"])
                                      and np.array_equal(source[f"{variable}_roles"], roles)))
        counts[variable] = {n: int((test & table[n]).sum()) for n in FLAG_TOTALS[variable]}
        metric = "circular_mae" if variable == "direction" else "mae"

        variable_record: dict = {}
        for site in HEADLINE_SITES:
            val_pred = validation_saved[f"{variable}_predictions"][:, column[site]]
            val_errors = np.full(len(labels), np.nan)
            val_errors[validation] = clip_errors(variable, labels[validation], val_pred[validation])
            again = json.loads(json.dumps(flag_records(variable, table, val_errors, val_pred, validation)))
            ok["validation_reproduced"].append(
                again == {key: v for key, v in breakdown[variable][site].items() if key != "plot_index"})

            pred = test_saved[f"{variable}_probe_predictions"][:, column[site]]
            ok["only_test_rows_predicted"].append(bool(np.isnan(pred[~test]).all() and np.isfinite(pred[test]).all()))
            errors = np.full(len(labels), np.nan)
            errors[test] = clip_errors(variable, labels[test], pred[test])
            for role in TEST_ROLES:
                point = test_scores[variable][role]["all"]["methods"][f"probe {site}"][metric]["point"]
                ok["test_scores_reproduced"].append(abs(float(errors[roles == role].mean()) - point) <= SCORE_TOLERANCE)

            record = {"plot_index": site_index(site)} | flag_records(variable, table, errors, pred, test)
            ok["finite"].append(all_finite(record))
            variable_record[site] = record
            summary.setdefault(variable, {})[str(site_index(site))] = summary_line(record)

        steer, reproduced = steering_by_flag(variable, steering[variable]["headline"], steering[variable]["k"])
        ok["steering_reproduced"].append(reproduced)
        ok["finite"].append(all_finite(steer))
        variable_record["steering"] = steer
        summary[variable]["steering"] = {
            flag: [e["n_flagged"], e.get("flagged", {}).get("reduction"), e.get("unflagged", {}).get("reduction"),
                   e["stratified_error_drop"] if isinstance(e.get("stratified_error_drop"), str)
                   else [round(c, 4) for c in e["stratified_error_drop"]["ci"]] if "stratified_error_drop" in e
                   else "no stratum"]
            for flag, e in steer["flags"].items()}
        result[variable] = variable_record

    criteria = {name: all(values) for name, values in ok.items()}
    return {
        "roles_scored": list(TEST_ROLES), "sites_scored": list(HEADLINE_SITES), "counts": counts,
        "summary": summary, "variables": result, "criteria": criteria, "passed": all(criteria.values()),
    }

def space_transform(own_scale: np.ndarray, target_scale: np.ndarray, kind: str):
    """Re-expression from a basis's own standardized space into a target space; the same callable for real and null."""
    return lambda q: re_express(q, own_scale, target_scale, kind)


def null_summary(samples: np.ndarray) -> dict:
    return {"median": float(np.median(samples)), "interval": [float(c) for c in percentile_interval(samples)]}


def check_subspace_overlap() -> dict:
    """Principal-angle overlap of the variables' probe subspaces (physics paper App. C.4 metrics), reverse readout
    validity, and direction steering specificity. Subspaces are train-fit only.

    Per site (indices 1, 9, 18; 9 = headline), pair and subspace (weights: nullspace Q rounds 1 and 1...K-1; patterns:
    train covariance directions), in the common space (one train scaler over all three sets) and the raw space: mean
    principal angle, overlap ||Q_AᵀQ_B||²_F / dim(B) both ways, Grassmann distance; nulls k_A / d and NULL_DRAWS random
    subspaces in the randomized set's own standardized train span, re-expressed like the real one; reading on overlap
    (null_reading). Validity (validation clips): speed / acceleration readouts on direction velocity clips <= 4 m/s vs
    clip distance (m); direction readout on speed / acceleration clips vs theta; R² >= VALIDITY_R2 = valid. Specificity
    (Phase 5 runs, index 18): per-clip mean |cross-readout change|, probes K-1 minus random K-1, paired clip bootstrap;
    read only with a valid readout. Passes if: Q orthonormal; round-1 predictions invariant under re-expression;
    covariance directions direct = transformed; null draws inside the span; angles = the SVD-cosine formula; full-space
    draws = k_A / d within 3 SE; specificity = steering_scores; everything finite.
    """
    tables = {v: load_joined(v) for v in DATASETS}
    ok: dict[str, list[bool]] = {name: [] for name in (
        "q_orthonormal", "weights_invariant", "patterns_consistent", "null_in_span", "angles_agree", "analytic_null",
        "specificity_reproduced", "finite")}
    sites_record, summary = {}, {}

    for site in HEADLINE_SITES:
        x = {v: np.asarray(site_features(tables[v]["activations"], site), np.float64) for v in DATASETS}
        pooled = train_scaler(np.concatenate([x[v] for v in DATASETS]),
                              np.concatenate([tables[v]["role"] for v in DATASETS]))
        d = len(pooled.scale_)
        targets = {"common": pooled.scale_, "raw": np.ones(d)}
        own, spans, scales, ks = {}, {}, {}, {}
        for v in DATASETS:
            seq = load_probe_sequence(v, site)
            roles, m = tables[v]["role"], seq.dims_per_round
            z = (x[v] - seq.mean) / seq.scale
            ok["q_orthonormal"].append(
                float(np.abs(seq.basis.T @ seq.basis - np.eye(seq.basis.shape[1])).max()) <= ALGEBRA_TOLERANCE)
            raw_pred = x[v] @ seq.maps[0]
            diff = pooled.transform(x[v]) @ rescale(seq.scale[:, None] * seq.maps[0], seq.scale, pooled.scale_,
                                                    "weights") - raw_pred
            ok["weights_invariant"].append(float((diff - diff.mean(axis=0)).std() / raw_pred.std()) <= ALGEBRA_TOLERANCE)
            y = probe_targets(v, tables[v]["label"])
            patterns = covariance_basis(z, y, roles)
            direct = covariance_basis(pooled.transform(x[v]), y, roles)
            via = re_express(patterns, seq.scale, pooled.scale_, "patterns")
            ok["patterns_consistent"].append(float(np.abs(direct @ direct.T - via @ via.T).max()) <= ALGEBRA_TOLERANCE)
            spans[v] = train_span(z, roles)
            draw = random_span_basis(spans[v], m, SEED)
            ok["null_in_span"].append(
                float(np.abs(draw - spans[v] @ (spans[v].T @ draw)).max()) <= ALGEBRA_TOLERANCE)
            own[v] = {"weights_1": seq.basis[:, :m], "weights_k_minus_1": seq.basis[:, :(seq.k - 1) * m],
                      "patterns": patterns}
            scales[v], ks[v] = seq.scale, seq.k

        site_record: dict = {"plot_index": site_index(site), "k": ks,
                             "span_rank": {v: int(spans[v].shape[1]) for v in DATASETS}, "pairs": {}}
        for a, b in OVERLAP_PAIRS:
            pair_record: dict = {}
            for name, kind in SUBSPACES:
                pair_record[name] = {}
                for space, target in targets.items():
                    ta, tb = space_transform(scales[a], target, kind), space_transform(scales[b], target, kind)
                    qa, qb = ta(own[a][name]), tb(own[b][name])
                    metrics = subspace_metrics(qa, qb)
                    svd = np.sort(np.arccos(np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), -1.0, 1.0)))
                    ok["angles_agree"].append(
                        float(np.abs(np.radians(metrics["angles_degrees"]) - svd).max()) <= ANGLE_TOLERANCE)
                    null_b = null_metrics(qa, spans[b], qb.shape[1], tb, NULL_DRAWS, SEED)  # B randomized
                    null_a = null_metrics(qb, spans[a], qa.shape[1], ta, NULL_DRAWS, SEED)  # A randomized
                    if site == STEERING_SITE and (a, b) == OVERLAP_PAIRS[0] and name == "weights_k_minus_1" \
                            and space == "common":
                        full = null_metrics(qa, np.eye(d), qb.shape[1], lambda q: q, NULL_DRAWS, SEED)["overlap_a_on_b"]
                        ok["analytic_null"].append(
                            abs(full.mean() - qa.shape[1] / d) <= 3 * full.std() / np.sqrt(NULL_DRAWS))
                    pair_record[name][space] = metrics | {
                        "dims": [int(qa.shape[1]), int(qb.shape[1])],
                        "analytic_null": {"overlap_a_on_b": qa.shape[1] / d, "overlap_b_on_a": qb.shape[1] / d},
                        "null_b_randomized": {key: null_summary(null_b[key]) for key in NULL_KEYS},
                        "null_a_randomized": {key: null_summary(null_a[key]) for key in NULL_KEYS},
                        "reading_a_on_b": null_reading(metrics["overlap_a_on_b"], null_b["overlap_a_on_b"]),
                        "reading_b_on_a": null_reading(metrics["overlap_b_on_a"], null_a["overlap_a_on_b"]),
                    }
            site_record["pairs"][f"{a}-{b}"] = pair_record
            if site == STEERING_SITE:
                summary.setdefault("overlap_idx9", {})[f"{a}-{b}"] = {
                    name: {space: [round(r["overlap_a_on_b"], 4), r["reading_a_on_b"], round(r["overlap_b_on_a"], 4),
                                   r["reading_b_on_a"], round(r["mean_angle_degrees"], 1)]
                           for space, r in pair_record[name].items()}
                    for name, _ in SUBSPACES}
        sites_record[site] = site_record

    with np.load(verified_artifact(STEERING_CHECKS, "steering_setup")) as f:
        setup = {key: f[key] for key in f.files}
    validity: dict = {}
    for read, on in VALIDITY_CHECKS:
        t = tables[on]
        rows = np.isin(t["role"], VALIDATION_ROLES)
        if on == "direction":
            rows &= (t["motion"] == "velocity") & (t["speed_mps"] <= MAX_VALIDITY_SPEED)
        entry: dict = {"n_clips": int(rows.sum())}
        for site in (STEERING_SITE, STEERING_READOUT):
            feats = np.asarray(site_features(t["activations"], site), np.float64)[rows]
            out = feats @ setup[f"readout_{read}_{site}_weights"] + setup[f"readout_{read}_{site}_offset"]
            if read == "direction":
                theta = t["theta_degrees"][rows]
                score = r2(probe_targets("direction", theta), out)
                extra = {"circular_mae": float(circular_errors(theta, angles_from_sincos(out)).mean())}
            else:
                true = clip_distance(t)[rows]
                pred = as_distance(read, out[:, 0])
                score = r2(true, pred)
                extra = {"mae_m": float(np.abs(pred - true).mean())}
            entry[str(site_index(site))] = {"r2": score, "valid": bool(score >= VALIDITY_R2)} | extra
        validity[f"{read}_on_{on}"] = entry

    steering = json.loads(STEERING_CHECKS.read_text())["steering_scores"]["result"]["variables"]
    specificity: dict = {}
    for steered, read in DIRECTION_SPECIFICITY:
        names = ("arm_kind", "arm_n", f"readout_{read}_{STEERING_READOUT}", f"readout_unedited_{read}_{STEERING_READOUT}")
        runs = []
        for half in ("seen", "unseen"):
            with np.load(verified_artifact(STEERING_CHECKS, f"steer_{steered}_{half}")) as f:
                runs.append({name: f[name] for name in names})
        kinds, ns, k = runs[0]["arm_kind"], runs[0]["arm_n"], steering[steered]["k"]
        value = readout_values(read, np.concatenate([r[names[2]] for r in runs]))
        base = readout_values(read, np.concatenate([r[names[3]] for r in runs]))
        change = np.abs(label_difference(read, value, base[:, None, None]))  # (clips, targets, arms)
        per_clip = {arm: change[:, :, (kinds == arm) & (ns == k - 1)].mean(axis=(1, 2)) for arm in ("probes", "random")}
        saved = steering[steered]["specificity_idx18"][read]
        ok["specificity_reproduced"].append(all(
            bool(np.isclose(per_clip[arm].mean(), saved[f"{arm}_{k - 1}"], rtol=1e-12, atol=0)) for arm in per_clip))
        diff = per_clip["probes"] - per_clip["random"]
        ci = [float(c) for c in percentile_interval(
            resampled_mean(diff, bootstrap_indices(len(diff), BOOTSTRAP_RESAMPLES, SEED)))]
        valid = validity[f"{read}_on_{steered}"][str(site_index(STEERING_READOUT))]["valid"]
        if not valid:
            reading = "readout not valid on these clips (not read)"
        else:
            reading = "cross-talk beyond a random edit" if ci[0] > 0.0 else "no cross-talk beyond a random edit"
        specificity[f"{steered}_to_{read}"] = {
            "arm": f"probes_{k - 1}", "n_clips": int(len(diff)), "probes": float(per_clip["probes"].mean()),
            "random": float(per_clip["random"].mean()), "difference": float(diff.mean()), "ci": ci,
            "valid_readout": valid, "reading": reading,
        }

    summary["validity_r2_idx9_idx18"] = {key: [round(e["9"]["r2"], 3), round(e["18"]["r2"], 3)]
                                         for key, e in validity.items()}
    summary["specificity"] = {key: [round(e["difference"], 4), [round(c, 4) for c in e["ci"]], e["reading"]]
                              for key, e in specificity.items()}
    result_body = {"sites": sites_record, "validity": validity, "specificity": specificity}
    ok["finite"].append(all_finite(result_body))
    criteria = {name: bool(values) and all(values) for name, values in ok.items()}
    return {
        "null_draws": NULL_DRAWS, "validity_r2": VALIDITY_R2,
        "counts": {"validity_clips": {key: e["n_clips"] for key, e in validity.items()}},
        "summary": summary, **result_body, "criteria": criteria, "passed": all(criteria.values()),
    }

CHECKS = {
    "figure_tubelet": check_figure_tubelet,
    "flag_breakdown": check_flag_breakdown,
    "flag_test": check_flag_test,
    "motion_type_transfer": check_motion_type_transfer,
    "motion_type_test": check_motion_type_test,
    "subspace_overlap": check_subspace_overlap,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("check", choices=sorted(CHECKS))
    name = parser.parse_args().check
    require_clean_code()  # after parsing, so --help still works with uncommitted code
    result = CHECKS[name]()
    save_result(OUT, name, result)
    print(json.dumps({"counts": result["counts"], "summary": result["summary"], "criteria": result["criteria"]},
                     indent=2))
    print(f"saved '{name}' -> {OUT.relative_to(REPO)}")
    if result.get("passed") is False:
        raise SystemExit(f"check '{name}' FAILED")


if __name__ == "__main__":
    main()