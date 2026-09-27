"""Robustness checks on stored activations and saved outputs (no model is run): direction's transfer between motion
types, and probe errors broken down by clip flag and per-tubelet motion.

Usage: python scripts/check_robustness.py <check>
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from vjepa_physics.data import DATASETS
from vjepa_physics.evidence import file_sha256, require_clean_code, save_result, verified_artifact
from vjepa_physics.extraction import SITES, plot_index
from vjepa_physics.joined import load_joined
from vjepa_physics.metrics import (
    angles_from_sincos, bootstrap_indices, circular_errors, percentile_interval, r2, resampled_mean, resampled_r2,
)
from vjepa_physics.probes import alpha_verdict, fit_probe, probe_scores, probe_targets, site_features
from vjepa_physics.reproducibility import SEED
from vjepa_physics.robustness import (
    MOTION_TYPES, clip_errors, distance_overlap, motion_masks, stratified_difference, within_range_trend,
    within_tubelet_px,
)
from vjepa_physics.steering import STEERING_SITE

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

CHECKS = {
    "flag_breakdown": check_flag_breakdown,
    "motion_type_transfer": check_motion_type_transfer,
    "motion_type_test": check_motion_type_test,
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