"""Robustness checks on stored activations and saved outputs (no model is run): direction's transfer between motion
types.

Usage: python scripts/check_robustness.py <check>
"""
import argparse
import json
from pathlib import Path

import numpy as np

from vjepa_physics.evidence import file_sha256, require_clean_code, save_result, verified_artifact
from vjepa_physics.extraction import SITES, plot_index
from vjepa_physics.joined import load_joined
from vjepa_physics.metrics import (
    angles_from_sincos, bootstrap_indices, circular_errors, percentile_interval, r2, resampled_mean, resampled_r2,
)
from vjepa_physics.probes import alpha_verdict, fit_probe, probe_targets, site_features
from vjepa_physics.reproducibility import SEED
from vjepa_physics.robustness import MOTION_TYPES, distance_overlap, motion_masks
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


CHECKS = {
    "motion_type_transfer": check_motion_type_transfer, 
    "motion_type_test": check_motion_type_test
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