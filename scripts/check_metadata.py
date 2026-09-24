"""Checks for clip metadata: fields, types and values, against the manifests and DATA.md.

Run one check at a time from the repo root, with .venv active:
    python scripts/check_metadata.py <check>

Each check prints its result and stores it under its own key in results/metadata/checks.json.
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path
from scipy import stats

from vjepa_physics.data import DATASETS, read_manifest, read_metadata, resolve
from vjepa_physics.evidence import save_result

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = REPO / "results/metadata/checks.json"

# DATA.md: all rows of the three manifests (1,500 + 1,536 + 1,536), and every clip is 16 frames at 24 fps.
EXPECTED_CLIPS = 4572
FPS, FRAMES = 24, 16

# DATA.md Labels table: each labelled dataset's magnitude field and value range.
MAGNITUDE_FIELD = {"speed": "speed_mps", "acceleration": "acceleration_mps2"}
MAGNITUDE_RANGE = {"speed": (0.25, 4.0), "acceleration": (0.25, 10.0)}
RANGE_TOLERANCE = 1e-9

# Motion types, and the one each labelled dataset must have (speed = constant speed).
MOTIONS = ("velocity", "acceleration")
DATASET_MOTION = {"speed": "velocity", "acceleration": "acceleration"}

NUMERIC = ("theta_degrees", "speed_mps", "acceleration_mps2")
CRITERIA = ("parses", "id_matches_manifest", "fps_and_frames", "fields_valid", "label_fields", "motion_consistent")
SAMPLE = 20  # problem lists are saved as a count plus the first few entries

# DATA.md: fields every metadata.json records, plus magnitude (the primary target) in the labelled datasets.
DOCUMENTED_FIELDS = (
    "primary_label", "theta_degrees", "motion", "speed_mps",
    "acceleration_mps2", "start_position_xy_m", "fps", "frames",
)
DOCUMENTED_LABEL_FIELDS = {"speed": ("magnitude",), "acceleration": ("magnitude",)}


def sample(items) -> dict:
    """Count and the first SAMPLE entries, so failures are diagnosable without huge JSON."""
    items = list(items)
    return {"count": len(items), "first": items[:SAMPLE]}


def is_real(value) -> bool:
    """A finite int or float; bool is excluded (it is a subclass of int)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_all() -> dict[str, list[tuple[int, dict | None, str | None]]]:
    """Per dataset, in manifest order: (row id, metadata or None, parse error or None)."""
    loaded = {}
    for dataset in DATASETS:
        root = DATA / dataset
        clips = []
        for row in read_manifest(root):
            try:
                clips.append((row["id"], read_metadata(resolve(root, row["metadata"])), None))
            except ValueError as e:
                clips.append((row["id"], None, str(e).replace(f"{REPO}/", "")))
        loaded[dataset] = clips
    return loaded


def clip_failures(dataset: str, row_id: int, meta: dict) -> dict[str, str]:
    """Criterion name -> reason, for every consistency criterion this clip fails (empty if none)."""
    failures = {}

    if type(meta.get("id")) is not int or meta["id"] != row_id:
        failures["id_matches_manifest"] = f"metadata id {meta.get('id')!r}"

    if meta.get("fps") != FPS or meta.get("frames") != FRAMES:
        failures["fps_and_frames"] = f"fps {meta.get('fps')!r}, frames {meta.get('frames')!r}"

    bad = [key for key in NUMERIC if not is_real(meta.get(key))]
    start = meta.get("start_position_xy_m")
    if not (isinstance(start, list) and len(start) == 2 and all(is_real(v) for v in start)):
        bad.append("start_position_xy_m")
    if not (isinstance(meta.get("motion"), str) and meta["motion"]):
        bad.append("motion")
    if not bad:
        if meta["speed_mps"] < 0 or meta["acceleration_mps2"] < 0:
            bad.append("negative speed or acceleration")
        if dataset == "direction" and not 0 <= meta["theta_degrees"] < 360:
            bad.append(f"theta {meta['theta_degrees']} outside [0, 360)")
    if bad:
        failures["fields_valid"] = ", ".join(bad)

    if dataset in MAGNITUDE_FIELD:
        magnitude = meta.get("magnitude")
        low, high = MAGNITUDE_RANGE[dataset]
        if meta.get("primary_label") != dataset:
            failures["label_fields"] = f"primary_label {meta.get('primary_label')!r}"
        elif not is_real(magnitude) or magnitude != meta.get(MAGNITUDE_FIELD[dataset]):
            failures["label_fields"] = f"magnitude {magnitude!r} != {MAGNITUDE_FIELD[dataset]} {meta.get(MAGNITUDE_FIELD[dataset])!r}"
        elif not low - RANGE_TOLERANCE <= magnitude <= high + RANGE_TOLERANCE:
            failures["label_fields"] = f"magnitude {magnitude} outside [{low}, {high}]"

    if "fields_valid" in failures:
        failures["motion_consistent"] = "not evaluated: fields invalid"
    else:
        motion, speed, accel = meta["motion"], meta["speed_mps"], meta["acceleration_mps2"]
        reasons = []
        if motion not in MOTIONS:
            reasons.append(f"unknown motion {motion!r}")
        elif motion == "velocity" and not (accel == 0 and speed > 0):
            reasons.append(f"velocity with speed {speed}, acceleration {accel}")
        elif motion == "acceleration" and not accel > 0:
            reasons.append(f"acceleration with acceleration {accel}")
        if dataset in DATASET_MOTION and motion != DATASET_MOTION[dataset]:
            reasons.append(f"{dataset} set clip with motion {motion!r}")
        if reasons:
            failures["motion_consistent"] = "; ".join(reasons)

    return failures


def span(values: list[float]) -> list[float] | None:
    return [min(values), max(values)] if values else None


def check_consistency() -> dict:
    """Every clip's metadata parses, matches its manifest row and DATA.md, and its values are coherent.

    Passes if, for every clip: the file is a strict JSON object (no duplicate keys, no NaN or
    Infinity); id equals the manifest id; fps 24 and frames 16; theta, speed and acceleration are
    finite numbers, speed and acceleration >= 0, start position is 2 finite numbers, motion is a
    non-empty string, and direction-set theta is in [0, 360); in the speed and acceleration sets,
    primary_label is the dataset name and magnitude equals speed_mps / acceleration_mps2 exactly and
    lies in DATA.md's range; motion is "velocity" (acceleration 0, speed > 0) or "acceleration"
    (acceleration > 0), and the speed / acceleration sets are all velocity / all acceleration. And
    all 4,572 clips are read. Ranges, motion counts and key sets are diagnostics.
    """
    per_dataset, total = {}, 0
    for dataset, clips in load_all().items():
        names = CRITERIA if dataset in MAGNITUDE_FIELD else tuple(n for n in CRITERIA if n != "label_fields")
        failed: dict[str, list[str]] = {name: [] for name in names}
        parsed, valid = [], []
        for row_id, meta, error in clips:
            if meta is None:
                failed["parses"].append(f"id {row_id}: {error}")
                continue
            parsed.append(meta)
            failures = clip_failures(dataset, row_id, meta)
            for name, reason in failures.items():
                failed[name].append(f"id {row_id}: {reason}")
            if "fields_valid" not in failures:
                valid.append(meta)
        total += len(clips)

        per_dataset[dataset] = {
            "criteria": {name: not failed[name] for name in names},
            "clips": len(clips),
            "problems": {name: sample(reasons) for name, reasons in failed.items()},
            "diagnostics": {
                "clips_with_valid_fields": len(valid),
                "ranges": {key: span([m[key] for m in valid]) for key in NUMERIC}
                | {
                    "start_x_m": span([m["start_position_xy_m"][0] for m in valid]),
                    "start_y_m": span([m["start_position_xy_m"][1] for m in valid]),
                },
                "motion_counts": dict(Counter(m["motion"] for m in valid)),
                "motion_by_nonzero_speed_and_acceleration": dict(Counter(
                    f"{m['motion']}: speed>0={m['speed_mps'] > 0}, acceleration>0={m['acceleration_mps2'] > 0}"
                    for m in valid
                )),
                "key_sets": dict(Counter(",".join(sorted(m)) for m in parsed)),
            },
        }

    criteria = {"all_clips_read": total == EXPECTED_CLIPS}
    return {
        "criteria": criteria,
        "clips_read": total,
        "expected_clips": EXPECTED_CLIPS,
        "per_dataset": per_dataset,
        "passed": all(criteria.values())
        and all(all(r["criteria"].values()) for r in per_dataset.values()),
    }

def check_documented_fields() -> dict:
    """Every metadata.json has the fields DATA.md lists for its dataset.

    DATA.md: every file records primary_label, theta_degrees, motion, speed_mps,
    acceleration_mps2, start_position_xy_m, fps and frames; the speed and acceleration sets also
    have magnitude, their primary target. Passes if every clip has all of its dataset's documented
    fields and all 4,572 clips are read. Fields present but not documented are recorded.
    """
    per_dataset, total = {}, 0
    for dataset, clips in load_all().items():
        required = DOCUMENTED_FIELDS + DOCUMENTED_LABEL_FIELDS.get(dataset, ())
        lacking, missing, undocumented = [], Counter(), Counter()
        for row_id, meta, _ in clips:
            absent = [field for field in required if meta is None or field not in meta]
            missing.update(absent)
            if absent:
                lacking.append(f"id {row_id}: {', '.join(absent)}")
            if meta is not None:
                undocumented.update(key for key in meta if key not in required)
        total += len(clips)
        per_dataset[dataset] = {
            "criteria": {"all_documented_fields_present": not lacking},
            "required_fields": list(required),
            "clips": len(clips),
            "clips_missing_fields": sample(lacking),
            "missing_count_per_field": dict(missing),
            "undocumented_fields_count": dict(undocumented),
        }

    criteria = {"all_clips_read": total == EXPECTED_CLIPS}
    return {
        "criteria": criteria,
        "clips_read": total,
        "per_dataset": per_dataset,
        "passed": all(criteria.values())
        and all(all(r["criteria"].values()) for r in per_dataset.values()),
    }
    
# The label each manifest could be sorted by: DATA.md's primary target per dataset.
LABEL_FIELD = {"direction": "theta_degrees", "speed": "magnitude", "acceleration": "magnitude"}


def runs(values: list[float]) -> list[int]:
    """Lengths of the runs of consecutive equal values, in order."""
    lengths: list[int] = []
    for i, value in enumerate(values):
        if i and value == values[i - 1]:
            lengths[-1] += 1
        else:
            lengths.append(1)
    return lengths


def check_sorted_by_label() -> dict:
    """Diagnostic: how each manifest's order relates to its label (DATA.md's primary target).

    Label: magnitude (speed, acceleration) or theta_degrees (direction), in manifest order.
    Records whether the label never decreases, the ids where it drops, the runs of equal labels,
    and Spearman / Pearson correlations of id vs label. No pass/fail: it describes the data, and
    splits never use head/tail whatever it shows.
    """
    per_dataset = {}
    for dataset, clips in load_all().items():
        if any(meta is None for _, meta, _ in clips):
            raise RuntimeError(f"{dataset}: unparsed metadata; run the consistency check first")
        ids = [row_id for row_id, _, _ in clips]
        labels = [meta[LABEL_FIELD[dataset]] for _, meta, _ in clips]
        drops = [ids[i] for i in range(1, len(labels)) if labels[i] < labels[i - 1]]
        lengths = runs(labels)
        per_dataset[dataset] = {
            "label": LABEL_FIELD[dataset],
            "clips": len(labels),
            "manifest_order_is_id_order": ids == sorted(ids),
            "non_decreasing": not drops,
            "ids_where_label_drops": sample(drops),
            "distinct_labels": len(set(labels)),
            "runs_of_equal_label": len(lengths),
            "run_length_counts": dict(sorted(Counter(lengths).items())),
            "spearman_id_vs_label": round(float(stats.spearmanr(ids, labels).statistic), 6),
            "pearson_id_vs_label": round(float(stats.pearsonr(ids, labels).statistic), 6),
        }
    return {"per_dataset": per_dataset}


CHECKS = {
    "consistency": check_consistency,
    "documented_fields": check_documented_fields,
    "sorted_by_label": check_sorted_by_label,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("check", choices=sorted(CHECKS))
    name = parser.parse_args().check
    result = CHECKS[name]()
    print(json.dumps(result, indent=2))
    save_result(OUT, name, result)
    print(f"saved '{name}' -> {OUT.relative_to(REPO)}")
    if result.get("passed") is False:
        raise SystemExit(f"check '{name}' FAILED")


if __name__ == "__main__":
    main()