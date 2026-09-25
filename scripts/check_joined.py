"""Checks for the joined per-variable tables: labels, split role, flags, tracked disk, activation hash.

Run one check at a time from the repo root, with .venv active:
    python scripts/check_joined.py <check>

Each check prints its result and stores it under its own key in results/joined/checks.json.
"""
import argparse
import csv
import json
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from vjepa_physics.data import DATASETS, load_dataset
from vjepa_physics.evidence import file_sha256, save_result, verified_artifact
from vjepa_physics.geometry import disk_centres
from vjepa_physics.joined import FLAG_NAMES, build_table
from vjepa_physics.splits import ROLES, read_splits

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = REPO / "results/joined/checks.json"
JOINED = REPO / "artifacts/joined"  # regenerable, git-ignored

EXTRACTION_CHECKS = REPO / "results/extraction/checks.json"
SPLITS_CHECKS = REPO / "results/splits/checks.json"
TRACKING_CHECKS = REPO / "results/tracking/checks.json"

EXPECTED_CLIPS = {"direction": 1500, "speed": 1536, "acceleration": 1536}  # DATA.md
# Split counts per role, in ROLES order (train, val_seen, val_unseen, test_seen, test_unseen): the split decision.
EXPECTED_ROLE_COUNTS = {
    "direction": (813, 203, 94, 203, 187),
    "speed": (832, 208, 96, 208, 192),
    "acceleration": (832, 208, 96, 208, 192),
}
# Flag counts in FLAG_NAMES order (exit, clipped, sub_patch_motion, frozen_start), from the flags check.
EXPECTED_FLAG_COUNTS = {"direction": (113, 199, 150, 92), "speed": (0, 0, 240, 1), "acceleration": (0, 0, 360, 267)}
MAPPING_TOLERANCE_PX = 1.0  # tracked vs predicted centre on fully visible frames (tracking's mapping criterion)


def build_all(out_dir: Path) -> dict[str, Path]:
    """Build every variable's table from hash-guarded sources; save each to out_dir/<variable>.npz.

    Sources: split file, flags table, tracking npz and activation id files, each read through its recorded
    hash; the activation file's SHA-256 is copied from its extraction record (the file itself is hashed
    when the table is loaded).
    """
    splits = read_splits(verified_artifact(SPLITS_CHECKS, "build"))
    with verified_artifact(TRACKING_CHECKS, "flags", "flags_table").open(newline="") as f:
        flags = list(csv.DictReader(f))
    with np.load(verified_artifact(TRACKING_CHECKS, "track")) as f:
        tracked = {key: f[key] for key in f.files}
    records = json.loads(EXTRACTION_CHECKS.read_text())

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for variable in DATASETS:
        key = f"extract_{variable}"
        ids = np.load(verified_artifact(EXTRACTION_CHECKS, key, "ids"))
        table = build_table(
            variable, ids, records[key]["result"]["artifact"]["sha256"],
            load_dataset(DATA, variable), splits, flags, tracked,
        )
        paths[variable] = out_dir / f"{variable}.npz"
        np.savez_compressed(paths[variable], **table)
    return paths


def check_build() -> dict:
    """Build and save the joined table of every variable, and check the saved files.

    Writes artifacts/joined/<variable>.npz. Every criterion is evaluated on the file as read back. Passes if,
    per variable: all clips present, ids distinct; every per-clip array has one entry per clip; role counts
    equal the split decision; flag counts equal the flags check; the tracked centre is within 1 px of the
    metadata prediction on every fully visible frame (matched by id, so a misaligned join fails); and a
    rebuild into a temporary folder is byte-identical. Records each file's path and SHA-256 under its variable.
    """
    built = build_all(JOINED)
    with tempfile.TemporaryDirectory() as tmp:
        again = build_all(Path(tmp))
        identical = {v: again[v].read_bytes() == built[v].read_bytes() for v in DATASETS}

    result: dict = {}
    passed = True
    for variable in DATASETS:
        path = built[variable]
        with np.load(path) as f:
            saved = {key: f[key] for key in f.files}
        n = len(saved["id"])
        by_id = {clip["id"]: clip for clip in load_dataset(DATA, variable)}
        predicted = np.stack([disk_centres(by_id[int(i)]) for i in saved["id"]])
        visible = (saved["area"] > 0) & ~saved["touches_border"]
        error = np.linalg.norm(saved["centre"] - predicted, axis=-1)[visible]
        roles = Counter(saved["role"].tolist())

        criteria = {
            "all_clips": n == EXPECTED_CLIPS[variable] and len(set(saved["id"].tolist())) == n,
            "one_entry_per_clip": all(a.shape[:1] == (n,) for k, a in saved.items() if k != "activations_sha256"),
            "role_counts_match": tuple(roles[r] for r in ROLES) == EXPECTED_ROLE_COUNTS[variable],
            "flag_counts_match": tuple(int(saved[name].sum()) for name in FLAG_NAMES) == EXPECTED_FLAG_COUNTS[variable],
            "tracking_within_1px": bool(error.size) and float(error.max()) <= MAPPING_TOLERANCE_PX,
            "rebuild_byte_identical": identical[variable],
        }
        passed = passed and all(criteria.values())
        result[variable] = {
            "criteria": criteria,
            "path": str(path.relative_to(REPO)),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
            "rows": n,
            "keys": sorted(saved),
            "activations_sha256": str(saved["activations_sha256"]),
            "tracking_max_error_px": round(float(error.max()), 3) if error.size else None,
        }
    result["passed"] = passed
    return result


CHECKS = {
    "build": check_build,
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