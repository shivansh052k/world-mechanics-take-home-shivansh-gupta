"""Checks for the joined per-variable tables: labels, split role, flags, tracked disk, activation hash.

Run one check at a time from the repo root, with .venv active:
    python scripts/check_joined.py <check>

Each check prints its result and stores it under its own key in results/joined/checks.json.
"""
import argparse
import csv
import json
import os
import resource
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from vjepa_physics.data import DATASETS, load_dataset
from vjepa_physics.evidence import file_sha256, save_result, verified_artifact
from vjepa_physics.geometry import disk_centres
from vjepa_physics.extraction import SITES, TIME_STEPS
from vjepa_physics.joined import FLAG_NAMES, build_table, load_joined
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

HIDDEN = 1024
NPY_HEADER_MAX = 4096  # a .npy header is a few hundred bytes at most; the rest of the file is the raw array
MIN_FREE_BYTES = 10 * 10**9  # free disk that must remain: 10 GB
MEMORY_FRACTION = 0.5  # the largest array loaded fully into RAM must keep peak RSS within half of physical memory


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


def artifact_records() -> list[tuple[Path, str, str]]:
    """(checks file, key, field) of every recorded artifact of the splits, extraction and joined checks."""
    records = [(SPLITS_CHECKS, "build", "artifact")]
    for variable in DATASETS:
        key = f"extract_{variable}"
        records += [(EXTRACTION_CHECKS, key, "artifact"), (EXTRACTION_CHECKS, key, "ids"), (OUT, "build", variable)]
    return records


def peak_rss_bytes() -> int:
    """Peak resident memory of this process so far (macOS reports ru_maxrss in bytes)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def check_storage() -> dict:
    """Every artifact still matches its recorded hash; the loader works; disk and memory fit the budget.

    Passes if: all 10 recorded artifacts (split file; per variable the activation array, its ids and the joined
    table) match their recorded SHA-256; load_joined returns, for every variable, a memory-mapped (clips, 26, 8,
    1024) float32 array with one row per table entry; each activation file is exactly its array's bytes plus a
    .npy header; at least 10 GB of disk stay free; and loading the largest activation array fully into RAM keeps
    peak process memory within half of physical memory. Sizes and memory figures are recorded.
    """
    hashed, failed, total_bytes = [], [], 0
    for checks, key, field in artifact_records():
        name = f"{checks.parent.name}/{key}/{field}"
        try:
            path = verified_artifact(checks, key, field)
        except (RuntimeError, KeyError, FileNotFoundError) as e:
            failed.append(f"{name}: {type(e).__name__}: {e}")
            continue
        hashed.append(name)
        total_bytes += path.stat().st_size

    round_trip, size_exact = {}, {}
    for variable in DATASETS:
        table = load_joined(variable)
        activations = table["activations"]
        n = EXPECTED_CLIPS[variable]
        round_trip[variable] = (
            isinstance(activations, np.memmap)
            and activations.shape == (n, len(SITES), TIME_STEPS, HIDDEN)
            and activations.dtype == np.float32
            and len(table["id"]) == n
        )
        data_bytes = n * len(SITES) * TIME_STEPS * HIDDEN * 4
        file_bytes = Path(str(activations.filename)).stat().st_size
        size_exact[variable] = data_bytes < file_bytes <= data_bytes + NPY_HEADER_MAX
        del table, activations

    largest = max(DATASETS, key=lambda v: EXPECTED_CLIPS[v])
    table = load_joined(largest)
    rss_before = peak_rss_bytes()
    in_ram = np.array(table["activations"])  # a full copy in RAM: the worst case for probing
    rss_after = peak_rss_bytes()
    in_ram_bytes = in_ram.nbytes
    del in_ram, table
    physical = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    free = shutil.disk_usage(REPO / "artifacts").free

    criteria = {
        "every_artifact_matches_record": not failed and len(hashed) == len(artifact_records()),
        "load_joined_round_trip": all(round_trip.values()),
        "activation_file_sizes_exact": all(size_exact.values()),
        "free_disk_at_least_10_gb": free >= MIN_FREE_BYTES,
        "full_array_fits_in_memory": rss_after <= MEMORY_FRACTION * physical,
    }
    return {
        "criteria": criteria,
        "artifacts_hashed": hashed,
        "artifacts_failed": failed,
        "artifacts_total_bytes": total_bytes,
        "artifacts_total_gib": round(total_bytes / 1024**3, 3),
        "round_trip": round_trip,
        "free_disk_bytes": free,
        "memory": {
            "largest_variable": largest,
            "array_in_ram_bytes": in_ram_bytes,
            "peak_rss_before_bytes": rss_before,
            "peak_rss_after_bytes": rss_after,
            "physical_bytes": physical,
        },
        "passed": all(criteria.values()),
    }


CHECKS = {
    "build": check_build,
    "storage": check_storage,
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