"""Checks for the supplied data files: manifests and folder structure.

Run one check at a time from the repo root, with .venv active:
    python scripts/check_data_files.py <check>

Each check prints its result and stores it under its own key in results/data_files/checks.json.
"""
import argparse
import json
from pathlib import Path

from vjepa_physics.data import DATASETS, MANIFEST, ROW_KEYS, read_manifest, resolve
from vjepa_physics.evidence import save_result

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = REPO / "results/data_files/checks.json"

# Row counts stated in DATA.md (Labels table).
EXPECTED_ROWS = {"direction": 1500, "speed": 1536, "acceleration": 1536}
# Files under data/ excluding .DS_Store, counted when the data fingerprint was taken
# (artifacts/manifests/data_fingerprint.sha256): 3 manifests + 2 files per clip.
EXPECTED_TOTAL_FILES = 9147
IGNORED = {".DS_Store"}  # written by Finder, not part of the supplied data
SAMPLE = 20  # problem lists are saved as a count plus the first few entries


def rel(path: Path) -> str:
    return path.relative_to(DATA).as_posix()


def sample(items) -> dict:
    """Count and the first SAMPLE entries (sorted), so failures are diagnosable without huge JSON."""
    items = sorted(items)
    return {"count": len(items), "first": items[:SAMPLE]}


def listing(root: Path) -> tuple[set[Path], set[Path]]:
    """All files and all folders under `root`, recursively, skipping IGNORED names."""
    files, folders = set(), set()
    for path in root.rglob("*"):
        if path.name not in IGNORED:
            (folders if path.is_dir() else files).add(path)
    return files, folders


def audit_dataset(dataset: str) -> tuple[dict, set[Path]]:
    """Criteria and diagnostics for one dataset, plus the set of files its manifest accounts for."""
    root = DATA / dataset
    try:
        rows = read_manifest(root)
    except ValueError as e:
        return {"criteria": {"rows_parse": False}, "parse_error": str(e).replace(f"{REPO}/", "")}, set()

    ids = [row["id"] for row in rows]
    path_errors, missing, split_folders, misnamed = [], [], [], []
    referenced: list[Path] = []
    for row in rows:
        pair = {}
        for key in ("video", "metadata"):
            try:
                pair[key] = resolve(root, row[key])
            except ValueError as e:
                path_errors.append(f"id {row['id']} {key}: {e}")
                continue
            path = pair[key]
            if (root / row[key]).is_symlink() or not path.is_file() or path.stat().st_size == 0:
                missing.append(f"id {row['id']} {key}: {row[key]}")
            referenced.append(path)
        if len(pair) == 2 and pair["video"].parent != pair["metadata"].parent:
            split_folders.append(f"id {row['id']}")
        folder = f"videos/scene_{row['id']:04d}"
        if (row["video"], row["metadata"]) != (f"{folder}/video.mp4", f"{folder}/metadata.json"):
            misnamed.append(f"id {row['id']}")

    expected_files = {root / MANIFEST, *referenced}
    needed_folders = {p for f in referenced for p in f.parents if p.is_relative_to(root) and p != root}
    files, folders = listing(root)
    raw = (root / MANIFEST).read_bytes()

    criteria = {
        "rows_parse": True,
        "row_count_matches": len(rows) == EXPECTED_ROWS[dataset],
        "ids_unique_and_contiguous": sorted(ids) == list(range(len(rows))),
        "paths_inside_dataset": not path_errors,
        "paths_distinct": len(set(referenced)) == len(referenced),
        "files_exist": not missing,
        "video_and_metadata_same_folder": not split_folders,
        "no_orphan_files": not (files - expected_files),
        "no_orphan_folders": not (folders - needed_folders),
    }
    return {
        "criteria": criteria,
        "rows": len(rows),
        "expected_rows": EXPECTED_ROWS[dataset],
        "problems": {
            "path_errors": sample(path_errors),
            "missing_symlinked_or_empty": sample(missing),
            "video_and_metadata_in_different_folders": sample(split_folders),
            "orphan_files": sample(map(rel, files - expected_files)),
            "orphan_folders": sample(map(rel, folders - needed_folders)),
        },
        "diagnostics": {
            "not_named_scene_id": sample(misnamed),
            "file_order_is_id_order": ids == sorted(ids),
            "extra_row_keys": sorted({k for row in rows for k in row} - set(ROW_KEYS)),
            "crlf_line_endings": b"\r" in raw,
        },
    }, expected_files


def check_manifests() -> dict:
    """Every clip is reachable through its manifest, and nothing in data/ is unaccounted for.

    Passes if, in every dataset: every manifest line parses; the row count matches DATA.md; ids
    are exactly 0 ... N - 1; every video and metadata path resolves inside data/<dataset>/, all
    paths are distinct, each is a non-empty regular file (not a symlink), and each row's video and
    metadata share a folder; no file or folder is left unreferenced. And across data/: only the
    three dataset folders at the top level, and the total file count equals the fingerprint's.
    .DS_Store is ignored everywhere. Folder naming (scene_<id>) and row order are diagnostics.
    """
    per_dataset, expected_files = {}, set()
    for dataset in DATASETS:
        per_dataset[dataset], files = audit_dataset(dataset)
        expected_files |= files

    top_level = sorted(p.name for p in DATA.iterdir() if p.name not in IGNORED)
    all_files, _ = listing(DATA)
    criteria = {
        "top_level_is_the_three_datasets": top_level == sorted(DATASETS)
        and all((DATA / name).is_dir() for name in top_level),
        "total_files_match_fingerprint_count": len(all_files) == EXPECTED_TOTAL_FILES,
        "every_file_accounted_for": all_files == expected_files,
    }
    return {
        "data_dir": str(DATA.relative_to(REPO)),
        "ignored_names": sorted(IGNORED),
        "criteria": criteria,
        "top_level": top_level,
        "total_files": len(all_files),
        "expected_total_files": EXPECTED_TOTAL_FILES,
        "per_dataset": per_dataset,
        "passed": all(criteria.values())
        and all(all(r["criteria"].values()) for r in per_dataset.values()),
    }


CHECKS = {
    "manifests": check_manifests,
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