"""Checks for the train / validation / test splits.

Run one check at a time from the repo root, with .venv active:
    python scripts/check_splits.py <check>

Each check prints its result and stores it under its own key in results/splits/checks.json.
"""
import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from vjepa_physics.data import DATASETS, LABEL_FIELD, load_dataset
from vjepa_physics.evidence import file_sha256, save_result
from vjepa_physics.reproducibility import SEED
from vjepa_physics.splits import (
    HELD_OUT, N_VALUES, PER_SEEN_VALUE, ROLES, SEEN_ROLES, VAL_UNSEEN, build_splits, read_splits, write_splits,
)

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = REPO / "results/splits/checks.json"
SPLITS = REPO / "artifacts/manifests/splits.csv"  # committed: every later step reads the splits from here

EXPECTED_CLIPS = 4572  # DATA.md: 1,500 + 1,536 + 1,536
# Clips per role, in ROLES order (train, val_seen, val_unseen, test_seen, test_unseen). Speed and acceleration:
# the two-level split's counts. Direction: the same rule applied to 36 angles x 23 + 28 x 24 clips.
EXPECTED_COUNTS = {
    "direction": (813, 203, 94, 203, 187),
    "speed": (832, 208, 96, 208, 192),
    "acceleration": (832, 208, 96, 208, 192),
}
SCHEME = {"direction": "direction", "speed": "magnitude", "acceleration": "magnitude"}


def check_build() -> dict:
    """Build the splits from metadata and SEED, save them, and check the saved file.

    Writes artifacts/manifests/splits.csv. Every criterion is evaluated on the file as read back.
    Passes if: every clip of every dataset appears exactly once and nothing else does; labels equal the
    metadata; value indices follow from the file's own labels; per dataset the role counts equal
    EXPECTED_COUNTS, val_unseen and test_unseen hold exactly the held-out value indices, and no seen
    role contains a held-out value; speed and acceleration: 16/4/4 for every seen value and the same
    role for every id in both sets; rebuilding with the same seed gives a byte-identical file.
    Held-out label values and counts are recorded.
    """
    clips = {dataset: load_dataset(DATA, dataset) for dataset in DATASETS}
    write_splits(build_splits(clips, SEED), SPLITS)
    rows = read_splits(SPLITS)

    expected_keys = [(dataset, clip["id"]) for dataset in DATASETS for clip in clips[dataset]]
    keys = [(r["dataset"], r["id"]) for r in rows]
    label_of = {(dataset, clip["id"]): clip[LABEL_FIELD[dataset]] for dataset in DATASETS for clip in clips[dataset]}
    role_of = {(r["dataset"], r["id"]): r["role"] for r in rows}

    per_dataset, dataset_criteria = {}, {}
    for dataset in DATASETS:
        mine = [r for r in rows if r["dataset"] == dataset]
        grid = np.unique([r["label"] for r in mine])
        counts = Counter(r["role"] for r in mine)
        values = {role: sorted({r["value_index"] for r in mine if r["role"] == role}) for role in ROLES}
        seen_values = set().union(*(values[role] for role in SEEN_ROLES))
        held = set(HELD_OUT[SCHEME[dataset]])
        val_unseen = sorted(VAL_UNSEEN[SCHEME[dataset]])

        criteria = {
            "value_index_follows_label": len(grid) == N_VALUES and all(
                r["value_index"] == int(np.searchsorted(grid, r["label"])) for r in mine
            ),
            "counts_match": tuple(counts[role] for role in ROLES) == EXPECTED_COUNTS[dataset],
            "unseen_values_as_decided": (
                values["val_unseen"] == val_unseen and values["test_unseen"] == sorted(held - set(val_unseen))
            ),
            "no_held_out_value_in_seen_roles": not (seen_values & held),
        }
        if dataset != "direction":
            per_value = Counter((r["value_index"], r["role"]) for r in mine if r["role"] in SEEN_ROLES)
            wanted = tuple(PER_SEEN_VALUE[role] for role in SEEN_ROLES)
            criteria["sixteen_four_four_per_seen_value"] = len(seen_values) == N_VALUES - len(held) and all(
                tuple(per_value[(v, role)] for role in SEEN_ROLES) == wanted for v in seen_values
            )
        dataset_criteria[dataset] = criteria
        per_dataset[dataset] = {
            "criteria": criteria,
            "counts": {role: counts[role] for role in ROLES},
            "seen_values": len(seen_values),
            "held_out_labels": {
                role: [float(grid[v]) for v in values[role]] for role in ("val_unseen", "test_unseen")
            },
        }

    speed_ids = [clip["id"] for clip in clips["speed"]]
    shared = [clip["id"] for clip in clips["acceleration"]] == speed_ids and all(
        role_of.get(("speed", i)) == role_of.get(("acceleration", i)) for i in speed_ids
    )

    with tempfile.TemporaryDirectory() as tmp:
        again = Path(tmp) / "splits.csv"
        write_splits(build_splits({dataset: load_dataset(DATA, dataset) for dataset in DATASETS}, SEED), again)
        rebuild_identical = again.read_bytes() == SPLITS.read_bytes()

    criteria = {
        "every_clip_exactly_once": (
            len(keys) == len(set(keys)) == EXPECTED_CLIPS and sorted(keys) == sorted(expected_keys)
        ),
        "labels_match_metadata": all(label_of.get(k) == r["label"] for k, r in zip(keys, rows)),
        "speed_acceleration_same_roles": shared,
        "rebuild_byte_identical": rebuild_identical,
    }
    return {
        "criteria": criteria,
        "seed": SEED,
        "per_dataset": per_dataset,
        "artifact": {"path": str(SPLITS.relative_to(REPO)), "rows": len(rows), "sha256": file_sha256(SPLITS)},
        "passed": all(criteria.values()) and all(all(c.values()) for c in dataset_criteria.values()),
    }


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