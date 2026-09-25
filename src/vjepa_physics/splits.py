"""Train / validation / test splits: whole label values held out, the remaining clips split within each value."""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from vjepa_physics.data import DATASETS, LABEL_FIELD, angle_octant, motion_group

ROLES = ("train", "val_seen", "val_unseen", "test_seen", "test_unseen")
SEEN_ROLES = ("train", "val_seen", "test_seen")
N_VALUES = 64  # distinct label values per dataset (DATA.md)

# Held-out label values, as indices into each dataset's sorted grid of 64 values. Speed and acceleration
# ("magnitude"): every 5th value from index 4, never an endpoint. Direction: evenly around the circle,
# avoiding the cardinal and diagonal angles (indices 0, 8, ..., 56).
HELD_OUT = {
    "magnitude": tuple(range(4, 60, 5)),
    "direction": (2, 7, 13, 18, 23, 29, 34, 39, 45, 50, 55, 61),
}
VAL_UNSEEN = {"magnitude": (9, 24, 39, 54), "direction": (2, 18, 34, 50)}

# Speed and acceleration: clips of each seen value per seen role (24 clips per value).
PER_SEEN_VALUE = {"train": 16, "val_seen": 4, "test_seen": 4}
# Direction: val_seen and test_seen each get 1 / EVAL_DENOMINATOR of the seen clips (4 of 24, as above).
EVAL_DENOMINATOR = 6

COLUMNS = ("dataset", "id", "label", "value_index", "role")


def value_indices(clips: list[dict], dataset: str) -> np.ndarray:
    """(clips,) index of each clip's label in the dataset's sorted grid of distinct labels.

    Raises ValueError unless the grid has exactly N_VALUES values.
    """
    labels = np.array([clip[LABEL_FIELD[dataset]] for clip in clips], dtype=float)
    grid = np.unique(labels)
    if len(grid) != N_VALUES:
        raise ValueError(f"{dataset}: {len(grid)} distinct labels, expected {N_VALUES}")
    return np.searchsorted(grid, labels)


def unseen_role(index: int, scheme: str) -> str | None:
    """'val_unseen' or 'test_unseen' for a held-out value index, None for a seen one."""
    if index in VAL_UNSEEN[scheme]:
        return "val_unseen"
    if index in HELD_OUT[scheme]:
        return "test_unseen"
    return None


def magnitude_roles(ids: list[int], index: np.ndarray, seed: int) -> dict[int, str]:
    """Role per clip id for a speed or acceleration set.

    Held-out values go whole to val_unseen / test_unseen. Each seen value's clips are shuffled and
    split 16 / 4 / 4 into train / val_seen / test_seen. Values are visited in index order and ids in
    increasing order, with one generator seeded by `seed`, so the result depends only on the inputs.
    Raises ValueError if a seen value does not have exactly 24 clips.
    """
    rng = np.random.default_rng(seed)
    by_value: dict[int, list[int]] = defaultdict(list)
    for clip_id, value in zip(ids, index.tolist()):
        by_value[value].append(clip_id)

    roles: dict[int, str] = {}
    per_value = sum(PER_SEEN_VALUE.values())
    for value in sorted(by_value):
        members = sorted(by_value[value])
        held = unseen_role(value, "magnitude")
        if held:
            roles.update(dict.fromkeys(members, held))
            continue
        if len(members) != per_value:
            raise ValueError(f"value index {value}: {len(members)} clips, expected {per_value}")
        shuffled = rng.permutation(members).tolist()
        start = 0
        for role, count in PER_SEEN_VALUE.items():
            roles.update(dict.fromkeys(shuffled[start:start + count], role))
            start += count
    return roles


def direction_roles(clips: list[dict], index: np.ndarray, seed: int) -> dict[int, str]:
    """Role per clip id for the direction set.

    Held-out angles go whole to val_unseen / test_unseen. Seen clips are stratified on motion group x
    angle octant: val_seen and test_seen each get round(seen / 6) clips in total. Each cell gets
    floor(n / 6) of each, and the remaining places go to the cells with the largest remainder n mod 6,
    ties broken by the seeded generator; each cell gets the same number for val_seen and test_seen.
    Cells and ids are visited in sorted order, so the result depends only on the inputs.
    """
    rng = np.random.default_rng(seed)
    roles: dict[int, str] = {}
    cells: dict[tuple[str, int], list[int]] = defaultdict(list)
    for clip, value in zip(clips, index.tolist()):
        held = unseen_role(value, "direction")
        if held:
            roles[clip["id"]] = held
        else:
            cells[(motion_group(clip), angle_octant(clip["theta_degrees"]))].append(clip["id"])

    keys = sorted(cells)
    seen = sum(len(cells[k]) for k in keys)
    target = (seen + EVAL_DENOMINATOR // 2) // EVAL_DENOMINATOR  # round half up, in integers
    per_cell = {k: len(cells[k]) // EVAL_DENOMINATOR for k in keys}
    remainder = {k: len(cells[k]) % EVAL_DENOMINATOR for k in keys}
    extra = target - sum(per_cell.values())
    if not 0 <= extra <= sum(r > 0 for r in remainder.values()):
        raise ValueError(f"cannot place {extra} extra clips across {len(keys)} cells")
    tie = rng.random(len(keys))
    order = sorted(range(len(keys)), key=lambda i: (-remainder[keys[i]], tie[i]))
    for i in order[:extra]:
        per_cell[keys[i]] += 1

    for k in keys:
        members = rng.permutation(sorted(cells[k])).tolist()
        n = per_cell[k]
        roles.update(dict.fromkeys(members[:n], "val_seen"))
        roles.update(dict.fromkeys(members[n:2 * n], "test_seen"))
        roles.update(dict.fromkeys(members[2 * n:], "train"))
    return roles


def build_splits(clips: dict[str, list[dict]], seed: int) -> list[dict]:
    """One row per clip of every dataset (DATASETS order, then manifest order): dataset, id, label, value_index, role.

    Speed and acceleration share one assignment, computed on speed and applied to acceleration by id.
    Raises ValueError if the two sets do not have the same ids with the same value indices.
    """
    index = {dataset: value_indices(clips[dataset], dataset) for dataset in DATASETS}
    speed_ids = [clip["id"] for clip in clips["speed"]]
    acceleration_ids = [clip["id"] for clip in clips["acceleration"]]
    if speed_ids != acceleration_ids or not np.array_equal(index["speed"], index["acceleration"]):
        raise ValueError("speed and acceleration differ in ids or value indices; a shared assignment is undefined")

    shared = magnitude_roles(speed_ids, index["speed"], seed)
    roles = {
        "speed": shared,
        "acceleration": shared,
        "direction": direction_roles(clips["direction"], index["direction"], seed),
    }
    return [
        {"dataset": dataset, "id": clip["id"], "label": clip[LABEL_FIELD[dataset]],
         "value_index": int(value), "role": roles[dataset][clip["id"]]}
        for dataset in DATASETS
        for clip, value in zip(clips[dataset], index[dataset].tolist())
    ]


def write_splits(rows: list[dict], path: Path) -> None:
    """Write split rows as CSV: fixed columns, rows in the given order, '\\n' line ends, so reruns are byte-identical."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(COLUMNS)
        writer.writerows([row[c] for c in COLUMNS] for row in rows)


def read_splits(path: Path) -> list[dict]:
    """Typed rows of a split file. Raises ValueError on unexpected columns or roles.

    Does not check the file's hash: callers get `path` from evidence.verified_artifact first.
    """
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(f"{path}: columns {reader.fieldnames}, expected {list(COLUMNS)}")
        rows = []
        for row in reader:
            if row["role"] not in ROLES:
                raise ValueError(f"{path}: unknown role {row['role']!r}")
            rows.append({
                "dataset": row["dataset"], "id": int(row["id"]), "label": float(row["label"]),
                "value_index": int(row["value_index"]), "role": row["role"],
            })
    return rows