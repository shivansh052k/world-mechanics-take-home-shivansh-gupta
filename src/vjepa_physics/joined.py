"""One table per variable joining each clip's activation row with its labels, split role, flags and tracked disk."""
import json
from collections.abc import Callable

import numpy as np

from vjepa_physics.data import LABEL_FIELD, angle_octant, motion_group
from vjepa_physics.evidence import repo_root, verified_artifact

FLAG_NAMES = ("exit", "clipped", "sub_patch_motion", "frozen_start")
JOINED_CHECKS = "results/joined/checks.json"  # key "build", one artifact entry per variable
EXTRACTION_CHECKS = "results/extraction/checks.json"  # keys "extract_<variable>": fields "artifact", "ids"


def aligned(rows: list, ids: list[int], get_id: Callable, source: str) -> list:
    """`rows` reordered to follow `ids`. Raises ValueError unless `rows` hold every id in `ids` exactly once."""
    by_id: dict[int, object] = {}
    for row in rows:
        clip_id = get_id(row)
        if clip_id in by_id:
            raise ValueError(f"{source}: id {clip_id} appears twice")
        by_id[clip_id] = row
    if len(ids) != len(set(ids)) or set(by_id) != set(ids):
        raise ValueError(f"{source}: ids differ from the activation ids")
    return [by_id[i] for i in ids]


def build_table(
    variable: str,
    activation_ids: np.ndarray,
    activations_sha256: str,
    clips: list[dict],
    split_rows: list[dict],
    flag_rows: list[dict],
    tracked: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Per-clip table for one variable, in the activation array's row order.

    `clips` = load_dataset(variable); `split_rows` = read_splits(...); `flag_rows` = the flags table's rows as
    read by csv.DictReader; `tracked` = the tracking npz's arrays. The last three may hold every dataset;
    only rows of `variable` are used. Raises ValueError if any source does not cover exactly the activation
    ids once each, if a split label differs from the metadata label, or if a flag is not "0" or "1".
    Keys: id, label, value_index, role, theta_degrees, speed_mps, acceleration_mps2, motion, the four flags
    (bool), centre (clips, 16, 2) px (col, row; NaN without disk), area (clips, 16), touches_border
    (clips, 16), activations_sha256 (the activation file this table belongs to); direction also group, octant.
    """
    ids = [int(i) for i in activation_ids.tolist()]
    meta = aligned(clips, ids, lambda m: m["id"], "metadata")
    split = aligned([r for r in split_rows if r["dataset"] == variable], ids, lambda r: r["id"], "split file")
    flags = aligned([r for r in flag_rows if r["dataset"] == variable], ids, lambda r: int(r["id"]), "flags table")
    track_rows = np.flatnonzero(tracked["dataset"] == variable).tolist()
    order = np.array(aligned(track_rows, ids, lambda k: int(tracked["id"][k]), "tracking"), dtype=np.int64)

    labels = np.array([m[LABEL_FIELD[variable]] for m in meta], dtype=float)
    if not np.array_equal(labels, np.array([r["label"] for r in split], dtype=float)):
        raise ValueError(f"{variable}: split labels differ from the metadata")
    if any(f[name] not in ("0", "1") for f in flags for name in FLAG_NAMES):
        raise ValueError(f"{variable}: a flag value is not '0' or '1'")

    table = {
        "id": np.array(ids, dtype=np.int64),
        "label": labels,
        "value_index": np.array([r["value_index"] for r in split], dtype=np.int64),
        "role": np.array([r["role"] for r in split]),
        "theta_degrees": np.array([m["theta_degrees"] for m in meta], dtype=float),
        "speed_mps": np.array([m["speed_mps"] for m in meta], dtype=float),
        "acceleration_mps2": np.array([m["acceleration_mps2"] for m in meta], dtype=float),
        "motion": np.array([m["motion"] for m in meta]),
        **{name: np.array([f[name] == "1" for f in flags]) for name in FLAG_NAMES},
        "centre": tracked["centre"][order],
        "area": tracked["area"][order],
        "touches_border": tracked["touches_border"][order],
        "activations_sha256": np.array(activations_sha256),
    }
    if variable == "direction":
        table["group"] = np.array([motion_group(m) for m in meta])
        table["octant"] = np.array([angle_octant(m["theta_degrees"]) for m in meta], dtype=np.int64)
    return table


def load_joined(variable: str) -> dict[str, np.ndarray]:
    """The saved table for one variable, plus key "activations": its (clips, 26, 8, 1024) array, memory-mapped.

    Every file is read through its recorded hash: the table (JOINED_CHECKS, key "build"), the activation array
    and its ids (EXTRACTION_CHECKS). Raises RuntimeError if the activation file is not the one the table was
    built from, or its ids or row count differ from the table's.
    """
    root = repo_root()
    with np.load(verified_artifact(root / JOINED_CHECKS, "build", variable)) as saved:
        table = {key: saved[key] for key in saved.files}

    extraction = root / EXTRACTION_CHECKS
    key = f"extract_{variable}"
    path = verified_artifact(extraction, key)  # file hash = the one recorded at extraction
    recorded = json.loads(extraction.read_text())[key]["result"]["artifact"]["sha256"]
    if recorded != str(table["activations_sha256"]):
        raise RuntimeError(f"{variable}: activation file {recorded} is not the one the table was built from")
    ids = np.load(verified_artifact(extraction, key, "ids"))
    activations = np.load(path, mmap_mode="r")
    if not np.array_equal(ids, table["id"]) or len(activations) != len(ids):
        raise RuntimeError(f"{variable}: activation rows are not aligned with the table")
    table["activations"] = activations
    return table