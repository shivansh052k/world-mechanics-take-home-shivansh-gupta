"""The supplied datasets: their names and a strict reader for their manifests."""
import json
from pathlib import Path

DATASETS = ("direction", "speed", "acceleration")
MANIFEST = "manifest.jsonl"
ROW_KEYS = ("id", "video", "metadata")


def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    """json object hook: a key given twice is an error (json.loads would silently keep the last)."""
    keys = [key for key, _ in pairs]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"duplicate keys {duplicates}")
    return dict(pairs)


def read_manifest(dataset_dir: str | Path) -> list[dict]:
    """Rows of `dataset_dir/manifest.jsonl`, in file order.

    Strict: raises ValueError, naming the line, if any line (other than the file's final newline)
    is not a JSON object, repeats a key, lacks `id`, `video` or `metadata`, has an `id` that is
    not an int, or a path that is not a string. Extra keys are kept. Paths are returned as
    written; DATA.md resolves them relative to `dataset_dir`.
    """
    path = Path(dataset_dir) / MANIFEST
    lines = path.read_text(encoding="utf-8").split("\n")
    if lines[-1] == "":
        lines.pop()  # the newline that ends the last row

    rows = []
    for number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line, object_pairs_hook=reject_duplicate_keys)
        except ValueError as e:  # JSONDecodeError is a ValueError, and so are duplicate keys
            raise ValueError(f"{path}:{number}: {e}") from None
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: expected a JSON object, got {type(row).__name__}")
        missing = [key for key in ROW_KEYS if key not in row]
        if missing:
            raise ValueError(f"{path}:{number}: missing keys {missing}")
        if type(row["id"]) is not int:  # bool is a subclass of int, so isinstance would accept True
            raise ValueError(f"{path}:{number}: id {row['id']!r} is not an int")
        for key in ("video", "metadata"):
            if not isinstance(row[key], str):
                raise ValueError(f"{path}:{number}: {key} {row[key]!r} is not a string")
        rows.append(row)
    return rows


def resolve(dataset_dir: str | Path, relative: str) -> Path:
    """Absolute path of a manifest entry, relative to `dataset_dir` (DATA.md), required to stay inside it.

    Raises ValueError if `relative` is absolute, or if the resolved path (after `..` and symlinks)
    is `dataset_dir` itself or lies outside it. Does not check that the file exists.
    """
    base = Path(dataset_dir).resolve()
    if Path(relative).is_absolute():
        raise ValueError(f"{relative!r} is absolute; manifest paths must be relative")
    target = (base / relative).resolve()
    if target == base or not target.is_relative_to(base):
        raise ValueError(f"{relative!r} resolves to {target}, outside {base}")
    return target


def reject_constant(name: str) -> None:
    """json parse_constant hook: NaN, Infinity and -Infinity are not valid JSON numbers."""
    raise ValueError(f"non-finite number {name}")


def read_metadata(path: str | Path) -> dict:
    """One clip's metadata.json as a dict.

    Strict: raises ValueError, naming the file, if it is not a JSON object, repeats a key, or
    contains NaN / Infinity (Python's json accepts those by default; standard JSON does not).
    Field names and values are not checked here.
    """
    path = Path(path)
    try:
        meta = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from None
    if not isinstance(meta, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(meta).__name__}")
    return meta