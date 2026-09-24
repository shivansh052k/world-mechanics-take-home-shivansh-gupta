"""Checks for result provenance: the code-only dirty flag.

Run from the repo root, with .venv active:
    python scripts/check_evidence.py <check>

Each check prints its result and stores it under its own key in results/evidence/checks.json.
"""
import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from vjepa_physics.evidence import code_changes, repo_root, save_result

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/evidence/checks.json"

# Files of the scratch repository; mirrors the real layout that matters for the flag.
SCRATCH_FILES = (
    "src/pkg/module.py", "scripts/run.py", "pyproject.toml", "requirements.lock.txt",
    "results/out.json", "docs/notes.md",
)

# Scenario -> (edits applied to a clean committed tree, expected dirty flag).
SCENARIOS = {
    "committed": ({}, False),
    "results_changed": ({"results/out.json": "changed\n", "results/new.json": "new\n"}, False),
    "docs_changed": ({"docs/notes.md": "changed\n"}, False),
    "src_edited": ({"src/pkg/module.py": "changed\n"}, True),
    "src_new_untracked": ({"src/pkg/extra.py": "new\n"}, True),
    "scripts_edited": ({"scripts/run.py": "changed\n"}, True),
    "pyproject_changed": ({"pyproject.toml": "changed\n"}, True),
    "lock_changed": ({"requirements.lock.txt": "changed\n"}, True),
}


def scratch_git(root: Path, *args: str) -> None:
    """Git in the scratch repository, with a fixed identity and no commit signing."""
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=check", "-c", "user.email=check@example.invalid",
         "-c", "commit.gpgsign=false", *args],
        capture_output=True, text=True, check=True,
    )


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def changes_from(cwd: Path, root: Path) -> list[str]:
    """code_changes(root) evaluated with the process working directory set to `cwd`."""
    previous = Path.cwd()
    os.chdir(cwd)
    try:
        return code_changes(root)
    finally:
        os.chdir(previous)


def check_dirty_flag() -> dict:
    """The dirty flag reacts to code and environment changes only, from any working directory.

    Every scenario runs in a scratch git repository in a temporary folder, so the real
    repository is never modified. Also records the real repository's current state.
    """
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        scratch_git(root, "init", "--quiet")
        for rel in SCRATCH_FILES:
            write(root / rel, "original\n")
        scratch_git(root, "add", "--all")
        scratch_git(root, "commit", "--quiet", "--message", "initial")

        for name, (edits, expected) in SCENARIOS.items():
            scratch_git(root, "reset", "--quiet", "--hard")
            scratch_git(root, "clean", "--quiet", "-fd")
            for rel, text in edits.items():
                write(root / rel, text)
            lines = code_changes(root)
            results[name] = {"dirty": bool(lines), "expected": expected, "paths": lines}

        # Same answer whatever the working directory (last scenario's edit is still in place).
        from_elsewhere = {
            "repo_root": changes_from(root, root),
            "subfolder": changes_from(root / "docs", root),
            "outside": changes_from(Path(tempfile.gettempdir()), root),
        }

    real_root_from_elsewhere = repo_root() == REPO
    previous = Path.cwd()
    os.chdir(tempfile.gettempdir())
    try:
        real_root_from_outside = repo_root() == REPO
    finally:
        os.chdir(previous)

    same_everywhere = len({tuple(v) for v in from_elsewhere.values()}) == 1 and bool(from_elsewhere["outside"])
    return {
        "scenarios": results,
        "same_answer_from_any_working_directory": same_everywhere,
        "real_repo_root_found_from_repo_and_outside": real_root_from_elsewhere and real_root_from_outside,
        "real_repo_code_changes_now": code_changes(REPO),
        "passed": bool(
            all(r["dirty"] == r["expected"] for r in results.values())
            and same_everywhere
            and real_root_from_elsewhere
            and real_root_from_outside
        ),
    }


CHECKS = {
    "dirty_flag": check_dirty_flag,
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