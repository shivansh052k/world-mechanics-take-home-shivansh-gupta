"""Save check results with provenance, so every stored result says how it was produced."""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

# Versions are read from package metadata, never by importing the packages
# (importing cv2 next to PyAV duplicates Objective-C classes on macOS).
PACKAGES = (
    "av", "numpy", "scipy", "torch", "torchvision", "transformers",
    "opencv-python-headless", "matplotlib",
)

# What produces results: the code and the pinned environment. Changes anywhere else
# (results/, docs/, ...) do not make the code that ran differ from the recorded commit.
CODE_PATHS = ("src", "scripts", "pyproject.toml", "requirements.lock.txt")

PACKAGE_DIR = Path(__file__).resolve().parent


def git(root: Path, *args: str) -> str:
    """Run git against `root` explicitly, independent of the working directory; return raw stdout."""
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
    ).stdout


def repo_root() -> Path:
    """Root of the git repository that contains this package."""
    return Path(git(PACKAGE_DIR, "rev-parse", "--show-toplevel").strip())


def code_changes(root: Path) -> list[str]:
    """Uncommitted or untracked changes under CODE_PATHS, as `git status --porcelain` lines."""
    return git(root, "status", "--porcelain", "--untracked-files=all", "--", *CODE_PATHS).splitlines()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_hashes(root: Path) -> dict:
    """SHA-256 of the running script and of every .py file in the package, plus one combined hash."""
    files = sorted(p for p in PACKAGE_DIR.rglob("*.py") if "__pycache__" not in p.parts)
    script = Path(sys.argv[0]).resolve()
    if script.suffix == ".py" and script.is_file():
        files.append(script)
    per_file = {p.relative_to(root).as_posix(): file_sha256(p) for p in files}
    listing = "".join(f"{path}\t{digest}\n" for path, digest in sorted(per_file.items()))
    return {"files": per_file, "combined_sha256": hashlib.sha256(listing.encode()).hexdigest()}


def provenance() -> dict:
    """UTC time, git commit, whether the code differs from it, code hashes, package versions."""
    root = repo_root()
    changes = code_changes(root)
    return {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git(root, "rev-parse", "HEAD").strip(),
        "git_dirty": bool(changes),
        "git_dirty_paths": changes,
        "code": code_hashes(root),
        "versions": {p: version(p) for p in PACKAGES},
    }


def save_result(out: Path, name: str, result: dict) -> None:
    """Store one check's result under its own key in a JSON file; other keys are kept."""
    out.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(out.read_text()) if out.exists() else {}
    data[name] = {"provenance": provenance(), "result": result}
    out.write_text(json.dumps(data, indent=2) + "\n")