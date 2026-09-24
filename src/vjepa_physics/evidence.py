"""Save check results with provenance, so every stored result says how it was produced."""
import json
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

PACKAGES = (
    "av", "numpy", "scipy", "torch", "torchvision", "transformers",
    "opencv-python-headless", "matplotlib",
)


def provenance(cwd: Path) -> dict:
    """UTC time, git commit and dirty flag of the repo containing `cwd`, and package versions."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True).stdout.strip()

    return {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "versions": {p: version(p) for p in PACKAGES},
    }


def save_result(out: Path, name: str, result: dict) -> None:
    """Store one check's result under its own key in a JSON file; other keys are kept."""
    out.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(out.read_text()) if out.exists() else {}
    data[name] = {"provenance": provenance(out.parent), "result": result}
    out.write_text(json.dumps(data, indent=2) + "\n")