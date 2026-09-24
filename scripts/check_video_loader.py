"""Sanity checks for video decoding and the clip loader.

Run one check at a time from the repo root, with .venv active:
    python scripts/check_video_loader.py inspect

Each check prints its result and stores it under its own key in
results/video_loader/checks.json, so that file holds the evidence for every check.
"""
import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import av

REPO = Path(__file__).resolve().parents[1]
CLIP = REPO / "data/speed/videos/scene_1000/video.mp4"  # speed 2.69 m/s, theta 230.625 deg
OUT = REPO / "results/video_loader/checks.json"


def provenance() -> dict:
    """When, on which commit, and with which decoder version a check ran."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()

    return {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "pyav": av.__version__,
    }


def save(name: str, result: dict) -> None:
    """Store one check's result under its own key; other checks' results are kept."""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    data[name] = {"provenance": provenance(), "result": result}
    OUT.write_text(json.dumps(data, indent=2) + "\n")
    print(f"saved '{name}' -> {OUT.relative_to(REPO)}")


def check_inspect() -> dict:
    """Raw stream facts from PyAV, before any loader exists.

    Expected: mpeg4 codec, yuv420p, 256x256, 16 frames, strictly increasing pts.
    """
    with av.open(str(CLIP)) as c:
        s = c.streams.video[0]
        frames = list(c.decode(video=0))
        return {
            "clip": str(CLIP.relative_to(REPO)),
            "container": c.format.name,
            "codec": s.codec_context.name,
            "pix_fmt": s.codec_context.pix_fmt,
            "size": [s.width, s.height],
            "stream_frames": s.frames,
            "average_rate": str(s.average_rate),
            "time_base": str(s.time_base),
            "decoded_frames": len(frames),
            "pts": [f.pts for f in frames],
            "frame_formats": sorted({f.format.name for f in frames}),
            "frame_sizes": sorted({(f.width, f.height) for f in frames}),
        }


CHECKS = {
    "inspect": check_inspect,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("check", choices=sorted(CHECKS))
    name = parser.parse_args().check
    result = CHECKS[name]()
    print(json.dumps(result, indent=2))
    save(name, result)


if __name__ == "__main__":
    main()