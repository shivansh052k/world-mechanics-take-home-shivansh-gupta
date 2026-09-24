"""Checks for running the frozen model end to end on a real clip.

Run one check at a time from the repo root, with .venv active:
    python scripts/check_forward.py <check>

Each check prints its result and stores it under its own key in results/forward/checks.json.
"""
import argparse
import json
from pathlib import Path

import torch

from vjepa_physics.evidence import save_result
from vjepa_physics.model import load_model, weights_fingerprint
from vjepa_physics.preprocess import preprocess_clip
from vjepa_physics.reproducibility import set_seeds
from vjepa_physics.video import load_clip

REPO = Path(__file__).resolve().parents[1]
CLIP = REPO / "data/speed/videos/scene_1000/video.mp4"  # speed 2.69 m/s, theta 230.625 deg
OUT = REPO / "results/forward/checks.json"

# In-memory weights fingerprint right after a clean load (results/model/checks.json, key "load").
REFERENCE_FINGERPRINT = "c865f524c1376e4452943b208d7d50ba588be9490604f235a3d9c9dc80804ede"

# 16 frames / tubelet 2 = 8 time steps; 256 / patch 16 = 16 x 16 patches; hidden size 1024.
EXPECTED_TOKENS_SHAPE = (1, 8 * 16 * 16, 1024)


def summary(t: torch.Tensor) -> dict:
    """Shape, dtype, device, finiteness and basic statistics of a tensor."""
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": t.device.type,
        "all_finite": bool(torch.isfinite(t).all()),
        "mean": round(float(t.mean()), 6),
        "std": round(float(t.std()), 6),
    }


def check_forward() -> dict:
    """One real clip through encoder and predictor on MPS.

    Expected: encoder and predictor outputs (1, 2048, 1024), every value finite, and the
    weights fingerprint unchanged by the forward pass (equal to the clean-load reference).
    """
    set_seeds()
    device = torch.device("mps")
    model, _ = load_model(device)
    before = weights_fingerprint(model)

    x = preprocess_clip(load_clip(CLIP)).unsqueeze(0).to(device)  # (1, 16, 3, 256, 256)
    with torch.inference_mode():
        out = model(pixel_values_videos=x)
    if out.predictor_output is None:
        raise RuntimeError("model returned no predictor output")

    encoder = out.last_hidden_state
    predictor = out.predictor_output.last_hidden_state
    target = out.predictor_output.target_hidden_state
    after = weights_fingerprint(model)

    outputs = {"encoder": encoder, "predictor": predictor, "predictor_target": target}
    return {
        "clip": str(CLIP.relative_to(REPO)),
        "input": summary(x),
        "outputs": {name: summary(t) for name, t in outputs.items()},
        "weights_fingerprint_before": before,
        "weights_fingerprint_after": after,
        "passed": bool(
            all(tuple(t.shape) == EXPECTED_TOKENS_SHAPE for t in outputs.values())
            and all(bool(torch.isfinite(t).all()) for t in outputs.values())
            and bool(torch.isfinite(x).all())
            and encoder.device.type == "mps"
            and before == after == REFERENCE_FINGERPRINT
        ),
    }


CHECKS = {
    "forward": check_forward,
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