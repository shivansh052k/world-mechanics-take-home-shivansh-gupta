"""Checks for capturing encoder activations with our own hooks.

Run one check at a time from the repo root, with .venv active:
    python scripts/check_activations.py <check>

Each check prints its result and stores it under its own key in results/activations/checks.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vjepa_physics.activations import GRID, capture_encoder
from vjepa_physics.evidence import save_result
from vjepa_physics.model import load_model
from vjepa_physics.preprocess import preprocess_clip
from vjepa_physics.video import load_clip

REPO = Path(__file__).resolve().parents[1]
CLIP = REPO / "data/speed/videos/scene_1000/video.mp4"  # speed 2.69 m/s, theta 230.625 deg
OUT = REPO / "results/activations/checks.json"


def count_forward_hooks(model: torch.nn.Module) -> int:
    return sum(len(module._forward_hooks) for module in model.modules())


def check_hidden_states() -> dict:
    """Our hooks line up with transformers' hidden_states, captured in the same forward pass.

    Expected (transformers 5.x): entry 0 = patch-embedding output; entry i = output of block i - 1
    for i = 1..24, so entry 24 = block 23 before the final LayerNorm; last_hidden_state =
    final LayerNorm(block 23). Also: our hooks are all removed afterwards.
    """
    model, _ = load_model("mps")
    x = preprocess_clip(load_clip(CLIP)).unsqueeze(0).to("mps")
    with torch.inference_mode():  # installs transformers' own (inactive) capture hooks first
        model(pixel_values_videos=x, skip_predictor=True, output_hidden_states=True)
    hooks_before = count_forward_hooks(model)

    with capture_encoder(model) as acts, torch.inference_mode():
        out = model(pixel_values_videos=x, skip_predictor=True, output_hidden_states=True)
        final_norm_of_last_block = model.encoder.layernorm(acts["block_23"])
    hooks_after = count_forward_hooks(model)

    names = ["embedding"] + [f"block_{i}" for i in range(24)]
    hidden_states = out.hidden_states or ()
    entry_matches = [bool(torch.equal(h, acts[n])) for h, n in zip(hidden_states, names)]
    last_is_normed = bool(torch.equal(out.last_hidden_state, final_norm_of_last_block))
    last_differs_from_block = not torch.equal(out.last_hidden_state, acts["block_23"])
    return {
        "clip": str(CLIP.relative_to(REPO)),
        "hidden_states_entries": len(hidden_states),
        "captured_names": len(acts),
        "entry_i_equals": dict(zip([f"{i}:{n}" for i, n in enumerate(names)], entry_matches)),
        "last_hidden_state_equals_final_norm_of_block_23": last_is_normed,
        "last_hidden_state_differs_from_block_23": last_differs_from_block,
        "forward_hooks_before_and_after": [hooks_before, hooks_after],
        "passed": bool(
            len(hidden_states) == 25
            and all(entry_matches)
            and last_is_normed
            and last_differs_from_block
            and hooks_before == hooks_after
        ),
    }


def most_deviant_patch(tokens_grid: np.ndarray) -> list[tuple[int, int]]:
    """Per time step, the (row, col) of the token farthest from the median token of the clip."""
    flat = tokens_grid.reshape(-1, tokens_grid.shape[-1])
    deviation = np.linalg.norm(flat - np.median(flat, axis=0), axis=1).reshape(tokens_grid.shape[:3])
    patches = []
    for t in range(GRID[0]):
        row, col = np.unravel_index(deviation[t].argmax(), deviation[t].shape)
        patches.append((int(row), int(col)))
    return patches


def check_token_layout() -> dict:
    """Token order is (time step, patch row, patch col), shown from the data.

    At the patch embedding, the token that deviates most from the median token (the disk's patch)
    must, for every time step t, lie on a 16 x 16 patch containing disk pixels in frame 2t or 2t + 1.
    Reading the tokens with rows and cols swapped, or with time reversed, must fail.
    """
    clip = load_clip(CLIP)
    model, _ = load_model("mps")
    x = preprocess_clip(clip).unsqueeze(0).to("mps")
    with capture_encoder(model) as acts, torch.inference_mode():
        model(pixel_values_videos=x, skip_predictor=True)
    tokens = acts["embedding"][0].cpu().numpy()  # (2048, 1024)

    disk = clip.max(axis=-1) > 128  # (16, 256, 256)
    disk_patches = disk.reshape(8, 2, 16, 16, 16, 16).any(axis=(1, 3, 5))  # (8, 16, 16)

    readings = {
        "time_row_col": tokens.reshape(8, 16, 16, -1),
        "time_col_row": tokens.reshape(8, 16, 16, -1).transpose(0, 2, 1, 3),
        "time_reversed": tokens.reshape(8, 16, 16, -1)[::-1],
    }
    hits = {}
    for name, grid in readings.items():
        patches = most_deviant_patch(grid)
        hits[name] = {
            "patch_row_col_per_time_step": [list(p) for p in patches],
            "on_disk": [bool(disk_patches[t, r, c]) for t, (r, c) in enumerate(patches)],
        }
    return {
        "clip": str(CLIP.relative_to(REPO)),
        "readings": hits,
        "disk_patches_per_time_step": [int(disk_patches[t].sum()) for t in range(8)],
        "passed": bool(
            all(hits["time_row_col"]["on_disk"])
            and not all(hits["time_col_row"]["on_disk"])
            and not all(hits["time_reversed"]["on_disk"])
        ),
    }


CHECKS = {
    "hidden_states": check_hidden_states,
    "token_layout": check_token_layout,
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