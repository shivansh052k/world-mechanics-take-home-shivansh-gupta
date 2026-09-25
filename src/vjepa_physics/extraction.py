"""Pooled encoder activations for probing: every site, averaged over the tokens of each time step."""
import torch
from transformers import VJEPA2Model

from vjepa_physics.activations import GRID, capture_encoder

# Sites in forward order: the patch embedding, the 24 blocks, and the encoder's final LayerNorm output.
BLOCK_SITES = tuple(f"block_{i}" for i in range(24))
SITES = ("embedding", *BLOCK_SITES, "final_norm")
TIME_STEPS = GRID[0]  # 8 time steps (tubelets of 2 frames)
TOKENS_PER_STEP = GRID[1] * GRID[2]  # 16 x 16 patches


def plot_index(site: str) -> int:
    """Layer index for plots = transformers' hidden_states index: 0 = embedding, 1-24 = block_0 ... block_23.

    Raises ValueError for the final norm, which has no hidden_states index and is plotted separately.
    """
    if site == "embedding":
        return 0
    if site in BLOCK_SITES:
        return BLOCK_SITES.index(site) + 1
    raise ValueError(f"{site!r} has no hidden_states index")


def pool_time_steps(tokens: torch.Tensor) -> torch.Tensor:
    """(B, 2048, D) tokens -> (B, 8, D): mean over the 256 tokens (16 x 16 patches) of each time step."""
    batch, n_tokens, dim = tokens.shape
    if n_tokens != TIME_STEPS * TOKENS_PER_STEP:
        raise ValueError(f"expected {TIME_STEPS * TOKENS_PER_STEP} tokens, got {n_tokens}")
    return tokens.reshape(batch, TIME_STEPS, TOKENS_PER_STEP, dim).mean(dim=2)


def pooled_activations(model: VJEPA2Model, x: torch.Tensor) -> torch.Tensor:
    """(B, 16, 3, 256, 256) encoder input on the model's device -> (B, 26, 8, 1024) float32 on the CPU.

    Axis 1 follows SITES. Pooling runs on the model's device, then the result moves to the CPU with no dtype
    change (a combined device + dtype move from MPS gave silent zeros once). Raises if a site is missing or
    any pooled value is non-finite or the result is all zero.
    """
    with capture_encoder(model) as acts, torch.inference_mode():
        out = model(pixel_values_videos=x, skip_predictor=True)
        outputs = dict(acts) | {"final_norm": out.last_hidden_state}
        if set(outputs) != set(SITES):
            raise RuntimeError(f"captured {sorted(outputs)}, expected {list(SITES)}")
        pooled = torch.stack([pool_time_steps(outputs[site]) for site in SITES], dim=1).to("cpu")
    if not torch.isfinite(pooled).all() or not pooled.any():
        raise RuntimeError("pooled activations are non-finite or all zero")
    return pooled


def all_token_mean(pooled: torch.Tensor) -> torch.Tensor:
    """(..., 8, D) per-time-step means -> (..., D) mean over all 2048 tokens (every time step has 256 tokens)."""
    return pooled.mean(dim=-2)