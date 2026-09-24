"""Capture encoder activations with our own forward hooks. The hooks only read; outputs are unchanged."""
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from transformers import VJEPA2Model

# Token grid for 16 frames at 256 x 256: 8 time steps (tubelets of 2 frames) x 16 x 16 patches.
# Token index = t * 256 + row * 16 + col (Conv3d patch embedding, flattened over time, rows, cols).
GRID = (8, 16, 16)


def as_grid(tokens: torch.Tensor) -> torch.Tensor:
    """(B, 2048, D) tokens -> (B, 8, 16, 16, D): [batch, time step, patch row, patch col, feature]."""
    return tokens.reshape(tokens.shape[0], *GRID, tokens.shape[-1])


@contextmanager
def capture_encoder(model: VJEPA2Model) -> Iterator[dict[str, torch.Tensor]]:
    """Record the patch-embedding output and every encoder block's output during a forward pass.

    Yields a dict that the forward pass fills: "embedding" -> (B, 2048, 1024), the input to
    block 0; "block_0" ... "block_23" -> each block's output (B, 2048, 1024), before the
    encoder's final LayerNorm. Tensors are detached and stay on the model's device. A second
    forward pass inside the same context overwrites them. All hooks are removed on exit.
    """
    captured: dict[str, torch.Tensor] = {}

    def record(name: str):
        def hook(module: torch.nn.Module, args: tuple, output) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured[name] = hidden.detach()
        return hook

    handles = [model.encoder.embeddings.register_forward_hook(record("embedding"))]
    handles += [
        block.register_forward_hook(record(f"block_{i}"))
        for i, block in enumerate(model.encoder.layer)
    ]
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()