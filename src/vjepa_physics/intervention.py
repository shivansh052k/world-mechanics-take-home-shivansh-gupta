"""Edit encoder activations during a forward pass: a hook writes a replacement back into the model."""
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch
from transformers import VJEPA2Model


def encoder_sites(model: VJEPA2Model) -> dict[str, torch.nn.Module]:
    """Editable sites, named as in capture_encoder: "embedding", "block_0" ... "block_23"."""
    sites: dict[str, torch.nn.Module] = {"embedding": model.encoder.embeddings}
    sites |= {f"block_{i}": block for i, block in enumerate(model.encoder.layer)}
    return sites


@contextmanager
def edit_encoder(
    model: VJEPA2Model, site: str, edit: Callable[[torch.Tensor], torch.Tensor]
) -> Iterator[None]:
    """Replace one site's output with edit(output) in every forward pass inside the context.

    `edit` receives the site's (B, 2048, 1024) hidden states and must return a tensor of the
    same shape, dtype and device, which then flows into the rest of the model. Blocks return
    (hidden states, attention weights); only the hidden states are replaced. The hook is
    prepended, so it runs before every other forward hook on that module (capture_encoder's
    and transformers' hidden_states hooks), and those all see the edited output. Removed on exit.
    """
    module = encoder_sites(model).get(site)
    if module is None:
        raise ValueError(f"unknown site {site!r}")

    def hook(module: torch.nn.Module, args: tuple, output):
        hidden = output[0] if isinstance(output, tuple) else output
        edited = edit(hidden)
        if (edited.shape, edited.dtype, edited.device) != (hidden.shape, hidden.dtype, hidden.device):
            raise ValueError(
                f"{site}: edit returned {tuple(edited.shape)} {edited.dtype} {edited.device}, "
                f"expected {tuple(hidden.shape)} {hidden.dtype} {hidden.device}"
            )
        return (edited, *output[1:]) if isinstance(output, tuple) else edited

    handle = module.register_forward_hook(hook, prepend=True)
    try:
        yield
    finally:
        handle.remove()
        

def run_blocks(model: VJEPA2Model, hidden: torch.Tensor, first: int, last: int) -> dict[str, torch.Tensor]:
    """Continue the encoder from given hidden states: run blocks first ... last (inclusive), return each output.

    `hidden` (B, 2048, 1024) must be what block `first` receives in a full forward pass, i.e. the output of block
    first - 1 (the patch embedding when first = 0), possibly edited. Each block is called as the encoder calls it,
    layer(hidden, None)[0]: no position mask, so positions come from the token index. Returns {"block_i": output}
    for i = first ... last, (B, 2048, 1024) on the model's device. The encoder's final LayerNorm is not applied.
    """
    blocks = model.encoder.layer
    if not 0 <= first <= last < len(blocks):
        raise ValueError(f"blocks {first}...{last} outside 0...{len(blocks) - 1}")
    if hidden.ndim != 3 or hidden.shape[-1] != blocks[first].hidden_size:
        raise ValueError(f"expected (B, tokens, {blocks[first].hidden_size}) hidden states, got {tuple(hidden.shape)}")
    outputs: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for i in range(first, last + 1):
            hidden = blocks[i](hidden, None)[0]
            outputs[f"block_{i}"] = hidden
    return outputs