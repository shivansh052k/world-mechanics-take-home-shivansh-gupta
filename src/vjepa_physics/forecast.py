"""Forecast future encoder tokens with the V-JEPA 2 predictor: encode the start of a clip, predict the rest."""
import torch
import torch.nn.functional as F
from transformers import VJEPA2Model


def encode(model: VJEPA2Model, x: torch.Tensor) -> torch.Tensor:
    """(B, T, 3, 256, 256) input -> (B, T / 2 * 256, 1024) encoder output, after its final LayerNorm."""
    with torch.inference_mode():
        return model(pixel_values_videos=x, skip_predictor=True).last_hidden_state


def predict(model: VJEPA2Model, context: torch.Tensor, target_positions: torch.Tensor) -> torch.Tensor:
    """Predictor output at `target_positions`, given the encoder tokens at positions 0 ... N - 1.

    `context` is (B, N, 1024): the encoder output for the clip's first N / 256 time steps. The
    predictor uses its context mask both to pick tokens out of `context` and as their positions,
    so the context must be a prefix of the clip. Returns (B, len(target_positions), 1024), in the
    order of `target_positions`.
    """
    batch, n_context, _ = context.shape
    context_mask = torch.arange(n_context, device=context.device).unsqueeze(0).repeat(batch, 1)
    target_mask = target_positions.to(context.device).unsqueeze(0).repeat(batch, 1)
    with torch.inference_mode():
        out = model.predictor(
            encoder_hidden_states=context, context_mask=[context_mask], target_mask=[target_mask]
        )
    return out.last_hidden_state


def training_target(tokens: torch.Tensor) -> torch.Tensor:
    """Non-affine layer norm over features: what the predictor was trained to match."""
    return F.layer_norm(tokens, (tokens.shape[-1],))