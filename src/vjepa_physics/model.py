"""Load the frozen V-JEPA 2 model at the pinned checkpoint, and fingerprint its weights."""
import hashlib
from typing import cast

import torch
from huggingface_hub import hf_hub_download
from transformers import VJEPA2Model

from vjepa_physics.checkpoint import MODEL_ID, REVISION, WEIGHTS_SHA256


def weights_file() -> str:
    """Local path of the cached model.safetensors at the pinned revision (never downloads)."""
    return hf_hub_download(MODEL_ID, "model.safetensors", revision=REVISION, local_files_only=True)


def file_sha256(path: str, chunk_bytes: int = 1 << 24) -> str:
    """SHA-256 of a file, read in 16 MB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def load_model(device: str | torch.device = "cpu", verify_file: bool = True) -> tuple[VJEPA2Model, dict]:
    """Frozen V-JEPA 2 (encoder + predictor): fp32, eval mode, no gradients.

    Returns the model and transformers' loading report. Raises if the weights file does not
    match the pinned SHA-256, or if any checkpoint key is missing, unexpected or mismatched.
    """
    if verify_file:
        digest = file_sha256(weights_file())
        if digest != WEIGHTS_SHA256:
            raise RuntimeError(f"model.safetensors SHA-256 {digest} != pinned {WEIGHTS_SHA256}")

    loaded = VJEPA2Model.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    model, info = cast(tuple[VJEPA2Model, dict], loaded)
    problems = {key: sorted(map(str, value)) for key, value in info.items() if value}
    if problems:
        raise RuntimeError(f"checkpoint did not load cleanly: {problems}")

    model.eval()
    model.requires_grad_(False)
    return model.to(device), info


def weights_fingerprint(model: torch.nn.Module) -> str:
    """SHA-256 over every tensor in the state dict (name, dtype, shape, bytes), in name order.

    The same value before and after an experiment proves the model was not modified.
    """
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()