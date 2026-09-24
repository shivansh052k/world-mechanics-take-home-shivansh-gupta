"""Turn decoded clips into V-JEPA 2 encoder inputs with the checkpoint's own video processor."""
from functools import lru_cache

import numpy as np
import torch
from transformers import VJEPA2VideoProcessor

from vjepa_physics.checkpoint import MODEL_ID, REVISION


@lru_cache(maxsize=2)
def load_processor(native: bool = True) -> VJEPA2VideoProcessor:
    """The checkpoint's video processor, from the local cache at the pinned revision.

    native=True turns resize and center-crop off, so 256 x 256 clips keep their exact pixels;
    rescale (1/255) and ImageNet normalization stay as shipped. native=False is the shipped
    default (resize shortest edge to 292, center-crop 256), kept only for comparison.
    """
    overrides = {"do_resize": False, "do_center_crop": False} if native else {}
    return VJEPA2VideoProcessor.from_pretrained(
        MODEL_ID, revision=REVISION, local_files_only=True, **overrides
    )


def preprocess_clip(clip: np.ndarray, native: bool = True) -> torch.Tensor:
    """(T, H, W, 3) uint8 RGB clip -> (T, 3, H, W) float32 encoder input on the CPU.

    The caller stacks clips into a batch (B, T, 3, H, W) and moves it to the device.
    """
    if clip.ndim != 4 or clip.shape[-1] != 3 or clip.dtype != np.uint8:
        raise ValueError(f"expected a (T, H, W, 3) uint8 clip, got {clip.shape} {clip.dtype}")
    batch = load_processor(native)(clip, input_data_format="channels_last", return_tensors="pt")
    return batch["pixel_values_videos"][0]