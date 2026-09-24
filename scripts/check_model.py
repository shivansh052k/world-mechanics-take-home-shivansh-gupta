"""Checks for loading the frozen V-JEPA 2 model at the pinned checkpoint.

Run one check at a time from the repo root, with .venv active:
    python scripts/check_model.py <check>

Each check prints its result and stores it under its own key in results/model/checks.json.
"""
import argparse
import gc
import json
import random
from pathlib import Path

import numpy as np
import torch

from vjepa_physics.checkpoint import WEIGHTS_SHA256
from vjepa_physics.reproducibility import SEED, set_seeds
from vjepa_physics.evidence import save_result
from vjepa_physics.model import file_sha256, load_model, weights_file, weights_fingerprint

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/model/checks.json"

# Expected architecture, from the checkpoint's config.json (revision b3c1679).
EXPECTED_CONFIG = {
    "num_hidden_layers": 24,
    "hidden_size": 1024,
    "num_attention_heads": 16,
    "patch_size": 16,
    "tubelet_size": 2,
    "crop_size": 256,
    "pred_hidden_size": 384,
    "pred_num_hidden_layers": 12,
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "drop_path_rate": 0.0,
}


def check_config() -> dict:
    """Loaded config and actual module structure match the documented architecture."""
    model, _ = load_model(verify_file=False)
    config = {key: getattr(model.config, key) for key in EXPECTED_CONFIG}
    structure = {
        "encoder_blocks": len(model.encoder.layer),
        "predictor_blocks": len(model.predictor.layer),
        "patch_embedding_weight_shape": list(model.encoder.embeddings.patch_embeddings.proj.weight.shape),
    }
    return {
        "config": config,
        "structure": structure,
        "passed": bool(
            config == EXPECTED_CONFIG
            and structure["encoder_blocks"] == 24
            and structure["predictor_blocks"] == 12
            and structure["patch_embedding_weight_shape"] == [1024, 3, 2, 16, 16]
        ),
    }


def check_load() -> dict:
    """Weights file matches the pinned SHA-256; the checkpoint loads cleanly; the model is frozen."""
    digest = file_sha256(weights_file())
    model, info = load_model(verify_file=False)  # file hash checked just above, recorded below
    params = list(model.parameters())
    return {
        "weights_file_sha256": digest,
        "matches_pinned_sha256": digest == WEIGHTS_SHA256,
        "loading_report": {key: sorted(map(str, value)) for key, value in info.items()},
        "parameters_total": sum(p.numel() for p in params),
        "parameters_encoder": sum(p.numel() for p in model.encoder.parameters()),
        "parameters_predictor": sum(p.numel() for p in model.predictor.parameters()),
        "state_dict_tensors": len(model.state_dict()),
        "dtypes": sorted({str(p.dtype) for p in params}),
        "device": str(params[0].device),
        "training_mode": model.training,
        "any_requires_grad": any(p.requires_grad for p in params),
        "attn_implementation": model.config._attn_implementation,
        "weights_fingerprint": weights_fingerprint(model),
        "passed": bool(
            digest == WEIGHTS_SHA256
            and not any(info.values())
            and not model.training
            and not any(p.requires_grad for p in params)
            and {str(p.dtype) for p in params} == {"torch.float32"}
        ),
    }


def check_fingerprint() -> dict:
    """Two independent loads give identical in-memory weights (nothing randomly initialized)."""
    first = weights_fingerprint(load_model(verify_file=False)[0])
    gc.collect()  # release the first model (about 1.3 GB) before loading the second
    second = weights_fingerprint(load_model(verify_file=False)[0])
    return {
        "first": first,
        "second": second,
        "identical": first == second,
        "passed": first == second,
    }


def draws() -> dict:
    """One sample from every seeded generator: Python, NumPy, torch CPU and torch MPS."""
    return {
        "python": random.random(),
        "numpy": np.random.rand(3).tolist(),
        "torch_cpu": torch.rand(3).tolist(),
        "torch_mps": torch.rand(3, device="mps").cpu().tolist() if torch.backends.mps.is_available() else None,
    }


def check_seeds() -> dict:
    """Re-seeding reproduces every generator's draws exactly; a different seed changes them."""
    set_seeds(SEED)
    first = draws()
    set_seeds(SEED)
    repeat = draws()
    set_seeds(SEED + 1)
    other = draws()
    return {
        "seed": SEED,
        "first": first,
        "repeat_identical": first == repeat,
        "other_seed_differs": {k: first[k] != other[k] for k in first},
        "mps_available": torch.backends.mps.is_available(),
        "passed": bool(
            first == repeat
            and all(first[k] != other[k] for k in first if first[k] is not None)
            and torch.backends.mps.is_available()
        ),
    }


CHECKS = {
    "config": check_config,
    "load": check_load,
    "fingerprint": check_fingerprint,
    "seeds": check_seeds,
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