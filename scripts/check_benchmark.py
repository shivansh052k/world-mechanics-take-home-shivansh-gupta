"""Speed and memory of the frozen model on MPS, to budget activation extraction and steering.

Run one check at a time from the repo root, with .venv active:
    python scripts/check_benchmark.py <check>

Each check prints its result and stores it under its own key in results/benchmark/checks.json.
"""
import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch

from vjepa_physics.activations import capture_encoder
from vjepa_physics.evidence import save_result
from vjepa_physics.model import load_model, weights_fingerprint
from vjepa_physics.preprocess import preprocess_clip
from vjepa_physics.reproducibility import set_seeds
from vjepa_physics.video import load_clip

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/benchmark/checks.json"
DEVICE = "mps"

# In-memory weights fingerprint right after a clean load (results/model/checks.json, key "load").
REFERENCE_FINGERPRINT = "c865f524c1376e4452943b208d7d50ba588be9490604f235a3d9c9dc80804ede"

# Timing does not depend on clip content; 16 consecutive speed clips divide evenly into every batch size.
CLIPS = [REPO / f"data/speed/videos/scene_{i:04d}/video.mp4" for i in range(1000, 1016)]
BATCH_SIZES = (1, 2, 4, 8)

TOTAL_CLIPS = 4572  # all three datasets
SITES = 25  # embedding + 24 blocks
HIDDEN = 1024
TIME_STEPS = 8
GIB = 1024**3

# Batch-size rule, fixed before running: among batch sizes whose MPS memory pool stays within
# MEMORY_FRACTION of Metal's recommended maximum, suggest the smallest whose time per clip is
# within TIME_SLACK of the fastest.
TIME_SLACK = 1.10
MEMORY_FRACTION = 0.5


def sci(value: float) -> float:
    """Round to 4 significant digits, for readable JSON."""
    return float(f"{value:.4g}")


def pooled_forward(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """(B, 16, 3, 256, 256) on MPS -> (B, 25, 1024) token-mean of every site, on the CPU.

    Stand-in for the extraction step: capture all 25 sites, mean over the 2048 tokens, move to CPU.
    """
    with capture_encoder(model) as acts, torch.inference_mode():
        model(pixel_values_videos=x, skip_predictor=True)
        pooled = torch.stack([acts[name].mean(dim=1) for name in acts], dim=1)
    return pooled.to("cpu")


def check_benchmark() -> dict:
    """Diagnostic: time and memory per clip on MPS, and a suggested extraction batch size.

    CPU stages (decode, preprocess) timed per clip. GPU stage (capture all sites, mean-pool,
    move to CPU) timed per batch at batch sizes 1, 2, 4, 8, after one untimed warm-up batch each,
    synchronizing MPS around every timed call. Memory: MPS pool after each batch size (emptied
    first; a high-water proxy, since MPS has no peak API), weights' footprint, peak process RSS.
    Also records whether batched pooled outputs equal single-clip ones exactly. Suggested batch
    size by the rule fixed above. No pass/fail.
    """
    set_seeds()
    model, _ = load_model(DEVICE, verify_file=False)
    torch.mps.synchronize()
    weights_gib = torch.mps.current_allocated_memory() / GIB

    decode_s, preprocess_s, inputs = [], [], []
    for path in CLIPS:
        t0 = time.perf_counter()
        clip = load_clip(path)
        t1 = time.perf_counter()
        inputs.append(preprocess_clip(clip))
        decode_s.append(t1 - t0)
        preprocess_s.append(time.perf_counter() - t1)
    all_inputs = torch.stack(inputs)  # (16, 16, 3, 256, 256) on the CPU

    reference = torch.cat([pooled_forward(model, x.unsqueeze(0).to(DEVICE)) for x in inputs])

    per_batch, raw = {}, {}
    for b in BATCH_SIZES:
        torch.mps.empty_cache()
        pooled_forward(model, all_inputs[:b].to(DEVICE))  # warm-up at this batch size, not timed
        times, outs = [], []
        for start in range(0, len(CLIPS), b):
            xb = all_inputs[start:start + b]
            torch.mps.synchronize()
            t = time.perf_counter()
            outs.append(pooled_forward(model, xb.to(DEVICE)))
            torch.mps.synchronize()
            times.append(time.perf_counter() - t)
        pool = torch.mps.driver_allocated_memory()
        pooled = torch.cat(outs)
        raw[b] = {"seconds_per_clip": float(np.median(times)) / b, "pool_bytes": pool}
        per_batch[str(b)] = {
            "batches_timed": len(times),
            "seconds_per_clip_median": sci(raw[b]["seconds_per_clip"]),
            "mps_pool_gib": sci(pool / GIB),
            "batched_equals_single": bool(torch.equal(pooled, reference)),
            "max_abs_diff_vs_single": sci((pooled - reference).abs().max().item()),
        }

    limit = MEMORY_FRACTION * torch.mps.recommended_max_memory()
    fits = [b for b in BATCH_SIZES if raw[b]["pool_bytes"] <= limit]
    suggested = None
    if fits:
        fastest = min(raw[b]["seconds_per_clip"] for b in fits)
        suggested = min(b for b in fits if raw[b]["seconds_per_clip"] <= TIME_SLACK * fastest)

    cpu_per_clip = float(np.median(decode_s) + np.median(preprocess_s))
    gpu_per_clip = raw[suggested]["seconds_per_clip"] if suggested else None
    return {
        "device": DEVICE,
        "clips": [str(p.relative_to(REPO)) for p in CLIPS],
        "decode_seconds_median": sci(float(np.median(decode_s))),
        "preprocess_seconds_median": sci(float(np.median(preprocess_s))),
        "per_batch_size": per_batch,
        "weights_gib_on_mps": sci(weights_gib),
        "recommended_max_gib": sci(torch.mps.recommended_max_memory() / GIB),
        "memory_limit_gib": sci(limit / GIB),
        "peak_process_rss_gib": sci(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / GIB),
        "memory_note": "MPS has no peak-memory API; mps_pool_gib is the allocator pool after emptying the cache and running that batch size (high-water proxy).",
        "rule": f"smallest batch size within {TIME_SLACK}x of the fastest time per clip, pool <= {MEMORY_FRACTION} x recommended max",
        "suggested_batch_size": suggested,
        "estimated_extraction_minutes_all_clips": (
            sci(TOTAL_CLIPS * (cpu_per_clip + gpu_per_clip) / 60) if gpu_per_clip else None
        ),
        "pooled_storage_gib_fp32": {
            "mean_over_tokens": sci(TOTAL_CLIPS * SITES * HIDDEN * 4 / GIB),
            "per_time_step": sci(TOTAL_CLIPS * SITES * TIME_STEPS * HIDDEN * 4 / GIB),
        },
        "weights_fingerprint_unchanged": weights_fingerprint(model) == REFERENCE_FINGERPRINT,
    }


CHECKS = {
    "benchmark": check_benchmark,
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