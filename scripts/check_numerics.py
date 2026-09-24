"""Numerical correctness checks for the frozen model: repeatability, batching, devices.

Run one check at a time from the repo root, with .venv active:
    python scripts/check_numerics.py <check>

Each check prints its result and stores it under its own key in results/numerics/checks.json.
"""
import argparse
import json
from pathlib import Path

import torch

from vjepa_physics.activations import capture_encoder
from vjepa_physics.evidence import save_result
from vjepa_physics.model import load_model
from vjepa_physics.preprocess import preprocess_clip
from vjepa_physics.reproducibility import set_seeds
from vjepa_physics.video import load_clip

REPO = Path(__file__).resolve().parents[1]
CLIP = REPO / "data/speed/videos/scene_1000/video.mp4"  # speed 2.69 m/s, theta 230.625 deg
OUT = REPO / "results/numerics/checks.json"
SECOND_CLIP = REPO / "data/acceleration/videos/scene_0382/video.mp4"  # accel. 2.57 m/s^2 from rest, theta 275.625 deg

# A numerical difference must be at least 1000x smaller than the difference between two real clips.
MAX_ERROR_TO_SIGNAL = 1e-3

# torch.testing.assert_close default tolerances for float32. Applied explicitly, because the
# compared copies are float64 (whose defaults are tighter).
FP32_RTOL, FP32_ATOL = 1.3e-6, 1e-5


def clip_input(path: Path) -> torch.Tensor:
    """(1, 16, 3, 256, 256) float32 model input for one clip, on the CPU."""
    return preprocess_clip(load_clip(path)).unsqueeze(0)


def run(model: torch.nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Every compared output of one forward pass, moved to the CPU as float64.

    Keys: "embedding", "block_0" ... "block_23" (own hooks), "final_norm" (encoder
    last_hidden_state) and "predictor" (predictor output); each (B, 2048, 1024).
    """
    with capture_encoder(model) as acts, torch.inference_mode():
        out = model(pixel_values_videos=x)
    if out.predictor_output is None:
        raise RuntimeError("model returned no predictor output")
    outputs = dict(acts) | {
        "final_norm": out.last_hidden_state,
        "predictor": out.predictor_output.last_hidden_state,
    }
    # Move first, then widen: MPS has no float64, so a combined .to("cpu", float64) is not safe.
    converted = {name: t.to("cpu").to(torch.float64) for name, t in outputs.items()}
    for name, t in converted.items():
        if not torch.isfinite(t).all() or t.norm() == 0:
            raise RuntimeError(f"{name}: degenerate output after moving to CPU (non-finite or all zero)")
    return converted


def sci(value: float) -> float:
    """Round to 4 significant digits, for readable JSON."""
    return float(f"{value:.4g}")


def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    """How far `a` is from the reference `b` (same shape, float64)."""
    diff = a - b
    tokens_a = a.reshape(-1, a.shape[-1])
    tokens_b = b.reshape(-1, b.shape[-1])
    return {
        "identical": bool(torch.equal(a, b)) and b.norm().item() > 0,
        "max_abs": sci(diff.abs().max().item()),
        "rel_fro": sci((diff.norm() / b.norm()).item()),
        "cosine": sci(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()),
        "min_token_cosine": sci(torch.nn.functional.cosine_similarity(tokens_a, tokens_b, dim=1).min().item()),
    }


def compare_runs(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict[str, dict]:
    return {name: compare(a[name], b[name]) for name in b}


def check_repeat() -> dict:
    """Same input twice on the same device gives identical outputs (MPS, then CPU).

    Expected: exact equality at every layer. If not, the size of the difference is recorded.
    """
    set_seeds()
    x = clip_input(CLIP)
    per_device = {}
    for device in ("mps", "cpu"):
        model, _ = load_model(device, verify_file=False)
        first = run(model, x.to(device))
        second = run(model, x.to(device))
        per_device[device] = compare_runs(second, first)
        del model, first, second
    return {
        "clip": str(CLIP.relative_to(REPO)),
        "all_identical": {d: all(v["identical"] for v in r.values()) for d, r in per_device.items()},
        "per_layer": per_device,
        "passed": all(v["identical"] for r in per_device.values() for v in r.values()),
    }


def check_batch() -> dict:
    """Two clips in one batch give the same outputs as each clip alone (MPS).

    Scale reference: the relative difference between the two clips' outputs at each layer.
    Passes if every layer is identical, or its batched-vs-single error is at most
    MAX_ERROR_TO_SIGNAL times that between-clip difference.
    """
    set_seeds()
    model, _ = load_model("mps", verify_file=False)
    xa, xb = clip_input(CLIP), clip_input(SECOND_CLIP)
    single_a = run(model, xa.to("mps"))
    single_b = run(model, xb.to("mps"))
    batched = run(model, torch.cat([xa, xb]).to("mps"))

    error_a = compare_runs({k: v[:1] for k, v in batched.items()}, single_a)
    error_b = compare_runs({k: v[1:] for k, v in batched.items()}, single_b)
    per_layer = {}
    for name in single_a:
        signal = ((single_a[name] - single_b[name]).norm() / single_b[name].norm()).item()
        worst = max(error_a[name]["rel_fro"], error_b[name]["rel_fro"])
        per_layer[name] = {
            "identical": error_a[name]["identical"] and error_b[name]["identical"],
            "rel_fro_clip_a": error_a[name]["rel_fro"],
            "rel_fro_clip_b": error_b[name]["rel_fro"],
            "min_token_cosine": min(error_a[name]["min_token_cosine"], error_b[name]["min_token_cosine"]),
            "between_clip_rel_fro": sci(signal),
            "error_to_signal": sci(worst / signal),
        }
    return {
        "clips": [str(CLIP.relative_to(REPO)), str(SECOND_CLIP.relative_to(REPO))],
        "all_identical": all(v["identical"] for v in per_layer.values()),
        "worst_error_to_signal": max(v["error_to_signal"] for v in per_layer.values()),
        "per_layer": per_layer,
        "passed": all(v["identical"] or v["error_to_signal"] <= MAX_ERROR_TO_SIGNAL for v in per_layer.values()),
    }


def outputs_on(
    device: str,
    inputs: list[torch.Tensor],
    dtype: torch.dtype = torch.float32,
    attn_implementation: str = "sdpa",
) -> list[dict[str, torch.Tensor]]:
    """Run each input alone on `device`; free the model before returning."""
    model, _ = load_model(device, verify_file=False, dtype=dtype, attn_implementation=attn_implementation)
    results = [run(model, x.to(device).to(dtype)) for x in inputs]
    del model
    return results


def check_attention() -> dict:
    """Diagnostic: sdpa (fused) vs eager (plain matmul + softmax) attention on MPS, each vs float64.

    Decision rule, stated in advance: keep sdpa unless eager is at least 2x closer to the
    float64 reference at block 23. No pass/fail.
    """
    set_seeds()
    inputs = [clip_input(CLIP), clip_input(SECOND_CLIP)]
    reference = outputs_on("cpu", inputs, dtype=torch.float64)
    sdpa = outputs_on("mps", inputs)
    eager = outputs_on("mps", inputs, attn_implementation="eager")

    def worst_rel(runs: list[dict[str, torch.Tensor]], refs: list[dict[str, torch.Tensor]], name: str) -> float:
        return max(compare(r[name], ref[name])["rel_fro"] for r, ref in zip(runs, refs))

    per_layer = {
        name: {
            "sdpa_vs_fp64": worst_rel(sdpa, reference, name),
            "eager_vs_fp64": worst_rel(eager, reference, name),
            "sdpa_vs_eager": worst_rel(sdpa, eager, name),
        }
        for name in reference[0]
    }
    for values in per_layer.values():
        values["eager_over_sdpa_error"] = sci(values["eager_vs_fp64"] / values["sdpa_vs_fp64"])
    return {
        "clips": [str(CLIP.relative_to(REPO)), str(SECOND_CLIP.relative_to(REPO))],
        "block_23_eager_over_sdpa_error": per_layer["block_23"]["eager_over_sdpa_error"],
        "keep_sdpa": per_layer["block_23"]["eager_over_sdpa_error"] > 0.5,
        "per_layer": per_layer,
    }


def check_precision() -> dict:
    """Diagnostic: how far fp32 on CPU and on MPS each are from a float64 reference (CPU).

    If CPU fp32 is about as far from float64 as MPS fp32 is, the MPS-vs-CPU difference is the
    ordinary error of fp32 arithmetic over 24 blocks, not an MPS defect. No pass/fail.
    """
    set_seeds()
    inputs = [clip_input(CLIP), clip_input(SECOND_CLIP)]
    reference = outputs_on("cpu", inputs, dtype=torch.float64)
    cpu = outputs_on("cpu", inputs)
    mps = outputs_on("mps", inputs)

    def worst_rel(runs: list[dict[str, torch.Tensor]], name: str) -> float:
        return max(compare(r[name], ref[name])["rel_fro"] for r, ref in zip(runs, reference))

    per_layer = {}
    for name in reference[0]:
        signal = ((reference[0][name] - reference[1][name]).norm() / reference[1][name].norm()).item()
        cpu_error, mps_error = worst_rel(cpu, name), worst_rel(mps, name)
        per_layer[name] = {
            "cpu_fp32_vs_fp64": cpu_error,
            "mps_fp32_vs_fp64": mps_error,
            "mps_over_cpu_error": sci(mps_error / cpu_error) if cpu_error > 0 else None,
            "between_clip_rel_fro": sci(signal),
            "cpu_error_to_signal": sci(cpu_error / signal),
            "mps_error_to_signal": sci(mps_error / signal),
        }
    return {
        "clips": [str(CLIP.relative_to(REPO)), str(SECOND_CLIP.relative_to(REPO))],
        "per_layer": per_layer,
    }


def check_devices() -> dict:
    """MPS vs CPU (the reference), per layer, on two clips.

    Passes if at every layer the MPS-vs-CPU error is at most MAX_ERROR_TO_SIGNAL times the
    difference between the two clips (on CPU). Also records whether torch's default fp32
    tolerances hold, and how many elements fall outside them.
    """
    set_seeds()
    inputs = [clip_input(CLIP), clip_input(SECOND_CLIP)]
    mps = outputs_on("mps", inputs)
    cpu = outputs_on("cpu", inputs)

    per_layer = {}
    for name in cpu[0]:
        signal = ((cpu[0][name] - cpu[1][name]).norm() / cpu[1][name].norm()).item()
        errors = [compare(m[name], c[name]) for m, c in zip(mps, cpu)]
        outside = [
            (~torch.isclose(m[name], c[name], rtol=FP32_RTOL, atol=FP32_ATOL)).double().mean().item()
            for m, c in zip(mps, cpu)
        ]
        worst = max(e["rel_fro"] for e in errors)
        per_layer[name] = {
            "rel_fro_per_clip": [e["rel_fro"] for e in errors],
            "max_abs_per_clip": [e["max_abs"] for e in errors],
            "min_token_cosine": min(e["min_token_cosine"] for e in errors),
            "between_clip_rel_fro": sci(signal),
            "error_to_signal": sci(worst / signal),
            "default_fp32_tolerance_holds": all(f == 0 for f in outside),
            "fraction_outside_default_tolerance": sci(max(outside)),
        }
    failing_default = [n for n, v in per_layer.items() if not v["default_fp32_tolerance_holds"]]
    return {
        "clips": [str(CLIP.relative_to(REPO)), str(SECOND_CLIP.relative_to(REPO))],
        "worst_error_to_signal": max(v["error_to_signal"] for v in per_layer.values()),
        "worst_rel_fro": max(max(v["rel_fro_per_clip"]) for v in per_layer.values()),
        "lowest_min_token_cosine": min(v["min_token_cosine"] for v in per_layer.values()),
        "layers_failing_default_fp32_tolerance": failing_default,
        "per_layer": per_layer,
        "passed": all(v["error_to_signal"] <= MAX_ERROR_TO_SIGNAL for v in per_layer.values()),
    }


CHECKS = {
    "repeat": check_repeat,
    "batch": check_batch,
    "devices": check_devices,
    "precision": check_precision,
    "attention": check_attention,
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