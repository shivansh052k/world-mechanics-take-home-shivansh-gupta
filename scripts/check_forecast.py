"""Checks for forecasting future encoder tokens with the V-JEPA 2 predictor.

Run one check at a time from the repo root, with .venv active:
    python scripts/check_forecast.py <check>

Each check prints its result and stores it under its own key in results/forecast/checks.json.
"""
import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from vjepa_physics.evidence import save_result
from vjepa_physics.forecast import encode, predict, training_target
from vjepa_physics.model import load_model, weights_fingerprint
from vjepa_physics.preprocess import preprocess_clip
from vjepa_physics.reproducibility import SEED, set_seeds
from vjepa_physics.video import load_clip

REPO = Path(__file__).resolve().parents[1]
CLIP = REPO / "data/speed/videos/scene_1000/video.mp4"  # speed 2.69 m/s, theta 230.625 deg
OUT = REPO / "results/forecast/checks.json"
DEVICE = "mps"

# In-memory weights fingerprint right after a clean load (results/model/checks.json, key "load").
REFERENCE_FINGERPRINT = "c865f524c1376e4452943b208d7d50ba588be9490604f235a3d9c9dc80804ede"

# 16 frames = 8 time steps x 256 tokens. Context = frames 0-7 = time steps 0-3 = tokens 0-1023;
# forecast targets = time steps 4-7 = tokens 1024-2047.
CONTEXT_FRAMES = 8
N_TOKENS = 2048
N_CONTEXT = 1024
HIDDEN = 1024
TOKENS_PER_STEP = 256
LAST_CONTEXT_STEP = 3
TARGET_STEPS = (4, 5, 6, 7)

# Clip sample: drawn in this dataset order by one seeded generator, without replacement.
DATASETS = ("speed", "acceleration", "direction")
CLIPS_PER_DATASET = 32
BOOTSTRAP_RESAMPLES = 10_000
LEAKAGE_NOTE = (
    "Splits do not exist yet, so some of these clips may later fall in test. Accepted: this check "
    "uses no labels and fits nothing."
)

# Each baseline type in two versions: from the frames-0-7-only encoding, and from the full-clip
# encoding (which has already seen the future frames, so it is advantaged). The stronger version
# (lower mean L1 within a dataset) decides.
BASELINES = {
    "copy_last_step": ("copy_context", "copy_full"),
    "mean_context_token": ("mean_context", "mean_full"),
}


def usable(t: torch.Tensor) -> bool:
    """Finite everywhere and not all zero."""
    return bool(torch.isfinite(t).all()) and bool(t.any())


def sci(value: float) -> float:
    """Round to 4 significant digits, for readable JSON."""
    return float(f"{value:.4g}")


def check_predictor_path() -> dict:
    """Our separate encode -> predict calls reproduce the model's own forward, and keep target order.

    Passes if: encode(x) equals the combined forward's encoder output exactly; predict with
    positions 0-2047 equals the combined forward's predictor output exactly; in the forecast
    setup (context = frames 0-7, targets 1024-2047) reversed targets give exactly the reversed
    rows, and the rows are not all identical (so the order test is not trivial); forecast shapes
    are (1, 1024, 1024), all outputs finite and non-zero; weights fingerprint unchanged.
    """
    set_seeds()
    model, _ = load_model(DEVICE, verify_file=False)
    x = preprocess_clip(load_clip(CLIP)).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        combined = model(pixel_values_videos=x)
    if combined.predictor_output is None:
        raise RuntimeError("model returned no predictor output")
    ours_encoder = encode(model, x)
    ours_predictor = predict(model, ours_encoder, torch.arange(N_TOKENS))

    context = encode(model, x[:, :CONTEXT_FRAMES])
    targets = torch.arange(N_CONTEXT, N_TOKENS)
    forecast = predict(model, context, targets)
    forecast_reversed = predict(model, context, targets.flip(0))

    early_full = ours_encoder[:, :N_CONTEXT]
    context_vs_full = ((context - early_full).norm() / early_full.norm()).item()
    fingerprint = weights_fingerprint(model)

    criteria = {
        "encoder_equals_combined": bool(torch.equal(ours_encoder, combined.last_hidden_state)),
        "predictor_equals_combined": bool(
            torch.equal(ours_predictor, combined.predictor_output.last_hidden_state)
        ),
        "reversed_targets_come_back_reversed": bool(torch.equal(forecast_reversed, forecast.flip(1))),
        "forecast_rows_not_all_identical": not bool(torch.equal(forecast, forecast.flip(1))),
        "forecast_shapes": (
            list(context.shape) == [1, N_CONTEXT, HIDDEN]
            and list(forecast.shape) == [1, N_TOKENS - N_CONTEXT, HIDDEN]
        ),
        "all_usable": all(usable(t) for t in (ours_encoder, ours_predictor, context, forecast)),
        "fingerprint_unchanged": fingerprint == REFERENCE_FINGERPRINT,
    }
    return {
        "clip": str(CLIP.relative_to(REPO)),
        "device": DEVICE,
        "criteria": criteria,
        "context_shape": list(context.shape),
        "forecast_shape": list(forecast.shape),
        "context_vs_full_clip_first_half_rel_diff": sci(context_vs_full),
        "weights_fingerprint": fingerprint,
        "passed": all(criteria.values()),
    }


def draw_clips() -> dict[str, list[dict]]:
    """CLIPS_PER_DATASET manifest rows per dataset, one seeded generator, DATASETS order; sorted by id."""
    rng = np.random.default_rng(SEED)
    drawn = {}
    for dataset in DATASETS:
        lines = (REPO / "data" / dataset / "manifest.jsonl").read_text().splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
        picks = rng.choice(len(rows), size=CLIPS_PER_DATASET, replace=False)
        drawn[dataset] = sorted((rows[i] for i in picks), key=lambda row: row["id"])
    return drawn


def timed(fn: Callable[..., torch.Tensor], *args) -> tuple[torch.Tensor, float]:
    """fn(*args) and its wall-clock seconds; MPS runs asynchronously, so synchronize on both sides."""
    torch.mps.synchronize()
    start = time.perf_counter()
    out = fn(*args)
    torch.mps.synchronize()
    return out, time.perf_counter() - start


def step_tokens(tokens: torch.Tensor, step: int) -> torch.Tensor:
    """(1, N, D) tokens -> (1, 256, D) tokens of one time step, in (row, col) order."""
    return tokens[:, step * TOKENS_PER_STEP:(step + 1) * TOKENS_PER_STEP]


def l1_per_step(prediction: torch.Tensor, target: torch.Tensor) -> list[float]:
    """Elementwise mean |prediction - target| for each target time step, as in the training loss."""
    diff = (prediction - target).abs().reshape(len(TARGET_STEPS), TOKENS_PER_STEP, HIDDEN)
    return diff.mean(dim=(1, 2)).tolist()


def token_std(tokens: torch.Tensor) -> float:
    """Mean over tokens of each token's standard deviation across features."""
    return tokens.std(dim=-1).mean().item()


def bootstrap_ci(diffs: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """95% percentile CI of the mean, resampling clips with replacement."""
    idx = rng.integers(0, len(diffs), size=(BOOTSTRAP_RESAMPLES, len(diffs)))
    low, high = np.percentile(diffs[idx].mean(axis=1), [2.5, 97.5])
    return float(low), float(high)


def compare(l1: dict[str, np.ndarray], rng: np.random.Generator) -> tuple[dict, dict[str, np.ndarray]]:
    """Per baseline type: stronger version, paired (predictor - baseline) differences, bootstrap CI."""
    summary, diffs_by_type = {}, {}
    for kind, versions in BASELINES.items():
        chosen = min(versions, key=lambda v: l1[v].mean())
        diffs = l1["predictor"] - l1[chosen]
        low, high = bootstrap_ci(diffs, rng)
        summary[kind] = {
            "mean_l1_per_version": {v: sci(l1[v].mean()) for v in versions},
            "chosen": chosen,
            "mean_diff_predictor_minus_baseline": sci(diffs.mean()),
            "ci95": [sci(low), sci(high)],
            "predictor_better": bool(high < 0),
        }
        diffs_by_type[kind] = diffs
    return summary, diffs_by_type


def check_forecast() -> dict:
    """The predictor forecasts time steps 4-7 from frames 0-7 better than fitting-free baselines.

    Per clip: encode frames 0-7 alone (context, tokens 0-1023); predict tokens 1024-2047; target =
    non-affine layer_norm of the full-clip encoder output at tokens 1024-2047 (the pretraining
    target). Baselines, layer-normed the same way: copy of time step 3, and the mean context
    token, each from the context encoding and from the full-clip encoding (tokens 0-1023 only);
    the stronger version per dataset decides. Metric: elementwise mean L1 per clip.
    Passes if, in every dataset separately, the 95% bootstrap CI (10,000 resamples over clips) of
    the mean paired difference (predictor L1 - baseline L1) lies entirely below 0 for both
    baseline types, and every output is finite, non-zero and correctly shaped, with the weights
    unchanged. Pooled results, per-step L1s, the zero predictor, token stds and timings are
    reported, not judged.
    """
    set_seeds()
    model, _ = load_model(DEVICE, verify_file=False)
    drawn = draw_clips()
    target_positions = torch.arange(N_CONTEXT, N_TOKENS)

    per_clip: dict[str, list[dict]] = {dataset: [] for dataset in DATASETS}
    seconds: dict[str, list[float]] = {"encode_full": [], "encode_context": [], "predict": []}
    all_usable = True
    warm_up = True
    for dataset, rows in drawn.items():
        for row in rows:
            x = preprocess_clip(load_clip(REPO / "data" / dataset / row["video"])).unsqueeze(0).to(DEVICE)
            full, t_full = timed(encode, model, x)
            context, t_context = timed(encode, model, x[:, :CONTEXT_FRAMES])
            prediction, t_predict = timed(predict, model, context, target_positions)
            if not warm_up:  # the first clip includes one-off MPS setup costs
                seconds["encode_full"].append(t_full)
                seconds["encode_context"].append(t_context)
                seconds["predict"].append(t_predict)
            warm_up = False

            raw_target = full[:, N_CONTEXT:]
            target = training_target(raw_target)
            candidates = {
                "predictor": prediction,
                "copy_context": training_target(step_tokens(context, LAST_CONTEXT_STEP)).repeat(1, len(TARGET_STEPS), 1),
                "copy_full": training_target(step_tokens(full, LAST_CONTEXT_STEP)).repeat(1, len(TARGET_STEPS), 1),
                "mean_context": training_target(context.mean(dim=1, keepdim=True)).expand_as(target),
                "mean_full": training_target(full[:, :N_CONTEXT].mean(dim=1, keepdim=True)).expand_as(target),
                "zero": torch.zeros_like(target),
            }
            all_usable = all_usable and all(usable(t) for t in (full, context, prediction, target)) and (
                tuple(prediction.shape) == tuple(target.shape) == (1, N_TOKENS - N_CONTEXT, HIDDEN)
            )
            per_clip[dataset].append({
                "id": row["id"],
                "l1_per_step": {name: l1_per_step(c, target) for name, c in candidates.items()},
                "predictor_token_std": token_std(prediction),
                "target_token_std": token_std(target),
                "raw_target_token_std": token_std(raw_target),
                "predictor_l1_vs_raw_target": (prediction - raw_target).abs().mean().item(),
            })

    rng = np.random.default_rng(SEED)
    methods = list(per_clip[DATASETS[0]][0]["l1_per_step"])
    per_dataset, pooled_diffs = {}, {kind: [] for kind in BASELINES}
    for dataset, clips in per_clip.items():
        steps = {m: np.array([c["l1_per_step"][m] for c in clips]) for m in methods}  # (clips, 4)
        l1 = {m: s.mean(axis=1) for m, s in steps.items()}  # equal-size steps: mean of steps = overall mean
        comparison, diffs = compare(l1, rng)
        for kind in BASELINES:
            pooled_diffs[kind].append(diffs[kind])
        per_dataset[dataset] = {
            "comparison": comparison,
            "mean_l1": {m: sci(v.mean()) for m, v in l1.items()},
            "mean_l1_per_target_step": {m: [sci(v) for v in s.mean(axis=0)] for m, s in steps.items()},
            "token_std": {
                key: sci(np.mean([c[key] for c in clips]))
                for key in ("predictor_token_std", "target_token_std", "raw_target_token_std")
            },
            "predictor_l1_vs_raw_target": sci(np.mean([c["predictor_l1_vs_raw_target"] for c in clips])),
            "per_clip_l1": [{"id": c["id"], **{m: sci(l1[m][i]) for m in methods}} for i, c in enumerate(clips)],
        }

    pooled = {}
    for kind, parts in pooled_diffs.items():
        diffs = np.concatenate(parts)
        low, high = bootstrap_ci(diffs, rng)
        pooled[kind] = {"mean_diff_predictor_minus_baseline": sci(diffs.mean()), "ci95": [sci(low), sci(high)]}

    fingerprint = weights_fingerprint(model)
    predictor_better = {
        dataset: {kind: r["comparison"][kind]["predictor_better"] for kind in BASELINES}
        for dataset, r in per_dataset.items()
    }
    return {
        "device": DEVICE,
        "seed": SEED,
        "clips_per_dataset": CLIPS_PER_DATASET,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "leakage_note": LEAKAGE_NOTE,
        "clip_ids": {dataset: [row["id"] for row in rows] for dataset, rows in drawn.items()},
        "predictor_better": predictor_better,
        "per_dataset": per_dataset,
        "pooled_report_only": pooled,
        "seconds_median": {name: sci(float(np.median(v))) for name, v in seconds.items()},
        "timed_clips": len(seconds["predict"]),
        "all_usable": all_usable,
        "weights_fingerprint": fingerprint,
        "passed": bool(
            all_usable
            and fingerprint == REFERENCE_FINGERPRINT
            and all(all(kinds.values()) for kinds in predictor_better.values())
        ),
    }


CHECKS = {
    "predictor_path": check_predictor_path,
    "forecast": check_forecast,
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