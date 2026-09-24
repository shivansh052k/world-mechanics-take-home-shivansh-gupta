"""Checks for writing activations back into the encoder through a hook.

Run one check at a time from the repo root, with .venv active:
    python scripts/check_intervention.py <check>

Each check prints its result and stores it under its own key in results/intervention/checks.json.
"""
import argparse
import json
from collections.abc import Callable
from pathlib import Path

import torch

from vjepa_physics.activations import capture_encoder
from vjepa_physics.evidence import save_result
from vjepa_physics.intervention import edit_encoder, encoder_sites
from vjepa_physics.model import load_model, weights_fingerprint
from vjepa_physics.preprocess import preprocess_clip
from vjepa_physics.reproducibility import SEED, set_seeds
from vjepa_physics.video import load_clip

REPO = Path(__file__).resolve().parents[1]
CLIP = REPO / "data/speed/videos/scene_1000/video.mp4"  # speed 2.69 m/s, theta 230.625 deg
OUT = REPO / "results/intervention/checks.json"
DEVICE = "mps"

# In-memory weights fingerprint right after a clean load (results/model/checks.json, key "load").
REFERENCE_FINGERPRINT = "c865f524c1376e4452943b208d7d50ba588be9490604f235a3d9c9dc80804ede"

# Edits that must change nothing: a new tensor with equal values, and adding exact zeros
# (the same add path a steering edit takes).
NO_OP_EDITS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "clone": lambda hidden: hidden.clone(),
    "add_zeros": lambda hidden: hidden + torch.zeros_like(hidden),
}


# Output names in forward order: an edit at one site can only change that site and what follows.
OUTPUT_ORDER = ["embedding", *(f"block_{i}" for i in range(24)), "final_norm", "predictor"]

# Positive-control edit size: each element of the random direction has a standard deviation of
# this fraction of the edited site's own activation std (small, but far above fp32 rounding).
DELTA_RELATIVE_STD = 1e-3


def count_forward_hooks(model: torch.nn.Module) -> int:
    return sum(len(module._forward_hooks) for module in model.modules())


def run(model: torch.nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """All 27 outputs of one forward pass, fp32 on the model's device.

    Keys: "embedding", "block_0" ... "block_23" (own hooks), "final_norm" (encoder
    last_hidden_state) and "predictor" (predictor output). Raises on non-finite or all-zero outputs.
    """
    with capture_encoder(model) as acts, torch.inference_mode():
        out = model(pixel_values_videos=x)
    if out.predictor_output is None:
        raise RuntimeError("model returned no predictor output")
    outputs = dict(acts) | {
        "final_norm": out.last_hidden_state,
        "predictor": out.predictor_output.last_hidden_state,
    }
    for name, t in outputs.items():
        if not torch.isfinite(t).all() or not t.any():
            raise RuntimeError(f"{name}: degenerate output (non-finite or all zero)")
    return outputs


def identical_to(outputs: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]) -> dict[str, bool]:
    return {name: bool(torch.equal(outputs[name], reference[name])) for name in reference}


def counted(edit: Callable[[torch.Tensor], torch.Tensor], calls: list[int]) -> Callable[[torch.Tensor], torch.Tensor]:
    """Wrap `edit` so every call is recorded in `calls` (proves the hook actually fired)."""
    def wrapped(hidden: torch.Tensor) -> torch.Tensor:
        calls.append(1)
        return edit(hidden)
    return wrapped


def sci(value: float) -> float:
    """Round to 4 significant digits, for readable JSON."""
    return float(f"{value:.4g}")


def rel_change(a: torch.Tensor, reference: torch.Tensor) -> float:
    """Relative Frobenius difference of `a` from `reference`."""
    return sci(((a - reference).norm() / reference.norm()).item())


def random_delta(reference: torch.Tensor, seed: int) -> torch.Tensor:
    """(1024,) seeded Gaussian vector, std = DELTA_RELATIVE_STD x std of `reference`, on its device.

    A random direction, not a constant shift: LayerNorm subtracts each token's feature mean,
    so a constant shift would be cancelled downstream and the control would say nothing.
    """
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn(reference.shape[-1], generator=generator, dtype=reference.dtype)
    return (z * DELTA_RELATIVE_STD * reference.std().item()).to(reference.device)


def append_edit(module: torch.nn.Module, delta: torch.Tensor):
    """Register the same `+ delta` edit the ordinary way (appended, no prepend): the negative control."""
    def hook(module: torch.nn.Module, args: tuple, output):
        if isinstance(output, tuple):
            return (output[0] + delta, *output[1:])
        return output + delta
    return module.register_forward_hook(hook)


def check_noop() -> dict:
    """Writing back an unchanged activation at any encoder site leaves all 27 outputs bit-identical.

    Passes if: a plain repeat is bit-exact; for every site (embedding, block_0 ... block_23) and
    both no-op edits (clone, add zeros), the hook fires once and every output equals the
    unhooked baseline exactly; afterwards the forward-hook count is unchanged, a plain run still
    equals the baseline, and the weights fingerprint equals the clean-load reference.
    """
    set_seeds()
    model, _ = load_model(DEVICE, verify_file=False)
    x = preprocess_clip(load_clip(CLIP)).unsqueeze(0).to(DEVICE)
    hooks_before = count_forward_hooks(model)

    baseline = run(model, x)
    repeat_identical = all(identical_to(run(model, x), baseline).values())

    per_site: dict[str, dict] = {}
    failures: list[str] = []
    for site in encoder_sites(model):
        per_site[site] = {}
        for edit_name, edit in NO_OP_EDITS.items():
            calls: list[int] = []
            with edit_encoder(model, site, counted(edit, calls)):
                matches = identical_to(run(model, x), baseline)
            per_site[site][edit_name] = {"identical": all(matches.values()), "hook_calls": len(calls)}
            failures += [f"{site}/{edit_name}/{name}" for name, ok in matches.items() if not ok]
            if len(calls) != 1:
                failures.append(f"{site}/{edit_name}/hook_calls={len(calls)}")

    after_identical = all(identical_to(run(model, x), baseline).values())
    hooks_after = count_forward_hooks(model)
    fingerprint = weights_fingerprint(model)
    return {
        "clip": str(CLIP.relative_to(REPO)),
        "device": DEVICE,
        "sites": len(per_site),
        "edits": list(NO_OP_EDITS),
        "repeat_identical": repeat_identical,
        "per_site": per_site,
        "failures": failures,
        "plain_run_after_edits_identical": after_identical,
        "forward_hooks_before_and_after": [hooks_before, hooks_after],
        "weights_fingerprint": fingerprint,
        "passed": bool(
            repeat_identical
            and len(per_site) == 25
            and not failures
            and after_identical
            and hooks_before == hooks_after
            and fingerprint == REFERENCE_FINGERPRINT
        ),
    }


def check_positive_control() -> dict:
    """An edit written back at a site changes exactly that site and everything after it.

    Edit: add a seeded random vector delta (one per site, same for every token). Passes if, for
    every site: delta is non-zero; every output before the site equals the baseline exactly; the
    output at the site equals baseline + delta exactly (so the write is what later hooks see);
    every output after the site, including final_norm and predictor, differs from the baseline;
    and the hook fires once. Afterwards: forward-hook count unchanged, a plain run equals the
    baseline, weights fingerprint equals the clean-load reference. Change sizes are observations.
    """
    set_seeds()
    model, _ = load_model(DEVICE, verify_file=False)
    x = preprocess_clip(load_clip(CLIP)).unsqueeze(0).to(DEVICE)
    hooks_before = count_forward_hooks(model)

    baseline = run(model, x)
    order_ok = list(baseline) == OUTPUT_ORDER

    per_site: dict[str, dict] = {}
    failures: list[str] = []
    for site in encoder_sites(model):
        position = OUTPUT_ORDER.index(site)
        before, after = OUTPUT_ORDER[:position], OUTPUT_ORDER[position + 1:]
        delta = random_delta(baseline[site], SEED)
        calls: list[int] = []
        # The lambda is only called inside this iteration's context, so it always sees this delta.
        with edit_encoder(model, site, counted(lambda hidden: hidden + delta, calls)):
            edited = run(model, x)

        result = {
            "delta_nonzero": bool(delta.any()),
            "upstream_identical": all(torch.equal(edited[n], baseline[n]) for n in before),
            "site_equals_baseline_plus_delta": bool(torch.equal(edited[site], baseline[site] + delta)),
            "unchanged_downstream": [n for n in after if torch.equal(edited[n], baseline[n])],
            "hook_calls": len(calls),
            "site_rel_change": rel_change(edited[site], baseline[site]),
            "min_downstream_rel_change": min(rel_change(edited[n], baseline[n]) for n in after),
            "final_norm_rel_change": rel_change(edited["final_norm"], baseline["final_norm"]),
            "predictor_rel_change": rel_change(edited["predictor"], baseline["predictor"]),
        }
        per_site[site] = result
        if not (
            result["delta_nonzero"]
            and result["upstream_identical"]
            and result["site_equals_baseline_plus_delta"]
            and not result["unchanged_downstream"]
            and result["hook_calls"] == 1
        ):
            failures.append(site)
        del edited

    after_identical = all(identical_to(run(model, x), baseline).values())
    hooks_after = count_forward_hooks(model)
    fingerprint = weights_fingerprint(model)
    return {
        "clip": str(CLIP.relative_to(REPO)),
        "device": DEVICE,
        "delta_relative_std": DELTA_RELATIVE_STD,
        "seed": SEED,
        "output_order_ok": order_ok,
        "per_site": per_site,
        "failures": failures,
        "plain_run_after_edits_identical": after_identical,
        "forward_hooks_before_and_after": [hooks_before, hooks_after],
        "weights_fingerprint": fingerprint,
        "passed": bool(
            order_ok
            and len(per_site) == 25
            and not failures
            and after_identical
            and hooks_before == hooks_after
            and fingerprint == REFERENCE_FINGERPRINT
        ),
    }


def check_hook_order() -> dict:
    """The edit runs before hooks registered earlier on the same module, so they record the edited value.

    Setup: transformers' hidden_states hooks are installed first (by one output_hidden_states
    call), and capture_encoder is opened before edit_encoder, so every recording hook is
    registered before the edit hook. Passes if: hidden_states of a plain call line up with the
    baseline (index alignment); at every site our capture and hidden_states both equal
    baseline + delta exactly and the hook fires once; negative control: the same edit registered
    without prepend leaves our capture at the site equal to the baseline (so the check can tell
    the two orders apart); afterwards the hook count is unchanged, a plain run equals the
    baseline, and the weights fingerprint equals the clean-load reference.
    """
    set_seeds()
    model, _ = load_model(DEVICE, verify_file=False)
    x = preprocess_clip(load_clip(CLIP)).unsqueeze(0).to(DEVICE)
    baseline = run(model, x)

    with torch.inference_mode():  # installs transformers' permanent hidden_states hooks
        plain = model(pixel_values_videos=x, skip_predictor=True, output_hidden_states=True)
    plain_hidden = plain.hidden_states or ()
    alignment_ok = len(plain_hidden) == 25 and all(
        torch.equal(h, baseline[name]) for h, name in zip(plain_hidden, OUTPUT_ORDER)
    )
    del plain, plain_hidden
    hooks_before = count_forward_hooks(model)

    per_site: dict[str, dict] = {}
    failures: list[str] = []
    for index, (site, module) in enumerate(encoder_sites(model).items()):
        delta = random_delta(baseline[site], SEED)
        expected = baseline[site] + delta

        calls: list[int] = []
        with capture_encoder(model) as acts:  # registered before the edit hook
            # The lambda is only called inside this iteration's context, so it always sees this delta.
            with edit_encoder(model, site, counted(lambda hidden: hidden + delta, calls)), torch.inference_mode():
                out = model(pixel_values_videos=x, skip_predictor=True, output_hidden_states=True)
        hidden = out.hidden_states or ()
        result = {
            "hidden_states_index": index,
            "capture_sees_edit": bool(torch.equal(acts[site], expected)),
            "hidden_states_sees_edit": len(hidden) == 25 and bool(torch.equal(hidden[index], expected)),
            "hook_calls": len(calls),
        }
        del out, hidden

        with capture_encoder(model) as acts:
            handle = append_edit(module, delta)
            try:
                with torch.inference_mode():
                    model(pixel_values_videos=x, skip_predictor=True)
            finally:
                handle.remove()
        result["appended_edit_capture_unedited"] = bool(torch.equal(acts[site], baseline[site]))

        per_site[site] = result
        if not (
            result["capture_sees_edit"]
            and result["hidden_states_sees_edit"]
            and result["hook_calls"] == 1
            and result["appended_edit_capture_unedited"]
        ):
            failures.append(site)

    after_identical = all(identical_to(run(model, x), baseline).values())
    hooks_after = count_forward_hooks(model)
    fingerprint = weights_fingerprint(model)
    return {
        "clip": str(CLIP.relative_to(REPO)),
        "device": DEVICE,
        "seed": SEED,
        "hidden_states_alignment_ok": alignment_ok,
        "per_site": per_site,
        "failures": failures,
        "plain_run_after_edits_identical": after_identical,
        "forward_hooks_before_and_after": [hooks_before, hooks_after],
        "weights_fingerprint": fingerprint,
        "passed": bool(
            alignment_ok
            and len(per_site) == 25
            and not failures
            and after_identical
            and hooks_before == hooks_after
            and fingerprint == REFERENCE_FINGERPRINT
        ),
    }
        

CHECKS = {
    "noop": check_noop,
    "positive_control": check_positive_control,
    "hook_order": check_hook_order,
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