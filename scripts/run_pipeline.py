"""Run the analysis pipeline, or any stage of it, in dependency order.

Each step is one saved check, `python scripts/check_<subject>.py <key>`, which writes its result with provenance to
results/<subject>/checks.json. Stages only read what earlier stages wrote. Some checks failed when they were recorded
and are kept on record on purpose; the runner therefore compares every new outcome with the recorded one, continues
past failures, and stops only when a check crashes before saving a result.

Usage:
    python scripts/run_pipeline.py --list
    python scripts/run_pipeline.py --stage probing --dry-run
    python scripts/run_pipeline.py --stage all
    python scripts/run_pipeline.py --stage figures

Needs a git clone (every result records its commit and code hashes), the supplied data in data/, and, from the
extraction stage on, the model in the Hugging Face cache and about 45 GB of free disk. A full run takes hours.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from vjepa_physics.evidence import require_clean_code

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
VARIABLES = ("direction", "speed", "acceleration")
HALVES = ("seen", "unseen")


def steps(subject: str, keys: str) -> list[tuple[str, str]]:
    """(subject, key) pairs for one script, in run order."""
    return [(subject, key) for key in keys.split()]


# stage -> (what it does, its steps in dependency order)
STAGES: dict[str, tuple[str, list[tuple[str, str]]]] = {
    "setup": (
        "verify loader, preprocessing, model, hooks, numerics, edits, predictor (nothing later reads it)",
        steps("video_loader", "inspect load repeat colour order opencv opencv_bicubic opencv_diff_stats "
                              "opencv_tolerance figure")
        + steps("preprocessing", "config manual identity default")
        + steps("model", "config load fingerprint seeds")
        + steps("forward", "forward")
        + steps("activations", "hidden_states token_layout")
        + steps("numerics", "repeat batch devices precision attention")
        + steps("intervention", "noop positive_control hook_order")
        + steps("forecast", "predictor_path forecast")
        + steps("evidence", "dirty_flag"),
    ),
    "audit": (
        "data audit: files, metadata, design, videos, disk tracking, per-clip flags",
        steps("data_files", "manifests fingerprint")
        + steps("metadata", "consistency documented_fields sorted_by_label")
        + steps("design", "value_grids design_balance start_positions label_independence distance_confound "
                          "figure_design")
        + steps("videos", "format uniform_frames duplicates decoders figure_contact_sheet")
        + steps("tracking", "track mapping documented_colour flags figure_tracking"),
    ),
    "extraction": (
        "splits, pooled activations at 26 sites for every clip (about 1 h 45 min on MPS), joined tables",
        steps("splits", "build balance")
        + steps("extraction", "pipeline extract_direction extract_speed extract_acceleration verify")
        + steps("joined", "build storage"),
    ),
    "probing": (
        "layer-wise probes, pixel floor, physics ceiling, bootstrap, one-time test, per-patch direction (39 GB)",
        steps("probes", "layer_curves shuffled_labels")
        + steps("baselines", "pixel_grams pixel_floor physics_ceiling")
        + steps("layer_curves", "bootstrap figure_layer_curves test_scores")
        + steps("patches", "extract_direction verify patch_probes patch_alpha_diagnostic patch_breakdown "
                           "patch_bootstrap spatial_generalization position_baseline patch_test_scores "
                           "figure_local_to_global"),
    ),
    "nullspace": (
        "iterative nullspace probing, controls, linear erasure, kernel recovery, one-time test",
        steps("nullspace", "nullspace_rounds leak_diagnostic covariance_exhaustion random_subspaces pc_subspaces "
                           "depth_profile fresh_probe_erasure alpha_sweep kernel_erasure kernel_shuffled_labels "
                           "kernel_grid_diagnostic kernel_hat_gap nullspace_test_scores figure_nullspace "
                           "figure_erasure"),
    ),
    "steering": (
        "multi-probe subspace steering on held-out clips (six runs on MPS) and its analyses",
        steps("steering", "steering_setup steering_cache")
        + [("steering", f"steer_{v}_{h}") for v in VARIABLES for h in HALVES]
        + steps("steering", "steering_scores steering_propagation steering_specificity steering_kernel "
                            "figure_steering"),
    ),
    "spline": (
        "activation manifolds, behavior readouts and isometry, spline steering (six runs on MPS), comparison",
        steps("manifolds", "manifold_loco speed_acceleration_manifold manifold_ladder manifold_dimension "
                           "direction_harmonics figure_manifolds")
        + steps("behavior", "behavior_readouts isometry isometry_local")
        + steps("spline_steering", "spline_setup")
        + [("spline_steering", f"spline_{v}_{h}") for v in VARIABLES for h in HALVES]
        + steps("spline_steering", "spline_scores spline_naturalness spline_held_out comparison_table "
                                   "figure_spline_paths figure_spline_profile"),
    ),
    "confounds": (
        "speed vs acceleration confound, motion-type transfer, clip flags, subspace overlap",
        steps("confounds", "cross_applied_probes matched_distance_classifier figure_confounds")
        + steps("robustness", "motion_type_transfer motion_type_test flag_breakdown figure_tubelet flag_test "
                              "subspace_overlap"),
    ),
    "final": (
        "data unchanged after the full pipeline; every saved result traced to committed code",
        steps("data_files", "fingerprint_final") + steps("evidence", "code_hash_check"),
    ),
}

# every step that writes a figure, in pipeline order; most read artifacts from the audit and extraction stages
FIGURE_STEPS = [s for _, stage in STAGES.values() for s in stage if s[1].startswith("figure")
                or s == ("preprocessing", "default")]


def recorded(subject: str, key: str) -> dict | None:
    """The saved record of one check, or None if it was never saved."""
    path = REPO / "results" / subject / "checks.json"
    return json.loads(path.read_text()).get(key) if path.exists() else None


def outcome(record: dict | None) -> str:
    """'passed', 'failed', 'no verdict' (a diagnostic without a pass criterion) or 'not recorded'."""
    if record is None:
        return "not recorded"
    passed = record["result"].get("passed")
    return "no verdict" if passed is None else "passed" if passed else "failed"


def run_step(subject: str, key: str) -> tuple[str, str, str]:
    """Run one check; return (recorded outcome, new outcome, status). A check that exits without saving a new
    record (its UTC stamp unchanged) crashed; one that saved a result counts even if it failed its criteria."""
    before = recorded(subject, key)
    code = subprocess.run([sys.executable, str(SCRIPTS / f"check_{subject}.py"), key], cwd=REPO).returncode
    after = recorded(subject, key)
    if after is None or (before is not None and after["provenance"]["utc"] == before["provenance"]["utc"]):
        return outcome(before), f"nothing saved (exit {code})", "crashed"
    expected, got = outcome(before), outcome(after)
    return expected, got, "as recorded" if expected in (got, "not recorded") else "differs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="print the stages and exit")
    parser.add_argument("--stage", nargs="+", choices=[*STAGES, "all", "figures"], help="stages to run, in order")
    parser.add_argument("--dry-run", action="store_true", help="print each step and its recorded outcome; run nothing")
    args = parser.parse_args()

    if args.list or not args.stage:
        for name, (about, stage) in STAGES.items():
            print(f"{name:<11}{len(stage):>4} steps  {about}")
        print(f"{'figures':<11}{len(FIGURE_STEPS):>4} steps  every figure from saved results (most need the audit "
              f"and extraction artifacts)")
        return

    names = [n for s in args.stage for n in (STAGES if s == "all" else [s])]  # "all" expands in place
    selected = [s for name in names for s in (FIGURE_STEPS if name == "figures" else STAGES[name][1])]
    if not args.dry_run:
        require_clean_code()  # every saved result must trace to a commit holding exactly its code

    rows = []
    for i, (subject, key) in enumerate(selected, 1):
        command = f"python scripts/check_{subject}.py {key}"
        if args.dry_run:
            print(f"[{i:>3}/{len(selected)}] {command:<62} recorded: {outcome(recorded(subject, key))}")
            continue
        print(f"\n[{i}/{len(selected)}] {command}", flush=True)
        expected, got, status = run_step(subject, key)
        rows.append((status, command, expected, got))
        if status == "crashed":
            print(f"stopped: {command} saved nothing; later steps may read its output")
            break

    if rows:
        print("\nSummary")
        for status, command, expected, got in rows:
            print(f"  {status:<12}{command:<62} recorded {expected:<13} now {got}")
        if any(status != "as recorded" for status, *_ in rows):
            raise SystemExit("some steps crashed or differ from their recorded outcome (see the summary)")


if __name__ == "__main__":
    main()