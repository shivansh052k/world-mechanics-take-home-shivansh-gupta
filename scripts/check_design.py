"""Checks for the dataset design: label grids, clips per value, and balance of the design factors.

Run one check at a time from the repo root, with .venv active:
    python scripts/check_design.py <check>

Each check prints its result and stores it under its own key in results/design/checks.json.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
from scipy import stats

import numpy as np

from vjepa_physics.data import DATASETS, LABEL_FIELD, load_dataset
from vjepa_physics.evidence import save_result

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = REPO / "results/design/checks.json"

# DATA.md: 1,500 + 1,536 + 1,536 clips; 64 label values per dataset; speed and acceleration ranges;
# 64 equally spaced directions in [0, 360).
EXPECTED_CLIPS = 4572
N_VALUES = 64
LABEL_RANGE = {"speed": (0.25, 4.0), "acceleration": (0.25, 10.0)}
DIRECTION_STEP = 360 / N_VALUES
TOLERANCE = 1e-9  # float slack for comparing generated values with DATA.md's round numbers


def sci(value: float) -> float:
    """Round to 4 significant digits, for readable JSON."""
    return float(f"{value:.4g}")


def check_value_grids() -> dict:
    """Each dataset's labels form the grid DATA.md states.

    Passes if: every dataset has exactly 64 distinct label values; speed spans exactly 0.25 to 4.0
    and acceleration 0.25 to 10.0; the 64 directions lie in [0, 360) and every gap between
    neighbouring angles around the circle (including the wrap from the last back to the first)
    is 360 / 64 = 5.625 degrees; and all 4,572 clips are counted (all within 1e-9). The full grids,
    their spacing, and clips per value are diagnostics.
    """
    per_dataset, total = {}, 0
    for dataset in DATASETS:
        values = np.array([clip[LABEL_FIELD[dataset]] for clip in load_dataset(DATA, dataset)], dtype=float)
        total += len(values)
        grid = np.unique(values)  # sorted distinct values
        per_value = Counter(values.tolist())
        histogram = Counter(per_value.values())
        usual = histogram.most_common(1)[0][0]

        criteria = {"sixty_four_values": len(grid) == N_VALUES}
        if dataset == "direction":
            gaps = np.diff(np.append(grid, grid[0] + 360))
            criteria["direction_equally_spaced"] = bool(
                len(grid) == N_VALUES
                and grid[0] >= 0
                and grid[-1] < 360
                and np.all(np.abs(gaps - DIRECTION_STEP) <= TOLERANCE)
            )
            spacing = {"gap_min": float(gaps.min()), "gap_max": float(gaps.max())}
        else:
            low, high = LABEL_RANGE[dataset]
            criteria["range_endpoints"] = bool(abs(grid[0] - low) <= TOLERANCE and abs(grid[-1] - high) <= TOLERANCE)
            steps = np.diff(grid)
            linear = np.linspace(grid[0], grid[-1], len(grid))
            spacing = {
                "step_min": float(steps.min()),
                "step_max": float(steps.max()),
                "step_if_linear": float((grid[-1] - grid[0]) / (len(grid) - 1)),
                "max_deviation_from_linear_grid": sci(float(np.abs(grid - linear).max())),
            }

        per_dataset[dataset] = {
            "criteria": criteria,
            "label": LABEL_FIELD[dataset],
            "clips": len(values),
            "distinct_values": len(grid),
            "grid": grid.tolist(),
            "spacing": spacing,
            "clips_per_value_histogram": {str(k): v for k, v in sorted(histogram.items())},
            "values_not_at_most_common_count": sorted(v for v, n in per_value.items() if n != usual),
        }

    criteria = {"all_clips_counted": total == EXPECTED_CLIPS}
    return {
        "criteria": criteria,
        "clips": total,
        "per_dataset": per_dataset,
        "passed": all(criteria.values())
        and all(all(r["criteria"].values()) for r in per_dataset.values()),
    }
    

def histogram(counter: Counter) -> dict[str, int]:
    """{count: how many keys have that count}, as a JSON-friendly dict."""
    return {str(k): v for k, v in sorted(Counter(counter.values()).items())}


def direction_group(clip: dict) -> str:
    """Direction-set motion group: motion type and its magnitude."""
    if clip["motion"] == "velocity":
        return f"velocity {clip['speed_mps']:g} m/s"
    return f"acceleration {clip['acceleration_mps2']:g} m/s^2"


def group_order(name: str) -> tuple[bool, float]:
    """Velocity groups first, each motion type by magnitude."""
    motion, magnitude, _ = name.split()
    return motion != "velocity", float(magnitude)


def check_design_balance() -> dict:
    """Diagnostic: how the design factors are crossed and balanced. No pass/fail (DATA.md states none of this).

    Speed and acceleration sets: the angle grid and clips per angle; whether every (value, angle)
    pair is unique; distinct angles per value and values per angle; whether both sets give each id
    the same angle; Pearson correlation of the label with cos and sin of the angle.
    Direction set: motion groups (motion type x magnitude) and their counts; the angle x group
    table with a chi-square test of independence and Cramer's V (expected counts are ~2 per cell,
    so the p-value is approximate); the octant x group table used for stratification; velocity
    clips per angle; correlation of speed / acceleration with cos and sin within each motion type.
    """
    clips = {dataset: load_dataset(DATA, dataset) for dataset in DATASETS}
    result: dict = {}

    for dataset in ("speed", "acceleration"):
        rows = clips[dataset]
        value = np.array([c["magnitude"] for c in rows])
        theta = np.array([c["theta_degrees"] for c in rows])
        angles = np.unique(theta)
        pairs = Counter(zip(value.tolist(), theta.tolist()))
        gaps = np.diff(np.append(angles, angles[0] + 360))
        result[dataset] = {
            "distinct_angles": len(angles),
            "angle_gap_min_max": [float(gaps.min()), float(gaps.max())],
            "clips_per_angle_histogram": histogram(Counter(theta.tolist())),
            "every_value_angle_pair_unique": len(pairs) == len(rows),
            "distinct_angles_per_value_histogram": histogram(Counter(v for v, _ in pairs)),
            "distinct_values_per_angle_histogram": histogram(Counter(t for _, t in pairs)),
            "corr_label_cos_theta": sci(stats.pearsonr(value, np.cos(np.deg2rad(theta))).statistic),
            "corr_label_sin_theta": sci(stats.pearsonr(value, np.sin(np.deg2rad(theta))).statistic),
        }

    speed_theta = [c["theta_degrees"] for c in clips["speed"]]
    accel_theta = [c["theta_degrees"] for c in clips["acceleration"]]
    result["speed_vs_acceleration_same_angle_by_id"] = {
        "clips_compared": min(len(speed_theta), len(accel_theta)),
        "same_length": len(speed_theta) == len(accel_theta),
        "mismatches": sum(a != b for a, b in zip(speed_theta, accel_theta)),
    }

    rows = clips["direction"]
    theta = np.array([c["theta_degrees"] for c in rows])
    groups = [direction_group(c) for c in rows]
    names = sorted(set(groups), key=group_order)
    angles = np.unique(theta)
    angle_index = {a: i for i, a in enumerate(angles.tolist())}
    table = np.zeros((len(angles), len(names)), dtype=int)
    octants = np.zeros((8, len(names)), dtype=int)
    for t, g in zip(theta.tolist(), groups):
        table[angle_index[t], names.index(g)] += 1
        octants[int(t // 45), names.index(g)] += 1
    chi = stats.chi2_contingency(table)
    velocity_per_angle = Counter(t for t, c in zip(theta.tolist(), rows) if c["motion"] == "velocity")

    def corr_within(motion: str, field: str) -> dict:
        sel = [c for c in rows if c["motion"] == motion]
        mag = np.array([c[field] for c in sel])
        rad = np.deg2rad([c["theta_degrees"] for c in sel])
        return {
            "cos": sci(stats.pearsonr(mag, np.cos(rad)).statistic),
            "sin": sci(stats.pearsonr(mag, np.sin(rad)).statistic),
        }

    result["direction"] = {
        "group_counts": {name: groups.count(name) for name in names},
        "angle_by_group": {
            "shape": list(table.shape),
            "cell_min": int(table.min()),
            "cell_max": int(table.max()),
            "zero_cells": int((table == 0).sum()),
            "chi2": sci(chi.statistic),
            "dof": int(chi.dof),
            "p_value_approximate": sci(chi.pvalue),
            "cramers_v": sci(float(np.sqrt(chi.statistic / (table.sum() * (min(table.shape) - 1))))),
        },
        "octant_by_group": {
            "octants_deg": [f"[{45 * k}, {45 * (k + 1)})" for k in range(8)],
            "counts_per_group": {name: octants[:, j].tolist() for j, name in enumerate(names)},
            "cell_min": int(octants.min()),
            "cell_max": int(octants.max()),
        },
        "velocity_clips_per_angle_min_max": [min(velocity_per_angle.values()), max(velocity_per_angle.values())],
        "angles_without_velocity_clips": len(angles) - len(velocity_per_angle),
        "corr_speed_with_angle_velocity_clips": corr_within("velocity", "speed_mps"),
        "corr_acceleration_with_angle_acceleration_clips": corr_within("acceleration", "acceleration_mps2"),
    }
    return result

CHECKS = {
    "value_grids": check_value_grids,
    "design_balance": check_design_balance,
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