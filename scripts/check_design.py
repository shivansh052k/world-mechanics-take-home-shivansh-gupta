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
from vjepa_physics.geometry import distance_travelled, frame_times, speed_at
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

# Flag rule for the start-position diagnostics, fixed before running (DATA.md sets no criterion):
# a test is flagged when p < FLAG_P (about 4 sigma), so ~50 tests in the step give almost no chance flags.
FLAG_P = 1e-4

# Pixel scale: 32 px per metre (verified on one clip; checked on every clip by disk tracking).
PX_PER_M = 32

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

def check_start_positions() -> dict:
    """Diagnostic: how start positions are spread in each dataset. No pass/fail (DATA.md states nothing).

    Per dataset: range and mean of x and y; whether all start positions are distinct;
    Kolmogorov-Smirnov test of x and of y against a uniform distribution on their observed range
    (approximate, since the range comes from the same data); correlation of x with y; counts per
    quadrant around the centre. A test is flagged if p < FLAG_P.
    """
    result: dict = {"flag_rule": f"p < {FLAG_P}"}
    for dataset in DATASETS:
        start = np.array([c["start_position_xy_m"] for c in load_dataset(DATA, dataset)], dtype=float)
        x, y = start[:, 0], start[:, 1]
        entry: dict = {
            "clips": len(start),
            "distinct_positions": len({tuple(p) for p in start.tolist()}),
        }
        for name, v in (("x", x), ("y", y)):
            low, high = float(v.min()), float(v.max())
            ks = stats.kstest(v, stats.uniform(loc=low, scale=high - low).cdf)
            entry[name] = {
                "min": sci(low),
                "max": sci(high),
                "mean": sci(float(v.mean())),
                "ks_uniform_statistic": sci(ks.statistic),
                "ks_p": sci(ks.pvalue),
                "flagged": bool(ks.pvalue < FLAG_P),
            }
        r = stats.pearsonr(x, y)
        entry["corr_x_y"] = {"r": sci(r.statistic), "p": sci(r.pvalue), "flagged": bool(r.pvalue < FLAG_P)}
        quadrants = Counter(f"{'+' if a >= 0 else '-'}x {'+' if b >= 0 else '-'}y" for a, b in start.tolist())
        entry["quadrant_counts"] = dict(sorted(quadrants.items()))
        result[dataset] = entry
    return result


def corr_entry(a: np.ndarray, b: np.ndarray) -> dict:
    """Pearson r and its p-value, flagged if p < FLAG_P."""
    r = stats.pearsonr(a, b)
    return {"r": sci(r.statistic), "p": sci(r.pvalue), "flagged": bool(r.pvalue < FLAG_P)}


def check_label_independence() -> dict:
    """Diagnostic: does where the disk starts depend on its labels? No pass/fail (DATA.md states nothing).

    Start features: x, y, |x|, |y|, distance from the centre, and the start position projected
    along the direction of motion (x cos(theta) + y sin(theta); negative = starts behind the centre
    relative to where it moves) and across it. Per dataset:
    - Pearson correlation of x, y, |x|, |y|, radius with the magnitude (speed / acceleration sets;
      within each motion type in the direction set) and with cos(theta), sin(theta);
    - correlation of the along / across projections with the magnitude, and a one-sample t-test of
      their mean against 0 (a generator keeping the disk in frame would start it behind);
    - one-way ANOVA of x and of y across the 64 label values (catches non-linear dependence).
    A test is flagged if p < FLAG_P. Later positions depend on the labels by construction; that is
    the distance confound, measured separately.
    """
    result: dict = {"flag_rule": f"p < {FLAG_P}"}
    for dataset in DATASETS:
        clips = load_dataset(DATA, dataset)
        x, y = np.array([c["start_position_xy_m"] for c in clips], dtype=float).T
        theta = np.deg2rad([c["theta_degrees"] for c in clips])
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        label = np.array([c[LABEL_FIELD[dataset]] for c in clips], dtype=float)
        everyone = np.ones(len(clips), dtype=bool)

        features = {"x": x, "y": y, "abs_x": np.abs(x), "abs_y": np.abs(y), "radius": np.hypot(x, y)}
        projections = {"along_motion": x * cos_t + y * sin_t, "across_motion": -x * sin_t + y * cos_t}
        if dataset == "direction":
            motion = np.array([c["motion"] for c in clips])
            magnitude = np.array([c["speed_mps"] + c["acceleration_mps2"] for c in clips])  # the other is 0
            magnitude_targets = {
                "speed_within_velocity_clips": (magnitude, motion == "velocity"),
                "acceleration_within_acceleration_clips": (magnitude, motion == "acceleration"),
            }
        else:
            magnitude_targets = {"magnitude": (label, everyone)}
        targets = magnitude_targets | {"cos_theta": (cos_t, everyone), "sin_theta": (sin_t, everyone)}

        correlations = {
            f"{f}_vs_{t}": corr_entry(fv[mask], tv[mask])
            for f, fv in features.items()
            for t, (tv, mask) in targets.items()
        }
        correlations |= {
            f"{p}_vs_{t}": corr_entry(pv[mask], tv[mask])
            for p, pv in projections.items()
            for t, (tv, mask) in magnitude_targets.items()
        }

        mean_tests = {}
        for p, pv in projections.items():
            test = stats.ttest_1samp(pv, 0.0)
            mean_tests[p] = {
                "mean_m": sci(float(pv.mean())),
                "t": sci(test.statistic),
                "p": sci(test.pvalue),
                "flagged": bool(test.pvalue < FLAG_P),
            }

        anova = {}
        grid = np.unique(label)
        for name, v in (("x", x), ("y", y)):
            test = stats.f_oneway(*(v[label == g] for g in grid))
            anova[name] = {
                "groups": len(grid),
                "F": sci(test.statistic),
                "p": sci(test.pvalue),
                "flagged": bool(test.pvalue < FLAG_P),
            }

        flagged = (
            [k for k, e in correlations.items() if e["flagged"]]
            + [f"mean_{k}" for k, e in mean_tests.items() if e["flagged"]]
            + [f"anova_{k}" for k, e in anova.items() if e["flagged"]]
        )
        result[dataset] = {
            "clips": len(clips),
            "tests": len(correlations) + len(mean_tests) + len(anova),
            "flagged": flagged,
            "correlations": correlations,
            "projection_mean_tests": mean_tests,
            "anova_by_label_value": anova,
        }
    return result

def kinematics(clips: list[dict]) -> dict[str, np.ndarray]:
    """Per clip, over the whole clip (frame 0 to the last frame): distance travelled (m), mean speed
    (distance / duration, m/s) and final speed (m/s)."""
    duration = np.array([frame_times(c["fps"], c["frames"])[-1] for c in clips])
    v = np.array([c["speed_mps"] for c in clips])
    a = np.array([c["acceleration_mps2"] for c in clips])
    distance = distance_travelled(v, a, duration)
    return {
        "duration_s": duration,
        "distance_m": distance,
        "mean_speed_mps": distance / duration,
        "final_speed_mps": speed_at(v, a, duration),
}


def min_max(values: np.ndarray, scale: float = 1.0) -> list[float]:
    return [sci(float(values.min()) * scale), sci(float(values.max()) * scale)]


def check_distance_confound() -> dict:
    """Diagnostic: how the magnitude labels are tied to distance travelled. No pass/fail.

    From metadata, with s = v t + a t^2 / 2 over the clip (frame k at t = k / fps). Speed and
    acceleration sets: correlation of the label with distance, mean speed and final speed, and the
    distance / label ratio (constant = exactly proportional). Across the two sets: the distance
    window both cover, the clips and values inside it, and for each acceleration value inside it the
    gap to the nearest speed value by distance (how closely the grids can be matched). Direction set:
    distance and final speed per motion group, and the correlation of motion type with distance.
    Distances also in pixels (PX_PER_M).
    """
    clips = {dataset: load_dataset(DATA, dataset) for dataset in DATASETS}
    kin = {dataset: kinematics(rows) for dataset, rows in clips.items()}
    result: dict = {"px_per_m": PX_PER_M}

    for dataset in ("speed", "acceleration"):
        label = np.array([c["magnitude"] for c in clips[dataset]])
        k = kin[dataset]
        ratio = k["distance_m"] / label
        result[dataset] = {
            "durations_s": sorted(set(k["duration_s"].tolist())),
            "corr_label_with": {
                q: sci(stats.pearsonr(label, k[q]).statistic)
                for q in ("distance_m", "mean_speed_mps", "final_speed_mps")
            },
            "distance_over_label_min_max": [float(ratio.min()), float(ratio.max())],
            "distance_m_min_max": min_max(k["distance_m"]),
            "distance_px_min_max": min_max(k["distance_m"], PX_PER_M),
        }

    ds, da = kin["speed"]["distance_m"], kin["acceleration"]["distance_m"]
    low, high = max(ds.min(), da.min()), min(ds.max(), da.max())
    inside_s = (ds >= low - TOLERANCE) & (ds <= high + TOLERANCE)
    inside_a = (da >= low - TOLERANCE) & (da <= high + TOLERANCE)
    speed_distances = np.unique(ds)
    accel_distances = np.unique(da[inside_a])
    gaps = np.array([np.abs(speed_distances - d).min() for d in accel_distances])
    result["speed_vs_acceleration_overlap"] = {
        "window_m": [float(low), float(high)],
        "window_px": [float(low) * PX_PER_M, float(high) * PX_PER_M],
        "speed_clips_inside": int(inside_s.sum()),
        "acceleration_clips_inside": int(inside_a.sum()),
        "speed_values_inside": len(np.unique(ds[inside_s])),
        "acceleration_values_inside": len(accel_distances),
        "nearest_speed_value_distance_gap_m": (
            {"median": sci(float(np.median(gaps))), "max": sci(float(gaps.max()))} if gaps.size else None
        ),
        "nearest_gap_px_max": sci(float(gaps.max()) * PX_PER_M) if gaps.size else None,
    }

    rows, k = clips["direction"], kin["direction"]
    groups = [direction_group(c) for c in rows]
    per_group = {}
    for name in sorted(set(groups), key=group_order):
        sel = np.array([g == name for g in groups])
        per_group[name] = {
            "clips": int(sel.sum()),
            "distance_m_min_max": min_max(k["distance_m"][sel]),
            "final_speed_mps_min_max": min_max(k["final_speed_mps"][sel]),
        }
    is_acceleration = np.array([c["motion"] == "acceleration" for c in rows], dtype=float)
    result["direction"] = {
        "durations_s": sorted(set(k["duration_s"].tolist())),
        "distance_by_group": per_group,
        "corr_acceleration_motion_with_distance": sci(stats.pearsonr(is_acceleration, k["distance_m"]).statistic),
        "distance_m_min_max": min_max(k["distance_m"]),
        "distance_px_min_max": min_max(k["distance_m"], PX_PER_M),
    }
    return result


CHECKS = {
    "value_grids": check_value_grids,
    "design_balance": check_design_balance,
    "start_positions": check_start_positions,
    "label_independence": check_label_independence,
    "distance_confound": check_distance_confound,
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