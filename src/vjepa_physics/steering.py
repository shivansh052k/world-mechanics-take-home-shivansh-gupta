"""Multi-probe subspace steering: the saved nullspace probe sequence at one layer, and the smallest shift of a
clip's pooled activation that makes the first n probes read a target value."""
import json
from dataclasses import dataclass

import numpy as np
import torch
from transformers import VJEPA2Model

from vjepa_physics.data import DATASETS
from vjepa_physics.evidence import repo_root, verified_artifact
from vjepa_physics.extraction import pool_time_steps
from vjepa_physics.intervention import run_blocks
from vjepa_physics.reproducibility import SEED
from vjepa_physics.metrics import angles_from_sincos
from vjepa_physics.probes import probe_targets

NULLSPACE_CHECKS = "results/nullspace/checks.json"  # key "nullspace_rounds": rounds.npz and each site's K
STEERING_SITE = "block_8"  # hidden_states index 9
SPACES = ("standardized", "raw")  # the space in which the shift's length is minimized

N_CLIPS = {"test_seen": 15, "test_unseen": 15}  # steered clips per variable and group
# Target value indices (0-63 on each dataset's value grid): 3 test-unseen and 2 seen values, spread over the range.
TARGET_VALUE_INDICES = {
    "direction": (7, 17, 29, 45, 58),  # 39.375, 95.625, 163.125, 253.125, 326.25 deg; 7, 29, 45 test-unseen
    "speed": (4, 16, 34, 46, 59),  # 4, 34, 59 test-unseen
    "acceleration": (4, 16, 34, 46, 59),
}
QUARTILE_SIZE = 16  # speed / acceleration test-seen clips are spread over value-index quartiles 0-15, ..., 48-63
RANDOM_SEEDS = 3  # random directions per clip, target and probe count

PATH_FRACTIONS = (0.25, 0.5, 0.75, 1.0)  # waypoints along a steering path (1.0 = the endpoint)

PROFILE_SITES = tuple(f"block_{i}" for i in range(8, 18))  # indices 9-18: steering site and the blocks after it
READOUT_ROLES = ("val_seen", "val_unseen")  # D-16: steering readouts are fit on validation clips only

@dataclass(frozen=True)
class ProbeSequence:
    """Rounds 1...K of one nullspace run. Round k's probe reads a raw pooled row x as x @ maps[k] + offsets[k]."""

    variable: str
    site: str
    k: int  # first round with val-seen R² < 0.1; later rounds are never used
    dims_per_round: int  # m: 2 for direction (sin, cos), else 1
    mean: np.ndarray  # (d,) train scaler
    scale: np.ndarray  # (d,)
    basis: np.ndarray  # (d, k * m) orthonormal removed directions (standardized space), in round order
    maps: np.ndarray  # (k, d, m) raw-space composite maps
    offsets: np.ndarray  # (k, m)

    def outputs(self, x: np.ndarray, n: int) -> np.ndarray:
        """(N, d) raw rows -> (N, n, m): what probes 1...n read."""
        check_count(self, n)
        return np.einsum("id,kdm->ikm", np.asarray(x, dtype=np.float64), self.maps[:n]) + self.offsets[:n]


def check_count(seq: ProbeSequence, n: int) -> None:
    """Raise ValueError unless 1 <= n <= K."""
    if not 1 <= n <= seq.k:
        raise ValueError(f"{seq.variable} {seq.site}: {n} probes requested; only rounds 1...{seq.k} exist")


def load_probe_sequence(variable: str, site: str = STEERING_SITE) -> ProbeSequence:
    """Rounds 1...K of the saved nullspace run, read through the recorded hash.

    K comes from the saved result (first round with val-seen R² < 0.1). Raises RuntimeError if K is missing, or if
    the run stopped (train covariance exhausted) at or before K, since those rounds would be undefined.
    """
    checks = repo_root() / NULLSPACE_CHECKS
    path = verified_artifact(checks, "nullspace_rounds")
    saved_results = json.loads(checks.read_text())
    record = saved_results["nullspace_rounds"]["result"][variable]["sites"][site]
    guard = saved_results["covariance_exhaustion"]["result"][variable]["sites"][site]
    k, m = record["summary"]["k"], record["dims_per_round"]
    exhausted = guard["guarded_run_exhausted_round"]
    if k is None or guard["k"] != k or (exhausted is not None and exhausted <= k):
        raise RuntimeError(f"{variable} {site}: K {k} (exhaustion check: K {guard['k']}, exhausted at {exhausted})")

    prefix = f"{variable}_{site}"
    with np.load(path) as saved:
        seq = ProbeSequence(
            variable, site, int(k), int(m),
            saved[f"{prefix}_mean"], saved[f"{prefix}_scale"],
            saved[f"{prefix}_basis"][:, : k * m],
            saved[f"{prefix}_maps"][:k], saved[f"{prefix}_offsets"][:k],
        )
    if seq.basis.shape[1] != k * m or seq.maps.shape != (k, len(seq.mean), m) or seq.offsets.shape != (k, m):
        raise RuntimeError(f"{variable} {site}: saved arrays do not hold rounds 1...{k}")
    return seq


def shift_matrix(seq: ProbeSequence, n: int, space: str = "standardized") -> np.ndarray:
    """(d, n * m) matrix C: a shift s measured in `space` changes probes 1...n's outputs by s @ C.

    raw: C = the composite maps side by side. standardized (s = raw shift / scale): C = scale * maps
    = P_{k-1} W_k, each probe's weights on the standardized rows after the earlier removals.
    """
    check_count(seq, n)
    if space not in SPACES:
        raise ValueError(f"space must be one of {SPACES}, got {space!r}")
    c = seq.maps[:n].transpose(1, 0, 2).reshape(len(seq.mean), n * seq.dims_per_round)
    return c * seq.scale[:, None] if space == "standardized" else c


def min_norm_shift(
    seq: ProbeSequence, x: np.ndarray, target: np.ndarray, n: int, space: str = "standardized"
) -> np.ndarray:
    """(N, d) raw shifts: per row, the shortest shift (length measured in `space`) that makes probes 1...n all read
    `target`.

    x: (N, d) raw pooled rows. target: (m,) or (N, m) in probe-target units (direction: (sin, cos)). The n * m
    probe directions are independent, so the system is solved exactly: s = r (CᵀC)⁻¹ Cᵀ, r = target - outputs.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    m = seq.dims_per_round
    goal = np.broadcast_to(np.asarray(target, dtype=np.float64).reshape(-1, m), (len(x), m))
    residual = (goal[:, None, :] - seq.outputs(x, n)).reshape(len(x), n * m)
    c = shift_matrix(seq, n, space)
    shift = np.linalg.solve(c.T @ c, residual.T).T @ c.T
    return shift * seq.scale if space == "standardized" else shift


def keyed_rng(*key: int) -> np.random.Generator:
    """A generator fixed by SEED and the integer key alone, so a draw never depends on how many draws came before."""
    return np.random.default_rng(np.random.SeedSequence([SEED, *key]))


def spread_counts(groups: np.ndarray, total: int, rng: np.random.Generator) -> dict[int, int]:
    """Clips per group: total // n_groups each, plus one for (total mod n_groups) groups picked by rng."""
    names = np.unique(groups)
    counts = {int(g): total // len(names) for g in names}
    for g in rng.choice(names, total % len(names), replace=False):
        counts[int(g)] += 1
    return counts


def steering_clips(table: dict, variable: str) -> dict[str, np.ndarray]:
    """Seeded choice of the clips to steer: {"test_seen": ids, "test_unseen": ids}, each sorted.

    test_seen is spread over value-index quartiles (speed, acceleration) or angle octants (direction), test_unseen
    over its held-out values; within a group, clips are drawn without replacement. Depends only on the table,
    SEED and the variable.
    """
    chosen = {}
    for group_index, role in enumerate(N_CLIPS):
        rows = np.flatnonzero(table["role"] == role)
        if role == "test_unseen":
            groups = table["value_index"][rows]
        elif variable == "direction":
            groups = table["octant"][rows]
        else:
            groups = table["value_index"][rows] // QUARTILE_SIZE
        rng = keyed_rng(DATASETS.index(variable), group_index)
        counts = spread_counts(groups, N_CLIPS[role], rng)
        picked = [rng.choice(rows[groups == g], c, replace=False) for g, c in counts.items()]
        chosen[role] = np.sort(table["id"][np.concatenate(picked)])
    return chosen


def steering_targets(table: dict, variable: str) -> dict[str, np.ndarray]:
    """Target values: value_index, label (degrees, m/s or m/s²) and unseen (True = a test-unseen value).

    Raises ValueError if a target value has no clips or mixed labels, is a validation-held-out value, or mixes
    held-out and seen roles.
    """
    indices = np.array(TARGET_VALUE_INDICES[variable])
    labels, unseen = [], []
    for i in indices:
        at = table["value_index"] == i
        values, roles = np.unique(table["label"][at]), set(table["role"][at].tolist())
        if len(values) != 1 or "val_unseen" in roles or ("test_unseen" in roles and roles != {"test_unseen"}):
            raise ValueError(f"{variable}: target value index {i}: labels {values}, roles {sorted(roles)}")
        labels.append(float(values[0]))
        unseen.append(roles == {"test_unseen"})
    return {"value_index": indices, "label": np.array(labels), "unseen": np.array(unseen)}


def covariance_map(z_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """(m, d) B of the train regression of standardized features on the target, z ≈ a + y B.

    B = (YcᵀYc)⁻¹ YcᵀZc (Yc, Zc centred) = Σ_yy⁻¹ Σ_yz; its rows span the train cross-covariance directions.
    """
    z = np.asarray(z_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64).reshape(len(z), -1)
    yc, zc = y - y.mean(axis=0), z - z.mean(axis=0)
    return np.linalg.solve(yc.T @ yc, yc.T @ zc)


def covariance_shift(seq: ProbeSequence, b: np.ndarray, x: np.ndarray, target: np.ndarray) -> np.ndarray:
    """(N, d) raw shifts σ ⊙ ((t − ŷ) B): each row moves the way train clips differ between the round-1 probe's
    reading ŷ of that row and the target t (direction: t on the unit circle). No label of the row is used."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    m = seq.dims_per_round
    goal = np.broadcast_to(np.asarray(target, dtype=np.float64).reshape(-1, m), (len(x), m))
    return ((goal - seq.outputs(x, 1)[:, 0, :]) @ b) * seq.scale


def random_probe_counts(k: int) -> tuple[int, ...]:
    """Probe counts with a random-direction arm: 1, K/2, K − 1 (the headline) and K."""
    return tuple(sorted({1, k // 2, k - 1, k}))


def random_key(variable: str, clip_id: int, target_index: int, n: int, seed_index: int) -> np.random.Generator:
    """The generator for one random-arm draw, keyed by (variable, clip, target, probe count, seed) only."""
    return keyed_rng(DATASETS.index(variable), clip_id, target_index, n, seed_index)


def random_shift(span: np.ndarray, length: float, scale: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(d,) raw shift σ ⊙ δ_z: δ_z = a uniformly random direction in `span` ((d, r) orthonormal, standardized space),
    of z-length `length`."""
    direction = span @ rng.standard_normal(span.shape[1])
    return length * direction / np.linalg.norm(direction) * scale


def arm_table(k: int) -> list[tuple[str, int, int]]:
    """Steering arms in run order, as (kind, n, seed): probes n = 1...K; covariance (n 0); random at each of
    random_probe_counts(K) x RANDOM_SEEDS (seed index 0...). Non-random arms have seed -1."""
    arms = [("probes", n, -1) for n in range(1, k + 1)] + [("covariance", 0, -1)]
    arms += [("random", n, s) for n in random_probe_counts(k) for s in range(RANDOM_SEEDS)]
    return arms


def arm_shifts(
    seq: ProbeSequence,
    b: np.ndarray,
    span: np.ndarray,
    variable: str,
    clip_id: int,
    target_index: int,
    x: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """(A, d) raw shifts for one clip and target, one per arm of arm_table(K), in that order.

    x: the clip's (d,) raw pooled row at the steering site. target: (m,) in probe-target units (direction: (sin, cos)).
    probes: standardized minimum-norm shift making probes 1...n read the target. covariance: covariance_shift with B.
    random: a keyed random direction in `span`, z-length = the probes shift with the same n for this clip and target.
    """
    x = np.asarray(x, dtype=np.float64).reshape(1, -1)
    probe_shifts = {n: min_norm_shift(seq, x, target, n)[0] for n in range(1, seq.k + 1)}
    shifts = []
    for kind, n, s in arm_table(seq.k):
        if kind == "probes":
            shifts.append(probe_shifts[n])
        elif kind == "covariance":
            shifts.append(covariance_shift(seq, b, x, target)[0])
        else:
            length = float(np.linalg.norm(probe_shifts[n] / seq.scale))
            shifts.append(random_shift(span, length, seq.scale, random_key(variable, clip_id, target_index, n, s)))
    return np.stack(shifts)


def steered_features(
    model: VJEPA2Model, cached: torch.Tensor, shifts: np.ndarray, first: int, last: int
) -> np.ndarray:
    """(A, last - first + 2, d) float64 probe features for each raw shift added to the tokens of `cached`.

    cached: the steering site's (1, 2048, d) output on the model's device. shifts: (A, d) = the same shift on every
    token, or (A, T, d) = one shift per time step, added to that step's 2048 / T tokens (token i is time step
    i // (2048 / T)). Each shift is cast to the model's fp32; blocks first...last then run on the result. Row 0 = the
    edited steering site, rows 1... = blocks first...last. Features are computed as the probes' are: per-time-step
    means on the device, moved to the CPU, widened to float64, then averaged over the 8 steps.
    """
    out = np.empty((len(shifts), last - first + 2, cached.shape[-1]))
    for a, shift in enumerate(np.asarray(shifts)):
        delta = torch.from_numpy(shift.astype(np.float32)).to(cached.device)
        if delta.ndim == 2:  # one shift per time step, on that step's tokens
            delta = torch.repeat_interleave(delta, cached.shape[1] // delta.shape[0], dim=0)
        edited = cached + delta
        blocks = run_blocks(model, edited, first, last)
        pooled = torch.stack([pool_time_steps(edited)] + [pool_time_steps(blocks[f"block_{i}"])
                                                          for i in range(first, last + 1)])
        out[a] = np.asarray(pooled.to("cpu")[:, 0], dtype=np.float64).mean(axis=1)
    return out

def readout_values(variable: str, readout: np.ndarray) -> np.ndarray:
    """(..., m) readout outputs -> (...) values in label units: degrees for direction (angle of (sin, cos)), else the output."""
    r = np.asarray(readout, dtype=np.float64)
    if variable == "direction":
        return angles_from_sincos(r.reshape(-1, 2)).reshape(r.shape[:-1])
    return r[..., 0]


def label_difference(variable: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a - b in label units; for direction the signed angle in [-180, 180) degrees."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return (d + 180.0) % 360.0 - 180.0 if variable == "direction" else d

def curve_parameter(variable: str, reading: np.ndarray) -> np.ndarray:
    """(N,) label values of (N, m) round-1 probe readings: degrees of (sin, cos) for direction, else the reading."""
    r = np.atleast_2d(np.asarray(reading, dtype=np.float64))
    return angles_from_sincos(r) if variable == "direction" else r[:, 0]


def clamp_to_curve(kind: str, values: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Parameters clamped to an open curve's fitted value range, and which were clamped (a loop needs none)."""
    u = np.asarray(u, dtype=np.float64)
    if kind == "loop":
        return u, np.zeros(u.shape, dtype=bool)
    return np.clip(u, values[0], values[-1]), (u < values[0]) | (u > values[-1])


def path_values(variable: str, start: float, target: float, fractions=PATH_FRACTIONS) -> np.ndarray:
    """(F,) waypoint labels start + f Δ, Δ = target - start (direction: on the shorter arc)."""
    return start + np.asarray(fractions, dtype=np.float64) * float(label_difference(variable, target, start))


def spline_path_shifts(curve, variable: str, start: float, target: float, fractions=PATH_FRACTIONS) -> np.ndarray:
    """(F, d) standardized shifts along the activation curve: S(start + f Δ) - S(start)."""
    return curve(path_values(variable, start, target, fractions)) - curve(np.array([start]))


def chord_shifts(curve, start: float, target: float, fractions=PATH_FRACTIONS) -> np.ndarray:
    """(F, d) standardized shifts along the straight chord with the same endpoint: f (S(target) - S(start))."""
    return np.outer(np.asarray(fractions, dtype=np.float64), curve(np.array([target]))[0] - curve(np.array([start]))[0])


def covariance_path_shifts(
    seq: ProbeSequence, b: np.ndarray, x: np.ndarray, target: np.ndarray, fractions=PATH_FRACTIONS
) -> np.ndarray:
    """(F, d) raw shifts f x covariance_shift; f = 1 is Phase 5's covariance arm exactly."""
    full = covariance_shift(seq, b, np.asarray(x, dtype=np.float64).reshape(1, -1), target)[0]
    return np.outer(np.asarray(fractions, dtype=np.float64), full)


def time_covariance_maps(z_steps: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(T, m, d) covariance maps of each time step's standardized train features z_steps (n, T, d) on the target.
    The map is linear in z, so their mean over steps is the all-token map."""
    return np.stack([covariance_map(z_steps[:, s], y) for s in range(z_steps.shape[1])])


def time_structured_shift(
    seq: ProbeSequence, maps: np.ndarray, x: np.ndarray, target: np.ndarray, reverse: bool = False
) -> np.ndarray:
    """(T, d) raw shifts σ ⊙ ((t - ŷ) B_s), one per time step s; ŷ = the round-1 reading of the clip's all-token row
    (as the covariance arm). reverse: step s gets step T - 1 - s's shift (control). Mean over steps = the covariance
    arm's shift."""
    x = np.asarray(x, dtype=np.float64).reshape(1, -1)
    goal = np.asarray(target, dtype=np.float64).reshape(seq.dims_per_round)
    per_step = np.einsum("m,smd->sd", goal - seq.outputs(x, 1)[0, 0], maps) * seq.scale
    return per_step[::-1].copy() if reverse else per_step

def spline_arm_table(variable: str) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Phase 6 arms as (kind, fraction): uniform (same shift on every token) and timed (one shift per time step).

    Uniform: spline path f = 0.25...1, chord f = 0.25...0.75 (f = 1 is the spline endpoint), covariance path
    f = 0.25...0.75 (f = 1 is Phase 5's covariance arm), and for speed / acceleration the free-spacing B line
    (endpoint). Timed: time-structured covariance and its time-reversed control.
    """
    uniform = [("spline", f) for f in PATH_FRACTIONS]
    uniform += [("chord", f) for f in PATH_FRACTIONS[:-1]] + [("covariance", f) for f in PATH_FRACTIONS[:-1]]
    if variable != "direction":
        uniform.append(("spacing_line", 1.0))
    return uniform, [("time_covariance", 1.0), ("time_reversed", 1.0)]


def spline_arm_shifts(
    seq: ProbeSequence, b: np.ndarray, maps: np.ndarray, curve, line, variable: str, values: np.ndarray,
    x: np.ndarray, target_label: float,
) -> dict:
    """Raw shifts of every Phase 6 arm for one clip and target, in spline_arm_table's order.

    x: the clip's (d,) raw pooled row at the steering site. The start = the round-1 probe's reading of x as a label
    (clamped to an open curve's range). Returns uniform (A_u, d), timed (A_t, T, d), the start and whether it was
    clamped. line: the free-spacing B line (None for direction).
    """
    kind = "loop" if variable == "direction" else "open"
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    goal = probe_targets(variable, np.array([target_label])).reshape(-1)
    start, clamped = clamp_to_curve(kind, values, curve_parameter(variable, seq.outputs(x[None], 1)[:, 0, :]))
    start = float(start[0])
    uniform = [spline_path_shifts(curve, variable, start, target_label) * seq.scale,
               chord_shifts(curve, start, target_label)[:-1] * seq.scale,
               covariance_path_shifts(seq, b, x, goal)[:-1]]
    if line is not None:
        uniform.append((line(np.array([target_label])) - line(np.array([start]))) * seq.scale)
    timed = np.stack([time_structured_shift(seq, maps, x, goal),
                      time_structured_shift(seq, maps, x, goal, reverse=True)])
    return {"uniform": np.concatenate(uniform), "timed": timed, "start": start, "clamped": bool(clamped[0])}


def ridge_readout_map(probe) -> tuple[np.ndarray, np.ndarray]:
    """Raw-space form of a fitted ridge probe: prediction = x @ weights + offset, weights (d, m), offset (m,)."""
    d = len(probe.scaler.scale_)
    weights = np.reshape(probe.ridge.coef_, (-1, d)).T / probe.scaler.scale_[:, None]
    offset = np.atleast_1d(probe.ridge.intercept_) - probe.scaler.mean_ @ weights
    return weights, offset