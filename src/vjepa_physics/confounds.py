"""Confound controls for the speed and acceleration sets: a shared distance-travelled scale and the overlap window of
distances both sets reach (acceleration clips start from rest, so within each set the label is proportional to the
distance travelled)."""
from collections.abc import Iterable

import numpy as np

from vjepa_physics.geometry import distance_travelled, frame_times

FPS, FRAMES = 24, 16  # every clip: 16 frames at 24 fps
CLIP_SECONDS = float(frame_times(FPS, FRAMES)[-1])  # frame 0 to the last frame: 15 / 24 s
PAIR = ("speed", "acceleration")
DISTANCE_PER_UNIT = {"speed": CLIP_SECONDS, "acceleration": CLIP_SECONDS**2 / 2}  # metres per m/s; per m/s² from rest

# Reading rule for an acceleration probe applied to speed clips (slope of cross-predicted on true distance, with CI):
READS_ACCELERATION_BELOW = 0.25  # whole CI below: reads acceleration itself (a constant-speed clip has a = 0 -> slope 0)
READS_SPEED_ABOVE = 0.5  # whole CI above: reads a speed / distance quantity (speed at tau -> slope T / (2 tau) >= 0.5)


def clip_distance(table: dict) -> np.ndarray:
    """(clips,) metres travelled from frame 0 to the last frame, from a joined table's speed and acceleration fields:
    v T + a T² / 2 with T = CLIP_SECONDS (speed set a = 0; acceleration set from rest, v = 0)."""
    return distance_travelled(table["speed_mps"], table["acceleration_mps2"], CLIP_SECONDS)


def overlap_window(distances: Iterable[np.ndarray]) -> tuple[float, float]:
    """(lo, hi): the distances every set reaches — the largest minimum and the smallest maximum."""
    distances = [np.asarray(d, dtype=np.float64) for d in distances]
    lo, hi = max(float(d.min()) for d in distances), min(float(d.max()) for d in distances)
    if lo > hi:
        raise ValueError(f"the distance ranges do not overlap: {lo} > {hi}")
    return lo, hi


def in_window(distance: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    """Mask of the clips whose distance lies inside the window, ends included."""
    d = np.asarray(distance, dtype=np.float64)
    return (d >= window[0]) & (d <= window[1])




def as_distance(variable: str, values: np.ndarray) -> np.ndarray:
    """Label values of `variable` (m/s or m/s²) -> metres a clip of that set with that label travels over the clip."""
    return np.asarray(values, dtype=np.float64) * DISTANCE_PER_UNIT[variable]


def expected_slope(probe_variable: str, tau_fraction: float) -> float:
    """Slope of cross-predicted on true distance if a probe reads the speed at time tau = tau_fraction x T.

    Speed probe on acceleration clips: the speed at tau is a tau, read as a constant speed -> slope 2 tau / T.
    Acceleration probe on speed clips: in its own set a = v(tau) / tau; on a constant-speed clip v(tau) = v -> slope
    T / (2 tau) (tau_fraction must be > 0). Distance, mean speed and mid-clip speed all give tau = T / 2, slope 1.
    """
    if probe_variable == "speed":
        return 2.0 * tau_fraction
    if probe_variable == "acceleration":
        return 1.0 / (2.0 * tau_fraction)
    raise ValueError(f"no cross-set reading for {probe_variable!r}")


def tau_fraction_from_slope(slope: float) -> float:
    """Speed probe on acceleration clips: tau / T = slope / 2 (0 = initial speed, 0.5 = mid-clip, 1 = final)."""
    return slope / 2.0


def reading_verdict(ci: tuple[float, float]) -> str:
    """Acceleration probe on speed clips, from the slope's 95% interval (fixed before any probe is applied)."""
    if ci[1] < READS_ACCELERATION_BELOW:
        return "reads acceleration"
    if ci[0] > READS_SPEED_ABOVE:
        return "reads a speed / distance quantity"
    return "mixed"


def matched_value_pairs(sparse: np.ndarray, dense: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pair every value of the sparser set with its nearest value of the other set, one to one.

    sparse, dense: sorted distinct distances (metres) of each set's values. Returns (N, 2) indices (into sparse, into
    dense) and the (N,) absolute gaps. Raises ValueError if two sparse values share a nearest dense value, since
    nearest-value pairing is then not one to one.
    """
    sparse, dense = np.asarray(sparse, dtype=np.float64), np.asarray(dense, dtype=np.float64)
    nearest = np.abs(sparse[:, None] - dense[None, :]).argmin(axis=1)
    if len(np.unique(nearest)) != len(nearest):
        raise ValueError("two values share their nearest partner; the pairing is not one to one")
    return np.stack([np.arange(len(sparse)), nearest], axis=1), np.abs(sparse - dense[nearest])



def clip_pairs(first: np.ndarray, second: np.ndarray, window: tuple[float, float]):
    """Per-clip pair ids for two sets' clip distances (first = the sparser set's): each set's distinct values inside the
    window are paired by matched_value_pairs. Returns (first ids, second ids, gaps); -1 = outside the window or a value
    left unpaired."""
    values = [np.unique(d[in_window(d, window)]) for d in (first, second)]
    pairs, gaps = matched_value_pairs(*values)
    ids = []
    for d, vals, column in zip((first, second), values, (0, 1)):
        pair_of_value = np.full(len(vals), -1)
        pair_of_value[pairs[:, column]] = np.arange(len(pairs))
        pid = np.full(len(d), -1)
        inside = in_window(d, window)
        pid[inside] = pair_of_value[np.searchsorted(vals, d[inside])]
        ids.append(pid)
    return ids[0], ids[1], gaps


def shared_pair_rows(first_ids: np.ndarray, second_ids: np.ndarray, first_mask: np.ndarray, second_mask: np.ndarray):
    """Masks of each set's rows (within its mask) whose pair also has rows in the other set's mask."""
    shared = np.intersect1d(first_ids[first_mask], second_ids[second_mask])
    shared = shared[shared >= 0]
    return first_mask & np.isin(first_ids, shared), second_mask & np.isin(second_ids, shared)


def pair_weights(pair_ids: np.ndarray) -> np.ndarray:
    """Weight 1 / (clips of the same pair) per clip of one class, so every pair weighs 1 in that class."""
    _, inverse, counts = np.unique(pair_ids, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


def weighted_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    """Mean over the two classes of the weighted share of that class predicted correctly."""
    y_true, y_pred, weights = np.asarray(y_true, bool), np.asarray(y_pred, bool), np.asarray(weights, np.float64)
    return float(np.mean([(weights[y_true == c] * (y_pred[y_true == c] == c)).sum() / weights[y_true == c].sum()
                          for c in (False, True)]))