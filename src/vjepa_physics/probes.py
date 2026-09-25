"""Linear probes on pooled activations: features, targets, a ridge fit on train clips only, and scores."""
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from vjepa_physics.extraction import SITES
from vjepa_physics.metrics import angles_from_sincos, circular_mae, mae, r2, sincos_targets

ALPHAS = np.logspace(-3, 7, 41)  # probe alpha grid
FIT_ROLE = "train"  # the only role any probe is fitted on
NO_SIGNAL_R2 = 0.05  # val-seen R² at or below which an upper-edge alpha is expected


def site_features(activations: np.ndarray, site: str, flatten: bool = False) -> np.ndarray:
    """Probe features of every clip at one site, as float64.

    `activations` is (clips, 26, 8, 1024), e.g. load_joined's memory-mapped array; only `site` is read.
    Default: (clips, 1024), the mean of the 8 per-time-step means = the mean over all 2048 tokens (every time
    step has 256 tokens). flatten=True: (clips, 8192), the 8 time steps side by side (a diagnostic only).
    """
    per_step = np.asarray(activations[:, SITES.index(site)], dtype=np.float64)
    if flatten:
        return per_step.reshape(len(per_step), -1)
    return per_step.mean(axis=1)


def probe_targets(variable: str, labels: np.ndarray) -> np.ndarray:
    """Direction: (clips, 2) = (sin, cos) of the angle in degrees. Speed, acceleration: (clips,) = the label."""
    if variable == "direction":
        return sincos_targets(labels)
    return np.asarray(labels, dtype=np.float64)


@dataclass(frozen=True)
class Probe:
    """A fitted probe: the train z-scoring, the ridge fit, and how many train clips it saw."""

    scaler: StandardScaler
    ridge: RidgeCV
    n_fit: int

    @property
    def alpha(self) -> float:
        return float(self.ridge.alpha_)

    @property
    def alpha_edge(self) -> str | None:
        """"lower" or "upper" if the chosen alpha is an end of the grid, else None."""
        alphas = np.asarray(self.ridge.alphas, dtype=float)
        if self.alpha == alphas[0]:
            return "lower"
        if self.alpha == alphas[-1]:
            return "upper"
        return None

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.ridge.predict(self.scaler.transform(x))


def fit_probe(x: np.ndarray, y: np.ndarray, roles: np.ndarray, alphas: np.ndarray = ALPHAS) -> Probe:
    """Fit a probe on the train rows only: z-score, then RidgeCV (efficient leave-one-out, one alpha for all outputs).

    `x`, `y` and `roles` cover the same clips in the same order; rows whose role is not "train" are never
    seen by the scaler or the ridge. Raises ValueError on mismatched lengths or if there is no train row.
    """
    roles = np.asarray(roles)
    if not len(x) == len(y) == len(roles):
        raise ValueError(f"lengths differ: x {len(x)}, y {len(y)}, roles {len(roles)}")
    fit_rows = roles == FIT_ROLE
    if not fit_rows.any():
        raise ValueError("no train rows to fit on")
    scaler = StandardScaler().fit(x[fit_rows])
    ridge = RidgeCV(alphas=alphas, fit_intercept=True).fit(scaler.transform(x[fit_rows]), y[fit_rows])
    return Probe(scaler, ridge, int(fit_rows.sum()))


def alpha_verdict(edge: str | None, val_seen_r2: float) -> str:
    """The alpha rule: "ok" inside the grid; "no_signal" for an upper edge with val-seen R² <= NO_SIGNAL_R2
    (expected: leave-one-out then predicts the train mean); "failure" for a lower edge, or an upper edge
    with val-seen R² above NO_SIGNAL_R2."""
    if edge is None:
        return "ok"
    if edge == "upper" and val_seen_r2 <= NO_SIGNAL_R2:
        return "no_signal"
    return "failure"


def probe_scores(variable: str, labels: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """DATA.md's metrics. Direction: R² on (sin, cos) and circular MAE in degrees. Others: R² and MAE."""
    if variable == "direction":
        return {
            "r2": r2(probe_targets(variable, labels), pred),
            "circular_mae": circular_mae(labels, angles_from_sincos(pred)),
        }
    return {"r2": r2(labels, pred), "mae": mae(labels, pred)}