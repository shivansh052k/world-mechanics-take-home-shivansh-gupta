"""Multi-probe steering: the minimum-norm shift makes probes 1...n read the target exactly, and is the shortest
such shift (compared with numpy's minimum-norm least-squares solution)."""
import numpy as np
import pytest

from vjepa_physics.steering import ProbeSequence, min_norm_shift


def synthetic_sequence(m: int, k: int = 3, d: int = 20, seed: int = 0) -> ProbeSequence:
    rng = np.random.default_rng(seed)
    return ProbeSequence(
        variable="speed" if m == 1 else "direction", site="block_8", k=k, dims_per_round=m,
        mean=rng.normal(size=d), scale=rng.uniform(0.5, 3.0, d), basis=np.zeros((d, k * m)),
        maps=rng.normal(size=(k, d, m)), offsets=rng.normal(size=(k, m)),
    )


@pytest.mark.parametrize("m", [1, 2])  # 2 = direction's (sin, cos)
def test_min_norm_shift_hits_the_target_with_the_shortest_edit(m):
    seq = synthetic_sequence(m)
    rng = np.random.default_rng(1)
    x, target = rng.normal(size=(5, 20)), rng.normal(size=m)
    for n in (1, seq.k):
        shift = min_norm_shift(seq, x, target, n)
        np.testing.assert_allclose(seq.outputs(x + shift, n), np.broadcast_to(target, (5, n, m)), atol=1e-10)
        # reference: numpy's minimum-norm solution of the underdetermined system in the standardized space
        c = seq.maps[:n].transpose(1, 0, 2).reshape(20, n * m) * seq.scale[:, None]
        residual = (target - seq.outputs(x, n)).reshape(5, n * m)
        reference = np.linalg.lstsq(c.T, residual.T, rcond=None)[0].T * seq.scale
        np.testing.assert_allclose(shift, reference, atol=1e-10)