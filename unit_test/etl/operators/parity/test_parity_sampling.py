"""Parity tests: evox_etl sampling operators vs the torch evox reference (torch allowed here).

Both sides run on the SAME numpy inputs: the etl side via ``etl.build`` +
``etl.run`` (numpy backend; etl has no eager mode), the torch side via
``torch.tensor`` inputs. Weights are compared with ``np.testing`` at 1e-6 and
``n_samples`` must be equal. The (30, 5) uniform_sampling case triggers the
Das-Dennis h2 branch on both sides.
"""

import numpy as np
import pytest

import etl

from evox.operators.sampling import gird as torch_gird
from evox.operators.sampling import uniform_sampling as torch_uniform_sampling
from evox_etl.operators.sampling import grid_sampling, uniform_sampling


def _run_etl(fn, *statics):
    """Trace the all-static ``fn`` and run it once on the numpy backend."""
    exe = etl.build(fn, *statics, backend="numpy")
    return etl.run(exe, *statics)


@pytest.mark.parametrize("n,m", [(20, 3), (30, 5), (6, 4)])
def test_uniform_sampling_parity(n, m):
    w, n_samples = _run_etl(uniform_sampling, n, m)
    tw, t_n_samples = torch_uniform_sampling(n, m)
    assert n_samples == t_n_samples
    np.testing.assert_allclose(w.numpy(), tw.numpy(), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("n,m", [(20, 3), (30, 5)])
def test_grid_sampling_parity(n, m):
    w, n_samples = _run_etl(grid_sampling, n, m)
    tw, t_n_samples = torch_gird.grid_sampling(n, m)
    assert n_samples == t_n_samples
    np.testing.assert_allclose(w.numpy(), tw.numpy(), rtol=1e-6, atol=1e-6)
