"""DEPRECATED test-compat stub — canonical RVEA selection + torch-faithful apd_fn.

``ref_vec_guided`` re-exports the canonical
``evox_etl.operators.selection.rvea_selection`` version (torch-parity verified).
``apd_fn`` and ``_cosine_similarity`` keep the torch-faithful implementations:
the canonical ``apd_fn`` gathers ``norm_obj`` with ``relu(x)`` while torch does
``norm_obj[x]`` (negative indices wrap to the last row). Latent for
``ref_vec_guided`` (those entries are masked afterwards) but pinned by the old
unit test — reported to the root agent as a canonical-operator bug.

``unit_test/etl/algorithms/test_shim_selection_rvea.py`` (sibling node, outside
this worker's write scope) still imports this module. Once the root agent
converts that test, DELETE this file.
"""
import etl
from typing import Tuple

from evox_etl.operators.selection.rvea_selection import ref_vec_guided

# --- torch-faithful private helpers (copied from the old shim, do not "fix") ---


def _cosine_similarity(a: etl.SymbolicTensor, b: etl.SymbolicTensor) -> etl.SymbolicTensor:
    """Cosine similarity along the last axis (row-normalized dot product)."""
    a = a / etl.norm(a, axis=-1, keepdims=True)
    b = b / etl.norm(b, axis=-1, keepdims=True)
    return etl.sum(a * b, axes=-1)


def apd_fn(
    x: etl.SymbolicTensor,
    y: etl.SymbolicTensor,
    z: etl.SymbolicTensor,
    obj: etl.SymbolicTensor,
    theta: float,
) -> etl.SymbolicTensor:
    """Compute the APD (Angle-Penalized Distance) for each solution/vector pair."""
    n, nv = z.shape[0], z.shape[1]
    # torch.gather(z, 0, relu(x)) == z[relu(x)[i, j], j]: etl.gather indexes along
    # an axis only (out[i, j, k] = z[idx[i, j], k]), so gather from a flattened view.
    flat_idx = etl.maximum(x, 0) * nv + etl.reshape(etl.cast(etl.arange(nv), etl.int64), (1, nv))
    selected_z = etl.reshape(etl.gather(etl.reshape(z, (n * nv,)), flat_idx, axis=0), (n, nv))
    left = (1 + obj.shape[1] * theta * selected_z) / etl.reshape(y, (1, y.shape[0]))
    norm_obj = etl.norm(obj, axis=1)
    right = etl.gather(norm_obj, x, axis=0)
    return left * right


__all__ = ["apd_fn", "ref_vec_guided"]
