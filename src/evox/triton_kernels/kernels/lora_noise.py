"""LoRA (Low-Rank Adaptation) noise utility functions.

These pure-PyTorch helpers build on :mod:`~evox.triton_kernels.kernels.philox`
to generate deterministic, non-overlapping random low-rank perturbation streams.
They are **not** Triton kernels themselves — they call
:func:`~evox.triton_kernels.kernels.philox.philox_normal` internally, which
dispatches to Triton on CUDA and to the PyTorch fallback elsewhere.

The functions support a ``batched / population`` first dimension so that, given
``pop_size`` seeds, one call produces ``pop_size`` independent LoRA factor
samples. This is useful for population-based (evolutionary) perturbation of
neural-network weights, where each individual gets its own low-rank delta.

Key ideas:

- :func:`compute_counter_offsets` partitions the Philox counter space so that
  different parameter blocks (or A/B sub-factors within a block) draw from
  non-overlapping streams.
- :func:`generate_lora_factors` produces the per-individual low-rank factors
  ``(A, B)`` (or a flat noise vector for 1-D weights).
- :func:`lora_delta_output` applies the low-rank delta to an input without ever
  materializing the full ``(out_features, in_features)`` weight delta.
- :func:`lora_gradient` estimates the gradient of a population objective with
  respect to a weight, given per-individual fitness and factors.
"""

import torch

from .philox import philox_normal


def _flatten_to_2d(weight_shape: tuple) -> tuple[int, int]:
    """Flatten a weight shape to ``(d, k)`` where ``k`` is the last dim.

    ``d`` is the product of all dimensions except the last; ``k`` is the last
    dimension. For a 1-D shape this returns ``(1, n)`` — but callers handle the
    1-D case separately, so this is only invoked for ``len >= 2``.

    :param weight_shape: The original weight tensor shape.
    :return: ``(d, k)`` where ``d = prod(shape[:-1])`` and ``k = shape[-1]``.
    """
    d = 1
    for s in weight_shape[:-1]:
        d *= s
    k = weight_shape[-1]
    return d, k


def _ceil_div4(x: int) -> int:
    """Round ``x`` up to the next multiple of 4 (Philox yields 4 values per call).

    :param x: An element count.
    :return: ``ceil(x / 4) * 4``.
    """
    return ((x + 3) // 4) * 4


def compute_counter_offsets(param_shapes: list[tuple], lora_rank: int) -> list[int]:
    """Compute non-overlapping Philox counter offsets for a list of parameter blocks.

    For each parameter block we compute how many Philox elements it consumes, then
    round up to a multiple of 4 (Philox produces 4 values per call). The offsets
    are cumulative so every sub-stream occupies a disjoint counter range.

    - **1-D block** ``(n,)``: consumes ``n`` elements.
    - **2-D or >2-D block** ``(d, k)`` (flattened): consumes
      ``lora_rank * k`` (factor ``A``) plus ``d * lora_rank`` (factor ``B``).

    :param param_shapes: List of weight shapes (each a tuple of ints).
    :param lora_rank: The LoRA rank ``r``.
    :return: A list of starting counter values, one per block (cumulative).
    """
    offsets = []
    cur = 0
    for shape in param_shapes:
        offsets.append(cur)
        if len(shape) == 1:
            n_elements = shape[0]
        else:
            d, k = _flatten_to_2d(shape)
            n_elements = lora_rank * k + d * lora_rank
        cur += _ceil_div4(n_elements)
    return offsets


def generate_lora_factors(
    seeds: torch.Tensor,
    weight_shape: tuple,
    rank: int,
    counter: int,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Generate deterministic LoRA factors for a batch of individuals.

    - **1-D weight** ``(n,)``: returns a flat noise tensor of shape
      ``(pop_size, n)`` drawn from :func:`philox_normal`.
    - **2-D / >2-D weight** ``(d, k)`` (flattened, ``d = prod(shape[:-1])``,
      ``k = shape[-1]``): returns a tuple ``(A, B)`` where

        - ``A`` has shape ``(pop_size, rank, k)`` (the "down" projection),
        - ``B`` has shape ``(pop_size, d, rank)`` (the "up" projection),

      drawn from two non-overlapping Philox sub-streams. The product
      ``B @ A`` reconstructs a low-rank delta of shape ``(d, k)`` per individual.

    :param seeds: 1-D ``int64`` tensor of per-individual seeds.
    :param weight_shape: The target weight tensor shape.
    :param rank: The LoRA rank ``r``.
    :param counter: Starting Philox counter for this block.
    :return: Flat noise tensor (1-D weight) or ``(A, B)`` tuple (≥2-D weight).
    """
    pop_size = seeds.shape[0]
    if len(weight_shape) == 1:
        n = weight_shape[0]
        noise = philox_normal(seeds, n, counter)
        return noise
    else:
        d, k = _flatten_to_2d(weight_shape)
        # Factor A: (rank, k) per individual — "down" projection.
        a_elements = rank * k
        a_offset = counter
        # Factor B: (d, rank) per individual — "up" projection, drawn from a
        # non-overlapping counter range immediately after A's stream.
        b_offset = counter + _ceil_div4(a_elements)
        b_elements = d * rank
        A = philox_normal(seeds, a_elements, a_offset).reshape(pop_size, rank, k)
        B = philox_normal(seeds, b_elements, b_offset).reshape(pop_size, d, rank)
        return (A, B)


def lora_delta_output(
    x: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Apply a batched low-rank delta to an input without materializing the weight.

    Given a batched input ``x`` and per-individual LoRA factors ``(A, B)``,
    compute ``sigma * (x @ Aᵀ) @ Bᵀ``. This is mathematically equivalent to
    ``sigma * x @ (B @ A)ᵀ`` (where ``B @ A`` is the low-rank weight delta) but
    avoids materializing the potentially huge ``(out_features, in_features)``
    matrix.

    :param x: Input tensor of shape ``(pop_size, batch, in_features)``.
    :param A: Factor ``A`` of shape ``(pop_size, rank, in_features)``.
    :param B: Factor ``B`` of shape ``(pop_size, out_features, rank)``.
    :param sigma: Scaling factor applied to the delta.
    :return: Output of shape ``(pop_size, batch, out_features)``.
    """
    # temp: (pop, batch, rank)
    temp = torch.bmm(x, A.transpose(-1, -2))
    # delta: (pop, batch, out_features)
    delta = sigma * torch.bmm(temp, B.transpose(-1, -2))
    return delta


def lora_gradient(
    fitness: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor | None,
    pop_size: int,
    sigma: float,
    weight_shape: tuple,
) -> torch.Tensor:
    """Estimate the weight gradient from population fitness and LoRA factors.

    This implements the population-based gradient estimate used in zeroth-order /
    evolution-strategies style optimization. For each individual ``i`` the
    perturbation applied to the weight is ``sigma * delta_W_i`` where
    ``delta_W_i = B_i @ A_i`` (2-D) or the flat noise (1-D). The gradient is the
    fitness-weighted average of these perturbations, normalized by
    ``pop_size * sigma``.

    - **1-D** (``B is None``): ``grad = (fitness / (pop_size * sigma)) @ noise``,
      reshaped to ``weight_shape``.
    - **2-D / >2-D**: ``grad = einsum('i,ijk,ikl->jl', fitness, B, A) /
      (pop_size * sigma)``, reshaped to ``weight_shape``.

    :param fitness: 1-D tensor of per-individual fitness values ``(pop_size,)``.
    :param A: For 2-D, factor ``A`` of shape ``(pop_size, rank, k)``; for 1-D,
        the flat noise of shape ``(pop_size, n)``.
    :param B: For 2-D, factor ``B`` of shape ``(pop_size, d, rank)``; for 1-D,
        ``None``.
    :param pop_size: The population size (number of individuals).
    :param sigma: The perturbation scaling factor used during evaluation.
    :param weight_shape: The target weight tensor shape for the output gradient.
    :return: Gradient tensor reshaped to ``weight_shape``.
    """
    if B is None:
        # 1-D weight: A is the flat noise (pop_size, n).
        grad = (fitness / (pop_size * sigma)) @ A
        return grad.reshape(weight_shape)
    else:
        # 2-D weight: grad = sum_i fitness_i * (B_i @ A_i) / (pop_size * sigma)
        grad = torch.einsum("i,ijk,ikl->jl", fitness, B, A) / (pop_size * sigma)
        return grad.reshape(weight_shape)
