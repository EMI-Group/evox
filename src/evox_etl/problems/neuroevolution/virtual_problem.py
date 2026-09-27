"""Functional ETL port of the torch ``VirtualProblem`` (virtual Gaussian-noise neuroevolution).

The torch ``src/evox/problems/neuroevolution/virtual_problem.py`` evaluates a
population of Gaussian-noise-perturbed neural networks WITHOUT materializing the
full perturbed population: it receives the tuple ``(center_flat, seeds, sigma)``
(not a ``(pop_size, dim)`` matrix) and regenerates each individual's weight/bias
noise on demand from its ``seeds`` entry inside a fused Triton kernel.  This
module reproduces the same payload protocol as a plain (traced) function

    ``evaluate(config, problem_state, payload) -> (fitness, problem_state)``

with ``payload = (center_flat, seeds, sigma)`` (``center_flat`` ``(dim,)`` f32,
``seeds`` ``(pop_size,)`` int64, ``sigma`` a static Python float) and ``fitness``
``(pop_size,)`` f32 under minimization semantics.  ``problem_state`` is the
shared empty :class:`~evox_etl.problems.numerical.state.ProblemState`
(the problem is stateless; no ``init`` is defined, so the workflow falls back to
an empty state).

The noise itself is the SHARED deterministic splitmix64 + Box-Muller stream of
:mod:`evox_etl.algorithms.so.es_variants.virtual_noise` (the deliberate
cross-subtree contract with the ``VirtualES``/``VirtualLoRAES`` algorithms).

Notes / deliberate differences from the torch reference
-------------------------------------------------------
1. **ETL trade-off (materialization).**  Torch's fused kernel never
   materializes ``(out, in)`` noise; etl has no fused matmul, so this port DOES
   materialize per-block ``(pop_size, out, in)`` noise for the full-noise mode
   (and the ``(pop_size, rank, k)``/``(pop_size, d, rank)`` LoRA factors for the
   LoRA mode).  That is still far smaller than a full ``(pop_size, dim)``
   population, so the O(dim) ALGORITHM state (a ``(dim,)`` center plus
   ``(pop_size,)`` seeds) is preserved.  Activations stay at
   ``(pop_size, batch, features)`` exactly like torch.
2. **LoRA mode is an addition.**  Torch's modern ``VirtualProblem`` has no LoRA
   mode (that lives in the separate ``virtual_lora_problem.py``); this port
   unifies both behind the single ``lora_rank`` config field, so the
   ``VirtualLoRAProblem`` alias is directly usable with the (distinct) LoRA
   algorithm.  ``lora_rank is None`` selects full noise + ``compute_offsets``;
   a positive ``lora_rank`` selects ``lora_factors`` + ``compute_counter_offsets``.
3. **Deterministic fixed batch.**  Torch round-robins a ``DataLoader`` iterator
   (``n_batch_per_eval``) and therefore carries mutable iterator state across
   calls.  This port has no DataLoader/iterator state: the full baked dataset
   (``inputs``/``targets``) is evaluated as ONE deterministic fixed batch, so
   ``evaluate`` is a pure function of its payload.

Not ported: torch's ``LeakyReLU``/``ELU``/``Softmax`` activations (their
non-default parameters / softmax axis cannot be expressed as plain config
leaves) — the supported set is ReLU/Tanh/Sigmoid/GELU/Identity.
"""

from __future__ import annotations

__all__ = [
    "SUPPORTED_ACTIVATIONS",
    "SUPPORTED_LOSSES",
    "SUPPORTED_REDUCTIONS",
    "VirtualProblemConfig",
    "VirtualProblem",
    "VirtualLoRAProblem",
    "make_virtual_problem",
    "evaluate",
]

import dataclasses
from typing import Any, Sequence

import etl
import etl.numpy as enp
import numpy as np

from evox_etl.algorithms._config_utils import require_choice, to_float_tuple
from evox_etl.algorithms.so.es_variants.virtual_noise import (
    compute_counter_offsets,
    compute_offsets,
    lora_factors,
    virtual_normal,
)
from evox_etl.problems.numerical.state import ProblemState

#: Activation names supported by the layer-by-layer virtual forward pass.
SUPPORTED_ACTIVATIONS = ("relu", "tanh", "sigmoid", "gelu", "identity")
#: Supported per-sample loss reductions (``mean`` | ``sum``).
SUPPORTED_REDUCTIONS = ("mean", "sum")
#: Supported per-sample losses (``mse`` | ``cross_entropy``).
SUPPORTED_LOSSES = ("mse", "cross_entropy")


@dataclasses.dataclass(frozen=True)
class VirtualProblemConfig:
    """Configuration of the virtual Gaussian-noise neuroevolution problem.

    Every field is a plain static leaf (Python scalars + flat float tuples), so
    the config is a plain pytree passed as an etl static argument.  Build it
    with :func:`make_virtual_problem`, which performs all normalization and
    validation.  ``layer_specs`` entries have the layout
    ``("linear", weight_index, bias_index_or_None, activation_name)`` where the
    indices address ``param_shapes`` (torch ``named_parameters()`` order).
    """

    #: Flat ordered tuple of parameter shapes (torch ``named_parameters()`` order).
    param_shapes: tuple[tuple[int, ...], ...]
    #: ``("linear", weight_idx, bias_idx_or_None, activation_name)`` per layer.
    layer_specs: tuple[tuple[Any, ...], ...]
    #: Flattened ``(batch_size, in_features)`` inputs.
    inputs: tuple[float, ...]
    #: Flattened ``(batch_size, out_features)`` (mse) or ``(batch_size,)`` (ce) targets.
    targets: tuple[float, ...]
    #: Number of input features of the first layer.
    in_features: int
    #: Number of output features of the last layer.
    out_features: int
    #: Number of samples in the single deterministic baked batch.
    batch_size: int
    #: Aggregation over the batch: ``"mean"`` | ``"sum"``.
    reduction: str = "mean"
    #: Per-sample loss: ``"mse"`` | ``"cross_entropy"``.
    loss: str = "mse"
    #: LoRA rank (``None`` = full Gaussian noise; positive int = low-rank mode).
    lora_rank: int | None = None


#: Class-free torch-parity aliases.
VirtualProblem = VirtualProblemConfig
VirtualLoRAProblem = VirtualProblem


def _require_positive_int(name: str, value: Any) -> int:
    """Return ``value`` as an ``int`` after checking it is a positive integer."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an int, got {value!r}")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return value


def _normalize_param_shapes(param_shapes: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    """Validate/normalize ``param_shapes`` into a tuple of positive-int tuples."""
    shapes: list[tuple[int, ...]] = []
    for i, shape in enumerate(param_shapes):
        if not isinstance(shape, (tuple, list)) or not shape:
            raise ValueError(f"param_shapes[{i}] must be a non-empty sequence of ints, got {shape!r}")
        dims = tuple(_require_positive_int(f"param_shapes[{i}][{j}]", d) for j, d in enumerate(shape))
        shapes.append(dims)
    if not shapes:
        raise ValueError("param_shapes must not be empty")
    return tuple(shapes)


def _normalize_layer_specs(
    layer_specs: Sequence[Sequence[Any]], param_shapes: tuple[tuple[int, ...], ...]
) -> tuple[tuple[Any, ...], ...]:
    """Validate/normalize ``layer_specs`` against ``param_shapes`` (indices in range)."""
    specs: list[tuple[Any, ...]] = []
    n_params = len(param_shapes)
    for i, spec in enumerate(layer_specs):
        if not isinstance(spec, (tuple, list)) or len(spec) != 4:
            raise ValueError(
                f"layer_specs[{i}] must be a 4-tuple (kind, weight_idx, bias_idx, activation), got {spec!r}"
            )
        kind, w_idx, b_idx, activation = spec
        if kind != "linear":
            raise ValueError(f"layer_specs[{i}][0] must be 'linear', got {kind!r}")
        w_idx = _require_positive_int(f"layer_specs[{i}] weight index", w_idx + 1) - 1
        if w_idx >= n_params:
            raise ValueError(f"layer_specs[{i}] weight index {w_idx} out of range (num params {n_params})")
        if len(param_shapes[w_idx]) != 2:
            raise ValueError(
                f"layer_specs[{i}] weight param_shapes[{w_idx}] must be 2-D, got {param_shapes[w_idx]}"
            )
        if b_idx is not None:
            b_idx = _require_positive_int(f"layer_specs[{i}] bias index", b_idx + 1) - 1
            if b_idx >= n_params:
                raise ValueError(f"layer_specs[{i}] bias index {b_idx} out of range (num params {n_params})")
            if param_shapes[b_idx] != (param_shapes[w_idx][0],):
                raise ValueError(
                    f"layer_specs[{i}] bias param_shapes[{b_idx}] must be "
                    f"({param_shapes[w_idx][0]},), got {param_shapes[b_idx]}"
                )
        if activation not in SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"layer_specs[{i}] activation must be one of {SUPPORTED_ACTIVATIONS!r}, got {activation!r}"
            )
        specs.append(("linear", w_idx, b_idx, activation))
    if not specs:
        raise ValueError("layer_specs must not be empty")
    return tuple(specs)


def make_virtual_problem(
    param_shapes: Sequence[Sequence[int]],
    layer_specs: Sequence[Sequence[Any]],
    inputs: Sequence[float] | np.ndarray,
    targets: Sequence[float] | np.ndarray,
    in_features: int,
    out_features: int,
    batch_size: int,
    reduction: str = "mean",
    loss: str = "mse",
    lora_rank: int | None = None,
) -> VirtualProblemConfig:
    """Normalize and validate a :class:`VirtualProblemConfig`.

    All array-like fields are flattened to float32-rounded ``float`` tuples
    (matching the float32 constants baked inside :func:`evaluate`) and every
    inconsistency raises ``ValueError``: ``param_shapes`` vs ``layer_specs``
    (weight rank / bias shape / index range), the layer feature chain
    (``in_features`` → … → ``out_features``), ``inputs``/``targets`` lengths vs
    ``batch_size``, the ``reduction``/``loss`` choices, and a positive
    ``lora_rank`` (or ``None``).
    """
    shapes = _normalize_param_shapes(param_shapes)
    specs = _normalize_layer_specs(layer_specs, shapes)
    in_features = _require_positive_int("in_features", in_features)
    out_features = _require_positive_int("out_features", out_features)
    batch_size = _require_positive_int("batch_size", batch_size)
    reduction = require_choice("reduction", reduction, SUPPORTED_REDUCTIONS)
    loss = require_choice("loss", loss, SUPPORTED_LOSSES)
    if lora_rank is not None:
        lora_rank = _require_positive_int("lora_rank", lora_rank)

    # Layer feature chain: first layer consumes in_features, each layer feeds the
    # next, the last one produces out_features.
    prev_out = in_features
    for i, (_, w_idx, _, _) in enumerate(specs):
        layer_out, layer_in = shapes[w_idx]
        if layer_in != prev_out:
            where = "in_features" if i == 0 else f"layer {i - 1} out_features"
            raise ValueError(
                f"layer {i} weight in_features {layer_in} does not match {where} ({prev_out})"
            )
        prev_out = layer_out
    if prev_out != out_features:
        raise ValueError(
            f"last layer out_features {prev_out} does not match out_features ({out_features})"
        )

    inputs = to_float_tuple(inputs, dtype=np.float32)
    targets = to_float_tuple(targets, dtype=np.float32)
    if len(inputs) != batch_size * in_features:
        raise ValueError(
            f"inputs length {len(inputs)} does not match batch_size * in_features "
            f"({batch_size * in_features})"
        )
    expected_targets = batch_size * out_features if loss == "mse" else batch_size
    if len(targets) != expected_targets:
        raise ValueError(
            f"targets length {len(targets)} does not match the expected {expected_targets} for "
            f"loss={loss!r}"
        )

    return VirtualProblemConfig(
        param_shapes=shapes,
        layer_specs=specs,
        inputs=inputs,
        targets=targets,
        in_features=in_features,
        out_features=out_features,
        batch_size=batch_size,
        reduction=reduction,
        loss=loss,
        lora_rank=lora_rank,
    )


def _constant(values: Sequence[float] | np.ndarray, shape: tuple[int, ...]) -> etl.SymbolicTensor:
    """Bake host values as a float32 graph constant of the given static shape."""
    arr = np.asarray(values, dtype=np.float32).reshape(shape)
    return etl.ops.constant(etl.core.tensor(arr))


def _activation(name: str, x: etl.SymbolicTensor) -> etl.SymbolicTensor:
    """Apply one supported element-wise activation (static dispatch on the name)."""
    if name == "identity":
        return x
    if name == "relu":
        return etl.relu(x)
    if name == "tanh":
        return etl.tanh(x)
    if name == "sigmoid":
        return etl.sigmoid(x)
    if name == "gelu":
        return etl.gelu(x)
    raise ValueError(f"unsupported activation {name!r}")  # pragma: no cover — validated in make_*


def _virtual_forward(
    config: VirtualProblemConfig,
    center_flat: etl.SymbolicTensor,
    seeds: etl.SymbolicTensor,
    sigma: float,
) -> etl.SymbolicTensor:
    """Layer-by-layer forward pass with on-demand perturbations — ``(pop_size, batch, out)``.

    For each Linear layer the per-individual perturbed weight/bias is regenerated
    from ``seeds`` (full noise or LoRA factors) and applied as a cheap center
    matmul plus a ``sigma``-scaled noise matmul; the ``(out, in)`` weight delta is
    never added to the center weight.  The input is a baked ``(batch, in)``
    constant that the first layer's noise matmul broadcasts against ``pop_size``.
    """
    pop = int(seeds.shape[0])
    lora_rank = config.lora_rank
    # Center-vector slicing always follows the full flat parameter layout;
    # the noise stream instead uses the LoRA counter offsets when active.
    param_offsets = compute_offsets(config.param_shapes)
    noise_offsets = (
        param_offsets
        if lora_rank is None
        else compute_counter_offsets(config.param_shapes, lora_rank)
    )

    # (batch, in_features) constant; the first matmul broadcasts it over pop_size.
    h = _constant(config.inputs, (config.batch_size, config.in_features))

    for _, w_idx, b_idx, activation in config.layer_specs:
        out_f, in_f = config.param_shapes[w_idx]
        w_offset = param_offsets[w_idx]
        w_center = etl.reshape(center_flat[w_offset : w_offset + out_f * in_f], (out_f, in_f))
        # base: h @ W^T (shared across individuals).
        out = etl.matmul(h, etl.transpose(w_center, (1, 0)))

        if lora_rank is None:
            noise = etl.reshape(
                virtual_normal(seeds, noise_offsets[w_idx], out_f * in_f), (pop, out_f, in_f)
            )
            delta = etl.matmul(h, etl.transpose(noise, (0, 2, 1)))
        else:
            a, b = lora_factors(seeds, (out_f, in_f), lora_rank, noise_offsets[w_idx])
            # delta = h @ A^T @ B^T  (== h @ (B @ A)^T), (pop_size, batch, out).
            delta = etl.matmul(etl.matmul(h, etl.transpose(a, (0, 2, 1))), etl.transpose(b, (0, 2, 1)))
        out = out + sigma * delta

        if b_idx is not None:
            b_offset = param_offsets[b_idx]
            b_center = center_flat[b_offset : b_offset + out_f]
            if lora_rank is None:
                b_noise = virtual_normal(seeds, noise_offsets[b_idx], out_f)  # (pop_size, out)
            else:
                b_noise = lora_factors(seeds, (out_f,), lora_rank, noise_offsets[b_idx])  # (pop, out)
            out = out + etl.reshape(b_center + sigma * b_noise, (pop, 1, out_f))
        h = _activation(activation, out)

    return h  # (pop_size, batch, out_features)


def _per_sample_loss(
    config: VirtualProblemConfig, out: etl.SymbolicTensor, pop: int
) -> etl.SymbolicTensor:
    """Per-sample loss ``(pop_size, batch_size)`` from the ``(pop, batch, out)`` logits."""
    if config.loss == "mse":
        target = _constant(config.targets, (config.batch_size, config.out_features))
        diff = out - target  # broadcast (batch, out) over pop_size
        return etl.mean(diff * diff, axes=2)

    # cross-entropy: targets are class indices; pick the target logit via a one-hot.
    target = etl.cast(_constant(config.targets, (config.batch_size,)), np.int32)
    classes = enp.reshape(enp.arange(config.out_features, dtype=np.int32), (1, config.out_features))
    onehot = etl.cast(etl.equal(enp.reshape(target, (config.batch_size, 1)), classes), np.float32)
    selected = etl.sum(out * enp.reshape(onehot, (1, config.batch_size, config.out_features)), axes=2)
    logit_max = etl.max(out, axes=2)  # (pop_size, batch)
    log_sum_exp = etl.log(
        etl.sum(etl.exp(out - enp.reshape(logit_max, (pop, config.batch_size, 1))), axes=2)
    ) + logit_max
    return log_sum_exp - selected


def evaluate(
    config: VirtualProblemConfig,
    problem_state: ProblemState,
    payload: tuple[etl.SymbolicTensor, etl.SymbolicTensor, float],
) -> tuple[etl.SymbolicTensor, ProblemState]:
    """Evaluate a virtual Gaussian-noise-perturbed population (traced).

    ``payload = (center_flat, seeds, sigma)`` — ``center_flat`` ``(dim,)`` f32,
    ``seeds`` ``(pop_size,)`` int64 and ``sigma`` a static Python float.  The
    full baked dataset is one deterministic fixed batch.  Returns
    ``(fitness, problem_state)`` with ``fitness`` ``(pop_size,)`` f32
    (minimization semantics); ``problem_state`` is threaded through unchanged.
    """
    center_flat, seeds, sigma = payload
    pop = int(seeds.shape[0])
    out = _virtual_forward(config, center_flat, seeds, sigma)
    per_sample = _per_sample_loss(config, out, pop)  # (pop_size, batch)
    if config.reduction == "mean":
        fitness = etl.mean(per_sample, axes=1)
    else:  # "sum"
        fitness = etl.sum(per_sample, axes=1)
    return fitness, problem_state
