## Intent

Crossover (recombination) operators for EvoX — a GPU-accelerated evolutionary computation framework on PyTorch. All operators are **pure, stateless functions** taking PyTorch tensors as input and returning tensors as output. They inherit from nothing; they are not `torch.nn.Module` subclasses.

This directory provides two families of crossover:

1. **Simulated Binary Crossover (SBX)** — for real-valued genetic algorithms, used primarily by multi-objective (MO) algorithms.
2. **Differential Evolution (DE) operators** — for differential evolution algorithms, providing the building blocks of DE mutation and recombination.

## API Surface

Exports (via `__init__.py`):

| Symbol | File | Signature |
|---|---|---|
| `simulated_binary` | `sbx.py` | `(x: Tensor, pro_c=1.0, dis_c=20.0) -> Tensor` |
| `simulated_binary_half` | `sbx_half.py` | `(x: Tensor, pro_c=1.0, dis_c=20.0) -> Tensor` |
| `DE_differential_sum` | `differential_evolution.py` | `(diff_padding_num: int, num_diff_vectors: Tensor, index: Tensor, population: Tensor) -> Tuple[Tensor, Tensor]` |
| `DE_binary_crossover` | `differential_evolution.py` | `(mutation_vector: Tensor, current_vector: Tensor, CR: Tensor) -> Tensor` |
| `DE_exponential_crossover` | `differential_evolution.py` | `(mutation_vector: Tensor, current_vector: Tensor, CR: Tensor) -> Tensor` |
| `DE_arithmetic_recombination` | `differential_evolution.py` | `(mutation_vector: Tensor, current_vector: Tensor, K: Tensor) -> Tensor` |

All are imported under the `evox.operators.crossover` namespace.

## SBX Family (`sbx.py`, `sbx_half.py`)

**Simulated Binary Crossover** — replicates the effect of single-point binary crossover but in a continuous (real-valued) space using a spread factor (beta) drawn from a polynomial distribution.

- **Input**: A 2D tensor `x` of shape `(n, d)` (n parents, d dimensions). The first `n//2` rows are treated as parent1, the remaining as parent2.
- **Parameters**:
  - `pro_c` (float): Crossover probability per gene. Default `1.0`.
  - `dis_c` (float): Distribution index controlling spread — smaller values produce offspring farther from parents. Default `20.0`.
- **Output**:
  - `simulated_binary`: `(n, d)` tensor — all n offspring (pairs of children from each parent pair).
  - `simulated_binary_half`: `(n/2, d)` tensor — one child per parent pair (half the offspring).

**Consumers**: NSGA2, NSGA3, RVEA, RVEAa, HYPE (`simulated_binary`); MOEAD (`simulated_binary_half`).

## DE Family (`differential_evolution.py`)

These are the core building blocks of Differential Evolution algorithms. They are composed by DE variant algorithms (SHADE, SADE, CODE) to perform the full DE mutation→crossover→selection cycle.

### `DE_differential_sum`
Computes the sum of difference vectors for DE mutation (`v = base + F * Σ(r1 − r2)`). This handles the `Σ(r1 − r2)` part.
- **Parameters**:
  - `diff_padding_num` (int): Maximum number of random indices to pre-select (for vectorized batching).
  - `num_diff_vectors` (Tensor): Scalar or per-individual tensor specifying how many difference vector pairs to sum.
  - `index` (Tensor): Current individual indices (to avoid self-selection).
  - `population` (Tensor): Full population.
- **Returns**: `(difference_sum, rand_indices[:, 0])` — the summed difference vectors and the index of the base vector.

### `DE_binary_crossover` (Binomial)
Standard binomial (uniform) crossover: each dimension is independently swapped from mutation vector to current vector with probability `CR`, with one random dimension guaranteed to cross over.
- **CR** (Tensor): Crossover probability (scalar or per-individual vector).

### `DE_exponential_crossover`
Crossover proceeds along a contiguous run of dimensions starting at a random index, with geometric-distributed run length parameterized by `CR`.
- **CR** (Tensor): Crossover probability (scalar or per-individual vector).

### `DE_arithmetic_recombination`
Linear recombination: `trial = current + K * (mutation - current)`.
- **K** (Tensor): Recombination coefficient (scalar or per-individual vector).

## Constraints

- **Pure functions only**: No state, no `self`, no `nn.Module` inheritance. All operators must be `torch.compile`- and `vmap`-friendly.
- **Device-aware**: All random tensors are created on the input tensor's device via `device=x.device` or `device=population.device`.
- **Vectorized**: All operators handle batched populations — no Python loops over individuals.
- **Tensor shapes**: SBX inputs/outputs are `(n, d)` 2D tensors. DE operators operate on `(pop_size, dim)` tensors with scalar or per-individual parameters broadcast appropriately.

## Routing Table

- `sbx.py` — `simulated_binary` implementation
- `sbx_half.py` — `simulated_binary_half` implementation (half-offspring variant of SBX)
- `differential_evolution.py` — All four DE operators (`DE_differential_sum`, `DE_binary_crossover`, `DE_exponential_crossover`, `DE_arithmetic_recombination`)
- `__init__.py` — Public re-exports
