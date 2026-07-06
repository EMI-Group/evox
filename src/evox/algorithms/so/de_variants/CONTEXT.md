# `src/evox/algorithms/so/de_variants/` — Differential Evolution Variants

## Intent
This directory contains the Differential Evolution (DE) family of single-objective optimization algorithms for EvoX. Each variant is a concrete subclass of `evox.core.Algorithm`, fully tensorized and GPU-accelerated via PyTorch. All algorithms operate on **bounded continuous spaces** defined by `lb`/`ub` tensors of shape `[dim]`.

## API Surface

All 6 variants are re-exported via `__init__.py`:

| Export | File | Description |
|---|---|---|
| `DE` | `de.py` | Classic Differential Evolution |
| `SHADE` | `shade.py` | Success-History based Adaptive DE |
| `CoDE` | `code.py` | Composite DE (multiple strategies + parameter pools) |
| `SaDE` | `sade.py` | Self-adaptive DE (online strategy probability learning) |
| `ODE` | `ode.py` | Opposition-based DE |
| `JaDE` | `jade.py` | Adaptive DE with per-individual F/CR + external archive |

## Architecture: Two Implementation Styles

### 1. Self-Contained (inline mutation/crossover)
**DE**, **ODE**, and **JaDE** implement mutation and crossover inline using raw `torch.randint` for parent vector selection and manual binary crossover. These are conceptually simpler and more self-contained.

- **DE** is the canonical base. It supports `base_vector="rand"` or `base_vector="best"`, configurable `num_difference_vectors`, and scalar `differential_weight` (F) and `cross_probability` (CR). Population initialized uniformly or via normal distribution.
- **ODE** extends DE with an **opposition-based step**: after standard DE mutation/crossover/selection, it computes `opposition_pop = lb + ub - pop`, evaluates opposites, and replaces individuals if opposites are better.
- **JaDE** uses **current-to-pbest/1** mutation (via its own `_select_rand_pbest_vectors`), with per-individual adaptive parameters `F_u` and `CR_u` updated via exponential moving average of successful F/CR values. Controlled by learning rate `c`.

### 2. Operator-Library-Based (modular operators)
**SHADE**, **CoDE**, and **SaDE** delegate mutation vector generation to shared operators from `evox.operators`:

| Operator | Source | Purpose |
|---|---|---|
| `DE_differential_sum` | `evox.operators.crossover` | Computes weighted sum of difference vectors with random parent selection |
| `DE_binary_crossover` | `evox.operators.crossover` | Binomial crossover between mutant and current vector |
| `DE_exponential_crossover` | `evox.operators.crossover` | Exponential (sequential) crossover |
| `DE_arithmetic_recombination` | `evox.operators.crossover` | Arithmetic recombination (weighted average) |
| `select_rand_pbest` | `evox.operators.selection` | Randomly selects a vector from the top-p% of the population |

**SHADE** is the simplest adaptive variant in this group: maintains a `Memory_FCR` (2 × pop_size) ring buffer of successful F/CR values. Each generation samples F and CR from normal distributions centered on memory entries, uses current-to-pbest/1 mutation, and updates the memory with weighted Lehmer/arithmetic means of successful parameters.

**CoDE** uses 3 fixed strategy+parameter combinations from a pool. Each strategy is a 4-bit code: `[base_vec_prim, base_vec_sec, diff_num, cross_strategy]`. Generates 3 trial vectors per individual and picks the best. Parameter pool defaults to 3 (F, CR) pairs: `[1, 0.1]`, `[1, 0.9]`, `[0.8, 0.2]`.

**SaDE** maintains a pool of 4 strategies and adapts their selection probabilities online based on success/failure counts over a learning period (`LP=50`). Also maintains a CR memory per strategy. Uses `torch.multinomial` for strategy sampling (with a `torch.randint` fallback for vmap compatibility). F is sampled from N(0.5, 0.3).

## Lifecycle

| Variant | Overrides `init_step`? | Behavior |
|---|---|---|
| DE, ODE, JaDE | Yes | Calls `self.evaluate(pop)` to seed initial fitness, then stops |
| SHADE, CoDE, SaDE | No (inherits default) | Default `init_step` calls `self.step()`; initial `fit` is `inf` so all trial vectors are accepted on first step |

## Shared Design Patterns

- **State**: `self.pop` (population), `self.fit` (fitness) always wrapped in `Mutable`
- **Bounds**: `self.lb`, `self.ub` stored as `[1, dim]` tensors
- **Hyperparameters**: Wrapped in `Parameter` (when present as tensor attributes)
- **Bound enforcement**: All variants use `evox.utils.clamp` to ensure trial vectors stay within `[lb, ub]`
- **Fitness evaluation**: Always via `self.evaluate()` — the proxy injected by the workflow
- **Selection**: Greedy one-to-one replacement (`new_fitness < self.fit` or `<=`)
- **Device**: All accept an optional `device` parameter, defaulting to the current PyTorch default device

## Constraints

- All variants must be pure PyTorch with no NumPy or CPU-bound Python loops.
- State tensors must use `Mutable`; hyperparameters must use `Parameter`.
- New DE variants should subclass `evox.core.Algorithm` and follow the `step()` lifecycle.
- The `evaluate()` proxy is set externally by the workflow — algorithms must never define it.
- Bounds (`lb`/`ub`) must be 1D tensors of equal shape and dtype.

## Routing Table

| Area | Path | Description |
|---|---|---|
| Classic DE | `de.py` | Base DE with rand/best base vector, configurable difference vectors |
| SHADE | `shade.py` | Success-history adaptive F/CR with current-to-pbest mutation |
| CoDE | `code.py` | Composite DE with 3 fixed strategies + parameter pool |
| SaDE | `sade.py` | Self-adaptive strategy probability learning online |
| ODE | `ode.py` | Opposition-based DE — adds opposite-point evaluation |
| JaDE | `jade.py` | Adaptive DE with per-individual F/CR + current-to-pbest mutation |
| Exports | `__init__.py` | Re-exports all 6 variants as a flat namespace |
