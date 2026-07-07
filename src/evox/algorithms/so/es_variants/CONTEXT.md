# `src/evox/algorithms/so/es_variants/` — Evolution Strategies Family

## Intent
This directory contains the Evolution Strategies (ES) family of single-objective optimizers. All ES variants are center-based — they maintain a **search distribution** (defined by a center/mean and a spread parameter) rather than maintaining an explicit population of individuals. At each generation they sample the distribution, evaluate fitness, compute a gradient estimate, and shift the distribution toward promising regions.

## The Center-Based ES Pattern

Every ES variant follows the same four-phase loop:

| Phase | Operation |
|---|---|
| **Sample** | Draw noise `z ~ N(0, Σ)` and form population `x = center + σ · z` |
| **Evaluate** | `fitness = self.evaluate(population)` (proxy injected by the workflow) |
| **Estimate** | Compute a search gradient from fitness-weighted perturbations |
| **Update** | Shift `center` (and optionally `σ`, `Σ`) via gradient descent or natural gradient |

**Key distinction from DE/PSO**: ES maintains no explicit population across generations — only the distribution parameters (`Mutable` tensors). The sampled population is ephemeral.

## Distribution Types (Spectrum of Complexity)

### Scalar σ — one global step size
Simplest. A single scalar controls the spread in all dimensions equally.

| Algorithm | File | Key Features |
|---|---|---|
| **OpenES** | `open_es.py` | Score-function estimator; optional Adam optimizer; mirrored sampling (`z, -z`); reference: "ES as a Scalable Alternative to RL" |
| **ARS** | `ars.py` | Augmented Random Search; elite-ratio selection from mirrored pairs; optional Adam; reference: "Simple random search provides a competitive approach to RL" |
| **GuidedES** | `guided_es.py` | Augments random search with a learned **surrogate gradient subspace** (QR-based); balances isotropic exploration with directional guidance; reference: "Guided evolutionary strategies" |
| **ASEBO** | `asebo.py` | Adaptive ES with **active subspaces** via SVD on historical gradients; auto-tunes the exploration/exploitation ratio (`alpha`); reference: "Adaptive ES-Active Subspaces" |
| **PersistentES** | `persistent_es.py` | Accumulates perturbations across `T` inner unrolled steps; resets accumulator periodically; for meta-learning / unrolled optimization; reference: "Unbiased Gradient Estimation in Unrolled Computation Graphs" |
| **NoiseReuseES** | `noise_reuse_es.py` | Reuses noise perturbations across inner-loop iterations (resets at interval `T`); similar motivation to PersistentES but with reuse rather than accumulation |

### Per-Dimension σ (diagonal covariance) — independent step size per dimension
The spread is a vector `σ ∈ R^d`, allowing axis-aligned scaling.

| Algorithm | File | Key Features |
|---|---|---|
| **DES** | `des.py` | Diagonal Evolution Strategy discovered via **meta-black-box optimization**; temperature-based softmax weighting over ranks; learned update rules for center and σ; reference: "Discovering Evolution Strategies via Meta-Black-Box Optimization" |
| **SNES** | `snes.py` | Separable Natural Evolution Strategy; natural gradient on **diagonal covariance**; supports two weighting schemes: temperature-softmax (`"temp"`) and recombination weights (`"recomb"`); reference: "Natural Evolution Strategies" |
| **SeparableNES** | `nes.py` | Exponential NES with **diagonal covariance**; full NES formulation with separate learning rates for mean and variance; uses recombination weights with log-rank clipping |
| **ESMC** | `esmc.py` | ES with **Monte Carlo** gradient estimation; uses a **baseline** population member (center itself) for variance reduction; per-dimension σ with decay; reference: "Learn2Hop: Learned Optimization on Rough Landscapes" |

### Full Covariance Matrix — captures pairwise parameter interactions
The most expressive form. The distribution shape can rotate and stretch in arbitrary directions.

| Algorithm | File | Key Features |
|---|---|---|
| **XNES** | `nes.py` | Exponential NES with **full covariance** via Cholesky factor `B`; natural gradient updates on mean, σ (global scale), and `B` (shape); three learning rates (`mean`, `var`, `B`); references: "Exponential Natural Evolution Strategies" |
| **CMAES** | `cma_es.py` | Full Covariance Matrix Adaptation ES; the **most sophisticated** variant: evolution paths (`p_σ`, `p_c`), cumulative step-size adaptation, rank-μ update, conditional eigen-decomposition (`torch.linalg.eigh`); reference: "The CMA Evolution Strategy: A Tutorial" |

## Shared Utilities

| File | Function | Used By | Purpose |
|---|---|---|---|
| `adam_step.py` | `adam_single_tensor(param, grad, exp_avg, exp_avg_sq, beta1, beta2, lr, ...)` | OpenES, ARS, ASEBO, GuidedES, PersistentES, NoiseReuseES, ESMC | Single-tensor Adam update with optional decoupled weight decay; all 7 use it for center updates when `optimizer="adam"` |
| `sort_utils.py` | `sort_by_key(keys, population)` | CMAES | Returns `(sorted_keys, sorted_population)` via `torch.argsort` |

## Common Design Conventions

- **`center_init` / `init_mean`**: All algorithms accept a 1D tensor defining the initial search center. This is stored as `Mutable`.
- **`sigma` / `noise_stdev`**: Controls initial spread. Stored as `Mutable` (if adaptive) or `Parameter` (if fixed).
- **`optimizer: Literal["adam"] | None`**: 7 variants support an optional Adam inner optimizer. When `"adam"`, they maintain `exp_avg` and `exp_avg_sq` `Mutable` buffers and use `adam_single_tensor`. When `None`, plain SGD with `lr` is used.
- **Mirrored sampling** (`z, -z`): Used by OpenES, ARS, ASEBO, GuidedES, PersistentES, NoiseReuseES, ESMC. Reduces variance by evaluating antithetic pairs.
- **`pop_size` must be even** when mirrored sampling is used.
- **Weighting schemes**: Ranges from simple rank-based (OpenES has none), elite-based (ARS), softmax-over-ranks (SNES/DES), log-rank recombination weights (XNES/SeparableNES), to μ-best truncation (CMAES).

## ESMC — Special Note
ESMC (class docstring incorrectly says "DES algorithm") includes the **center itself** in the population (`torch.zeros(1, dim)` appended to noise) as a baseline for variance reduction, then computes `min(f_i, baseline) - min(f_j, baseline)` as the differential fitness signal.

## Imports from `evox.core`
All variants import `Algorithm`, `Mutable`, and `Parameter` from `evox.core`:
- `Algorithm` — base class (extends `torch.nn.Module` via `ModuleBase`)
- `Mutable` — wraps state tensors (center, sigma, covariance factors, optimizer moments)
- `Parameter` — wraps hyperparameters (learning rates, decay factors, Adam betas)

## Constraints
- All operations are **pure PyTorch** — no NumPy, no Python control flow dependent on tensor values.
- `step()` must be compatible with `torch.compile` and `vmap`.
- `self.evaluate(pop)` is set externally by the workflow; not defined here.
- The population sampled in `step()` is ephemeral; only distribution parameters persist across generations via `Mutable`.
- New ES variants should follow the center-based pattern and use shared utilities (`adam_step.py`, `sort_utils.py`) where applicable.

## Routing Table

| Area | File | Description |
|---|---|---|
| OpenES | `open_es.py` | Score-function ES, optional Adam, mirrored sampling |
| ARS | `ars.py` | Augmented Random Search with elite ratio |
| XNES | `nes.py` (class `XNES`) | Exponential NES with full covariance (Cholesky factor) |
| SeparableNES | `nes.py` (class `SeparableNES`) | Exponential NES with diagonal covariance |
| SNES | `snes.py` | Separable NES, temperature or recombination weights |
| DES | `des.py` | Diagonal ES discovered via meta-learning |
| CMAES | `cma_es.py` | Full CMA-ES with evolution paths and eigen-decomposition |
| ASEBO | `asebo.py` | Adaptive ES with SVD-based active subspaces |
| GuidedES | `guided_es.py` | Guided ES with QR-based surrogate gradient subspace |
| PersistentES | `persistent_es.py` | ES with persistent perturbations across unrolled steps |
| NoiseReuseES | `noise_reuse_es.py` | ES reusing perturbations across inner iterations |
| ESMC | `esmc.py` | ES with Monte Carlo estimation and baseline subtraction |
| VirtualLoRAES | `virtual_lora_es.py` | Virtual-population ES using Philox seeds + LoRA low-rank noise for neuroevolution; passes (center, seeds, sigma) tuple to evaluate |
| Shared — Adam | `adam_step.py` | Single-tensor Adam optimizer step |
| Shared — Sorting | `sort_utils.py` | Fitness-based population sorting |
