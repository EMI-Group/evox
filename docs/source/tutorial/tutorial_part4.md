# 4. Algorithms

This chapter is the algorithm catalogue of the **experimental EvoX 2.0 pre-release**, the functional
rewrite of EvoX on the EvoX Compiler (`evoxc`). Everything here is imported from `evox_etl`:

```python
import evox_etl                      # the workflow / monitor surface
from evox_etl.algorithms import PSO  # the algorithms
```

```{note}
EvoX 2.0 replaces the 1.x object-oriented API: `ModuleBase`, `Parameter`, `Mutable` and `torch.compile`
are gone, and so is the habit of instantiating a class and calling `algorithm.step()` yourself. The
algorithm *names* are unchanged, so a ported script usually only swaps the constructor call.
```

## What an algorithm is in 2.0

An algorithm is a **frozen config dataclass** plus a small set of **plain functions** operating on
tensor state:

```python
@dataclasses.dataclass(frozen=True)
class PSO:
    pop_size: int
    lb: tuple[float, ...]
    ub: tuple[float, ...]
    w: float = 0.6
    phi_p: float = 2.5
    phi_g: float = 0.8
```

Three things follow from that shape, and they explain almost every difference from 1.x:

1. **The config you build is what you hand to the workflow.** `StdWorkflow(algorithm=..., problem=...)`
   takes the *config object itself* — not a class, not an instance with behaviour. Configs hold only
   plain Python scalars and flat float tuples (never tensors), so the compiler treats the whole
   config as a static.
2. **The behaviour lives in module-level functions.** Every algorithm module exposes the same step
   protocol: `init(config, key) -> state`, `init_step(config, state, evaluate) -> state`,
   `step(config, state, evaluate) -> state`, and optionally `final_step`. Each owns **one full
   generation** (generate candidates → `evaluate(candidates)` → update state); the workflow wires
   them together and decides, host-side, whether generation 0 uses `init_step` and the last
   generation uses `final_step`.
3. **State is tensors, and its shapes must not change.** The workflow compiles one step graph per
   step variant and reuses it for the whole run, so a state leaf that changes shape or dtype
   mid-run forces a recompile — which is why algorithms carry fixed-shape buffers (some keep an
   `offspring` buffer purely so the pytree never changes).

You never call `init`/`step` yourself; the entire interaction is two lines
(`workflow = StdWorkflow(config, problem, ...)` then `workflow.run(generations=N, seed=42)`), as in
the four runnable examples below.

## Naming rules

The names are deliberately torch-compatible, but the *kind* of object behind each name differs by
family. Reading the four rules below is enough to avoid every import error:

- **A bare class name is the frozen config** you pass to the workflow: `PSO`, `DE`, `NSGA2`, …
  `PSO(...)` and `DE(...)` construct dataclass instances, and that instance is the workflow's
  `algorithm=` argument.
- **The ES family defines its configs with a `*Config` suffix** and re-exports them under the
  torch-style bare names. Both spellings are importable from `evox_etl.algorithms.so` and from
  `evox_etl.algorithms`, and they are *the same object*:

  ```python
  from evox_etl.algorithms.so import CMAES, CMAESConfig
  assert CMAES is CMAESConfig      # identical class objects
  ```

  The same pairing holds for `OpenES`/`OpenESConfig`, `XNES`/`XNESConfig`, `SNES`, `ARS`, `ASEBO`,
  `DES`, `ESMC`, `GuidedES`, `NoiseReuseES`, `PersistentES` and `SeparableNES`. Note that the bare
  aliases are what appear in `__all__`; the `*Config` names are importable by explicit import.
- **The MO family exposes only `*Config` names plus `make_*` from `evox_etl.algorithms.mo`**;
  `evox_etl.algorithms` adds the bare aliases on top (`NSGA2`, `NSGA3`, `MOEAD`, `RVEA`, `RVEAa`,
  `HypE`). So `from evox_etl.algorithms.mo import NSGA2` fails, while
  `from evox_etl.algorithms import NSGA2` works.
- **`make_*` is the recommended constructor.** It normalizes array-like `lb`/`ub` (and other
  array-like fields) into flat tuples of plain Python floats, validates the hyperparameters by
  raising `ValueError`, and fills derived defaults. For example:

  ```python
  from evox_etl.algorithms import make_pso, make_de, make_virtual_es

  make_pso(pop_size=10, lb=[0.0, 0.0], ub=[1.0])        # ValueError: lb and ub must have the same shape
  make_de(pop_size=0, lb=[0.0], ub=[1.0])               # ValueError: pop_size must be at least 4
  make_virtual_es(((8, 4), (8,), (1, 8), (1,)), 256, (0.0,) * 65, 0.1, 0.1)
  # ValueError: center_init length must equal the total number of parameters ... expected 49, got 65
  ```

  Direct construction (`PSO(pop_size=10, lb=(0.0, 0.0), ub=(1.0, 1.0))`) still works and is what the
  workflow receives either way, but it stores its fields verbatim: no normalization, no validation,
  and no derived defaults. Use `make_*` in application code.

## Four configs that actually run

The snippets below were executed against this pre-release (numpy backend, `seed=42`). They differ
only in the config and the problem, which is the whole point of the design.

**Differential evolution** on the 10-D Sphere function:

```python
import numpy as np
from evox_etl import StdWorkflow, EvalMonitorConfig
from evox_etl.algorithms import make_de
from evox_etl.problems.numerical import Sphere

dim = 10
config = make_de(pop_size=100, lb=np.full(dim, -10.0), ub=np.full(dim, 10.0))
workflow = StdWorkflow(config, Sphere(), monitor=EvalMonitorConfig(), opt_direction="min")
workflow.run(generations=200, seed=42)

print("best fitness:", float(np.asarray(workflow.monitor.get_best_fitness()).reshape(-1)[0]))
# best fitness: 3.7793165574839804e-07
```

**CMA-ES** on the same problem. Note the arguments: `mean_init` and `sigma` are required, and the
config carries no `dim` field, so pass `dim` to the monitor explicitly — the workflow can infer
`pop_size` from the config but not the search-space dimension:

```python
from evox_etl.algorithms import make_cma_es

config = make_cma_es(mean_init=np.zeros(dim), sigma=0.5, pop_size=40)
workflow = StdWorkflow(config, Sphere(), monitor=EvalMonitorConfig(dim=dim), opt_direction="min")
workflow.run(generations=200, seed=42)

print("best fitness:", float(np.asarray(workflow.monitor.get_best_fitness()).reshape(-1)[0]))
# best fitness: 7.022894162158854e-24
```

**PSO** on the multimodal 10-D Rastrigin function:

```python
from evox_etl.algorithms import make_pso
from evox_etl.problems.numerical import Rastrigin

config = make_pso(pop_size=200, lb=np.full(dim, -5.12), ub=np.full(dim, 5.12))
workflow = StdWorkflow(config, Rastrigin(), monitor=EvalMonitorConfig(), opt_direction="min")
workflow.run(generations=500, seed=42)

print("best fitness:", float(np.asarray(workflow.monitor.get_best_fitness()).reshape(-1)[0]))
# best fitness: 0.994964599609375   (Rastrigin is multimodal; ~1.0 is a normal landing spot)
```

**NSGA-II** on the bi-objective DTLZ2 problem. Multi-objective configs need `n_objs`, the
direction is one entry per objective, and results come back as a Pareto set rather than a single
best solution:

```python
from evox_etl.algorithms import make_nsga2
from evox_etl.problems.numerical import DTLZ2

config = make_nsga2(pop_size=100, n_objs=2, lb=np.zeros(12), ub=np.ones(12))
monitor = EvalMonitorConfig(full_sol_history=True)          # get_pf() needs the solution history
workflow = StdWorkflow(config, DTLZ2(d=12, m=2), monitor=monitor, opt_direction=["min", "min"])
workflow.run(generations=100, seed=42)

pf_solutions, pf_fitness = workflow.monitor.get_pf()
print(pf_solutions.shape, pf_fitness.shape)                 # (787, 12) (787, 2)
```

## The catalogue

The pre-release ships **27 single-objective configs and 6 multi-objective configs** — the
"30+ algorithms" claim of the EvoX ecosystem. Every row below was read off the package: the config
class, the `make_*` constructor, and the facts that actually distinguish the variants.

### Differential evolution — 6 configs

API reference: {doc}`evox_etl.algorithms.so.de_variants <../apidocs/evox_etl/evox_etl.algorithms.so.de_variants>`

| Config class | `make_*` constructor | Notes |
|---|---|---|
| `DE` | `make_de` | Classic DE: `base_vector` selects the base vector (`"rand"` or `"best"`), `num_difference_vectors=1` how many difference pairs are summed, `differential_weight=0.5` (a scalar when there is one difference vector, otherwise a sequence of that length), `cross_probability=0.9` the binomial crossover rate. Setting **both** `mean` and `stdev` switches the initial population from uniform-in-bounds to a clamped Gaussian. |
| `SHADE` | `make_shade` | Success-history adaptive DE: F and CR are drawn from a memory of recent successful values. `diff_padding_num=9` is the number of donor rows sampled per individual; `pop_size` must be at least 9. |
| `CoDE` | `make_code` | Composite DE: each individual tries every `(F, CR)` pair in `param_pool` (default three pairs) and keeps the best, so one generation evaluates a `3 * pop_size` batch in a single `evaluate` call. `replace` is kept for API parity only and is unused. |
| `SaDE` | `make_sade` | Self-adaptive DE: per-individual F/CR are sampled from adaptive pools, and the strategy probabilities are re-estimated every `LP=50` generations. `pop_size` must be at least 9. |
| `ODE` | `make_ode` | Opposition-based DE: after the usual DE selection it evaluates the opposition population `lb + ub - pop`, hence **two `evaluate` calls per generation**. Constructor arguments match `make_de`. |
| `JaDE` | `make_jade` | Adaptive DE with per-individual `F`/`CR` vectors that are updated from successes; `c=0.1` is the adaptation rate. Exposes `num_difference_vectors`, `mean`, `stdev`. |

### Evolution strategies — 14 configs

API reference: {doc}`evox_etl.algorithms.so.es_variants <../apidocs/evox_etl/evox_etl.algorithms.so.es_variants>`

| Config class | `make_*` constructor | Notes |
|---|---|---|
| `OpenES` | `make_open_es` | Plain (OpenAI-style) ES: `learning_rate` and `noise_stdev` are **required**, plus `pop_size` and `center_init`. `mirrored_sampling=True` reuses antithetic noise; `optimizer="adam"` or `None` (no adaptive optimizer). |
| `XNES` | `make_xnes` | Exponential NES on a full covariance matrix: needs `init_mean` and `init_covar`; `pop_size`, the recombination weights and the three learning rates are derived from the dimension when omitted. `covar_as_cholesky=True` keeps the covariance factorized. |
| `SeparableNES` | `make_separable_nes` | Diagonal-covariance NES: needs `init_mean` and `init_std`; O(dim) memory instead of O(dim²). |
| `SNES` | `make_snes` | Separable NES with utility weights: `pop_size` and `center_init` required, plus `sigma`, `lrate_mean`, and `weight_type` — `"temp"` (softmax over ranks at `temperature=12.5`, the default) or `"recomb"` (recombination weights). |
| `ARS` | `make_ars` | Augmented random search: samples mirrored perturbations, ranks them, and moves along the elite average. `elite_ratio=0.1`, `lr=0.05`, `sigma=0.03`, `optimizer="adam"`/`None`. |
| `ASEBO` | `make_asebo` | Adaptive ES with active subspaces for black-box optimization: the update happens inside a `subspace_dims`-dimensional subspace. Extra fields `lr_decay`, `lr_limit`, `sigma_decay`, `sigma_limit`; the two `lr_*` ones are kept for API parity and unused, as in the torch original. |
| `DES` | `make_des` | Ranked-direction ES: `pop_size` and `center_init` required, `temperature=12.5` and `sigma_init=0.1`. |
| `ESMC` | `make_esmc` | The DES variant from Learn2Hop: mirrored sampling, so **`pop_size` must be odd** (a `ValueError` otherwise). |
| `GuidedES` | `make_guided_es` | Mirrored sampling inside the subspace spanned by the stored gradients, of dimension `subspace_dims`; the default learning rate is tuned for that use (`lr=60`). |
| `NoiseReuseES` | `make_noise_reuse_es` | Reuses each noise sample across `T=100` generations with `K=10` inner steps, cutting sampling cost in online settings. |
| `PersistentES` | `make_persistent_es` | Persistent perturbations for unrolled computation graphs: perturbations are accumulated over `T=100` generations instead of resampled (`K=10`). |
| `CMAES` | `make_cma_es` | The full CMA-ES: `mean_init` and `sigma` are required, `pop_size` and the recombination `weights` are derived from the dimension when omitted. No `dim` field — the dimension is inferred from `mean_init`. |
| `VirtualES` | `make_virtual_es` | Training-free virtual ES over a *virtual* (never materialized) population: needs `param_shapes` (one shape per weight block) and `learning_rate`/`noise_stdev`. See the dedicated section below. |
| `VirtualLoRAES` | `make_virtual_es` | **Alias of the `VirtualES` config**, exactly like 1.x — so it is built with `make_virtual_es`. The distinct low-rank variant is the separate `VirtualLoRAESConfig` (`make_virtual_lora_es`), covered in the dedicated section below. |

### Particle swarm optimization — 7 configs

API reference: {doc}`evox_etl.algorithms.so.pso_variants <../apidocs/evox_etl/evox_etl.algorithms.so.pso_variants>`

| Config class | `make_*` constructor | Notes |
|---|---|---|
| `PSO` | `make_pso` | Global-best PSO. Fields are exactly `pop_size, lb, ub, w=0.6, phi_p=2.5, phi_g=0.8` (inertia / cognitive / social) — **there is no `dim` field**, the dimension is `len(lb)`. |
| `CLPSO` | `make_clpso` | Comprehensive-learning PSO: each particle follows a different dimension-wise exemplar, refreshed with probability `learning_probability=0.05`; `inertia_weight=0.5`, `const_coefficient=1.5`. |
| `CSO` | `make_cso` | Competitive swarm optimizer: after each generation only the losing half (`pop_size // 2` "students") learns from the winners and from the swarm center (`phi=0.0`). |
| `DMSPSOEL` | `make_dms_pso_el` | Dynamic multi-swarm PSO with elitist learning — and **no `pop_size`**: swarms are sized by `dynamic_sub_swarm_size=10`, `dynamic_sub_swarms_num=5` and `following_sub_swarm_size=10`. Sub-swarms regroup every `regrouped_iteration_num=50` steps and switch to a global-best strategy after `0.9 * max_iteration`. |
| `FSPSO` | `make_fs_pso` | Feature-selection PSO: the standard velocity update (`inertia_weight`, `cognitive_coefficient`, `social_coefficient`) plus a per-coordinate random offset applied with probability `mutate_rate=0.01`; optional Gaussian init via `mean`/`stdev`. |
| `SLPSOGS` | `make_sl_pso_gs` | Social-learning PSO, Gaussian demonstrator sampling: particles learn from a sampled demonstrator chosen with `demonstrator_choice_factor=0.7`, scaled by `social_influence_factor=0.2`. |
| `SLPSOUS` | `make_sl_pso_us` | Same social-learning scheme with **uniform** demonstrator sampling; identical constructor arguments to `make_sl_pso_gs`. |

### Multi-objective — 6 configs

These all take `pop_size`, `n_objs`, `lb`, `ub` and accept custom operators (next section but one).
API pages: {doc}`nsga2 <../apidocs/evox_etl/evox_etl.algorithms.mo.nsga2>`,
{doc}`nsga3 <../apidocs/evox_etl/evox_etl.algorithms.mo.nsga3>`,
{doc}`moead <../apidocs/evox_etl/evox_etl.algorithms.mo.moead>`,
{doc}`rvea <../apidocs/evox_etl/evox_etl.algorithms.mo.rvea>`,
{doc}`rveaa <../apidocs/evox_etl/evox_etl.algorithms.mo.rveaa>`,
{doc}`hype <../apidocs/evox_etl/evox_etl.algorithms.mo.hype>`.

| Config class | `make_*` constructor | Notes |
|---|---|---|
| `NSGA2Config` | `make_nsga2` | Non-dominated sorting + crowding distance. Fields: `pop_size, n_objs, lb, ub, selection_op, mutation_op, crossover_op`. |
| `NSGA3Config` | `make_nsga3` | Reference-point niching for many-objective problems. Same fields as NSGA-II plus `data_type`, which is the decision-variable dtype: `None` (float32 population, the default) or the builtin `bool` for a boolean population — anything else, notably `torch.bool`, raises `ValueError`. |
| `MOEADConfig` | `make_moead` | Decomposition-based (PBI) MOEA: one sub-problem per weight vector, updated one-by-one inside a single traced generation loop. No extra hyperparameters beyond the shared four. |
| `RVEAConfig` | `make_rvea` | Reference-vector guided EA with an APD selection criterion. Adds `alpha=2.0` (exponent of the APD penalty), `fr=0.1` (reference vectors are re-adapted every `round(1 / fr)` generations) and `max_gen=100`, the horizon that normalizes the generation counter in `theta = (gen / max_gen) ** alpha` — set it to the run length you intend. |
| `RVEAaConfig` | `make_rveaa` | RVEA with reference-vector regeneration (`a`-variant): the effective population size becomes the Das–Dennis count of the reference vectors, and `alpha`/`fr`/`max_gen` mean the same as in RVEA. |
| `HypEConfig` | `make_hype` | Hypervolume estimation by Monte-Carlo sampling: `n_sample=10000` sampled points, and it has **no `selection_op`** — only `mutation_op`/`crossover_op` are injectable. |

```{tip}
`opt_direction` must be a list with one entry per objective for multi-objective runs (e.g.
`["min", "min"]`); a single string means single-objective. All algorithms minimize internally, so
`opt_direction` is applied by the workflow, not by the algorithm.
```

## The training-free virtual ES family

`VirtualES` and `VirtualLoRAES` are the *training-free* members of the ES family: no gradient, and —
unlike every other algorithm here — no materialized `(pop_size, dim)` population. The state holds one
`(dim,)` center plus `(pop_size,)` integer seeds, and each individual's Gaussian perturbation is
regenerated deterministically from its seed on demand, so memory drops from O(pop_size · dim) to
O(dim) — which is what makes evolving large weight tensors practical.

Naming, precisely:

- `VirtualES` and `VirtualLoRAES` are **both** the `VirtualESConfig` config (the second is the 1.x
  compatibility alias); `VirtualESConfig` is importable under its own name too.
- The distinct low-rank variant is `VirtualLoRAESConfig`, built by `make_virtual_lora_es`. For every
  ≥ 2-D block it perturbs `delta = B @ A` with `A: (rank, k)` and `B: (d, rank)` instead of a full
  Gaussian matrix, shrinking the search space from `d·k` to `rank·(d + k)`; 1-D blocks keep full
  Gaussian noise.

They only make sense together with a virtual problem: the evaluate payload is `(center, seeds, sigma)`
instead of a candidate matrix, so the problem must know how to rebuild the perturbed network from the
seeds. That is what `make_virtual_problem(...)` builds (see {doc}`tutorial_part5`), and the pair runs
like any other workflow:

```python
import numpy as np
from evox_etl import StdWorkflow
from evox_etl.algorithms import make_virtual_es
from evox_etl.problems.neuroevolution import make_virtual_problem

# A 4 -> 8 -> 1 ReLU network; one shape per weight block, in named_parameters() order.
param_shapes = ((8, 4), (8,), (1, 8), (1,))
layer_specs = (("linear", 0, 1, "relu"), ("linear", 2, 3, "identity"))

rng = np.random.default_rng(0)
inputs = rng.uniform(-1.0, 1.0, size=(8, 4)).astype(np.float32)
targets = (inputs @ np.array([1.0, -0.5, 0.25, 0.75], np.float32)).reshape(8, 1)

problem = make_virtual_problem(
    param_shapes=param_shapes, layer_specs=layer_specs,
    inputs=inputs.reshape(-1), targets=targets.reshape(-1),
    in_features=4, out_features=1, batch_size=8,
)

dim = 8 * 4 + 8 + 1 * 8 + 1          # 49: total parameter count, derived by make_virtual_es too
config = make_virtual_es(
    param_shapes, pop_size=256, center_init=(0.0,) * dim, learning_rate=0.1, noise_stdev=0.1
)
workflow = StdWorkflow(config, problem, opt_direction="min")
state = workflow.run(generations=100, seed=0)

print(float(np.asarray(state.algorithm_state.best_fitness.numpy())))   # 0.0020579006522893906
```

```{warning}
Do not pair a virtual ES config with an ordinary problem. `VirtualES` hands
`(center, seeds, sigma)` to `evaluate`, and only a virtual problem (`VirtualProblem` /
`VirtualLoRAProblem`, built with `make_virtual_problem`) knows how to consume that payload.
```

## Injecting custom operators into the MO algorithms

The six multi-objective constructors accept their genetic operators as keyword arguments, so you can
replace the default selection/crossover/mutation without touching the algorithm:

- `selection_op` — accepted by `make_nsga2`, `make_nsga3`, `make_moead`, `make_rvea`, `make_rveaa`
  (**not** by `make_hype`, which has no selection operator of its own).
- `mutation_op` — accepted by all six.
- `crossover_op` — accepted by all six.

`None` (the default) means "use the built-in operator", and `make_*` raises `ValueError` if you pass
something that is neither `None` nor callable. The injected object must be a **plain function
callable inside the traced step**, matching the keyed signature of the default it replaces:

```python
def selection_op(key, n_parents, fitness): ...   # NSGA-II/III, MOEA/D default: tournament_selection_multifit
def crossover_op(key, x): ...                    # default: simulated_binary
def mutation_op(key, x, lb, ub): ...             # default: polynomial_mutation
```

RVEA and RVEAa are the exception: their default `selection_op` is `ref_vec_guided`, which takes no
key — `selection_op(x, f, v, theta)`.

A runnable example — an identity mutation (useful for ablations) plus a crossover that collapses
everything towards the middle of the box:

```python
import numpy as np
from evox_etl import StdWorkflow
from evox_etl.algorithms import make_nsga2
from evox_etl.problems.numerical import DTLZ2

def identity_mutation(key, x, lb, ub):
    return x

def constant_crossover(key, x):
    return x * 0.0 + 0.5

config = make_nsga2(
    pop_size=100, n_objs=2, lb=np.zeros(12), ub=np.ones(12),
    mutation_op=identity_mutation, crossover_op=constant_crossover,
)
workflow = StdWorkflow(config, DTLZ2(d=12, m=2), opt_direction=["min", "min"])
workflow.run(generations=20, seed=0)
```

Because the operators are frozen with the rest of the hyperparameters, swapping one means building a
new config — there is no in-place replacement. Operator signatures, defaults and the full list of
available operators live in {doc}`the operator guide <../guide/advanced/operators>`.

## Choosing an algorithm

- **Continuous single-objective, moderate dimension (≤ ~50)** → `CMAES`; it is the strongest default
  and needs only a mean and a step size. `DE`/`JaDE`/`SHADE` are the robust population-based
  alternative, at one evaluation per individual per generation.
- **Cheap, noisy or non-differentiable black box where sample budget matters** → `NoiseReuseES` /
  `PersistentES` (sample reuse), `ARS` (mirrored sampling), or `OpenES` for an Adam-style update.
- **Large parameter vectors (network weights) without gradients** → the training-free `VirtualES`, or
  `VirtualLoRAESConfig` when most blocks are matrices and a low-rank perturbation is acceptable.
- **Multi-objective with 2–3 objectives** → `NSGA2` (cheapest, well understood). **Many-objective
  (4+)** → `NSGA3`, `RVEA`/`RVEAa` or `HypE`, built for reference-vector / hypervolume selection.
- **Feature-selection-shaped problems** → `FSPSO`, whose mutation step is built for that setting.

```{note}
All algorithms minimize internally. For maximization, pass `opt_direction="max"` (or a per-objective
list) to `StdWorkflow` and let the workflow apply the sign change — never negate fitness inside a
problem.
```

## Where to go next

- {doc}`tutorial_part5` — problems: numerical benchmarks, virtual/neuroevolution problems, and how
  `evaluate` is called for each algorithm family.
- {doc}`tutorial_part6` — multi-objective optimization in depth: Pareto fronts, metrics, and
  monitor history.
- {doc}`the operator guide <../guide/advanced/operators>` — the operator library behind
  `selection_op` / `crossover_op` / `mutation_op`.
