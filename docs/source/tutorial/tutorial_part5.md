# 5. Problems

A problem in EvoX 2.0 is **not a class**. It is a *module* that defines two things:

- a **frozen config dataclass** named after the problem (`Sphere()`, `DTLZ2(d=12)`, `CEC2022(problem_number=3, dimension=10)`), holding every parameter as data;
- **plain module-level functions** in that *same* module — `evaluate(config, problem_state, pop)` and, only for stateful problems, `init(config, key)`.

Nothing is stored on the config and nothing is mutated. `evaluate` returns `(fitness, problem_state)`, and the population is a tensor of shape `(pop_size, dim)` that the compiler traces through your formulas.

The **algorithm never calls the problem directly**. It receives an opaque `evaluate` *closure*, and the workflow owns the single compile site:

```
solution_transform → problem.evaluate → opt-direction scaling → fitness_transform → monitor
```

That closure is built once by `StdWorkflow` and threaded through the traced step graph, so a problem function only ever runs inside an `evoxc` trace — its shape and dtype are baked into the compiled program (see {doc}`tutorial_part2` for the workflow itself).

```{warning}
Calling a problem function outside a trace raises `evoxc`'s `TraceError` — `evaluate(Sphere(), ProblemState(), x)` on a NumPy array does **not** work. Always go through `evoxc.build`/`evoxc.run` or a `StdWorkflow`.
```

## 5.1 Numerical benchmark functions

`evox_etl.problems.numerical` re-exports the nine basic single-objective functions (plus the raw `*_func` kernels such as `sphere_func(x)`):

| Config | Constructor arguments worth setting | Global minimum at | Commonly used box *(convention — not in the code)* |
|---|---|---|---|
| `Sphere()` | `shift`, `affine` | `[0, …, 0]` | `[-5.12, 5.12]` |
| `Ackley()` | `a=20.0`, `b=0.2`, `c=2π`, `shift`, `affine` | `[0, …, 0]` | `[-32.768, 32.768]` |
| `Rastrigin()` | `shift`, `affine` | `[0, …, 0]` | `[-5.12, 5.12]` |
| `Rosenbrock()` | `shift`, `affine` | `[1, …, 1]` | `[-30, 30]` |
| `Griewank()` | `shift`, `affine` | `[0, …, 0]` | `[-600, 600]` |
| `Levy()` | `shift`, `affine` | `[1, …, 1]` | `[-10, 10]` |
| `Schwefel()` | `shift`, `affine` | `[420.9687, …, 420.9687]` | `[-500, 500]` |
| `Zakharov()` | `shift`, `affine` | `[0, …, 0]` | `[-5, 10]` |
| `Ellipsoid()` | `shift`, `affine` | `[0, …, 0]` | `[-5.12, 5.12]` |

Two properties matter more than the formulas:

- **Every one of them is single-objective and dimension-agnostic.** There is no `d` field: the formulas reduce over `axis=1`, so the width of `pop` *is* the dimension. Fitness comes back as `(pop_size,)` float32.
- **They apply no bounding.** These are raw mathematical functions of `x`, defined everywhere on ℝᵈ, and the port keeps that behaviour deliberately. If you want a box-constrained search you give the box to the **algorithm**: `make_pso(pop_size=64, lb=[-10.0, -10.0], ub=[10.0, 10.0])` (bounds are 1-D, one value per dimension). A candidate that leaves the box is only pulled back by whatever boundary handling that algorithm implements — the problem will happily evaluate it.

You can drive a built-in problem on its own, without a workflow, to check your understanding of the protocol:

```python
import numpy as np
import evoxc
from evox_etl.problems.numerical import ProblemState, Sphere
from evox_etl.problems.numerical.basic import evaluate

config = Sphere()
x = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]], dtype=np.float32)

spec = evoxc.core.TensorSpec(x.shape, x.dtype)        # shape AND dtype are static
exe = evoxc.build(evaluate, config, ProblemState(), spec)
fitness, problem_state = evoxc.run(exe, config, ProblemState(), x)

print(fitness.numpy())      # [0. 9.]
print(problem_state)        # ProblemState()
```

The same config reaches the problem through the *module* it is defined in, so `Sphere` is used exactly like `DTLZ2` or your own problem: `StdWorkflow(algorithm, Sphere())`.

## 5.2 DTLZ1–DTLZ7 (multi-objective)

The seven DTLZ problems live in `evox_etl.problems.numerical.dtlz`. Their configs share three integer fields:

| Config | `d` (decision variables) | `m` (objectives) | `ref_num` (Pareto-front samples) |
|---|---|---|---|
| `DTLZ1()` | 7 | 3 | 1000 |
| `DTLZ2()` | 12 | 3 | 1000 |
| `DTLZ3()` | 12 | 3 | 1000 |
| `DTLZ4()` | 12 | 3 | 1000 |
| `DTLZ5()` | 12 | 3 | 1000 |
| `DTLZ6()` | 12 | 3 | 1000 |
| `DTLZ7()` | 21 | 3 | 1000 |

- **`m`** is the number of objectives: `evaluate` returns fitness of shape `(pop_size, m)`, and the algorithm's own `n_obj` must agree with it.
- **`d`** is the number of decision variables. The defaults are the classic `d = m - 1 + k` choices (`k = 5` for DTLZ1, `k = 10` for DTLZ2–6, `k = 20` for DTLZ7). `evaluate` reads the width from the population you hand it, so `d` is the *contract* — build your initial population with `d` columns.
- **`ref_num`** is the requested number of reference Pareto-front points. The front is generated with the Das–Dennis uniform sampler, which picks the smallest regular grid that covers the request, so the actual row count is *about* `ref_num * m` rather than exact: `ref_num=1000, m=3` yields 2926 rows for DTLZ1–4 and 3000 rows for DTLZ5/6.

The reference Pareto front is a **module-level function** `pf(config)` — there is no `DTLZ` base class and no `.pf()` method. Since `pf` takes no tensor inputs, trace it with just the config and materialise the result with `.numpy()`:

```python
import evoxc
from evox_etl.problems.numerical.dtlz import DTLZ2, pf

config = DTLZ2()                          # m=3, ref_num=1000
exe = evoxc.build(pf, config)             # no tensor specs: pf(config) only
front = evoxc.run(exe, config).numpy()    # host-side ndarray

print(front.shape)                        # (2926, 3)
print(front.min(), front.max())           # (1e-06, 1.0) — all objectives in [0, 1]
```

The per-variant functions `pf_dtlz1` … `pf_dtlz7` are exported too (DTLZ3/DTLZ4 reuse `pf_dtlz2`'s normalised sphere, exactly like the original class hierarchy), but `pf` dispatches on the config type for you. The ndarray you get back is the natural argument to the monitor's overlay: `workflow.monitor.plot(problem_pf=front)` on a multi-objective run (see {doc}`tutorial_part6`).

```{tip}
`pf` is traced, not eagerly executed, so it needs the same static shapes as `evaluate`. With `d`/`m` fixed in the config this is always satisfied: the number of rows and columns is decided by `ref_num` and `m` alone.
```

## 5.3 CEC2022

`CEC2022` is the single-objective CEC 2022 suite (12 functions). Both fields are **required**:

```python
import numpy as np
import evoxc
from evox_etl.problems.numerical import CEC2022, ProblemState
from evox_etl.problems.numerical.cec2022 import evaluate

config = CEC2022(problem_number=3, dimension=10)
x = np.random.default_rng(0).normal(size=(4, 10)).astype(np.float32)

spec = evoxc.core.TensorSpec(x.shape, x.dtype)
exe = evoxc.build(evaluate, config, ProblemState(), spec)
fitness, _ = evoxc.run(exe, config, ProblemState(), x)
print(fitness.numpy())      # (4,) float32
```

Constraints the config enforces when you construct it:

- `problem_number` selects the function, `1`–`12` (an unknown number raises `ValueError` at evaluation).
- `dimension` must be one of **2, 10 or 20**.
- Functions **6, 7 and 8 are not defined for `dimension=2`** and are rejected.

The suite is **single-objective**: fitness has shape `(pop_size,)`. `evaluate` also asserts that the population width equals `dimension` ("Dimension mismatch! Expect 10, got …"), so the `d` you give the algorithm must match the config exactly.

```{note}
CEC2022 bakes its per-function rotation matrix, shift vector and (for functions 6–8) shuffle index into the graph as constants at **trace** time. The first trace of a given `(problem_number, dimension)` therefore pays a one-off file-load, and the compiled program is specific to that pair — switching functions means a new config, hence a new compiled step graph.
```

## 5.4 Shifted and affine variants

Every basic function carries two optional fields inherited from the `ShiftAffineNumericalProblem` base config: `shift` (a 1-D vector) and `affine` (a square matrix). When present they are applied to the input **first**, in this order:

```
x' = (x + shift) @ affine      # shift added, then rotated/scaled
fitness = f_true(x')
```

So `Sphere(shift=np.ones(3))` is minimised at `x = [-1, -1, -1]`, not at the origin. Both values are validated at construction time (square `affine`, 1-D `shift`, matching dimensions) and baked as graph constants, which is how a "shifted/rotated benchmark" becomes a single config field instead of a wrapper problem.

`ShiftAffineNumericalProblem()` itself is the *base* config, not a problem you can run: the evaluator dispatches on the concrete type and raises `TypeError: unknown numerical problem config: ShiftAffineNumericalProblem`. Use it for `isinstance`-free reasoning about the shared fields, and instantiate `Sphere`/`Ackley`/`Rastrigin`/… to actually evaluate.

## 5.5 Virtual (Gaussian-noise neuroevolution) problems

The neuroevolution side is served by a *virtual* problem: it never receives a `(pop_size, dim)` population. It receives a compact payload and regenerates each individual's weight/bias perturbation from its seed, which is what allows the training-free virtual ES family of {doc}`tutorial_part4` to keep an `O(dim)` state (a center vector plus a seed per individual) instead of an `O(pop_size * dim)` population.

Two names, one config: `VirtualProblem` and `VirtualLoRAProblem` are both aliases of `VirtualProblemConfig`, and the `lora_rank` field selects the mode — `lora_rank=None` is full Gaussian noise, a positive `lora_rank` is the low-rank factorised update.

### The payload protocol

```text
payload = (center_flat, seeds, sigma)
```

| Part | Shape / type | Meaning |
|---|---|---|
| `center_flat` | `(dim,)` float32 | The shared center network, flattened in `named_parameters()` order |
| `seeds` | `(pop_size,)` int64 | One RNG seed per individual; the problem derives that individual's noise from it deterministically |
| `sigma` | static Python `float` | Noise scale applied to every perturbation |

`evaluate` returns `(pop_size,)` float32 fitness with **minimisation** semantics, and the problem is stateless (it reuses the shared empty `ProblemState` and defines no `init`). This is the exact payload `make_virtual_es` / `make_virtual_lora_es` hand to their `evaluate` closure — the pair is a shared contract, so the algorithm config's `param_shapes` and the problem config's `param_shapes` must be identical.

### `make_virtual_problem`

Build the config through the constructor, which normalises and validates everything (it raises `ValueError` on any inconsistency):

| Argument | Type | Meaning |
|---|---|---|
| `param_shapes` | sequence of int-tuples | Weight/bias shapes in `named_parameters()` order, e.g. `((8, 4), (8,), (1, 8), (1,))` |
| `layer_specs` | sequence of 4-tuples | `("linear", weight_idx, bias_idx_or_None, activation)` per layer |
| `inputs` | flat float sequence/array | The baked batch, `batch_size * in_features` values |
| `targets` | flat float sequence/array | `batch_size * out_features` values for `loss="mse"`, `batch_size` for `loss="cross_entropy"` |
| `in_features` / `out_features` | int | First layer's input width / last layer's output width (the feature chain is validated) |
| `batch_size` | int | Rows in the single deterministic batch |
| `reduction` | `"mean"` \| `"sum"` | How the batch losses are aggregated into one fitness |
| `loss` | `"mse"` \| `"cross_entropy"` | Per-sample loss |
| `lora_rank` | `None` \| positive int | `None` = full Gaussian noise, int = low-rank (LoRA) mode |

Supported activations are `"relu"`, `"tanh"`, `"sigmoid"`, `"gelu"` and `"identity"`.

Because the dataset is baked into the config, `evaluate` is a pure function of the payload — there is no data-loader iterator state, so a virtual problem is reproducible by construction:

```python
import numpy as np
import evoxc
from evox_etl.problems.neuroevolution import make_virtual_problem
from evox_etl.problems.neuroevolution.virtual_problem import evaluate
from evox_etl.problems.numerical import ProblemState

shapes = ((8, 4), (8,), (1, 8), (1,))    # a 4 -> 8 -> 1 MLP, named_parameters() order
layers = (("linear", 0, 1, "relu"), ("linear", 2, 3, "identity"))
dim = 8 * 4 + 8 + 1 * 8 + 1          # 41

rng = np.random.default_rng(0)
inputs = rng.uniform(-1.0, 1.0, size=(8, 4)).astype(np.float32)
targets = (inputs @ np.array([1.0, -0.5, 0.25, 0.75], dtype=np.float32)).reshape(8, 1)

problem = make_virtual_problem(
    param_shapes=shapes,
    layer_specs=layers,
    inputs=inputs,
    targets=targets,
    in_features=4, out_features=1, batch_size=8,
    loss="mse", reduction="mean", lora_rank=None,
)

center = np.zeros(dim, dtype=np.float32)
seeds = np.arange(16, dtype=np.int64)
sigma = 0.05

payload_spec = (
    evoxc.core.TensorSpec(center.shape, center.dtype),
    evoxc.core.TensorSpec(seeds.shape, seeds.dtype),
    sigma,                                    # static leaf, not a traced tensor
)
exe = evoxc.build(evaluate, problem, ProblemState(), payload_spec)
fitness, _ = evoxc.run(exe, problem, ProblemState(), (center, seeds, sigma))
print(fitness.numpy().shape)                  # (16,) — one fitness per seed
```

```{note}
A full optimisation run needs one more input pair you must supply yourself: the `param_shapes`/`layer_specs` description has to match the network your `center` vector describes, and `sigma` is part of the payload, so it must be re-passed by value on every `evoxc.run`. There is no `ParamsAndVector` helper in 2.0 — you flatten your network's parameters (and unpack the optimised center) by hand. The end-to-end wiring is `StdWorkflow(make_virtual_es(...), make_virtual_problem(...))`.
```

## 5.6 Writing your own problem

The step protocol is the whole contract (`evox_etl.core.problem.Problem` documents it; problems are duck-typed and do **not** inherit from anything):

| Function | Signature | Required |
|---|---|---|
| `evaluate` | `(config, problem_state, pop) -> (fitness, problem_state)` | yes |
| `init` | `(config, key) -> state` or `(state, key)` | only for stateful problems |

`fitness` is `(pop_size,)` **or** `(pop_size, n_obj)`; both are accepted. Stateless problems return the state they were given unchanged.

Two rules decide whether the workflow can find your functions:

1. The config and `evaluate` must live in the **same module** — the workflow resolves it with `importlib.import_module(type(config).__module__)`.
2. Everything inside `evaluate` is traced: use `evoxc` / `evoxc.numpy` ops (not NumPy, not Python branching on tensor values). Test with a Python `if` on the *config* is fine — the config is a static argument.

Here is a complete, runnable custom problem. Put the module in `my_problem.py`:

```python
"""my_problem.py — a minimal custom problem: shifted quadratic bowl."""
from __future__ import annotations

import dataclasses

import evoxc.numpy as enp


@dataclasses.dataclass(frozen=True)
class QuadraticBowl:
    """f(x) = sum((x - center)**2) + offset, minimised at x = center."""

    center: float = 1.0
    offset: float = 0.0


def evaluate(config, problem_state, pop):
    """Return (fitness, problem_state); fitness has shape (pop_size,)."""
    diff = pop - config.center
    fitness = enp.sum(diff * diff, axis=1) + config.offset
    return fitness, problem_state
```

and optimise it with any single-objective algorithm — no registration step, no base class:

```python
import my_problem
from evox_etl import StdWorkflow
from evox_etl.algorithms.so import make_pso

problem = my_problem.QuadraticBowl(center=1.0)   # the dimension comes from the population
algo = make_pso(pop_size=64, lb=[-10.0, -10.0], ub=[10.0, 10.0])

wf = StdWorkflow(algo, problem, opt_direction="min")
state = wf.init(seed=0)
for _ in range(50):
    state = wf.step(state)

best = state.algorithm_state
print(best.global_best_location.numpy())   # [1.000024 1.000022]
print(best.global_best_fit.numpy())        # 1.0604992e-09
```

```{tip}
Keep the config small and static. Fields may be Python scalars, tuples, `None` or NumPy arrays (arrays are legal static leaves), but any field you read inside `evaluate` becomes part of the compiled program's constants — changing it means a new trace, while changing a *tensor state* does not.
```

## 5.7 Where to go next

- {doc}`tutorial_part4` — the algorithms that consume these problems, including the virtual ES family that produces the `(center_flat, seeds, sigma)` payload.
- {doc}`tutorial_part6` — multi-objective workflows, Pareto-front overlays and compiled backends.
- {doc}`../guide/developer/custom-alg-pro` — the developer guide for writing your own problem module.
- {doc}`../guide/advanced/hpo` — wrapping a problem for hyperparameter optimisation.

EvoX 2.0 ships **30+ benchmark problems**: basic numerical functions, DTLZ1–7, CEC2022, plus virtual (Gaussian-noise neuroevolution) problems.
