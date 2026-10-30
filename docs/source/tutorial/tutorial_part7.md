# 7. Migrating from EvoX 1.x

This page is for you if you already have an EvoX 1.x script and want it running on EvoX 2.0.

EvoX 2.0 is an **experimental pre-release**. It is imported as `evox`, it is a functional rewrite built on the `evoxc` compiler, and it replaces the torch object model that 1.x shipped.
The 1.x torch tree was deleted when 2.0 landed (it survives only in the released 1.x distributions on PyPI), so it receives no fixes and no new features; the 2.0 public surface lives in `src/evox/` under the same `evox` import name.

"The 1.x API is removed" means exactly that, not "deprecated with a warning": there is no `ModuleBase`, no `Parameter` and no `Mutable`; no `evox.core.compile`, `vmap` or `use_state`; no class-component `StdWorkflow` (the constructor keeps its name and argument order, but every component it receives is now a frozen config dataclass); and no distributed workflow switch.
`evox` imports nothing from the 1.x torch tree and nothing from torch, so 1.x code does not merely warn at import time — it simply is not there.

```{note}
The vocabulary does **not** change, and that is why migration is mostly mechanical. Three shapes change:

1. **Classes → frozen config dataclasses.** `PSO(...)` in 1.x was a `torch.nn.Module` subclass; in 2.0 `PSO` is an immutable dataclass of hyperparameters, usually built by a `make_*` constructor.
2. **Mutating methods → functions that take and return state.** `wf.step()` mutated tensors hidden inside modules; `wf.step(state)` returns the next `WorkflowState`.
3. **Ambient RNG → an explicit key.** `torch.manual_seed(...)` is gone; the run's key is a tensor leaf inside the state, seeded by `seed=` on `init`/`run`/`fit`.
```

## Mapping table

| 1.x | 2.0 | note |
|---|---|---|
| `from evox.algorithms import PSO`; `PSO(pop_size=50, lb=-32*torch.ones(d), ub=32*torch.ones(d), device=...)` | `from evox.algorithms import make_pso`; `make_pso(pop_size=50, lb=[-32.0]*d, ub=[32.0]*d)` | Frozen config dataclass built by the `make_*` constructor; bounds are flat float tuples; no `dim` and no `device` argument |
| `StdWorkflow(algorithm=algo, problem=prob, monitor=EvalMonitor())` with class components | `StdWorkflow(make_pso(...), Sphere(), monitor=EvalMonitorConfig())` | Same name and argument order; the components are config dataclasses instead of modules |
| `wf.init_step()`; `for _ in range(n): wf.step()` — mutates internals | `state = wf.init(seed=42)`; `state = wf.init_step(state)`; `state = wf.step(state)` | State threading; `wf.run(n, seed=...)` / `wf.fit(...)` run the whole loop for you |
| `torch.manual_seed(42)` before the run | `wf.run(seed=42)` / `wf.init(seed=42)`; `evox.key(seed)` for anything you build yourself | Keys live in the state; there is no global seed |
| `hv(objs, ref)` | `hv(key, objs, ref, num_sample=100000)` | Leading key added; still must run inside a trace |
| `from evox.utils import switch, clamp, clamp_int, clip, maximum, minimum, lexsort, nanmin, nanmax, randint` | `from evox.operators.jit_fix_operator import switch, clamp, clamp_int, clip, maximum, minimum, lexsort, nanmin, nanmax, randint` | NOT re-exported from `evox.utils`; `randint` gains a leading `key` |
| `evox.__version__` (undefined in 1.x — raises `AttributeError`) | `evox.__version__` (same import name, now defined) | A plain string read from the installed `evox` metadata: `2.0.0a1` for this pre-release, `0.0.0.dev0` in a bare source checkout |
| `monitor = EvalMonitor(topk=1)`; `monitor.get_best_fitness().item()` | `monitor=EvalMonitorConfig(topk=1)`; `wf.monitor.get_best_fitness()` | You construct the *config*; the workflow builds the host-side wrapper at `init` time |
| `self.w = Parameter(w, device=device)`; `self.fit = Mutable(torch.zeros(...))` in `class MyAlgo(Algorithm)` | config field `w: float = 0.6`; `fit` field of the `<Algo>State` dataclass | Config fields are static Python values; state fields are traced tensor leaves |
| `@evox.core.compile` / `@evox.core.vmap` on operators and algorithm methods | nothing | The workflow compiles one step graph per step variant; `vmap` has no analogue |

## Before and after

### Algorithms: class → frozen config

```python
# 1.x
import torch
from evox.algorithms import PSO

dim = 10
algo = PSO(pop_size=50, lb=-32 * torch.ones(dim), ub=32 * torch.ones(dim))
```

```python
# 2.0
from evox.algorithms import make_pso

DIM = 10
algo = make_pso(pop_size=50, lb=[-32.0] * DIM, ub=[32.0] * DIM)
```

`PSO` is itself importable and is a frozen dataclass with fields `pop_size, lb, ub, w, phi_p, phi_g`; `make_pso` is the same thing plus bound validation and `list`/`ndarray` → flat `float` tuple normalization.
In 1.x the bounds *also* carried the dimensionality (`dim = lb.shape[0]`); in 2.0 the dimension is implied by the length of `lb`/`ub`, which is why `PSO` has no `dim` argument.

### Problems: subclass → module with `evaluate`

```python
# 1.x
import torch
from evox.core import Problem

class TwoObjectiveProblem(Problem):
    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        x = X[:, 0]
        return torch.stack([x ** 2, (x - 2) ** 2], dim=1)
```

```python
# 2.0 — put BOTH of these in my_problems.py
import dataclasses

import evoxc.numpy as enp


@dataclasses.dataclass(frozen=True)
class TwoObjective:
    """f1 = x^2, f2 = (x - 2)^2 over a 1-D search space."""


def evaluate(config, state, pop):
    x = pop[:, 0]
    return enp.stack([x ** 2, (x - 2.0) ** 2], axis=1), state
```

```{warning}
The config dataclass and the `evaluate` function must live in the **same** module — the workflow resolves the function via `type(config).__module__` — and `pop` is **symbolic** while tracing, so the body uses `evoxc` ops (`evoxc.numpy` as `enp` above), never numpy or torch.
```

`evaluate` is a plain module-level function, not a method: it returns `(fitness, state)`, so a stateless problem returns `state` untouched (the workflow supplies an `EmptyState` when the module defines no `init`).
Handed to a multi-objective workflow, this problem works end to end: `wf.run(generations=60, seed=0)` followed by `wf.monitor.get_pf_fitness()` yields the Pareto-front estimate as a `(k, 2)` array.

### Workflow: mutating steps → state threading

```python
# 1.x
from evox.workflows import EvalMonitor, StdWorkflow

monitor = EvalMonitor()
wf = StdWorkflow(algorithm=algo, problem=prob, monitor=monitor)
wf.init_step()
for _ in range(99):
    wf.step()
```

```python
# 2.0
from evox import EvalMonitorConfig, StdWorkflow

wf = StdWorkflow(algorithm=algo, problem=prob, monitor=EvalMonitorConfig())
state = wf.init(seed=42)
state = wf.init_step(state)
for _ in range(99):
    state = wf.step(state)
```

`wf.step()` without an argument is still accepted — it re-uses the state remembered from the previous call — so a loop that ignores the return value keeps working.
What you cannot do is reach into the workflow for a mutated population: after the conversion, `algo.pop` no longer exists, and the tensors you want are leaves of `WorkflowState` (the algorithm's population lives in `state.algorithm_state`).

```{tip}
Prefer `wf.run(generations=100, seed=42)` or `wf.fit(generations=100, seed=42)` whenever you do not need per-generation host-side work: they initialize, dispatch `init_step` / `step` / `final_step` correctly, and drain the monitor's asynchronous read-backs at the end.
```

### Seeding: global seed → explicit key

```python
# 1.x
import torch

torch.manual_seed(0)
wf.init_step()
```

```python
# 2.0
from evox import key

state = wf.init(seed=0)          # workflow-level seeding
k = key(0)                        # a 0-d int64 evoxc tensor, for code you build yourself
```

`key(seed)` is `evoxc.random.key`; the key is threaded through `WorkflowState.key` and split by each component that needs randomness (population init, operator sampling, Monte Carlo metrics), so a run is reproducible from `seed=` alone regardless of what else ran in the process.

### Monitor: object → config plus host-side wrapper

```python
# 1.x
monitor = EvalMonitor(topk=1)
wf = StdWorkflow(algo, prob, monitor)
wf.init_step()

best = monitor.get_best_fitness().item()      # torch tensor -> Python float
sol = monitor.get_best_solution()             # torch tensor
```

```python
# 2.0
from evox import EvalMonitorConfig, StdWorkflow

wf = StdWorkflow(algo, prob, monitor=EvalMonitorConfig(topk=1))
wf.run(generations=100, seed=42)

best = wf.monitor.get_best_fitness()          # Python float
sol = wf.monitor.get_best_solution()          # numpy array
```

The object you construct is now the **config**; the accessor object appears as `wf.monitor` once the workflow has run `init` (which fills in `pop_size`, `dim`, `n_obj`, `multi_obj` and `opt_direction` for you). Note the missing `.item()`: `get_best_fitness()` already returns a Python `float`, and the solution accessors return numpy arrays because history is recorded host-side.
`wf.monitor.plot()` is still there — it delegates to `evox.vis_tools` and needs the optional `plotly` dependency.

### Metrics: `hv` gains a key

```python
# 1.x
from evox.metrics import hv

value = hv(objs, ref)                          # torch tensors in, tensor out
```

```python
# 2.0
import numpy as np
import evoxc

from evox import hv, key

objs = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
ref = np.array([2.0, 2.0], dtype=np.float32)
value = float(evoxc.evaluate(hv, key(0), objs, ref).numpy())
```

```text
3.245039939880371
```

The Monte Carlo sampler needs randomness, so `hv(key, objs, ref, num_sample=100000)` takes the key first; the same key and the same points give the same estimate, and `num_sample` is a static trace-time argument. `evoxc.evaluate(fn, *concrete_tensors)` is the short host-side form (build a graph from the argument shapes, then run it).
`gd`, `igd`, `gd_plus` and `igd_plus` keep their 1.x signatures — `gd(objs, pf)`, `igd(objs, pf, p=1.0)` — and only need the same wrapper, because evoxc has no eager mode and every `evox` metric must run inside a trace.

### Safe math helpers: `evox.utils` → `evox.operators.jit_fix_operator`

```python
# 1.x
from evox.utils import clamp, maximum, randint

x = clamp(x, lb, ub)
x = maximum(x, 0.0)
r = randint(0, 10, (4,))
```

```python
# 2.0
from evox import key
from evox.operators.jit_fix_operator import clamp, maximum, randint

x = clamp(x, lb, ub)
x = maximum(x, 0.0)
r = randint(key(0), 0, 10, (4,))    # leading key
```

The math is the same relu/`select` composition that keeps operator fusion safe, and the same caveat applies: these are trace-time functions, callable only inside an active graph. `randint` is the one signature that changes, because evoxc randomness is stateless.

### The whole program: PSO on Sphere

```python
# 1.x — pso_sphere.py
import torch
from evox.algorithms import PSO
from evox.problems.numerical import Sphere
from evox.workflows import EvalMonitor, StdWorkflow

torch.manual_seed(42)
dim = 10
algo = PSO(pop_size=50, lb=-32 * torch.ones(dim), ub=32 * torch.ones(dim))
wf = StdWorkflow(algorithm=algo, problem=Sphere(), monitor=EvalMonitor())

wf.init_step()
for gen in range(1, 100):
    wf.step()
    if gen % 20 == 0:
        print(f"gen {gen:3d}  best fitness {wf.monitor.get_best_fitness().item():10.4f}")
```

```python
# 2.0 — pso_sphere.py
from evox import EvalMonitorConfig, StdWorkflow
from evox.algorithms import make_pso
from evox.problems import Sphere

DIM = 10
wf = StdWorkflow(
    algorithm=make_pso(pop_size=50, lb=[-32.0] * DIM, ub=[32.0] * DIM),
    problem=Sphere(),
    monitor=EvalMonitorConfig(),
    opt_direction="min",
)

state = wf.init(seed=42)          # seeds the whole run; no global seed
state = wf.init_step(state)       # generation 1
for gen in range(1, 100):
    state = wf.step(state)
    if gen % 20 == 0:
        print(f"gen {gen:3d}  best fitness {wf.monitor.get_best_fitness():10.4f}")

print("final best solution:", wf.monitor.get_best_solution())
print("generations run:", int(state.generation.numpy()))
```

```text
gen  20  best fitness    16.7388
gen  40  best fitness     0.0656
gen  60  best fitness     0.0011
gen  80  best fitness     0.0000
final best solution: [ 5.0371531e-05 -4.5477755e-05  6.2948122e-05  5.0059789e-05
  1.9440666e-04 -1.2963169e-04 -1.1773363e-05 -7.5407603e-05
 -2.3137491e-04  8.7571068e-05]
generations run: 100
```

Sphere has its optimum at the origin, so a correct PSO drives the best fitness towards `0.0` and the best solution towards zero — the same qualitative behaviour as 1.x, reached through configs and state instead of modules and mutations.

If you do not need the per-generation print, the whole run collapses to one call: build the workflow with `num_generations=100` and use `best = wf.fit(seed=42)`, which returns the best fitness (`1.327e-07` for this seed) after running the loop for you.

## Removed in 2.0

| Removed | Why / what to use instead |
|---|---|
| `evox.core.ModuleBase`, `Parameter`, `Mutable` | The class hierarchy is gone: components are frozen config dataclasses plus plain functions. `evox.core.Algorithm` / `Problem` / `Monitor` are documentation-only `Protocol`s, so there is nothing to inherit. Static hyperparameters become config fields, mutable tensors become `<Component>State` fields. |
| `evox.core.compile`, `evox.core.vmap`, `evox.core.use_state` | The workflow compiles one step graph per step variant and per state shape; `vmap` has no analogue because a component owns a whole generation in one traced call, so a population is already a batch. |
| Distributed workflow: `StdWorkflow(enable_distributed=True, group=...)` | `evox`'s `StdWorkflow` accepts neither argument, and there is no `torch.distributed` integration. Device placement is `device=` plus `backend=`. |
| Brax / MuJoCo-Playground / supervised-learning problems in `evox.problems.neuroevolution` | These wrapped external libraries through `torch`; only the virtual Gaussian-noise ES family was ported (`evox.problems.neuroevolution.VirtualProblem`, `VirtualLoRAProblem`). |
| `evox.utils.ParamsAndVector`, `VmapInfo`, `register_vmap_op` | Parameter↔vector conversion and custom-vmap registration existed to serve `torch.func`/`vmap`; 2.0 has plain pytrees instead (`evox.core.tree_flatten/tree_leaves/tree_map/tree_unflatten`) and no `vmap` to register operators for. |
| `evox.triton_kernels` | The optional Triton package has no 2.0 counterpart; the virtual-ES noise generation it accelerated is a plain evoxc graph (`evox.algorithms.so.es_variants.virtual_noise`). |
| `nn.Module` transforms for `solution_transform` / `fitness_transform` | 2.0 takes plain callables (`Callable[[Any], Any]`); `torch.nn.Sequential` compositions must become ordinary Python functions. |

Where a replacement does exist:

- **Safe math helpers** → `evox.operators.jit_fix_operator` (traced, same math as 1.x's `evox.utils`).
- **Plotting** → `evox.vis_tools` (`plot_obj_space_1d`, `plot_obj_space_2d`, `plot_obj_space_3d`, `plot_dec_space`) or `wf.monitor.plot()`; both need the optional `plotly` extra.
- **Hyperparameter optimization** → `evox.problems.hpo_wrapper.HPOProblemWrapper`, a host-side rewrite: it substitutes candidate values into inner configs, rebuilds and runs an inner `StdWorkflow`, and reads back the inner monitor's best fitness. Its limitation matters for migration planning — it is **not** an `evox` `Problem`, so it cannot be the outer problem of a `StdWorkflow` (you drive it from host code such as `random_search`), it evaluates candidates one at a time instead of batched via `vmap`, only tunes plain static config fields, and re-traces the inner graph for every distinct candidate.

## Migration checklist

1. Pin the import: keep `import evox...` and confirm it now resolves to 2.0 (`evox.__version__` reports `2.0.0a1` when the distribution is installed) in a separate environment before touching any logic.
2. Convert algorithms: pick the matching config dataclass and build it with `make_*`; turn tensor bounds into flat `float` lists and drop `dim`/`device` arguments.
3. Convert problems: move each `class MyProblem(Problem)` body into a module-level `evaluate(config, state, pop)` next to its frozen config dataclass, replacing torch/numpy ops with `evoxc` ops.
4. Convert monitors: `EvalMonitor(...)` → `EvalMonitorConfig(...)`, and read results through `wf.monitor` after `init`/`run`/`fit` (drop the trailing `.item()`).
5. Thread state: `wf.init(seed=...)` then `state = wf.step(state)`; prefer `wf.run(...)`/`wf.fit(...)` wherever the loop body has no host-side work.
6. Replace seeding: delete `torch.manual_seed(...)`, pass `seed=` to `init`/`run`/`fit`, and use `evox.key(seed)` for graphs you build yourself.
7. Fix `hv` calls: add the leading key and wrap the call in `evoxc.evaluate` (or `evoxc.build` + `evoxc.run`); check that `gd`/`igd` calls are wrapped the same way.
8. Replace removed helpers: `evox.utils.switch/clamp/...` → `evox.operators.jit_fix_operator`, and rewrite anything relying on `ModuleBase`, `Parameter`, `Mutable`, `compile`, `vmap`, `register_vmap_op`, `ParamsAndVector`, the Triton package or the distributed switch.
9. Run the smoke test from {doc}`tutorial_part1` on the converted script; if it passes, your script is on the 2.0 API.

## Where to go next

Workflow and monitor details — including `EvalMonitorConfig` history options, `wf.monitor` accessors and plotting — are covered in {doc}`tutorial_part3`.
For the reasoning behind configs, state and the step protocol, read the architecture guide at {doc}`../guide/developer/architecture`.
Compiler backends and device placement (`backend=`, `device=`, `compile_options=`) are described in {doc}`../guide/advanced/backends`.
If you maintain a plugin package, the 2.0 extension mechanism (the `evox_ext` namespace package) is documented in {doc}`../guide/advanced/extensions`.
