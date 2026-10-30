# Porting MATLAB Code to EvoX

```{warning} Experimental EvoX 2.0 pre-release
These pages target the **EvoX 2.0 pre-release** (import name `evox_etl`, compiled by `evoxc`).
They do **not** cover the EvoX 1.x torch API (`import evox`, `ModuleBase`, `Parameter`/`Mutable`,
`torch.*`, `#evox.compile`), which was removed in 2.0 and is documented nowhere in these
pages. APIs may still change between pre-releases.
```

This page is for people arriving from MATLAB. It keeps the part of the mental model that transfers — an
evolutionary algorithm is a bounded loop over a *matrix* of candidates, and the whole matrix moves at
once — and rewrites the part that does not: EvoX 2.0 is NumPy-shaped Python, with tensors from the
`evoxc` compiler, frozen dataclasses instead of `struct`s, and explicit random keys instead of a global
`rand()`.

Every Python sample below runs as printed; the algorithm in the last section is a complete, working port
of the MATLAB PSO shown above it.

## Syntax differences

### Array creation and indexing

**MATLAB**

- 1-based indexing.
- Vectors and matrices are declared with square brackets and semicolons (`[1 2 3; 4 5 6]`); `rand()`
  returns values in $[0, 1)$.
- Slicing uses `(start:end)` with 1-based indices.

**EvoX / Python**

- 0-based indexing, everywhere.
- Arrays inside an algorithm are `evoxc` tensors. They are created from a random key
  (`evoxc.random.uniform(key, shape, 0.0, 1.0, evoxc.float32)`), from graph constants
  (`evoxc.ops.constant(...)`), or from another tensor by `expand_dims`/`reshape`/`concatenate`.
  NumPy-style constructors and shape helpers live in `evoxc.numpy`, conventionally imported as `enp`
  (`enp.zeros`, `enp.full`, `enp.reshape`, `enp.expand_dims`).
- Slicing uses `[start:end]` with 0-based indices, e.g. `pop[0]` is the first row.
- NumPy itself is a host-side tool in EvoX 2.0: use it to validate config values and to build data
  before the run, never for the arithmetic inside a step.

### Matrix computation

**MATLAB**

- `*` is linear-algebraic matrix multiplication; `.*` multiplies element-wise.
- `/` is matrix right division; `.^` is element-wise power.
- Trailing and leading dimensions of length 1 are **ignored**.
- Broadcasting is implicit: dimensions are extended automatically.

**EvoX / Python**

- Matrix multiplication is `@` or `evoxc.matmul`.
- `*` is element-wise, over the same or broadcastable shapes.
- `/` is element-wise division; `**` is element-wise power.
- Dimensions of length 1 are **preserved** and treated as broadcast dimensions.
- Broadcasting follows NumPy rules and is explicit about it: a `(dim,)` vector does not line up with a
  `(pop_size, dim)` matrix by accident — write `enp.expand_dims(vector, axis=0)` to make the row
  explicit.

### Functions and definitions

**MATLAB**

- A function is introduced by the `function` keyword; one file holds several functions, and the primary
  one usually shares the file name.
- Anonymous functions (`@(x) sum(x.^2)`) are the inline tool.

**EvoX / Python**

- Plain `def` functions in a module. EvoX 2.0 has no algorithm base class to inherit: `init`, `step` and
  the optional `init_step`/`final_step` are *module-level functions* that the workflow looks up on the
  module where your config dataclass is defined.
- Static configuration is a `@dataclass(frozen=True)` (the `struct` you pass around), and the mutable
  working set is another frozen dataclass whose fields are tensors. "Mutating" a field means returning
  `dataclasses.replace(state, field=value)`.
- `lambda` still exists for short callables, but multi-line lambdas are not allowed.

### Control flow

**MATLAB**

- `for i = 1:N ... end`, `if`/`elseif`/`else`, 1-based loops.

**EvoX / Python**

- `for i in range(N):`, `if`/`elif`/`else`, indentation for scoping, no `end` keyword.
- Python control flow may not depend on tensor *values* inside a step: the step graph is traced once and
  compiled. Decide branches host-side (before the run), or use `evoxc.cond`/`evoxc.while_loop` inside
  the graph.

### Printing and comments

**MATLAB**

- `fprintf()` for formatted output; `%` starts a comment.

**EvoX / Python**

- `print()` with f-strings (`print(f"Iteration = {i}, global best = {best:.6f}")`); `#` starts a comment.

### Multi-line coding

**MATLAB**

- A trailing `...` continues a statement on the next line.

**EvoX / Python**

- A trailing `\` continues a statement, but it is rarely needed: inside parentheses or brackets the
  expression simply continues to the next line.

## The PSO example in MATLAB

A MATLAB PSO, written the way it usually is: a struct for the state, one function to set it up, one to
advance it, and a driver that owns the loop.

```matlab
function [] = example_pso()
    pso = init_pso(100, [-10, -10], [10, 10], 0.6, 2.5, 0.8);
    test_fn = @(x) (sum(x .* x, 2));
    for i = 1:20
        pso = step_pso(pso, test_fn);
        fprintf("Iteration = %d, global best = %f\n", i, pso.global_best_fitness);
    end
end


function [self] = init_pso(pop_size, lb, ub, w, phi_p, phi_g)
    self = struct();
    self.pop_size = pop_size;
    self.dim = length(lb);
    self.w = w;
    self.phi_p = phi_p;
    self.phi_g = phi_g;
    % setup
    range = ub - lb;
    population = rand(self.pop_size, self.dim);
    population = range .* population + lb;
    velocity = rand(self.pop_size, self.dim);
    velocity = 2 .* range .* velocity - range;
    self.lb = lb;
    self.ub = ub;
    % mutable
    self.population = population;
    self.velocity = velocity;
    self.local_best_location = population;
    self.local_best_fitness = Inf(self.pop_size, 1);
    self.global_best_location = population(1, :);
    self.global_best_fitness = Inf;
end


function [self] = step_pso(self, evaluate)
    % Evaluate
    fitness = evaluate(self.population);
    % Update the local best
    compare = find(self.local_best_fitness > fitness);
    self.local_best_location(compare, :) = self.population(compare, :);
    self.local_best_fitness(compare) = fitness(compare);
    % Update the global best
    values = [self.global_best_location; self.population];
    keys = [self.global_best_fitness; fitness];
    [min_val, min_index] = min(keys);
    self.global_best_location = values(min_index, :);
    self.global_best_fitness = min_val;
    % Update velocity and position
    rg = rand(self.pop_size, self.dim);
    rp = rand(self.pop_size, self.dim);
    velocity = self.w .* self.velocity ...
        + self.phi_p .* rp .* (self.local_best_location - self.population) ...
        + self.phi_g .* rg .* (self.global_best_location - self.population);
    population = self.population + velocity;
    self.population = min(max(population, self.lb), self.ub);
    self.velocity = min(max(velocity, self.lb), self.ub);
end
```

Three habits stand out, and each one has a counterpart in EvoX 2.0: the state travels as one `struct`
(→ a frozen state dataclass), the loop lives in `example_pso` (→ the workflow owns it), and randomness
comes from a global generator (→ an explicit `key` that is split at every use).

## The same PSO in EvoX 2.0

### Using the built-in PSO

PSO already exists in `evox_etl.algorithms`, so the shortest port is a `make_*` constructor plus a
`StdWorkflow`. The workflow evaluates the population (`test_fn` above), keeps the generation counter,
and applies the optimization direction:

```python
import numpy as np

from evox_etl import EvalMonitorConfig, StdWorkflow
from evox_etl.algorithms import make_pso
from evox_etl.problems.numerical import Sphere

workflow = StdWorkflow(
    algorithm=make_pso(pop_size=100, lb=np.full(2, -10.0), ub=np.full(2, 10.0)),
    problem=Sphere(),  # sum(x ** 2), minimized at the origin
    monitor=EvalMonitorConfig(),
    opt_direction="min",
    num_generations=20,
)
workflow.run(seed=42)  # 20 generations, the workflow owns the loop
print(f"global best = {workflow.monitor.get_best_fitness():.6f}")
```

`make_pso` validates the bounds on the host and freezes a config dataclass; `Sphere()` is a problem
config dataclass. Everything the MATLAB version carries as a struct or a function handle is one of those
two objects, and `workflow.run()` is the `for i = 1:20` loop.

### Writing your own algorithm

When no built-in matches, write the algorithm yourself — this is the `init_pso`/`step_pso` pair,
translated. Two dataclasses replace the struct: the frozen **config** (hyperparameters, fixed for the
whole run) and the frozen **state** (tensors, one per generation). `init` builds the state,
`init_step` performs the first generation, and `step` performs every later one. All of them are plain
module-level functions.

```python
"""example_pso.py — a hand-written PSO for EvoX 2.0 (evox_etl)."""

from dataclasses import dataclass, replace

import evoxc
import evoxc.numpy as enp
import evoxc.random as random
import numpy as np

from evox_etl import EvalMonitorConfig, StdWorkflow
from evox_etl.problems.numerical import Sphere


@dataclass(frozen=True)
class PSOConfig:
    pop_size: int
    lb: tuple[float, ...]
    ub: tuple[float, ...]
    w: float = 0.6
    phi_p: float = 2.5
    phi_g: float = 0.8


@dataclass(frozen=True)
class PSOState:
    pop: evoxc.SymbolicTensor
    velocity: evoxc.SymbolicTensor
    fit: evoxc.SymbolicTensor
    local_best_location: evoxc.SymbolicTensor
    local_best_fit: evoxc.SymbolicTensor
    global_best_location: evoxc.SymbolicTensor
    global_best_fit: evoxc.SymbolicTensor
    key: evoxc.SymbolicTensor


def make_pso(pop_size, lb, ub, w=0.6, phi_p=2.5, phi_g=0.8):
    """Validate the hyperparameters on the host and freeze the config."""
    lb = tuple(float(v) for v in np.asarray(lb).reshape(-1))
    ub = tuple(float(v) for v in np.asarray(ub).reshape(-1))
    if len(lb) != len(ub):
        raise ValueError("lb and ub must have the same length")
    return PSOConfig(pop_size=pop_size, lb=lb, ub=ub, w=w, phi_p=phi_p, phi_g=phi_g)


def _bounds(config):
    """Bake the bounds as (1, dim) graph constants (NumPy lives host-side only)."""
    lb = evoxc.ops.constant(evoxc.core.tensor(np.asarray(config.lb, np.float32).reshape(1, -1)))
    ub = evoxc.ops.constant(evoxc.core.tensor(np.asarray(config.ub, np.float32).reshape(1, -1)))
    return lb, ub


def min_by(values, keys):
    """Index-min over concatenated candidates; returns (value, key)."""
    values = evoxc.concatenate(values, axis=0)
    keys = evoxc.concatenate(keys, axis=0)
    index = enp.reshape(evoxc.argmin(keys, axis=0), (1,))
    return (
        enp.reshape(evoxc.gather(values, index, axis=0), tuple(values.shape[1:])),
        enp.reshape(evoxc.gather(keys, index, axis=0), ()),
    )


def init(config, key):
    pop_size, dim = config.pop_size, len(config.lb)
    key, subkey = random.split(key)
    lb, ub = _bounds(config)
    span = ub - lb
    key_pop, key_vel = random.split(subkey)
    pop = span * random.uniform(key_pop, (pop_size, dim), 0.0, 1.0, evoxc.float32) + lb
    velocity = 2 * span * random.uniform(key_vel, (pop_size, dim), 0.0, 1.0, evoxc.float32) - span
    inf = enp.full((pop_size,), float("inf"), dtype=evoxc.float32)
    return PSOState(
        pop=pop,
        velocity=velocity,
        fit=inf,
        local_best_location=pop,
        local_best_fit=inf,
        global_best_location=pop[0],
        global_best_fit=enp.full((), float("inf"), dtype=evoxc.float32),
        key=key,
    )


def init_step(config, state, evaluate):
    fitness = evaluate(state.pop)
    global_best_location, global_best_fit = min_by([state.pop], [fitness])
    return replace(
        state,
        fit=fitness,
        local_best_fit=fitness,
        global_best_location=global_best_location,
        global_best_fit=global_best_fit,
    )


def step(config, state, evaluate):
    pop_size, dim = config.pop_size, len(config.lb)
    lb, ub = _bounds(config)
    better = state.local_best_fit > state.fit
    local_best_location = evoxc.select(
        enp.expand_dims(better, axis=1), state.pop, state.local_best_location
    )
    local_best_fit = evoxc.select(better, state.fit, state.local_best_fit)
    global_best_location, global_best_fit = min_by(
        [enp.expand_dims(state.global_best_location, axis=0), state.pop],
        [enp.expand_dims(state.global_best_fit, axis=0), state.fit],
    )
    key, subkey = random.split(state.key)
    key_rg, key_rp = random.split(subkey)
    rg = random.uniform(key_rg, (pop_size, dim), 0.0, 1.0, evoxc.float32)
    rp = random.uniform(key_rp, (pop_size, dim), 0.0, 1.0, evoxc.float32)
    velocity = (
        config.w * state.velocity
        + config.phi_p * rp * (local_best_location - state.pop)
        + config.phi_g * rg * (enp.expand_dims(global_best_location, axis=0) - state.pop)
    )
    pop = evoxc.clamp(state.pop + velocity, lb, ub)
    velocity = evoxc.clamp(velocity, lb, ub)
    return replace(
        state,
        pop=pop,
        velocity=velocity,
        fit=evaluate(pop),
        local_best_location=local_best_location,
        local_best_fit=local_best_fit,
        global_best_location=global_best_location,
        global_best_fit=global_best_fit,
        key=key,
    )


def report(i, state):
    """Print the algorithm's own best-so-far fitness (the `fprintf` line)."""
    best = state.algorithm_state.global_best_fit
    print(f"Iteration = {i}, global best = {float(best.numpy()):.6f}")


def main():
    dim = 10
    workflow = StdWorkflow(
        algorithm=make_pso(100, np.full(dim, -10.0), np.full(dim, 10.0)),
        problem=Sphere(),
        monitor=EvalMonitorConfig(),
        opt_direction="min",
        num_generations=20,
    )
    # the `for i = 1:20` loop, driven by hand
    state = workflow.init(seed=42)
    # the first generation: the module's init_step (evaluate the initial pop)
    state = workflow.init_step(state)
    report(1, state)
    for i in range(2, 21):
        state = workflow.step(state)
        report(i, state)


if __name__ == "__main__":
    main()
```

Notes on the translation, in the order the MATLAB code makes them:

- **`self.population` and friends** are now fields of an immutable `PSOState`; the `self.x = ...`
  assignments become one `replace(state, x=...)` at the end of `step`.
- **`rand(self.pop_size, self.dim)`** becomes `random.uniform(key, shape, 0.0, 1.0, evoxc.float32)`
  with a key derived by `random.split`. Same distribution, but the randomness is an input, so a run is
  reproducible from `seed=42` and the step graph can be compiled once and replayed.
- **`Inf(self.pop_size, 1)`** becomes `enp.full((pop_size,), float("inf"), dtype=evoxc.float32)`; an
  empty-shape constant is `enp.full((), ...)`.
- **`[a; b]` and `[a, b]`** concatenate in MATLAB; here it is `evoxc.concatenate([...], axis=0)`, and the
  operands must already agree on rank — that is what `enp.expand_dims(x, axis=0)` is for.
  `values(min_index, :)` row extraction becomes `evoxc.gather(values, index, axis=0)` plus a `reshape`.
- **`min(max(x, lb), ub)`** is `evoxc.clamp(x, lb, ub)`.
- **`evaluate(self.population)`** is the `evaluate` argument the workflow passes into `step`: it applies
  your problem's fitness, the optimization direction and the monitor. You never call the objective
  function directly.
- **`for i = 1:20`** belongs to the driver above: `workflow.run(20)` performs exactly that loop
  (`init_step`, then `step`, then an optional `final_step`). Driving `init_step`/`step` yourself is the
  same loop unrolled, which is how the per-generation printout above reaches the algorithm's own
  `global_best_fit`.
- **Reading a tensor back** to Python requires `.numpy()` — and on a non-CPU device, first
  `.to(evoxc.core.Device("cpu"))`. Readbacks are host-side work: keep them out of the generation loop or
  accept the cost, as in any accelerator program.

```{note} Two habits that do not survive the move
`global` variables and a mutable module-level RNG have no place in a traced step: the step graph must be a
function of its inputs. Likewise, per-individual loops (`for p = 1:pop_size`) are pointless — the whole
population is one tensor and every PSO expression above is a single vectorized op over it. That is the
actual reason EvoX exists: `evaluate` sees all candidates at once, and on a GPU that one call is the
parallel work.
```

## Where to go next

{doc}`../tutorial/tutorial_part2` covers the config/state/step contract in full,
{doc}`../guide/developer/custom-alg-pro` walks through a custom algorithm and problem inside the
framework, {doc}`../guide/advanced/operators` lists the operator catalog you can compose from, and
{doc}`selecting_gpu` explains how to run any of it on a GPU.
