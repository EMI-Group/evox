# Operators

Every variation mechanism in EvoX 2.0 — sampling, selection, crossover, mutation — is a plain, stateless Python
function that takes `evoxc` tensors in and returns `evoxc` tensors out. Operators hold no state, no
hyperparameters and no configuration: an algorithm keeps the population, the fitness and the PRNG key in its own
state dataclass and calls the operators it needs from inside its `step` function. That is why a single
`polynomial_mutation` serves NSGA-II, RVEA and your own algorithm without any subclassing, and why an operator can
be swapped by passing a different callable (see {doc}`../developer/custom-alg-pro`).

The package is split into `sampling`, `selection`, `crossover` and `mutation` — one module per operator family —
plus `jit_fix_operator`, a module of compiler-safe scalar helpers. Each section below lists the exact signatures
and return shapes as they are defined in the source.

## Sampling — `evox_etl.operators.sampling`

| Signature | Returns | Semantics |
| --- | --- | --- |
| `uniform_sampling(n, m)` | `(w, n_samples)` | Das–Dennis simplex-lattice weight vectors; `w` is `(n_samples, m)` `float32` with unit row sums (entries floored at `1e-6`). Deterministic — no key. |
| `grid_sampling(n, m)` | `(w, num_samples)` | Regular grid of `ceil(n ** (1 / m)) ** m` points over `[0, 1]^m` (endpoints included); columns are reversed to match the reference ordering. Deterministic. |
| `latin_hypercube_sampling(key, n, lb, ub, smooth=True)` | `(n, d)` | Latin hypercube inside the box given by the 1-D `lb`/`ub` (same shape and dtype); `d = lb.shape[0]`. |
| `latin_hypercube_sampling_standard(key, n, d, smooth=True)` | `(n, d)` | The same sampler in the unit hypercube; `smooth=True` jitters inside each cell, `smooth=False` returns cell centres. |

```{note}
The two `latin_hypercube_*` functions are key-first: they take an `evoxc` PRNG key and split it internally.
`uniform_sampling`/`grid_sampling` use no key and return `(tensor, int)`, the int being the number of rows
actually produced — for `uniform_sampling` typically *fewer* than the requested `n`.
```

## Selection — `evox_etl.operators.selection`

All selection operators assume **minimization**: the lowest fitness wins.

| Signature | Returns | Semantics |
| --- | --- | --- |
| `tournament_selection(key, n_round, fitness, tournament_size=2)` | `int32 (n_round,)` | `n_round` independent tournaments over the 1-D `fitness`; winner per round = `argmin`. |
| `tournament_selection_multifit(key, n_round, fitnesses, tournament_size=2)` | `int32 (n_round,)` | Same, but `fitnesses` is a **Python list** of 1-D tensors compared lexicographically — the *last* entry is the primary key. |
| `select_rand_pbest(key, percent, population, fitness)` | `(pop_size, dim)` | For each individual, one random personal best drawn from the top `max(int(pop_size * percent), 1)` of `population` by `fitness`. |
| `non_dominate_rank(x)` | `int32 (n,)` | 0-based non-domination ranks of the `(n, m)` objective matrix `x`; the first Pareto front is rank 0. |
| `crowding_distance(costs, mask)` | `(n,)` | Per-solution crowding distance over `(n, m)` `costs`; `mask` is a 1-D bool tensor selecting the front (`None` = every row). Masked-out rows get `-inf`, front extremes `inf`. |
| `nd_environmental_selection(x, f, topk)` | `(x, f, rank, crowding_dis)` | NSGA-II environmental selection: rank first, crowding distance as tie-breaker; all four outputs are truncated to `topk`. |
| `ref_vec_guided(x, f, v, theta)` | `(next_x, next_f)` | RVEA APD-based selection against the reference vectors `v` (`(r, m)`); one survivor per reference vector, with `NaN` rows for partitions that received no solution. |

```{note}
`dominate_relation(x, y)` and `apd_fn(...)` live in `evox_etl.operators.selection.non_dominate` and
`evox_etl.operators.selection.rvea_selection`, but — mirroring the reference — neither is re-exported by
`evox_etl.operators.selection`; only the seven functions above are public there, and two of them
(`crowding_distance`, `non_dominate_rank`) are also re-exported at the `evox_etl.operators` top level.
```

## Crossover — `evox_etl.operators.crossover`

| Signature | Returns | Semantics |
| --- | --- | --- |
| `DE_differential_sum(key, diff_padding_num, num_diff_vectors, index, population, F=None, replace=False)` | `(difference_sum, rand_indices[:, 0])` | Sum of `num_diff_vectors` difference vectors drawn from `population`; `num_diff_vectors`/`index` are int tensors of shape `(pop_size,)` and `replace=False` keeps an individual from picking itself. `F` (scalar or `(pop_size,)`) scales the sum. |
| `DE_binary_crossover(key, mutation_vector, current_vector, CR)` | `(n, d)` | Per-gene uniform crossover with probability `CR` (a `(n,)` `CR` is broadcast over genes) plus one forced gene from the mutant. |
| `DE_exponential_crossover(key, mutation_vector, current_vector, CR)` | `(n, d)` | Geometric, contiguous-block variant of the same crossover. |
| `DE_arithmetic_recombination(mutation_vector, current_vector, K)` | `(n, d)` | `current + K * (mutation - current)`; deterministic, no key. |
| `simulated_binary(key, x, pro_c=1.0, dis_c=20.0)` | `(2 * (n // 2), d)` | SBX: parent `i` of the first half is paired with parent `n//2 + i`; both children are returned, so `c1 + c2 == p1 + p2`. |
| `simulated_binary_half(key, x, pro_c=1.0, dis_c=20.0)` | `(n // 2, d)` | Same pairing, but only `mid + beta * half_diff` is kept — one child per pair. |

## Mutation — `evox_etl.operators.mutation`

| Signature | Returns | Semantics |
| --- | --- | --- |
| `polynomial_mutation(key, x, lb, ub, pro_m=1.0, dis_m=20.0)` | `(n, d)` | Bounded polynomial mutation; `x` is first clamped into `[lb, ub]` and the result stays inside it. The per-gene rate is `pro_m / d`, so `pro_m=1.0` mutates roughly one gene per individual, not every gene. |

## The pure-function contract

Everything in `evox_etl.operators` obeys the same five rules, which is what makes them reusable and traceable:

- **Stateless.** No module-level mutable state, no `self`, no cached buffers, nothing to reset between
  generations; every quantity an operator needs arrives as an argument.
- **Randomness is explicit.** Anything random takes an `evoxc` PRNG key as its first parameter and splits it
  itself (`evoxc.random.split` / `evoxc.random.split_n`); keys are never consumed, so the same key plus the same
  inputs always returns the same output.
- **Static shapes only.** Shapes are fixed at trace time and nothing depends on runtime tensor *values* for
  indexing or branching — where the reference branches, the port uses `evoxc.select` (or `evoxc.while_loop` in
  `non_dominate_rank`), which is how non-dominated sorting stays traceable.
- **Trace-only.** Operators are plain Python functions, **not** `@evoxc.defn`s, and `evoxc` has no eager mode:
  calling one outside a trace raises `TraceError: No active trace ...`. Reach them through `evoxc.build`/
  `evoxc.run` or from inside another traced function — see {doc}`backends`.
- **Deterministic for a fixed key**, so seeding the key once replays a whole operator chain bit-for-bit.

## Compiler-safe helpers: `evox_etl.operators.jit_fix_operator`

The algorithm ports do not call raw `evoxc` scalar ops at their variation sites; they call these fifteen helpers,
because the compiler promotes dtypes differently from PyTorch (`float32 * int32` → `float64`, `int32 + python int`
→ `int64`) and a few torch ops have no direct `evoxc` counterpart. They are **deliberately not** re-exported by
`evox_etl.operators` or `evox_etl.utils` — import from `evox_etl.operators.jit_fix_operator` explicitly.

| Helper | Semantics |
| --- | --- |
| `switch(label, values)` | Per-element pick from a list of tensors by integer `label` (`label <= 0` → `values[0]`, `label >= len(values) - 1` → the last one). Replaces a `where` chain. |
| `clamp(a, lb, ub)` | Elementwise clamp into tensor bounds `[lb, ub]`, dtype-safe for ints. |
| `clamp_float(a, lb, ub)` | Clamp into scalar float bounds. |
| `clamp_int(a, lb, ub)` | Clamp into scalar int bounds, preserving `a`'s dtype. |
| `clip(a)` | Clamp into `[0, 1]`. |
| `lexsort(keys, dim=-1)` | Stable multi-key argsort; like NumPy's, the **LAST** key is primary and earlier keys only break ties. |
| `maximum(a, b)` / `minimum(a, b)` | Elementwise max/min of two tensors. |
| `maximum_float(a, b)` / `minimum_float(a, b)` | Same, against a scalar float. |
| `maximum_int(a, b)` / `minimum_int(a, b)` | Same, against a scalar int, dtype preserved. |
| `nanmax(input_tensor, dim=-1, keepdim=False)` | NaN-ignoring max, returning a **`(values, indices)` pair**. |
| `nanmin(input_tensor, dim=-1, keepdim=False)` | NaN-ignoring min, returning a **`(values, indices)` pair**. |
| `randint(key, low, high, size, dtype=None)` | Keyed `randint` with torch-style bounds (`low`/`high` may be tensors). |

```{tip}
Reach for these helpers — not the raw operators — whenever you combine tensors of different dtypes. `evoxc.relu`
promotes an integer tensor to `float64`, so `clamp`, `maximum` and `minimum` are composed through a dtype-aware
`relu` internally and give you your original dtype back. `lexsort` is what you want for "sort by rank, then by
crowding distance" orderings — the *last* key is primary. A private `_take_along_axis` also lives in the module.
```

## Using operators inside a custom algorithm

An operator is just a function, so a custom `step` is nothing more than a few calls inside one traced function.
Everything is traced once with `evoxc.build` — tensor arguments are declared with `evoxc.core.TensorSpec`, static
arguments (ints, floats, `None`, configs) are passed as plain values — and then replayed with `evoxc.run`:

```python
import numpy as np, evoxc
import evoxc.random as random
from evoxc import core
from evox_etl.operators.jit_fix_operator import clamp
from evox_etl.operators.mutation import polynomial_mutation

def mutate_step(key, pop, lb, ub):
    """One step: mutate the whole batch, then clamp every child into bounds."""
    k_mut, _ = random.split(key)
    return clamp(polynomial_mutation(k_mut, pop, lb, ub), lb, ub)

KEY, BND = core.TensorSpec((), np.dtype("int64")), core.TensorSpec((4,), np.dtype("float32"))
POP = core.TensorSpec((8, 4), np.dtype("float32"))
exe = evoxc.build(mutate_step, KEY, POP, BND, BND, backend="numpy")

pop = np.random.default_rng(0).uniform(-1.5, 1.5, (8, 4)).astype(np.float32)
lb, ub = np.full(4, -1.0, np.float32), np.full(4, 1.0, np.float32)
out = evoxc.run(exe, np.array(7, dtype=np.int64), pop, lb, ub)

print(out.numpy().shape, out.numpy().dtype, "|", type(out).__module__ + "." + type(out).__name__)
print(np.round(pop[0], 4), "->", np.round(out.numpy()[0], 4))
```

```
(8, 4) float32 | evoxc.core.tensor.Tensor
[ 0.4109 -0.6906 -1.3771 -1.4504] -> [ 0.4109 -0.6906 -1.     -1.    ]
```

`evoxc.run` hands back concrete `evoxc.core.tensor.Tensor` objects, so read results with `.numpy()` (`np.asarray`
on a tensor is not the supported path). Note the last two genes: that row came in outside the box and `clamp` —
not the mutation — pulled it back to `-1.0`; the same key always replays the same array. `switch` is the
idiomatic way to pick a tensor per element without a `where` chain, clamping out-of-range labels to the ends:

```python
import numpy as np, evoxc
from evoxc import core
from evox_etl.operators.jit_fix_operator import switch

def pick(label, a, b):
    return switch(label, [a, b])

LBL, VEC = core.TensorSpec((5,), np.dtype("int32")), core.TensorSpec((5,), np.dtype("float32"))
exe = evoxc.build(pick, LBL, VEC, VEC, backend="numpy")
out = evoxc.run(exe, np.array([0, 1, 7, -1, 1], dtype=np.int32),
                np.full(5, 10.0, np.float32), np.full(5, 20.0, np.float32))
print(out.numpy())   # [10. 20. 20. 10. 20.]
```

## Injecting custom operators into MO algorithms

The multi-objective constructors accept replacement operators, so a research variant is a config change rather
than a fork. `None` (the default) means "use the built-in operator", and an all-`None` config is byte-identical
to the built-in behavior — same RNG draw order, same results.

| Constructor | `selection_op` | `mutation_op` | `crossover_op` | Built-in defaults |
| --- | --- | --- | --- | --- |
| `make_nsga2(pop_size, n_objs, lb, ub, ...)` | yes | yes | yes | `tournament_selection_multifit` / `polynomial_mutation` / `simulated_binary` |
| `make_nsga3(pop_size, n_objs, lb, ub, ..., data_type=None)` | yes | yes | yes | same as NSGA-II |
| `make_moead(pop_size, n_objs, lb, ub, ...)` | accepted but ignored | yes | yes | — / `polynomial_mutation` / `simulated_binary_half` |
| `make_rvea(pop_size, n_objs, lb, ub, alpha=2.0, fr=0.1, max_gen=100, ...)` | yes | yes | yes | `ref_vec_guided` / `polynomial_mutation` / `simulated_binary` |
| `make_rveaa(pop_size, n_objs, lb, ub, alpha=2.0, fr=0.1, max_gen=100, ...)` | yes | yes | yes | same as RVEA |
| `make_hype(pop_size, n_objs, lb, ub, n_sample=10000, ...)` | not exposed | yes | yes | — / `polynomial_mutation` / `simulated_binary` |

A custom callable must match the keyed signature of the operator it replaces: `selection_op(key, n_parents, fitnesses)`
for the tournament-style algorithms (NSGA-II, NSGA-III), or `selection_op(x, f, v, theta)` — **no key** — for the
reference-vector ones (RVEA, RVEAa); `crossover_op(key, x)` (parents in, children out); and
`mutation_op(key, x, lb, ub)`, with `lb`/`ub` arriving as baked `(dim,)` `float32` tensors.

```python
import numpy as np
from evox_etl.algorithms.mo import make_nsga2
from evox_etl.operators.crossover import simulated_binary

cfg = make_nsga2(pop_size=20, n_objs=2,
                 lb=np.zeros(4, dtype=np.float32), ub=np.ones(4, dtype=np.float32),
                 crossover_op=simulated_binary)    # None -> the built-in SBX
print(cfg.crossover_op is simulated_binary)        # True
print(cfg.selection_op, cfg.mutation_op)           # None None
print(cfg.lb, cfg.ub)                              # (0.0, 0.0, 0.0, 0.0) (1.0, 1.0, 1.0, 1.0)

try:
    make_nsga2(pop_size=20, n_objs=2, lb=[0.0] * 4, ub=[1.0] * 4, crossover_op="sbx")
except ValueError as err:
    print(err)   # crossover_op must be callable or None, got 'sbx'
```

The constructors normalize `lb`/`ub` to flat tuples of plain Python floats and reject a non-callable op field at
construction time, so a typo fails immediately instead of at trace time. The resulting config is a frozen
dataclass registered with `evoxc` as a **zero-child pytree node**: callables travel as one opaque static config
node, not as tensor leaves — which is why `evoxc.build`/`evoxc.run` can carry a function reference at all. A
non-`None` callable is therefore only valid in a field the config actually declares: `make_hype` has no
`selection_op`, and `make_moead`'s `selection_op` is accepted for signature parity but never called.

```{note}
For the multi-objective quality indicators that consume these outputs (hypervolume, IGD, GD), see {doc}`metrics`;
for a walkthrough of composing operators into a full run, see {doc}`../../tutorial/tutorial_part4`.
```

```{warning}
This page documents the **experimental EvoX 2.0 pre-release** API: the functional `evox_etl` package running on the
`evoxc` tensor-graph compiler. Operator names, argument order and the `jit_fix_operator` helper set may still
change between pre-release builds, and there is no eager mode to fall back on — code written against the 1.x
PyTorch API will not run here. Pin the version you build against and re-read this page when you upgrade.
```
