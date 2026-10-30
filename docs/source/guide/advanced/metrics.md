# Multi-objective metrics

`evox_etl.metrics` holds the classic multi-objective quality indicators, ported to `evoxc`
tensors: seven plain Python functions that take an achieved objective matrix, the reference
front, and (for the stochastic estimators) a PRNG key, and return one **float32 scalar** each.

| Function | Signature | Measures |
|---|---|---|
| `gd` | `gd(objs, pf)` | generational distance — convergence, lower is better |
| `gd_plus` | `gd_plus(objs, pf, p=1.0)` | dominance-respecting GD+ |
| `igd` | `igd(objs, pf, p=1.0)` | inverted generational distance — convergence + spread |
| `igd_plus` | `igd_plus(objs, pf, p=1.0)` | dominance-respecting IGD+ |
| `hv` | `hv(key, objs, ref, num_sample=100000)` | Monte-Carlo hypervolume (bounding cube) |
| `bounding_cube_monte_carlo_hv` | same as `hv` | the estimator `hv` delegates to |
| `each_cube_monte_carlo_hv` | `(key, objs, ref, num_sample=100000)` | per-point-cube Monte-Carlo hypervolume |

`gd`, `gd_plus`, `igd`, `igd_plus` and `hv` are also re-exported from the `evox_etl` root
package; the two Monte-Carlo estimators are reachable only via `evox_etl.metrics`. For the front
these numbers score, and for plotting them, see {doc}`viz` and {doc}`operators`.

## Arguments and shapes

Every indicator takes `objs` (`(n, m)`, one row per achieved solution, all objectives
**minimized**) and `pf` (`(k, m)`, the reference front — the target compared against in the GD
family, the set to be covered in the IGD family). Hypervolume adds `ref` (`(m,)`), a scalar
`int64` PRNG key from `evoxc.random.key(seed)`, and the static scalars `p` (float, default
`1.0`; larger values weight the worst distances more) and `num_sample` (int, default `100000`).

All outputs are `float32` scalars — the package casts explicitly, since evoxc's dtype promotion
differs from PyTorch's — and all are minimization-based, so pass un-negated fitness from the
opt-direction-resolved workflow.

## Generational distance: `gd` and `gd_plus`

`gd(objs, pf)` builds the full `(n, k)` pairwise Euclidean distance matrix
(`pairwise_euclidean`, broadcast `expand_dims` arithmetic — evoxc has no `cdist`) and returns
`norm(min_k(d_ij)) / n`: the L2 norm of each solution's distance to its closest reference point,
divided by `n`. The norm runs over the whole vector, so one far outlier dominates the score.

`gd_plus(objs, pf, p=1.0)` replaces the plain distance with a dominance-aware one:
`sqrt(sum(max(pf - obj, 0)^2))` per solution — a front point *worse* than the solution
contributes nothing on that objective, so being better than the reference is never punished —
aggregated as `(sum(min_dis ** p) / n) ** (1 / p)`. It is `0` exactly when every achieved
solution weakly dominates at least one point of `pf`.

## Inverted generational distance: `igd` and `igd_plus`

`igd(objs, pf, p=1.0)` reverses the roles: it distances `pf` to `objs` (a `(k, n)` matrix) and
averages over the **front** points, `(mean(min_n(d) ** p)) ** (1 / p)`. Because every front point
must be reached, IGD also punishes a sparse or truncated front — the standard IGD motivation.

`igd_plus(objs, pf, p=1.0)` is the dominance-respecting variant: the per-front-point distance is
`sqrt(sum(max(obj - pf, 0)^2))` (reversed subtraction relative to `gd_plus`), averaged over the
`k` front points, `(sum(min_dis ** p) / k) ** (1 / p)`. In both `*_plus` variants the `p` power
is applied *before* the division.

## Hypervolume: `hv` and the two Monte-Carlo estimators

`hv(key, objs, ref, num_sample=100000)` takes a **leading PRNG key** (evoxc's RNG is key-based, so
stochastic functions need an explicit key first) and delegates to `bounding_cube_monte_carlo_hv`.

The bounding-cube estimator first maps each point through the internal `_distances_to_ref`
helper, `max(ref - objs, 0)`: objectives are minimized, so that is the extent of the box a point
dominates towards the reference point, and clipping at zero means a point not better than `ref`
along an objective contributes a **zero** extent there, hence adds nothing to the volume (the old
`|objs - ref|` behaviour, crediting points outside the reference cube, is gone). It then takes the
elementwise maximum over points as the bounding-cube edges `bound`, draws `num_sample` uniform
points in `[0, bound]`, counts those dominated (in *all* objectives) by at least one point, and
returns `count / num_sample * prod(bound)`.

`each_cube_monte_carlo_hv(key, objs, ref, num_sample=100000)` uses one cube per point instead of
one bounding cube: `split_n` keys, `num_sample // n` samples inside each point's own box, and
each dominated sample's credit split among the boxes containing it (`1 / dom_count`) to avoid
double counting. Same quantity, different variance; it divides by zero if `num_sample < n`.

## The evaluation model: traced graph functions

These are **traced graph functions**, not host-side NumPy helpers — there is no eager mode, they
only build IR. From an ordinary script you compile a one-shot executable with `evoxc.build` and
run it with `evoxc.run`, which returns concrete tensors (use `.numpy()`). Static scalars such as
`p` and `num_sample` must be passed to `evoxc.build` **and** re-passed by value to `evoxc.run`:

```python
import numpy as np
import evoxc
from evoxc.core import TensorSpec
from evox_etl.metrics import gd, gd_plus, igd, igd_plus

objs = np.array([[1.0, 0.0], [0.6, 0.6], [0.0, 1.0]], dtype=np.float32)  # (n=3, m=2)
pf = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=np.float32)    # (k=3, m=2)
specs = [TensorSpec(objs.shape, np.float32), TensorSpec(pf.shape, np.float32)]

for fn in (gd, igd):
    exe = evoxc.build(fn, *specs, backend="numpy")
    print(fn.__name__, float(evoxc.run(exe, objs, pf).numpy()))

exe = evoxc.build(gd_plus, *specs, 2.0, backend="numpy")   # p is a static arg
print("gd_plus p=2", float(evoxc.run(exe, objs, pf, 2.0).numpy()))
```

```
gd 0.047140464186668396
igd 0.047140464186668396
gd_plus p=2 0.0
```

The achieved row `(0.6, 0.6)` is worse than the front point `(0.5, 0.5)`, yet `gd_plus` reports
`0.0`: every solution dominates or matches its closest front point, so the clipped distances
vanish (`gd` and `igd` coincide here only because the two sets are symmetric).

## Worked example: IGD and hypervolume on DTLZ2

Run a small NSGA2 on DTLZ2, read the achieved front from the monitor, trace the problem's `pf`
helper for the true front, then score both.

```python
import numpy as np
import evoxc
from evoxc.core import TensorSpec

from evox_etl import StdWorkflow, EvalMonitorConfig
from evox_etl.algorithms import make_nsga2
from evox_etl.problems.numerical import DTLZ2
from evox_etl.problems.numerical.dtlz import pf as dtlz_pf
from evox_etl.metrics import igd, hv

problem = DTLZ2(d=7, m=2)
algo = make_nsga2(pop_size=40, n_objs=2, lb=np.zeros(7, dtype=np.float32),
                  ub=np.ones(7, dtype=np.float32))
wf = StdWorkflow(algo, problem, monitor=EvalMonitorConfig(multi_obj=True),
                 opt_direction=["min", "min"], backend="numpy")
wf.run(generations=40, seed=0)

objs = np.asarray(wf.monitor.get_pf_fitness(), dtype=np.float32)   # (n, 2), un-negated
front = evoxc.run(evoxc.build(dtlz_pf, problem), problem).numpy()  # (k, 2) true front

igd_exe = evoxc.build(igd, TensorSpec(objs.shape, np.float32),
                      TensorSpec(front.shape, np.float32), backend="numpy")
print("IGD:", float(evoxc.run(igd_exe, objs, front).numpy()))

ref = (1.1 * front.max(axis=0)).astype(np.float32)
hv_exe = evoxc.build(hv, TensorSpec((), np.int64), TensorSpec(objs.shape, np.float32),
                     TensorSpec(ref.shape, np.float32), 200000, backend="numpy")
print("ref:", ref, "HV:", float(evoxc.run(hv_exe, evoxc.random.key(0), objs, ref, 200000).numpy()))
```

```
IGD: 0.006482047960162163
ref: [1.1 1.1] HV: 0.41628843545913696
```

- `EvalMonitorConfig(multi_obj=True)` puts the monitor in Pareto mode; `get_pf_fitness()` returns
  the non-dominated subset of everything evaluated, host-side and already un-negated. With
  `full_fit_history=False` the history is empty and the call warns, then fails.
- `pf` is a graph function of the problem config only — `evoxc.build(pf, config)` then
  `evoxc.run(exe, config)` (see `unit_test/etl/problems/test_dtlz.py::run_pf`); DTLZ2's front lies
  on the unit sphere, so `front.max(axis=0) == [1, 1]`.
- Hypervolume is stochastic: another key moves the estimate by well under `1 %` at
  `num_sample=200000`. Fix the key and the sample budget when comparing runs.

## Choosing a reference point

The reference point defines *what* is measured; it is not a free parameter:

- Objectives are minimized, so `ref` must be **worse than every point you care about** —
  strictly, or boundary points contribute a zero extent on the touched axis.
- Points not dominating `ref` add **nothing** (`_distances_to_ref` clips their extent to zero);
  if every point is clipped on some axis the bounding cube collapses and the estimate is `0.0`.
- Hypervolume is **unnormalized** — a volume in objective space — so numbers from different
  `ref` values (or different objective scales) are not comparable.

Sweeping `ref` on the front from the example shows both failure modes:

| `ref` | hypervolume |
|---|---|
| `0.5 * worst corner` | `0.0000` — every achieved point is clipped away |
| `0.9 * worst corner` | `0.0790` — only points inside the shrunk cube count |
| `1.0 * worst corner` (exactly the front's corner) | `0.2053` — corner points touch the boundary |
| `1.1 * worst corner` | `0.4163` — the recommended, safe choice |
| `2.0 * worst corner` | `3.2001` — larger, but a different question |

The `1.1 *` convention is the practical default: a small margin beyond the worst front corner
makes every front point strictly dominate `ref`, at the cost of sampling a slightly larger cube
(more budget in empty space, so slightly higher variance). A too-small reference silently zeroes
the score; a far-away one reports a larger volume comparable with nothing else.

```{note}
These indicators are trace-only. Any HPO (see {doc}`hpo`), benchmarking or plotting script must
wrap each call in `evoxc.build` / `evoxc.run` (as above) before it has a number to print, log or
plot, and values come back to the host as plain NumPy. Cache the executables when a metric is
evaluated repeatedly — rebuilding per call dominates the runtime on a small front.
```

```{warning}
This page documents the **experimental EvoX 2.0 pre-release**. `evox_etl` is a functional rewrite
running on the `evoxc` tensor-graph compiler, so these signatures may still change before 2.0. For
custom algorithms and problems see {doc}`../developer/custom-alg-pro`; for the end-to-end
multi-objective walkthrough see {doc}`../../tutorial/tutorial_part5`. The 1.x `torch`-based `evox`
package is scheduled for removal and is not covered here.
```
