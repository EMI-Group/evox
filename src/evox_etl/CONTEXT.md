# evox_etl — Functional EvoX on ETL

## Intent
A functional rewrite of the EvoX framework on the ETL tensor library (`etl`, foreign
repo `/mnt/local-ssd/bchuang/etl`, installed editable in the shared venv). Replaces the
torch OOP design (`src/evox/`, kept untouched as reference) with plain functional code:
frozen config dataclasses, namedtuple/dataclass tensor states, `@etl.defn` pure
functions, separate `init(config, key) -> state` functions, and a compile-once
StdWorkflow loop.

**Binding spec: `DESIGN.md` in this directory — read it fully before writing any code.**

## API Surface
- `evox_etl.core`: `Algorithm`/`Problem`/`Monitor` protocols (duck-typed), `StdWorkflow`,
  `WorkflowState`, state helpers.
- `evox_etl.algorithms`: SO (de/es/pso variants) + MO (nsga2, nsga3, moead, rvea,
  rveaa, hype) — each module = config dataclass + `init/ask/tell` defn functions.
- `evox_etl.operators`: pure functions (sampling, selection, crossover, mutation).
- `evox_etl.problems.numerical`: basic, dtlz, cec2022.
- `evox_etl.metrics`: igd, gd, hv. `evox_etl.workflows`: std_workflow, eval_monitor.
- `evox_etl.utils`: tree helpers, min_by, dominate_relation, pairwise distances,
  parse_opt_direction, rank.

## Constraints
- NO eager tensor ops — everything inside `@etl.defn`/traces (ETL has no eager mode).
- Config fields are Python scalars partialized away at compile time; state leaves are
  tensors ONLY. Minimization semantics internally (workflow applies opt_direction).
- No torch/numpy in framework code (numpy allowed only to load cec2022 input data).
- Etl issues found while implementing: record under "ETL issues found" below and
  escalate to the root agent (do NOT edit the etl repo from here).

## Routing Table
| Area | Path | Notes |
|---|---|---|
| Core (protocols, workflow, monitor, state) | `core/` | Foundation — implement FIRST |
| Operators (pure functions) | `operators/` | sampling/selection/crossover/mutation |
| Algorithms SO | `algorithms/so/` | de_variants, es_variants, pso_variants |
| Algorithms MO | `algorithms/mo/` | nsga2, nsga3, moead, rvea, rveaa, hype |
| Numerical problems | `problems/numerical/` | basic, dtlz, cec2022 |
| Metrics | `metrics/` | igd, gd, hv |
| Workflow + EvalMonitor | `workflows/` | std_workflow, eval_monitor |
| Utilities | `utils/` | functional helpers |
| Tests | `../unit_test/etl/` | sibling — mirrors this package |
| Benchmarks (torch vs etl) | `../benchmarks/etl_vs_torch/` | sibling — comparison harness |
| Reference (torch) impl | `../evox/` | sibling — READ-ONLY, never modify |
| ETL library | (foreign repo `/mnt/local-ssd/bchuang/etl`) | read via the venv install; fixes only by root agent |

## Environment
Shared venv interpreter: `/mnt/local-ssd/bchuang/evox/.venv/bin/python` (python3.11,
torch 2.6.0+cu124 CUDA works, evox+etl editable, pytest). Tests:
`/mnt/local-ssd/bchuang/evox/.venv/bin/python -m pytest unit_test/etl -q`.
GPUs: 3× RTX A6000 (scan `nvidia-smi` for the most-free GPU before GPU runs).

## ETL issues found (escalated to root agent)
1. **np.ndarray fields in config dataclasses are rejected by `etl.build`** as static
   trace values ("neither a TensorSpec nor a static Python value"; static values may
   be None/bool/int/float/complex/str/Enum/dtype/slice/...). DESIGN.md §4.1 says
   config `lb`/`ub` become numpy arrays — not possible as-is; workaround used by the
   de_variants ports: normalize array fields to tuples of plain floats in
   `__post_init__` (callers still pass np arrays). Either update DESIGN.md or have
   etl accept ndarray as a static value.
   Repro: `etl.build(lambda cfg: cfg, MyConfig(lb=np.zeros(3)), backend="numpy")`
   → `etl.core.errors.TraceError` at `_flatten_specs`.
2. **`etl.select` does not numpy-broadcast a `(n,)` condition against `(n, m)`
   branches** (`cannot broadcast incompatible dims n and m`; the `(n,)` is aligned
   as `(1, n)`). numpy/torch broadcast `(n,)` → `(n, 1)` against `(n, m)` fine.
   Workaround: always `enp.expand_dims(cond, 1)` first (scalar conds broadcast OK).
   Repro: `etl.build(lambda x: etl.select(etl.sum(x, axes=1) < 0.5, 0.0, x),
   TensorSpec((3, 4), float32))` → ShapeError.
