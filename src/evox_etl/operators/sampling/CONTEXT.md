# evox_etl/operators/sampling — plain functional sampling operators

## Intent
ETL port of torch evox sampling operators (`src/evox/operators/sampling/`,
read-only). All functions are PLAIN Python functions (no `@etl.defn`) that compose
inside traces — see `../../DESIGN.md` §4.3 for the binding rule.

## API Surface
- `uniform.py`: `uniform_sampling(n, m) -> (w: float32 (n', m), n_samples: int)` —
  Das-Dennis sampling, deterministic, static-Python combinatorics; constants via
  `etl.constant(etl.core.tensor(...))`.
- `gird.py` (torch file name typo kept): `grid_sampling(n, m) -> (w: float32, n_samples)` —
  deterministic grid; no etl meshgrid → per-axis reshape+broadcast+stack.
- `latin_hypercube.py`:
  - `latin_hypercube_sampling_standard(key, n, d, smooth=True)` — key FIRST (two
    random draws, keys split via `random.split`), no device arg. torch gather→
    flatten trick: `etl.gather(reshape(cells,(-1,)), perms*d + col_idx)`.
  - `latin_hypercube_sampling(key, n, lb, ub, smooth=True)` — `lb[None,:]` →
    `etl.reshape(lb, (1, -1))` (etl getitem supports NO None/newaxis indexing).
- `__init__.py`: mirrors torch `__all__` + imports exactly.

## Gotchas (verified against etl)
- `etl.reshape` requires a TUPLE shape (lists raise TypeError).
- `etl.getitem` supports only static ints/slices — no `None`/newaxis/ellipsis.
- `etl.build` has no static-args support; static Python ints go via
  `functools.partial` closures. Returning static Python ints alongside tensors
  (e.g. `n_samples`) works.
- `etl.random.key(seed)` returns a SCALAR int64 tensor (shape `()`) — key specs
  are `etl.core.TensorSpec((), "int64")`.
- Weak Python scalars (int/float) keep float32 promotion (matches torch).

## Constraints
No torch/numpy imports. Deterministic functions must match torch bit-exactly
(verified vs `evox.operators.sampling` incl. the h2 branch of uniform_sampling).
Smoke tests live in `/tmp` throwaway scripts; unit tests are maintained by a
separate agent under `unit_test/etl/operators/`.
