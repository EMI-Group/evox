# evox_etl/operators/crossover — functional crossover operators

## Intent
Port of torch evox crossover operators (`../../../evox/operators/crossover/`, read-only)
as PLAIN functions (no `@etl.defn` — see `../../DESIGN.md` §4.3): rng-using functions
gain `key` as first param, all functions use `etl`/`etl.numpy as enp`/`etl.random as
random` only (no torch/numpy in framework code).

## API Surface (implemented)
- `differential_evolution.py`: `DE_differential_sum(key, diff_padding_num, num_diff_vectors, index, population, F=None, replace=False)`, `DE_binary_crossover(key, mutation_vector, current_vector, CR)`, `DE_exponential_crossover(key, mutation_vector, current_vector, CR)`, `DE_arithmetic_recombination(mutation_vector, current_vector, K)` (deterministic, no key).
- `sbx.py`: `simulated_binary(key, x, pro_c=1.0, dis_c=20.0)` → (n2*2, m).
- `sbx_half.py`: `simulated_binary_half(key, x, pro_c=1.0, dis_c=20.0)` → (n2, m).
- `__init__.py` mirrors torch `__all__` + imports.

## Constraints / gotchas (verified against etl 0.1.0)
- **No `.ndim` on SymbolicTensor** — only `.shape` (tuple of static ints). Rank checks
  use `getattr(x, "shape", None) is not None and len(x.shape) == k` (also tolerates
  python-scalar statics).
- **`[:, None]` newaxis is NOT supported** by etl getitem (`too many indices (2) for
  rank 1`). Use `enp.expand_dims(a, axis)` instead. `[None, :]` likewise → `expand_dims(a, 0)`.
- **`etl.gather(x, indices, axis=0)` = numpy-take** (out.shape = x.shape[:axis] +
  indices.shape + x.shape[axis+1:]). For torch `gather(x, dim, idx)` (out[i,j] =
  x2d[idx[i,j], j]) use the flatten trick: `etl.gather(etl.reshape(tiled, (-1,)),
  idx_flat)` with int64 indices (see `DE_exponential_crossover`).
- **No `etl.zeros` inside traces** — initialize via `etl.select(cond, a, 0.0)`
  select-chains (see `simulated_binary` beta init).
- **`enp.log1p`/`enp.floor` don't exist** on `etl.numpy` — they live at `etl` top
  level (`etl.log1p`, `etl.floor`). `enp.sum(a, axis=..., ...)` keyword only.
- **`random.key(seed)` works EAGERLY** (concrete Tensor) — keys are graph inputs via
  `TensorSpec((), 'int64')`. `TensorSpec` is a plain dataclass (shape, dtype) — no
  `from_tensor` classmethod.
- **SBX beta can exceed 1** (μ>0.5 branch: `(2-2μ)^(-1/(η+1)) > 1`) — offspring may
  legitimately lie OUTSIDE [min(p1,p2), max(p1,p2)]; same as torch. Exact invariant
  instead: `c1 + c2 == p1 + p2` (symmetry around mid).
- Static python ints/floats (shapes, pro_c, dis_c, diff_padding_num) are fine:
  static slices + constant broadcast; scalar F/CR/K may be python floats.
- DE_differential_sum keeps the torch self-pick fix (map index→pop_size-1) unless
  `replace=True`; F=None reproduces the exact old torch behavior, else
  `difference_sum * F` (F expanded to (pop,1) when rank-1).

## Test strategy
Smoke-tested on etl numpy backend (throwaway script, not committed): exact parity vs
torch for `DE_arithmetic_recombination` (K scalar/(pop,)/(pop,1), atol 1e-6);
determinism (same key → same output) and children-from-parents for DE binary/expo
crossover; `DE_differential_sum` verified against a numpy rebuild from the identical
internal random draw (F=None/scalar/(pop,), replace True/False); SBX verified against
a numpy reimplementation on identical draws + symmetry + pro_c=0 edge case.
Unit tests live in `../../../unit_test/etl/operators/` (separate agent).
