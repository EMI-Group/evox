# evox_etl/algorithms — SO and MO algorithms in functional style

## Intent
Port of all 34 torch evox algorithms (read-only reference in `../../../evox/
algorithms/`) to `init/ask/tell` `@etl.defn` functions + frozen config dataclasses.
See `../../DESIGN.md` §4-5.

## Shims
`_shim_crossover.py` — plain-function etl port of the torch crossover
operators (`simulated_binary`, `simulated_binary_half`, `DE_differential_sum`,
`DE_binary_crossover`, `DE_exponential_crossover`, `DE_arithmetic_recombination`)
with key-first RNG params. Temporary: algorithm ports import from here until
`evox_etl.operators.crossover` lands, then it should be deleted.
`_shim_utils.py` — utility shims (written in parallel; import it for
`minimum_int` etc. once it exists).

## Routing Table
| Area | Path |
|---|---|
| Selection shims (tournament + pbest, plain functions) | `_shim_selection_basic.py` |
| DE variants (code, de, jade, ode, sade, shade) | `so/de_variants/` |
| ES variants (adam_step, ars, asebo, cma_es, des, esmc, guided_es, nes, noise_reuse_es, open_es, persistent_es, snes, sort_utils, virtual_lora_es) | `so/es_variants/` |
| PSO variants (clpso, cso, dms_pso_el, fs_pso, pso, sl_pso_gs, sl_pso_us, utils) | `so/pso_variants/` |
| MO algorithms (nsga2, nsga3, moead, rvea, rveaa, hype) | `mo/` |
| Tests | `../../unit_test/etl/` | sibling — mirrors this package |

## Notes for agents (verified against etl 0.1.0 — do not re-investigate)
- `etl.gather(x, idx, axis)` is numpy `take` semantics (index array applied to
  every row), NOT torch `gather`/`take_along_axis`. For row-local selection use
  `_take_along_axis` from `_shim_selection_basic.py` (flatten-trick, 2-D).
- etl has NO `unbind` / `expand_dims` / `squeeze` / `take_along_axis` — use
  `__getitem__` slices/ints (trace fine) and `enp.reshape` instead.
- RNG: functions take the key as an explicit first argument; allocate ONE
  `random.split` subkey per draw (`key_rand, _ = random.split(key)`), same draw
  order and distribution as torch. `random.randint(key, shape, low, high,
  dtype=etl.int32)` draws WITH replacement (high exclusive). Keys must be graph
  inputs (spec shape=(), dtype int64) wrapped via `etl.core.from_numpy` at run.
- `random.split` returns TWO keys (JAX-style). `etl.argsort(x, axis, stable=True)`
  and `etl.argmin(x, axis=...)` return int64 indices.
- The torch evox `lexsort` (`src/evox/utils/jit_fix_operator.py` lines 216-252)
  is empirically LAST-key primary (numpy lexsort convention), despite its
  docstring prose — verified against torch. Port it 1:1 (`_lexsort` in
  `_shim_selection_basic.py`), do not "fix" the key order.
- Both `list` and `tuple` pytrees work as `etl.build` specs and `etl.run` inputs.
- Static ints (n_round, tournament_size, pop_size, top_p_num) are read from
  tensor `.shape` at trace time — fine on the numpy backend.
- Self-test for the selection shims lives at
  `../../unit_test/etl/algorithms/test_shim_selection_basic.py` (to be created by
  the unit_test node agent): seed 1026 is pinned because with-replacement draws
  only guarantee every full-size tournament row contains the argmin candidate for
  that seed (verified: w_full all 5, w_multi all 1).
