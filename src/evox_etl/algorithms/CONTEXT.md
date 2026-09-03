# evox_etl/algorithms — SO and MO algorithms in functional style

## Intent
Port of all 34 torch evox algorithms (read-only reference in `../../../evox/
algorithms/`) to `init/ask/tell` plain functions + frozen config dataclasses.
See `../../DESIGN.md` §4-5.

## Shared shims (module-level files in this directory)
Temporary local fallbacks for operator/util functions while `evox_etl.operators`
and `evox_etl.utils` are being built in parallel. Algorithm ports import from
these until the real modules land.

- `_shim_utils.py` — torch util ports: clamp, clamp_float, clamp_int, maximum,
  minimum (+int variants), lexsort, nanmin, nanmax, randint (keyed).
- `_shim_crossover.py` — simulated_binary, simulated_binary_half,
  DE_differential_sum, DE_binary_crossover, DE_exponential_crossover,
  DE_arithmetic_recombination (key-first RNG).
- `_shim_mutation_sampling.py` — polynomial_mutation, uniform_sampling.
- `_shim_selection_basic.py` — tournament_selection, tournament_selection_multifit,
  select_rand_pbest + `_take_along_axis` helper.
- `_shim_selection_nd.py` — dominate_relation, non_dominate_rank,
  crowding_distance, nd_environmental_selection.
- `_shim_selection_rvea.py` — apd_fn, ref_vec_guided (+ local clamp_float/
  maximum/nanmin).
- `_staged_unit_test_shim_selection_nd.py` — verified pytest suite (5 tests,
  green on the etl numpy backend, checked bit-for-bit against torch) STAGED
  here because the sibling tests node (`../unit_test/etl/`) was outside the
  shim agent's write scope. `git mv` it to
  `unit_test/etl/algorithms/test_shim_selection_nd.py`.
- `test_shim_selection_rvea.py` — same situation; move to
  `unit_test/etl/algorithms/test_shim_selection_rvea.py` when writable.

## Routing Table
| Area | Path |
|---|---|
| Selection shims (tournament + pbest, plain functions) | `_shim_selection_basic.py` |
| DE variants (code, de, jade, ode, sade, shade) | `so/de_variants/` |
| ES variants (adam_step, ars, asebo, cma_es, des, esmc, guided_es, nes, noise_reuse_es, open_es, persistent_es, snes, sort_utils, virtual_lora_es) | `so/es_variants/` |
| PSO variants (clpso, cso, dms_pso_el, fs_pso, pso, sl_pso_gs, sl_pso_us, utils) | `so/pso_variants/` |
| MO algorithms (nsga2, nsga3, moead, rvea, rveaa, hype) | `mo/` |
| Tests | `../../unit_test/etl/` | sibling — mirrors this package |

## Notes for agents (verified against etl — do not re-investigate)
- `etl.gather(x, idx, axis)` is numpy `take` semantics (index array applied to
  every row), NOT torch `gather`/`take_along_axis`. For row-local selection use
  `_take_along_axis` from `_shim_selection_basic.py` (flatten-trick, 2-D).
  etl.gather indexes along one axis only (out = indices.shape + x.shape[axis+1:]);
  torch-style `torch.gather(z, 0, idx)` needs a flattened-index gather + reshape.
- etl has NO `unbind` / `expand_dims` / `squeeze` / `take_along_axis` — use
  `__getitem__` slices/ints (trace fine) and `enp.reshape` / `enp.expand_dims`
  instead.
- RNG: functions take the key as an explicit first argument; allocate ONE
  `random.split` subkey per draw (`key_rand, _ = random.split(key)`), same draw
  order and distribution as torch. `random.randint(key, shape, low, high,
  dtype=etl.int32)` draws WITH replacement (high exclusive). Keys must be graph
  inputs (spec shape=(), dtype int64) wrapped via `etl.core.from_numpy` at run.
- `random.split` returns TWO keys (JAX-style). `etl.argsort(x, axis, stable=True)`
  and `etl.argmin(x, axis=...)` return int64 indices.
- The torch evox `lexsort` (`src/evox/utils/jit_fix_operator.py` lines 216-252)
  is empirically LAST-key primary (numpy lexsort convention), despite its
  docstring prose — verified against torch. Port it 1:1, do not "fix" the key order.
- Both `list` and `tuple` pytrees work as `etl.build` specs and `etl.run` inputs.
- Static ints (n_round, tournament_size, pop_size, top_p_num) are read from
  tensor `.shape` at trace time — fine on the numpy backend.
- Self-test for the selection shims lives at
  `../../unit_test/etl/algorithms/test_shim_selection_basic.py` (to be created by
  the unit_test node agent): seed 1026 is pinned because with-replacement draws
  only guarantee every full-size tournament row contains the argmin candidate for
  that seed (verified: w_full all 5, w_multi all 1).
- `etl.while_loop` passes the carry pytree as ONE argument (unpack inside
  cond/body); loop-carried dtypes must match exactly (python-int promotion →
  wrap in `etl.cast`).
- `etl.topk` returns (values, indices); numpy scalars (np.float32) are NOT valid
  graph operands — use Python floats; static args must be passed to BOTH
  `etl.build` AND `etl.run` (run validates all signature args).
- `enp.arccos` does not exist — use `etl.acos`; `etl.relu` promotes ints to
  float64 — use `etl.maximum(x, 0)` for int-preserving relu; no `&`/`!=`/`%`
  overloads on SymbolicTensor — use `enp.logical_and`, `etl.not_equal`,
  `etl.remainder`.
- `etl.zeros`/`etl.full` return CONCRETE tensors in traces (illegal operands) —
  use `enp.zeros`/`enp.full` (symbolic); `etl.sum/mean` take `axes=` (not
  `axis=`), while `etl.norm`/`etl.min` take `axis=`/`keepdims`.
- `_shim_selection_rvea.py`: `theta` is a Python float (static arg passed to
  BOTH `etl.build` and `etl.run`).
