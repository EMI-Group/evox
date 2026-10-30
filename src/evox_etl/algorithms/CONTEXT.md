# evox_etl/algorithms — SO and MO algorithms in functional style

## Intent
Port of all 34 torch evox algorithms (read-only reference in `../../../evox/
algorithms/`) to the step protocol: `init`/`init_step`/`step`/`final_step`
plain functions (`step(config, state, evaluate) -> state` owns one full
generation) + frozen config dataclasses. See `../../DESIGN.md` §4-5 and
`../core/algorithm.py` (the protocol doc).

## Canonical operator imports
- Algorithms import crossover, sampling, selection, and mutation operators from
  `evox_etl.operators.*` (canonical, torch-parity-verified modules).
- The jit-fix utils (clamp family, lexsort, nanmin/nanmax, randint,
  `_take_along_axis`) are imported from
  `evox_etl.operators.jit_fix_operator` (canonical home — a port of the torch
  `evox/utils/jit_fix_operator.py` helpers).

## Routing Table
| Area | Path |
|---|---|
| DE variants (code, de, jade, ode, sade, shade) | `so/de_variants/` |
| ES variants (adam_step, ars, asebo, cma_es, des, esmc, guided_es, nes, noise_reuse_es, open_es, persistent_es, snes, sort_utils) | `so/es_variants/` |
| Virtual ES family (virtual_es, virtual_lora_es, virtual_noise, _virtual_common) | `so/es_variants/` | O(dim) center+seeds virtual-population ES: `VirtualES`/`VirtualLoRAES` + `make_virtual_es` (virtual_es.py), the DISTINCT low-rank `VirtualLoRAESConfig` + `make_virtual_lora_es` (virtual_lora_es.py), and the shared deterministic noise generator (virtual_noise.py) |
| PSO variants (clpso, cso, dms_pso_el, fs_pso, pso, sl_pso_gs, sl_pso_us, utils) | `so/pso_variants/` |
| MO algorithms (nsga2, nsga3, moead, rvea, rveaa, hype) | `mo/` |
| Shared config-constructor helpers (`to_float_tuple`, `normalize_bounds`, `require_gt`/`require_ge`/`require_between`/`require_choice`, `bake_float32_constant`, `bake_bounds`) | `_config_utils.py` |
| Tests | `../../unit_test/etl/` | sibling — mirrors this package |

## Torch-parity state (audited against `../../evox/algorithms/`)
- Module coverage is complete: all 32 torch algorithm modules have a port module (incl. helper modules `adam_step`, `es_variants/sort_utils`, `pso_variants/utils`); no torch algorithm is missing.
- Every algorithm module defines `make_*`, `init(config, key)` and `step(config, state, evaluate)`; `init_step` exists exactly where torch has it (DE/ODE/JaDE, all 7 PSO, all 6 MO) and is absent in the greedy ES family, whose torch counterpart also lacks it (the ES family has `record_step` instead, matching torch).
- Constructor-parameter parity is 1:1 (minus `device`) for all 32 algorithms; the single deliberate drop is `HypE.selection_op` (inert in torch — overwritten by `tournament_selection`).
- Bare torch names are exported at `evox_etl.algorithms`, `evox_etl.algorithms.so`, and in `de_variants`/`pso_variants`; they are ABSENT at `algorithms.mo` and `algorithms.so.es_variants` (those export `*Config` only, although torch's `mo/__init__.py`/`es_variants/__init__.py` export the bare names).
- `virtual_lora_es.py` defines only `VirtualLoRAESConfig` — there is no `virtual_lora_es.VirtualLoRAES` symbol; the torch-parity bare `VirtualLoRAES` is the `VirtualES` alias.
- Port-owned state tensors are float32 by construction (configs normalize to float32 tuples + `bake_*` constants), so torch float64 workflows lose precision.

## Notes for agents (verified against evoxc — do not re-investigate)
- All algorithm configs have module-level `make_*` constructors (policy in `../DESIGN.md` §4.1), exported through `so/__init__.py` and `algorithms/__init__.py`.
- `evoxc.gather(x, idx, axis)` is numpy `take` semantics (index array applied to
  every row), NOT torch `gather`/`take_along_axis`. For row-local selection use
  `_take_along_axis` from `evox_etl.operators.jit_fix_operator` (flatten-trick, 2-D).
  evoxc.gather indexes along one axis only (out = indices.shape + x.shape[axis+1:]);
  torch-style `torch.gather(z, 0, idx)` needs a flattened-index gather + reshape.
- evoxc has NO `unbind` / `expand_dims` / `squeeze` / `take_along_axis` — use
  `__getitem__` slices/ints (trace fine) and `enp.reshape` / `enp.expand_dims`
  instead.
- RNG: functions take the key as an explicit first argument; allocate ONE
  `random.split` subkey per draw (`key_rand, _ = random.split(key)`), same draw
  order and distribution as torch. `random.randint(key, shape, low, high,
  dtype=evoxc.int32)` draws WITH replacement (high exclusive). Keys must be graph
  inputs (spec shape=(), dtype int64) wrapped via `evoxc.core.from_numpy` at run.
- `random.split` returns TWO keys (JAX-style). `evoxc.argsort(x, axis, stable=True)`
  and `evoxc.argmin(x, axis=...)` return int64 indices.
- The torch evox `lexsort` (`src/evox/utils/jit_fix_operator.py` lines 216-252)
  is empirically LAST-key primary (numpy lexsort convention), despite its
  docstring prose — verified against torch. Port it 1:1, do not "fix" the key order.
- Both `list` and `tuple` pytrees work as `evoxc.build` specs and `evoxc.run` inputs.
- Static ints (n_round, tournament_size, pop_size, top_p_num) are read from
  tensor `.shape` at trace time — fine on the numpy backend.
- `evoxc.while_loop` passes the carry pytree as ONE argument (unpack inside
  cond/body); loop-carried dtypes must match exactly (python-int promotion →
  wrap in `evoxc.cast`).
- `evoxc.topk` returns (values, indices); numpy scalars (np.float32) are NOT valid
  graph operands — use Python floats; static args must be passed to BOTH
  `evoxc.build` AND `evoxc.run` (run validates all signature args).
- `enp.arccos` does not exist — use `evoxc.acos`; `evoxc.relu` promotes ints to
  float64 — use `evoxc.maximum(x, 0)` for int-preserving relu; no `&`/`!=`/`%`
  overloads on SymbolicTensor — use `enp.logical_and`, `evoxc.not_equal`,
  `evoxc.remainder`.
- `evoxc.zeros`/`evoxc.full` return CONCRETE tensors in traces (illegal operands) —
  use `enp.zeros`/`enp.full` (symbolic); `evoxc.sum/mean` take `axes=` (not
  `axis=`), while `evoxc.norm`/`evoxc.min` take `axis=`/`keepdims`.
- `ref_vec_guided`/`apd_fn` (canonical
  `evox_etl.operators.selection.rvea_selection`): `theta` is a Python float
  (static arg passed to BOTH `evoxc.build` and `evoxc.run`). `apd_fn` is
  torch-faithful: `selected_z` gathers with `relu(x)` but the `norm_obj` gather
  uses the RAW indices (negative indices wrap to the last row, numpy-take
  semantics — same as torch `norm_obj[x]`).
