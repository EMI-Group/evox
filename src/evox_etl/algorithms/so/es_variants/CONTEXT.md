# evox_etl/algorithms/so/es_variants — functional ES-family ports

## Intent
Plain-function (`init`/`ask`/`tell`) ports of the torch evox ES-family algorithms
(read-only reference: `src/evox/algorithms/so/es_variants/`). Follows `../../../../DESIGN.md`.

## Status (ported so far)
- `adam_step.py` — `adam_single_tensor` helper (torch 1:1).
- `sort_utils.py` — `sort_by_key` helper.
- `esmc.py` — ESMCConfig/ESMCState + init/ask/tell (DES algorithm; pop_size must be odd).
- `asebo.py` — ASEBOConfig/ASEBOState + init/ask/tell (active subspaces; `lr_decay`/`lr_limit` kept for API parity but unused, as in torch).

## Notes for agents (verified — do not re-investigate)
- All functions are PLAIN (no `@etl.defn`); traced via `etl.build`/`etl.run`
  (backend="numpy" in tests). Relative import `from .adam_step import adam_single_tensor`
  works (PEP 420 namespace package, no `__init__.py` anywhere under `src/evox_etl`).
- `etl.svd(x)` returns reduced `(U, S, Vh)` with Vh of shape `(k, n)` — the TRANSPOSE
  of torch's `Vt` (n, k). The asebo port follows the JAX/evosax decomposition
  (`U2 = Vh[:pop//2]`, `UUT = dot(transpose(U2), U2)` → (dim, dim)) as the binding
  spec prescribes.
- `etl.transpose(x, axes)` requires axes as a TUPLE, not a list.
- Broadcast gotcha: `(k, n) * signs(k,)` does NOT align over the k axis — use
  `enp.expand_dims(signs, axis=1)`; `(m, k) * signs(k,)` aligns fine (trailing axis).
- Dtypes: `etl.eye(n)` is already float32; `etl.svd` preserves float32; python
  float/scalar-tensor mixes stay float32.
- Tests: `unit_test/etl/algorithms/so/es_variants/test_*.py` — smoke tests drive
  init/ask/tell through `helpers.run_generations` (no torch).
