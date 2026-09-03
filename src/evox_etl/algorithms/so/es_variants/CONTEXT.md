# evox_etl/algorithms/so/es_variants — ES family, functional style

## Intent
Ports of the torch evox ES-family algorithms (read-only reference in
`../../../../evox/algorithms/so/es_variants/`) to `init/ask/tell` plain
functions + frozen config/state dataclasses. See `../../../../DESIGN.md`
§4-5 for the binding spec.

## Routing Table
| Area | Path |
|---|---|
| NES (XNES + SeparableNES) | `nes.py` |
| Adam step helper | `adam_step.py` |
| Sort-by-key helper | `sort_utils.py` |
| Tests | `../../../../unit_test/etl/algorithms/so/es_variants/` — sibling, mirrors this dir |

## Notes for agents (verified — do not re-investigate)
- `nes.py` holds BOTH algorithms; the shared module-level `init`/`ask`/`tell`
  dispatch on the (static) config type via `isinstance` so
  `run_generations` (helpers) can drive either one.
- Configs normalize array fields to flat f32 tuples in `__post_init__`
  (ndarray leaves are rejected as static `etl.build` args); host-side asserts
  (`init_std.shape == (dim,)`, recombination weights descending) run on the
  ORIGINAL arrays before normalization. Arrays are re-baked in-graph via
  `etl.ops.constant(etl.core.tensor(np.asarray(...)))`.
- Torch bug NOT replicated: the default pop_size formula uses
  `4 + math.floor(3 * math.log(self.dim))` with `self.dim` undefined at that
  point — the ports use the local `dim` (commented in `nes.py`).
- `etl.clamp` requires BOTH bounds — torch `clip(x, 0)` → `etl.clamp(x, 0.0, np.inf)`.
- `etl.eye(n)` returns float32. `etl.transpose(x, (1, 0))` needs a TUPLE.
- tell-math parity against the torch formulas (same injected noise) is
  verified bit-close for both algorithms (init bit-exact, tell ~1e-5).
- Smoke tests import the module directly (`import ...nes as nes`) and call
  `run_generations(nes, cfg, SphereConfig(dim), n_gens, seed)`; the venv
  interpreter must be used (`/mnt/local-ssd/bchuang/evox/.venv/bin/python`).
  `evox_etl` is NOT pip-installed in the venv — pytest's conftest path
  shims make it importable under pytest only (standalone scripts need
  `sys.path.insert(0, "src")`).
