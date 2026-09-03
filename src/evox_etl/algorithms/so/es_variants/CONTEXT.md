# evox_etl/algorithms/so/es_variants — ES variants in functional style

## Intent
Ports of the torch evox ES-variant algorithms (read-only reference in
`../../../evox/algorithms/so/es_variants/`) to `init/ask/tell` plain functions
+ frozen config dataclasses. See `../../DESIGN.md` §4-5.

## Status (implemented so far)
- `adam_step.py` — `adam_single_tensor` helper (plain function; beta1/beta2/lr
  float kwargs; called with beta1=0.9, beta2=0.999, lr=<algo lr>).
- `sort_utils.py` — `sort_by_key` (etl.gather = numpy take semantics).
- `guided_es.py` — `GuidedESConfig`/`GuidedESState` + `init/ask/tell`.
- `noise_reuse_es.py` — `NoiseReuseESConfig`/`NoiseReuseESState` + `init/ask/tell`.
- `persistent_es.py` — `PersistentESConfig`/`PersistentESState` + `init/ask/tell`.

## Conventions (verified — do not re-investigate)
- No `__init__.py` here (PEP 420 namespace package) — tests import the dotted
  module directly, never the package `__init__`.
- Config `center_init` accepts np.ndarray or tuple; `__post_init__` normalizes
  it to a flat float32 tuple. Bake inside functions via
  `etl.ops.constant(etl.core.tensor(np.asarray(config.center_init, dtype=np.float32)))`.
- Every state carries `exp_avg`/`exp_avg_sq` (dim,) regardless of optimizer;
  they are updated ONLY when `config.optimizer == "adam"` (Python `if` on the
  static config is fine inside traces). `best_fitness` (f32 scalar, init
  `np.inf`) is updated every tell; `key` is the last state field.
- RNG: `key, subkey = random.split(state.key)` at top of ask (or in init when
  drawing); one split per draw (`key, subkey2 = random.split(key)`), store the
  advanced key back. NEVER `random.key` inside a trace.
- `etl.transpose(x, (1, 0))` — axes MUST be a tuple (lists raise TypeError).
- `etl.select(cond, a, b)` broadcasts a scalar pred over tensor branches
  (verified) and accepts a Python `0` as a branch.
- `etl.qr(x)` returns `(Q, R)` reduced, Q first — matches `torch.linalg.qr`.

## Routing Table
| Area | Path |
|---|---|
| GuidedES | `guided_es.py` |
| NoiseReuseES | `noise_reuse_es.py` |
| PersistentES | `persistent_es.py` |
| Adam helper / sort helper | `adam_step.py`, `sort_utils.py` |
| Tests | `../../unit_test/etl/algorithms/so/es_variants/` |
