# evox_etl/algorithms/so/es_variants — ES-family algorithms (step protocol)

## Intent
The 2.0 functional ES family on the STEP protocol.
Binding spec: `../../../../DESIGN.md` (§4-5) +
`evox_etl.core.algorithm` (the protocol doc). Each module exposes a frozen
`XConfig` dataclass (dumb: plain static leaves only, array fields stored as
f32 tuples — construct via the module-level `make_X` constructor, which
normalizes and validates once), a frozen `XState` dataclass (tensor leaves
only: 1.x state names + sampling intermediates + `best_fitness` f32
scalar + `key` last), and plain `init(config, key) -> state` /
`step(config, state, evaluate) -> state` functions. `step` owns the WHOLE
generation: sample the candidates and stash the sampling intermediates into
the state (`replace(...)`), then `fitness = evaluate(candidates)`
(workflow-injected traced closure: solution_transform → problem evaluate →
opt-direction scaling → fitness_transform; minimization semantics; treated as
opaque, never stored), then update the distribution from
(intermediates, fitness) and return the new state — one full generation.
`best_fitness` (min over all evaluated fitness, updated in step) is an
evox_etl addition.
**No ES module defines `init_step`/`final_step`** — no algorithm in this family
has a distinct first or last generation, so the workflow's `init_step()`/
`final_step()` fall back to `step` everywhere here.
**Every ES module defines the optional `record_step(config, state, candidate,
fitness)` aux-history hook** — a PLAIN, module-level, host-side function (NOT
`@evoxc.defn`); the returned dict of already-carried ``evoxc`` tensors feeds
`EvalMonitor.aux_history` (see "Aux-history hook" below).

## Files
| File | Contents |
|---|---|
| `adam_step.py` | `adam_single_tensor` (used by 7 algorithms when optimizer="adam") |
| `sort_utils.py` | `sort_by_key(keys, population)` (CMA-ES) |
| `cma_es.py` | CMAESConfig/State — full CMA-ES, conditional eigh decomposition via `evoxc.cond` |
| `open_es.py` | OpenESConfig/State — mirrored sampling, SGD/adam |
| `ars.py` | ARSConfig/State — elite ratio, unbiased std (ddof=1) |
| `snes.py` | SNESConfig/State — weight_type "temp"/"recomb"; `_softmax` shared with des.py |
| `des.py` | DESConfig/State — ranks softmax weighting |
| `nes.py` | XNESConfig + SeparableNESConfig/States — `init`/`step` dispatch on config type via isinstance |
| `guided_es.py` | GuidedESConfig/State — QR basis `(dim, subspace_dims)` of the gradient rows' span (factorises `grad_subspace.T`) |
| `noise_reuse_es.py` | NoiseReuseESConfig/State — perturbation reuse, T/K counter |
| `persistent_es.py` | PersistentESConfig/State — perturbation accumulation + reset |
| `esmc.py` | ESMCConfig/State — baseline member; pop_size must be ODD |
| `asebo.py` | ASEBOConfig/State — SVD active subspaces (fixed b448084a semantics, see Design Decisions); lr_decay/lr_limit are config-only (unused) |
| `virtual_noise.py` | SHARED deterministic generator (do not modify): `compute_offsets`, `compute_counter_offsets`, `virtual_normal`, `lora_factors` |
| `virtual_es.py` | VirtualESConfig/State + make_virtual_es — O(dim) memory center+seeds virtual-population ES; bare aliases `VirtualES = VirtualESConfig` / `VirtualLoRAES = VirtualES` |
| `virtual_lora_es.py` | VirtualLoRAESConfig/State + make_virtual_lora_es — DISTINCT low-rank variant (extra `lora_rank`, `B @ A` perturbations for ≥2-D blocks) |
| `_virtual_common.py` | Shared virtual-family helpers: `normalize_param_shapes`, `param_dim`, `draw_seeds`, `update_center` |

`__init__.py` exports the 14 configs + their 14 `make_*` constructors, plus the
bare aliases `VirtualES` / `VirtualLoRAES` (both = `VirtualESConfig`; the distinct
low-rank config is `VirtualLoRAESConfig`).

## Config construction (make_* constructors)
- All 14 configs are dumb frozen dataclasses with NO `__post_init__` (13 files;
  nes.py holds XNESConfig + SeparableNESConfig, virtual_es.py/virtual_lora_es.py
  the virtual family). Array fields store flat tuples of f32-rounded Python floats
  (`param_shapes` is a tuple of int tuples); nes.py `init_covar` stays a nested
  tuple-of-tuples (small local `_to_float_matrix` — the flat helper cannot express
  nesting).
- 14 module-level constructors live in the same module as their config and are
  exported from `es_variants/__init__.py`: make_cma_es (cma_es.py), make_open_es,
  make_ars, make_snes (snes.py), make_des (des.py), make_xnes +
  make_separable_nes (nes.py), make_guided_es, make_noise_reuse_es,
  make_persistent_es, make_esmc, make_asebo, make_virtual_es (virtual_es.py),
  make_virtual_lora_es (virtual_lora_es.py). Param names/defaults follow the
  documented hyperparameters (there is no device field).
- Shared host-side helpers live in `../_config_utils.py` (`to_float_tuple`,
  `bake_float32_constant`, `require_gt/ge/between/choice`); validation raises
  ValueError naming the parameter (never bare asserts). make_* rounds ALL array
  inputs — tuple AND ndarray — to f32 uniformly (package decision; float64/tuple
  input no longer passes through unrounded).
- Eager derived defaults computed inside make_* (so `getattr(cfg, ...)` works):
  make_xnes/make_separable_nes fill pop_size = 4+⌊3·ln dim⌋ and the lr defaults
  from dim; make_asebo fills subspace_dims = len(center_init) when None.
  NOT hoisted (still trace-time): cma_es pop_size default (`_derive()` in
  init/step) and guided_es subspace_dims (derived inline in init/step) —
  cma_es' cfg.pop_size None is handled by bench_so.py:132-135; core/workflow.py
  `_discover_pop_size` relies on the nes make_* eager fill.
- `_softmax` is byte-identical in des.py/snes.py; single home is snes.py, des.py
  imports it (des → snes dependency mirrors the algorithm lineage).
- Direct `Config(...)` construction bypasses normalization/validation — only
  valid for already-normalized values. For XNES/SeparableNES/ASEBO direct
  construction also leaves the eager derived fields (`pop_size`,
  `learning_rate_*`, `subspace_dims`) as None → trace-time
  TypeError/TraceError ("unsupported operand type NoneType",
  "dynamic-length shapes (None, …)"): construct via make_xnes/
  make_separable_nes/make_asebo (or pass explicit values).

## Virtual (training-free) ES family — PORTED
`virtual_es.py` / `virtual_lora_es.py` implement the virtual-population ES
family: the state carries only a `(dim,)` center plus `(pop_size,)` int64 seeds,
and the full-parameter Gaussian perturbations are REGENERATED deterministically
from those seeds by the shared `virtual_noise.py` generator (O(dim) memory
instead of O(pop*dim)). The 1.x Philox counter PRNG has no evoxc analogue
(evoxc.random is key/split-only), so: `virtual_noise` is an in-graph splitmix64 +
Box-Muller generator whose cumulative per-block offsets make
`virtual_normal(seeds, 0, dim)` identical to concatenating per-block calls.
- `step` resamples seeds, then calls the opaque workflow-injected
  `evaluate((center, seeds, sigma))` payload (sigma = the PYTHON float
  `config.noise_stdev`, static), then rebuilds the SAME noise and forms the
  fitness-weighted ES gradient `sum_i f_i * noise_i / (pop * sigma)`.
- `virtual_lora_es` uses `lora_factors` instead: ≥2-D `(d,k)` blocks get
  `delta = B @ A` (`A (rank,k)`, `B (d,rank)`, batched `evoxc.dot`), 1-D blocks keep
  flat full Gaussian noise; parts are raveled row-major and concatenated.
- `_virtual_common.py` holds the SHARED config/seed/update logic:
  `normalize_param_shapes`, `param_dim`, `draw_seeds`, `update_center` (plain SGD
  or Adam via `adam_single_tensor(..., 0.9, 0.999, lr)` + `best_fitness`), so no
  update logic is duplicated between the two modules.
- `exp_avg`/`exp_avg_sq` are ALWAYS carried at `(dim,)` f32 zeros; when
  `optimizer is None` they are passed through unchanged (no zero-size tensors).
- `dim` is an evox_etl addition filled eagerly by `make_*` (`sum(prod(shape))`)
  for `_discover_pop_size`/monitor completion; `init`/`step` recompute it with
  `param_dim`.
- No `init_step`/`final_step` (the workflow falls back to `step`). Both virtual
  modules DO define the optional `monitor_candidate(payload) -> (pop_size, dim)`
  hook (the workflow's monitor bridge), which broadcasts the `(dim,)` center to
  the `(pop_size, dim)` candidate the `EvalMonitorConfig` concatenates with its
  elite buffer; it is only called when a monitor is configured, so the
  monitor-less hot path stays O(dim). They also define `record_step` (returns
  `{"center": state.center}`; see "Aux-history hook").

## Aux-history hook (`record_step`)
Every module here defines the OPTIONAL plain, module-level, host-side
`record_step(config, state, candidate, fitness) -> dict[str, evoxc_tensor]` hook
(contract: `evox_etl.core.algorithm`; consumed host-side by
`evox_etl.core.workflow._record_history`). It receives the POST-step algorithm
state plus the monitor's `latest_solution`/`latest_fitness` (or `None`) and
returns ``evoxc`` tensors ALREADY carried in the state — no tracing, no Python
scalars:
- `{"center": state.center}` — ars.py, open_es.py, virtual_es.py,
  virtual_lora_es.py
- `{"center": state.center, "sigma": state.sigma}` — des.py, esmc.py,
  guided_es.py (alpha deliberately NOT recorded),
  noise_reuse_es.py, persistent_es.py, snes.py
- `{"center": state.center, "sigma": state.sigma, "alpha": state.alpha}` —
  asebo.py
- `{"mean": state.mean, "sigma": state.sigma}` — cma_es.py
- nes.py — one module-level hook that dispatches on the config type via
  isinstance (mirroring its `init`/`step`): XNES → `{"mean","sigma","B"}`,
  SeparableNES → `{"mean","sigma"}`.
The workflow only calls it (and the monitor only stores it) when a monitor is
configured AND `EvalMonitorConfig.full_pop_history` is True.

## Routing Table
| Area | Path |
|---|---|
| Family tests — evoxc-only smoke tests | `unit_test/etl/algorithms/so/es_variants/` (sibling — read-only here, escalate writes to the parent agent) |
| Shared test driver / toy problems | `unit_test/etl/algorithms/helpers.py` (sibling — read-only) |

## Design Decisions (current state)
### 1. ASEBO uses the right singular vectors (no sign juggling)
`asebo.py step()`:
`subspace_ready = gen_counter > subspace_dims` is computed ONCE from the
PRE-increment counter and gates BOTH `UUT`/`UUT_ort` and `alpha`.
`X = grad_subspace - mean(X, axes=0)`; `_, S, Vh = evoxc.svd(X)`;
`U = Vh[:subspace_dims]` (ROWS of the right singular vectors — `evoxc.svd`
returns the numpy `linalg.svd` convention, so no transpose and no sign
multiply); `tol = eps(f32) * max(X.shape) * S[0]`;
`informative = (S[:subspace_dims] > tol)`;
`UUT = (U.T * informative) @ U`; `UUT_ort = I - UUT`;
`effective_dims = clamp(sum(informative), min=1.0)`.
`cov = (alpha/dim) I + ((1-alpha)/effective_dims) UUT + 1e-6 I` — NO sigma factor;
sigma instead multiplies the sampled perturbation (`z_plus = sigma * (noise @
chol.T)` with `noise (half, dim)`, no per-vector normalisation).
`alpha = norm(theta_grad @ UUT_ort) / (norm(theta_grad @ UUT) + 1e-8)`, clamped to
[0, 1] and pinned to 1.0 while not ready.
The freshly computed `UUT`/`UUT_ort` are stored in the returned state and feed
that same generation's alpha (the pre-fix port divided by the never-written
all-zero state buffer and never stored them).

### 2. ASEBO is well-posed for any `subspace_dims <= dim` and any run length
The centred gradient buffer has rank at most `subspace_dims - 1`, so the trailing
direction(s) are dropped by the singular-value tolerance and `effective_dims`
floors the normalisation at 1; the `1e-6 * I` jitter keeps `cov` positive
definite for every alpha in [0, 1].
Measured on Sphere (20 generations, seed 0, center_init 5, sigma 0.5, lr 1.0):
(dim=10, pop=8, subspace_dims=4) 248.03 → 54.00, (dim=10, pop=8, default
subspace_dims=10) 248.03 → 51.83, (dim=12, pop=6, subspace_dims=3) 298.85 → 74.86
— all finite, alpha in [0, 1], `best_fitness` (a running min) non-increasing.

### 3. CMA-ES's rank-one covariance update = TRUE outer product
`cma_es.py` step() builds
`evoxc.dot(enp.expand_dims(p_c, 1), enp.expand_dims(p_c, 0))` (`evoxc.dot` needs
rank ≥ 2), i.e. a genuine (dim, dim) outer product; the 1-D `p_c @ p_c.T`
scalar-dot form is a plain bug. Measured on the sibling CMA-ES config (Sphere
dim 40, 80 gens): outer product 3.96 / 2.60 (seeds 0 / 1).

## Known Issues
- `asebo.py step` uses `evoxc.svd` (U, S, Vh), which evoxc's stablehlo-v1 exporter
  DEFERS (`BackendError` on compiled backends iree/xla); the numpy backend is
  fine. The NSGA3 fix pattern (eigh-based equivalents, see
  `src/evox_etl/algorithms/mo/nsga3.py`) is the reference if asebo is ever
  benchmarked on compiled backends.
- `open_es` with `mirrored_sampling=False` and a large lr can overflow to a NaN
  best-fitness — inherent to the algorithm, not a step-protocol regression.

## Notes for agents (verified — do not re-investigate)
- All functions are PLAIN (no `@evoxc.defn`); traced via `evoxc.build`/`evoxc.run`.
- The step functions are bitwise-identical to the pre-conversion two-phase
  protocol on the numpy backend (validated per module over 10–20 generations)
  EXCEPT `cma_es.py`, `asebo.py` and `ars.py`, whose numerics follow the
  documented formulas (see Design Decisions / git history).
- `nes.py` computes the local `dim` before using it; the 1.x code read
  `self.dim` before it exists when `pop_size=None` (commented in nes.py).
- `evoxc.transpose(x, axes)` requires axes as a TUPLE (a list raises TypeError).
- Broadcast gotcha: `(k, n) * signs(k,)` does NOT align over the k axis — use
  `enp.expand_dims(signs, axis=1)`; `(m, k) * signs(k,)` aligns fine.
- `evoxc.svd` returns reduced `(U, S, Vh)` with Vh (k, n) — the numpy
  `linalg.svd` convention (Vh is the transpose of V; the two coincide in shape
  but not in value for a square matrix). Every module here consumes Vh directly,
  including `asebo.py`.
- `guided_es.py` factorises `evoxc.transpose(state.grad_subspace, (1, 0))` (the
  stored gradients are ROWS), so `evoxc.qr` returns Q `(dim, subspace_dims)`
  (verified: 3×10 → Q (10,3), 10×10 → Q (10,10)); factorising the un-transposed
  `(subspace_dims, dim)` buffer yields a `(subspace_dims, subspace_dims)` basis
  and `c * Q @ eps_subspace` then raises a ShapeError whenever
  `subspace_dims < dim`.
- `evoxc.qr` returns `(Q, R)` reduced, Q first. `evoxc.cond` supports pytree outputs.
- `evoxc.dot` requires rank ≥ 2 BUT also supports BATCHED 3-D matmul: `evoxc.dot(B
  (pop,d,rank), A (pop,rank,k)) -> (pop,d,k)` works (numpy backend; used by
  `virtual_lora_es.py` for the per-individual `B @ A` delta — no broadcast+sum
  workaround needed).
- `evoxc.eye(n)` is already float32; svd/eigh preserve f32; `evoxc.clamp` requires
  BOTH bounds (use evoxc.maximum for one-sided clamps); use `math.*` never `np.*`
  for Python scalars at trace time (numpy scalars are illegal operands).
- Config array baking inside traced functions uses
  `bake_float32_constant` from `../_config_utils.py` (canonical); a few inline
  `evoxc.ops.constant(evoxc.core.tensor(np.asarray(x, dtype=np.float32)))` blocks
  remain (snes/des center, nes init_covar) — byte-identical, leave as-is unless
  touching the file anyway. Numpy import is sanctioned for this + np.dtype only.

## Tests
Environment: `uv venv` + `uv pip install -e . pytest plotly pandas` (see
`../../CONTEXT.md` "Environment"), then run with `PYTHONPATH=src` from the repo root.
- **Family suite**: `unit_test/etl/algorithms/so/es_variants/` — the evoxc-only
  smoke tests (`test_*.py`, driven by `unit_test/etl/algorithms/helpers.py`; the
  repo-root/`src` sys.path shim is `conftest.py`).
- **Aux-history test**: `test_record_step_aux.py` drives `cma_es`/`des` through
  `StdWorkflow` + `EvalMonitorConfig(full_pop_history=True)` and pins the
  `record_step` → `aux_history` aux channel (keys, one entry/generation, shapes,
  final-state match). These ES states expose neither `population`/`pop` nor a
  `dim` config field, so `dim` (and `pop_size` for `cma_es`'s `None` default) must
  be passed EXPLICITLY to `EvalMonitorConfig`.
- Gate: `PYTHONPATH=src /tmp/venv/bin/python -m pytest
  unit_test/etl/algorithms/so/es_variants -q`.
