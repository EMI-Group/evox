# evox_etl/algorithms/so/es_variants — functional ES-family ports (step protocol)

## Intent
Plain-function ports of the torch evox ES-family algorithms (read-only
reference: `src/evox/algorithms/so/es_variants/`) to the post-1.0 STEP
protocol. Binding spec: `../../../../DESIGN.md` (§4-5) +
`evox_etl.core.algorithm` (the protocol doc). Each module exposes a frozen
`XConfig` dataclass (dumb: plain static leaves only, array fields stored as
f32 tuples — construct via the module-level `make_X` constructor, which
normalizes and validates once), a frozen `XState` dataclass (tensor leaves
only: torch Mutable names + sampling intermediates + `best_fitness` f32
scalar + `key` last), and plain `init(config, key) -> state` /
`step(config, state, evaluate) -> state` functions. `step` owns the WHOLE
generation: sample the candidates and stash the sampling intermediates into
the state (`replace(...)`), then `fitness = evaluate(candidates)`
(workflow-injected traced closure: solution_transform → problem evaluate →
opt-direction scaling → fitness_transform; minimization semantics; treated as
opaque, never stored), then update the distribution from
(intermediates, fitness) and return the new state — mirroring torch
`Algorithm.step` exactly.
`best_fitness` (min over all evaluated fitness, updated in step) is an
evox_etl addition — torch has no such field.
**No ES module defines `init_step`/`final_step`** — the torch reference has
no algorithm-level overrides in this family (all use the base class
fallbacks), so the workflow's `init_step()`/`final_step()` fall back to
`step` everywhere here.

## Files
| File | Contents |
|---|---|
| `adam_step.py` | `adam_single_tensor` (used by 7 algorithms when optimizer="adam") |
| `sort_utils.py` | `sort_by_key(keys, population)` (CMA-ES) |
| `cma_es.py` | CMAESConfig/State — full CMA-ES, conditional eigh decomposition via `etl.cond` |
| `open_es.py` | OpenESConfig/State — mirrored sampling, SGD/adam |
| `ars.py` | ARSConfig/State — elite ratio, unbiased std (ddof=1) |
| `snes.py` | SNESConfig/State — weight_type "temp"/"recomb"; `_softmax` shared with des.py |
| `des.py` | DESConfig/State — ranks softmax weighting |
| `nes.py` | XNESConfig + SeparableNESConfig/States — `init`/`step` dispatch on config type via isinstance |
| `guided_es.py` | GuidedESConfig/State — QR surrogate-gradient subspace |
| `noise_reuse_es.py` | NoiseReuseESConfig/State — perturbation reuse, T/K counter |
| `persistent_es.py` | PersistentESConfig/State — perturbation accumulation + reset |
| `esmc.py` | ESMCConfig/State — baseline member; pop_size must be ODD |
| `asebo.py` | ASEBOConfig/State — SVD active subspaces; lr_decay/lr_limit config-only (unused in torch too) |
| `virtual_noise.py` | SHARED deterministic generator (do not modify): `compute_offsets`, `compute_counter_offsets`, `virtual_normal`, `lora_factors` |
| `virtual_es.py` | VirtualESConfig/State + make_virtual_es — O(dim) memory center+seeds virtual-population ES; torch-parity `VirtualES = VirtualESConfig` / `VirtualLoRAES = VirtualES` aliases |
| `virtual_lora_es.py` | VirtualLoRAESConfig/State + make_virtual_lora_es — DISTINCT low-rank variant (extra `lora_rank`, `B @ A` perturbations for ≥2-D blocks) |
| `_virtual_common.py` | Shared virtual-family helpers: `normalize_param_shapes`, `param_dim`, `draw_seeds`, `update_center` |
| `tests/` | in-node torch-parity suite (see Tests) |

`__init__.py` exports the 14 configs + their 14 `make_*` constructors, plus the
torch-style bare aliases `VirtualES` / `VirtualLoRAES` (both = `VirtualESConfig`,
mirroring torch's shadowing of the distinct low-rank class; the low-rank config is
`VirtualLoRAESConfig`).

## Config construction (make_* constructors)
- All 12 configs are dumb frozen dataclasses with NO `__post_init__` (11 files;
  nes.py holds XNESConfig + SeparableNESConfig). Array fields store flat tuples of
  f32-rounded Python floats; nes.py `init_covar` stays a nested tuple-of-tuples
  (small local `_to_float_matrix` — the flat helper cannot express nesting).
- 12 module-level constructors live in the same module as their config and are
  exported from `es_variants/__init__.py`: make_cma_es (cma_es.py), make_open_es,
  make_ars, make_snes (snes.py), make_des (des.py), make_xnes +
  make_separable_nes (nes.py), make_guided_es, make_noise_reuse_es,
  make_persistent_es, make_esmc, make_asebo. Param names/defaults = the torch
  `__init__` kwargs (minus device).
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
`virtual_es.py` / `virtual_lora_es.py` implement the torch virtual-population ES
family: the state carries only a `(dim,)` center plus `(pop_size,)` int64 seeds,
and the full-parameter Gaussian perturbations are REGENERATED deterministically
from those seeds by the shared `virtual_noise.py` generator (O(dim) memory
instead of O(pop*dim)). Torch parity reached WITHOUT the Philox counter PRNG
(etl.random is key/split-only): `virtual_noise` is an in-graph splitmix64 +
Box-Muller generator whose cumulative per-block offsets make
`virtual_normal(seeds, 0, dim)` identical to concatenating per-block calls.
- `step` resamples seeds, then calls the opaque workflow-injected
  `evaluate((center, seeds, sigma))` payload (sigma = the PYTHON float
  `config.noise_stdev`, static), then rebuilds the SAME noise and forms the
  fitness-weighted ES gradient `sum_i f_i * noise_i / (pop * sigma)`.
- `virtual_lora_es` uses `lora_factors` instead: ≥2-D `(d,k)` blocks get
  `delta = B @ A` (`A (rank,k)`, `B (d,rank)`, batched `etl.dot`), 1-D blocks keep
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
- No `init_step`/`final_step`/`record_step` (torch VirtualES has none; the
  workflow falls back to `step`).

## Routing Table
| Area | Path |
|---|---|
| In-node parity tests (torch allowed) | `tests/` — `tests/parity/test_parity.py` is the only torch-importing file; `tests/conftest.py` is the sys.path shim |
| Smoke tests, this family (no torch) | `unit_test/etl/algorithms/so/es_variants/` (sibling — read-only here, escalate writes to the parent agent) |
| Convergence-parity tests vs torch | `unit_test/etl/algorithms/parity/` (sibling — read-only here, escalate writes to the parent agent) |
| Shared test driver / toy problems | `unit_test/etl/algorithms/helpers.py` (sibling — read-only) |
| Torch reference for every module here | `src/evox/algorithms/so/es_variants/` (read-only, never modify) |

## Design Decisions (torch-parity choices — current state)
### 1. ASEBO's SVD block mirrors the reference verbatim, including two API traps
`asebo.py step()` reproduces `src/evox/algorithms/so/es_variants/asebo.py:95-107`
exactly:
- `signs` is a **(k,k) sign MATRIX**, not a length-k vector: the reference writes
  `torch.sign(U[max_abs_cols, :])`, where advanced indexing on dim 0 selects k
  ROWS (row i = sign of row `max_abs_cols[i]` of U), then multiplies `U` and `Vt`
  element-wise. The port uses `etl.sign(etl.gather(U, max_abs_cols, axis=0))`
  (`etl.gather` is numpy-take semantics, i.e. exactly `U[max_abs_cols, :]`).
- The reference's `Vt` is the **deprecated `torch.svd` third output = V (= Vh.T)**,
  NOT Vh; `etl.svd` returns Vh. The port therefore transposes to the reference's
  convention **before** the sign multiply (`Vt = etl.transpose(Vh, (1, 0))`);
  transposing after the multiply is NOT equivalent.
Both are required: without them the port's `UUT_ort` differs from the reference by
≈ 1.2 (missing transpose) / ≈ 0.8 (sign vector) on an identical X. Shape law: the
reference's `U * signs` / `Vt * signs` broadcasts only when
`subspace_dims == dim` — the port keeps that constraint (no generalisation), so
ASEBO here is defined only for the default `subspace_dims == dim`.

### 2. ASEBO's alpha degenerates EXACTLY like the reference — deliberately not "fixed"
The `alpha` denominator is `state.UUT`, i.e. the MASKED UUT stored a few lines
earlier: the UUT mask reads the PRE-increment `gen_counter` while alpha's own mask
reads the POST-increment one — the same ordering the reference has (`where` on its
local UUT before the counter bump, `where` on alpha after). So in the generation
where `gen_counter` first exceeds `subspace_dims` the denominator is the all-zero
matrix and **alpha = inf**, reproducing the reference's division by its
never-refreshed all-zero `self.UUT`. torch then raises `LinAlgError` in
`cholesky` at generation `subspace_dims + 2`; numpy's cholesky NaN-propagates
instead and the port surfaces the failure one generation later as an `etl.svd`
non-convergence error. Verified with the real torch algorithm (dim=10, pop=8,
sigma=0.5, lr=1.0, seed 0): alpha=1.0 through generation 10, alpha=inf at 11,
LinAlgError at 12. Running a *finite* alpha instead does not help: measured, an
unmasked fresh UUT in the denominator makes `cov` non-positive-definite at
generation 12 anyway (alpha can exceed 1, so `(1-alpha)*UUT` goes negative).
**Runs beyond `subspace_dims` generations are unsupported on both sides.**

### 3. CMA-ES's rank-one covariance update = TRUE outer product
The reference was fixed upstream (commit 86e8fa63) from the 1-D `p_c @ p_c.T`
scalar-dot quirk to `torch.outer(p_c, p_c)`; `cma_es.py` step() builds
`etl.dot(enp.expand_dims(p_c, 1), enp.expand_dims(p_c, 0))` (`etl.dot` needs
rank ≥ 2). Measured on the sibling CMA-ES parity config (Sphere dim 40, 80 gens):
outer product 3.96 / 2.60 (seeds 0 / 1) vs the old scalar form 20.05 / 24.78 vs
torch 5.07 — i.e. the old scalar replication of the removed quirk was the
divergence. Any future "replicate the quirk" reasoning here is obsolete: check
the reference source, not this file's history.

## Known Issues
- ASEBO is defined only for `subspace_dims == dim`, and numerically usable only
  for ≤ `subspace_dims` generations (Design Decisions 1 and 2).
- ASEBO's subspace block is float32-degenerate on a REAL gradient history:
  `X = grad_subspace - mean(grad_subspace, axes=0)` always has columns summing to
  zero (rank ≤ sub−1), so the smallest singular value ≈ 0 and the `argmax`/signs
  land on float32 ties — numpy's and torch's pipelines can disagree by 0.2-1.0 in
  `UUT_ort` on identical ops. Parity of that block is therefore asserted on a
  well-posed constructed X in `tests/parity/test_parity.py`; note the block is
  also INERT for `gen_counter <= subspace_dims` (alpha is forced to 1.0 and UUT is
  masked to zero, so `cov` is isotropic and the sampled population, fitness and
  center are independent of the SVD block).
- `asebo.py step` uses `etl.svd` (U, S, Vh), which etl's stablehlo-v1 exporter
  DEFERS (`BackendError` on compiled backends iree/xla); the numpy backend is
  fine. The NSGA3 fix pattern (eigh-based equivalents, see
  `src/evox_etl/algorithms/mo/nsga3.py`) is the reference if asebo is ever
  benchmarked on compiled backends.
- `open_es` with `mirrored_sampling=False` and a large lr can overflow to a NaN
  best-fitness — pre-existing in the reference, not a step-protocol regression.

## Notes for agents (verified — do not re-investigate)
- All functions are PLAIN (no `@etl.defn`); traced via `etl.build`/`etl.run`.
- The step ports are bitwise-identical to the pre-conversion two-phase protocol
  on the numpy backend (validated per module over 10–20 generations) EXCEPT
  `cma_es.py`, `asebo.py` and `ars.py`, whose numerics were corrected to match the
  torch reference (see Design Decisions / git history).
- Known torch bugs NOT replicated: XNES/SeparableNES use `self.dim` before it
  exists when pop_size=None (torch nes.py lines 42-43, 154) — ports use the
  local `dim` (commented in nes.py).
- `etl.transpose(x, axes)` requires axes as a TUPLE (a list raises TypeError).
- Broadcast gotcha: `(k, n) * signs(k,)` does NOT align over the k axis — use
  `enp.expand_dims(signs, axis=1)`; `(m, k) * signs(k,)` aligns fine.
- `etl.svd` returns reduced `(U, S, Vh)`, Vh (k, n) — the numpy/`torch.linalg.svd`
  convention, NOT `torch.svd(some=True)`: the deprecated `torch.svd` third output
  is **V** (n, k), the transpose (its docstring: `input = U diag(S) V^H`; verified
  on torch 2.12.1 with a (4,6) input → third output (6,4); `torch.linalg.svd(...,
  full_matrices=False).Vh == V.T`). The two conventions coincide in shape — but
  not in value — for a SQUARE matrix (as in `asebo.py`, where
  `subspace_dims == dim`): `|V − Vh| ≈ 1.0` for a random 10×10. `asebo.py`
  transposes etl's Vh back to `torch.svd`'s V (Design Decisions 1); every other
  module here consumes Vh directly (matching `torch.linalg.svd`).
- `etl.qr` returns `(Q, R)` reduced, Q first. `etl.cond` supports pytree outputs.
- `etl.eye(n)` is already float32; svd/eigh preserve f32; `etl.clamp` requires
  BOTH bounds (use etl.maximum for one-sided clamps); use `math.*` never `np.*`
  for Python scalars at trace time (numpy scalars are illegal operands).
- Config array baking inside traced functions uses
  `bake_float32_constant` from `../_config_utils.py` (canonical); a few inline
  `etl.ops.constant(etl.core.tensor(np.asarray(x, dtype=np.float32)))` blocks
  remain (snes/des center, nes init_covar) — byte-identical, leave as-is unless
  touching the file anyway. Numpy import is sanctioned for this + np.dtype only.

## Tests
Environment (no pre-provisioned venv): `uv venv` + `uv pip install <etl-source-copy>
pytest numpy`, then `PYTHONPATH="src:<torch-site-packages>"` (torch comes from the
shared evox venv; no GPU here).
- **In-node parity (torch allowed, numpy backend)**: `tests/parity/test_parity.py` —
  the ONLY torch-importing file in this node; `tests/conftest.py` is the
  relocation-ready repo-root/`src` sys.path shim (pattern:
  `../../workflows/tests/conftest.py`; force-add the files — `tests/` is
  `.gitignore`d, as for the sibling `metrics/tests/`). It injects IDENTICAL noise on
  both sides (monkeypatch `torch.randn` + `etl.random.normal` → an in-graph
  `etl.ops.constant`) to make torch and etl trajectories comparable, and covers:
  ASEBO (state parity + the subspace block vs the reference ops, on a well-posed X)
  with discriminating-power guards (~1.2 / ~0.8), ARS odd `pop_size=5, elite_ratio=0.9`
  (torch elite count 2 vs the old integer-division 1, Δcenter 1.8e-2) and CMA-ES
  2-step state parity (ΔC 6e-8; the pre-fix scalar form shifts C by 0.12).
  Run: `python -m pytest src/evox_etl/algorithms/so/es_variants/tests -q` (3 tests).
- **Sibling smoke (no torch)**: `unit_test/etl/algorithms/so/es_variants/test_*.py`
  — 22 tests, green; driven by `unit_test/etl/algorithms/helpers.py`.
- **Sibling parity (torch)**: `unit_test/etl/algorithms/parity/test_cma_es.py`
  (+ `test_open_es.py`), the sole red tests of the etl suite here. `test_cma_es.py`
  fails BOTH parametrizations for an ENVIRONMENT reason on the TORCH side, not a
  port defect, and this node cannot edit that sibling file (write scope — escalate
  to the parent agent): `CMAES.step` → `_conditional_decomposition` calls
  `torch.cond(...)`, which under torch 2.12.1 raises `UncapturedHigherOrderOpError`
  while capturing the **`_decomposition`** branch — its first line
  `C = (C + C.T) / 2` takes a transposed VIEW of the cond operand, dynamo lifts that
  view as a second graph input aliasing the operand, and HOP tracing rejects
  input-to-input aliasing (a later clone inside the branch does NOT help). The torch
  side dies at `workflow.init_step()` before any etl code runs. The etl side is
  correct and converges BETTER than torch (Design Decisions 3). Fix: on this file's
  configs `decomp_per_iter == 1`, so `_decomposition` is always the taken branch —
  monkeypatch `CMAES._conditional_decomposition` to
  `lambda self, iteration, C: self._decomposition(C)`, guarded by
  `assert int(alg.decomp_per_iter) == 1`, which is mathematically identical and
  drives both parametrizations green (torch 5.06529 / 5.64848 vs etl 3.96482 /
  2.59804, seeds 0 / 1). The same monkeypatch pattern is already used in
  `tests/parity/test_parity.py`; the full write-up lives in
  `unit_test/etl/algorithms/parity/CONTEXT.md` (Known Issues).
