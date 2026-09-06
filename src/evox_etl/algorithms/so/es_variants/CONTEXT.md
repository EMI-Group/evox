# evox_etl/algorithms/so/es_variants — functional ES-family ports

## Intent
Plain-function (`init`/`ask`/`tell`) ports of the torch evox ES-family algorithms
(read-only reference: `src/evox/algorithms/so/es_variants/`). Binding spec:
`../../../../DESIGN.md` (§4-5). Each module exposes a frozen `XConfig` dataclass
(torch `__init__` signature minus `device`; array args as f32 tuples after
`__post_init__` normalization — etl.build REJECTS ndarray static leaves), a
frozen `XState` dataclass (tensor leaves only: torch Mutable names + ask→tell
intermediates + `best_fitness` f32 scalar + `key` last), and plain
`init(config, key)` / `ask(config, state) -> (population, state)` /
`tell(config, state, fitness)` functions. `best_fitness` (min over all evaluated
fitness, updated in tell) is an evox_etl addition — torch has no such field.

## Files
| File | Contents |
|---|---|
| `adam_step.py` | `adam_single_tensor` (used by 7 algorithms when optimizer="adam") |
| `sort_utils.py` | `sort_by_key(keys, population)` (CMA-ES) |
| `cma_es.py` | CMAESConfig/State — full CMA-ES, conditional eigh decomposition via `etl.cond` |
| `open_es.py` | OpenESConfig/State — mirrored sampling, SGD/adam |
| `ars.py` | ARSConfig/State — elite ratio, unbiased std (ddof=1) |
| `snes.py` | SNESConfig/State — weight_type "temp"/"recomb" |
| `des.py` | DESConfig/State — ranks softmax weighting |
| `nes.py` | XNESConfig + SeparableNESConfig/States — `init/ask/tell` dispatch on config type via isinstance |
| `guided_es.py` | GuidedESConfig/State — QR surrogate-gradient subspace |
| `noise_reuse_es.py` | NoiseReuseESConfig/State — perturbation reuse, T/K counter |
| `persistent_es.py` | PersistentESConfig/State — perturbation accumulation + reset |
| `esmc.py` | ESMCConfig/State — baseline member; pop_size must be ODD |
| `asebo.py` | ASEBOConfig/State — SVD active subspaces; lr_decay/lr_limit config-only (unused in torch too) |

`__init__.py` exports the 12 configs (VirtualLoRAES intentionally skipped).

## Config `__post_init__` audit (input for the planned __post_init__-removal refactor)
- All 12 config dataclasses here define `__post_init__` (11 files; nes.py has TWO —
  `XNESConfig` at nes.py:65 and `SeparableNESConfig` at nes.py:200, each normalizing its
  own array fields and deriving its own dim-based pop_size/lr defaults). No config in
  this node lacks one and no State dataclass has one. Every body is IDEMPOTENT
  (conditional `isinstance(np.ndarray)` guards, or tuple re-normalization is a fixed
  point) — relevant because `dataclasses.replace` re-invokes `__post_init__`.
- Purposes: (1) Config()-time `assert` validation — ars.py:44 (pop>1, elite_ratio),
  asebo.py:40 (pop>1, optimizer whitelist), cma_es.py:38 (sigma>0, pop_size>0),
  des.py:35 (pop>1), esmc.py:36 (pop odd>1, optimizer), guided_es.py:42 (pop even>1),
  noise_reuse_es.py:44 (pop>1), open_es.py:43 (stdev/lr/pop/mirrored-even/optimizer),
  persistent_es.py:45 (pop even>1), snes.py:39 (pop>1, weight_type), nes.py:65/200
  (positivity, recomb-weights descending order / init_std + recomb shape); (2) normalize
  np.ndarray array fields to flat f32 tuples — CONDITIONAL on isinstance for
  ars/as/des/esmc/guided_es/noise_reuse_es/open_es/persistent_es/snes, UNCONDITIONAL
  (re-rounds tuple input to f32 too) for cma_es (mean_init, weights) and both nes
  configs; (3) eagerly DERIVE defaults — XNES/SeparableNES pop_size = 4+⌊3·ln dim⌋ +
  learning-rate defaults from dim, ASEBO subspace_dims = len(center_init).
- Derivation-policy split: eager at Config()-time (nes, asebo) vs deferred to trace time
  (cma_es `_derive()` recomputes the pop_size default + 9 scalars per init/ask/tell;
  guided_es re-derives subspace_dims inline at init:71 and ask:99). core/workflow.py
  `_discover_pop_size` (line 247) relies on nes' eager pop_size fill for monitor
  auto-completion; cma_es' None pop_size is already handled by bench_so.py:132-135.
- Every in-repo construction site passes KEYWORD np.ndarray args (np.full/np.eye/rng
  arrays): the 12 smoke-test files, parity test_cma_es.py:54 + test_open_es.py:64,
  benchmarks/etl_vs_torch/bench_so.py:117/123. No positional or no-arg constructions
  exist (all configs have ≥2 required fields) and docs/README construct none.
- Type-annotation drift: ars/open_es declare `center_init: tuple[float, ...]` though all
  callers pass np.ndarray; asebo/des/esmc/snes declare `np.ndarray` though post-init
  storage is a tuple; guided_es/noise_reuse_es/persistent_es/cma_es declare honest
  unions; nes declares `Any`. Field names/order mirror the torch `__init__` exactly in
  all 12 configs (center_init vs mean_init vs init_mean is torch's own drift).
- Normalization is documented in the class docstring of ars/open_es/guided_es/
  noise_reuse_es/persistent_es/cma_es/XNESConfig, but SILENT for asebo, esmc, des, snes
  and SeparableNESConfig (des/snes carry only an inline code comment).
- Duplication to factor: the `object.__setattr__` + `tuple(np.asarray(x, dtype=F32)
  .tolist())` idiom appears 15+ times; the `etl.ops.constant(...)` center-bake block is
  duplicated in 10 init()s (only nes factored `_constant_1d`); `_softmax` is
  verbatim-duplicated (des.py:57, snes.py:63); the exp_avg/exp_avg_sq/best_fitness=inf/
  zero-noise init block repeats in ~10 init()s. No mutable-default args, module-level
  closures, or state-hiding classes anywhere in the node.

## Skipped
`virtual_lora_es.py` is NOT ported: it needs the torch Philox counter-stream
PRNG (`evox.triton_kernels.kernels.philox`), LoRA factor/gradient utilities
(`lora_noise.py`, uses torch.einsum), and a `(center, seeds, sigma)` tuple
evaluate protocol hard-coded into torch `StdWorkflow._evaluate` +
`VirtualLoRAProblem` (nn.Sequential + DataLoader). etl.random is key/split-only
(no jump-ahead counter streams) and evox_etl's contract is tensor-only
`(n, dim)` populations. Full analysis: see `src/evox_etl/algorithms/CONTEXT.md`.

## Known Issues
- `asebo.py:117` uses `etl.svd` (full reduced SVD: U, S, Vh), which etl's
  stablehlo-v1 exporter DEFERS (`BackendError` on compiled backends
  iree/xla). Runs fine on the etl-numpy backend (unit tests green) but would
  fail to export like the NSGA3 `matrix_rank`/`solve` blocker did. The NSGA3
  fix pattern (eigh-based equivalents, see `src/evox_etl/algorithms/mo/nsga3.py`)
  is the reference if asebo is ever benchmarked on compiled backends;
  reconstructing a full two-factor SVD (U and Vh) via eigh is possible but was
  deliberately NOT attempted here because it's unstable for rank-deficient
  inputs and asebo is not currently in the benchmark suite.

## Notes for agents (verified — do not re-investigate)
- All functions are PLAIN (no `@etl.defn`); traced via `etl.build`/`etl.run`.
- Known torch bugs NOT replicated: XNES/SeparableNES use `self.dim` before it
  exists when pop_size=None (torch nes.py lines 42-43, 154) — ports use the
  local `dim` (commented in nes.py).
- `etl.transpose(x, axes)` requires axes as a TUPLE (a list raises TypeError).
- Broadcast gotcha: `(k, n) * signs(k,)` does NOT align over the k axis — use
  `enp.expand_dims(signs, axis=1)`; `(m, k) * signs(k,)` aligns fine.
- `etl.svd` returns reduced `(U, S, Vh)`, Vh (k, n) — matches torch.svd(some=True).
- `etl.qr` returns `(Q, R)` reduced, Q first. `etl.cond` supports pytree outputs.
- `etl.eye(n)` is already float32; svd/eigh preserve f32; `etl.clamp` requires
  BOTH bounds (use etl.maximum for one-sided clamps); use `math.*` never `np.*`
  for Python scalars at trace time (numpy scalars are illegal operands).
- Config array baking inside functions:
  `etl.ops.constant(etl.core.tensor(np.asarray(config.center_init, dtype=np.float32)))`
  — numpy import is sanctioned for this + np.dtype only.

## Design Decisions

### CMA-ES deliberately replicates torch's scalar-dot-product `p_c @ p_c.T` quirk
The torch reference (`src/evox/algorithms/so/es_variants/cma_es.py:128`)
computes `p_c @ p_c.T` with **1-D** `p_c`: torch's deprecated `.T` is a no-op
on 1-D tensors, so this is a scalar dot product `‖p_c‖²` that broadcasts
additively into **every element** of `C` — an isotropic `c_1`-scaled C
inflation, NOT the canonical rank-one outer product. `cma_es.py` `tell()`
mirrors this exactly (`pc_norm_sq = etl.sum(p_c * p_c)`; 0-d, since
`etl.dot` needs rank ≥ 2) instead of the mathematically "correct" outer
product, because the parity contract is exact mirroring of torch's actual
behavior (the outer product stalls on plateaus: `CMAES/Ackley/100x10`,
mean_init=50, sigma=25, pop 100, key 42, 100 gens, stalls at best ≈ 20.56
while the scalar form converges to ≈ 0.0007). Verified on the numpy and
iree backends. Note: even with identical injected noise, later steps diverge
chaotically (float32 reduction order) — parity is qualitative (both converge
to ≈ 0), not bitwise.

## Tests
- Smoke (no torch): `unit_test/etl/algorithms/so/es_variants/test_*.py` — drive
  init/ask/tell via `helpers.run_generations` on Sphere; local conftest.py shims
  `import helpers`.
- Parity (torch allowed): `unit_test/etl/algorithms/parity/test_cma_es.py` and
  `test_open_es.py` — torch StdWorkflow+EvalMonitor vs etl; margin
  etl_best ≤ torch_best*1.1 + 1e-3; tuning rationale in
  `unit_test/etl/algorithms/parity/CONTEXT.md`.
- Gate: `/mnt/local-ssd/bchuang/evox/.venv/bin/python -m pytest
  unit_test/etl/algorithms/so/es_variants
  unit_test/etl/algorithms/parity/test_cma_es.py
  unit_test/etl/algorithms/parity/test_open_es.py -q`
