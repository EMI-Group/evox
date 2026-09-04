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

## Skipped
`virtual_lora_es.py` is NOT ported: it needs the torch Philox counter-stream
PRNG (`evox.triton_kernels.kernels.philox`), LoRA factor/gradient utilities
(`lora_noise.py`, uses torch.einsum), and a `(center, seeds, sigma)` tuple
evaluate protocol hard-coded into torch `StdWorkflow._evaluate` +
`VirtualLoRAProblem` (nn.Sequential + DataLoader). etl.random is key/split-only
(no jump-ahead counter streams) and evox_etl's contract is tensor-only
`(n, dim)` populations. Full analysis: see `src/evox_etl/algorithms/CONTEXT.md`.

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

## Known Issues

### CMA-ES rank-one term diverges from the torch reference (Ackley stall)
`cma_es.py` `tell()` line ~241 computes the rank-one covariance term as the
canonical outer product `rank_one = expand_dims(p_c,1) * expand_dims(p_c,0)`.
The torch reference (`src/evox/algorithms/so/es_variants/cma_es.py:128`) writes
`p_c @ p_c.T` where `p_c` is **1-D**, so torch evaluates a scalar dot product
`‖p_c‖²` (deprecation-warned `.T` no-op on 1-D) that broadcasts additively into
**every element** of `C`: effectively
`C += c_1 * ‖p_c‖² * ones(dim, dim)` — an isotropic C inflation, NOT a rank-one
update. The port "fixed" this to the mathematically intended outer product,
which diverges from the reference: without the isotropic inflation, C/sigma
decay on flat plateaus and the algorithm stalls. Concretely,
`CMAES/Ackley/100x10` (mean_init=50, sigma=25, seed 42, 100 gens) stalls at
best ≈ 20.56 on all etl backends (bit-identical — deterministic divergence,
not RNG) while torch-cpu reaches 0.0036. Verified: with identical injected
noise, etl matches torch to ~1e-6 through step 1 and converges to ≈0.003 with
the scalar-broadcast C update; with the etl's own key 42 the same update gives
best ≈ 0.00065 (fixes the stall).
**Fix (parity contract is exact torch mirroring):** replace the rank-one term
with the torch-faithful scalar — `pc_norm_sq = etl.sum(p_c * p_c)` (0-d;
`etl.dot` needs rank ≥ 2) and use `p.c_1 * (pc_norm_sq + (1 - h_sigma) *
p.c_c * (2 - p.c_c) * state.C)` in the C update (0-d + matrix broadcasts on
the numpy backend; verify on iree/xla). Do NOT "fix" the torch side to the
outer product — parity means matching torch's actual behavior.
Note: even with identical noise, later steps diverge chaotically (float32
reduction order) — parity is qualitative (both converge to ≈0), not bitwise.

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
