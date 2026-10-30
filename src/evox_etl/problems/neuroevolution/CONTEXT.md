# evox_etl/problems/neuroevolution — virtual-population neuroevolution problems

## Intent
Functional-ETL port of the torch virtual-population neuroevolution problem
(`../../../evox/problems/neuroevolution/`, read-only reference): a Gaussian-noise
ES problem that evaluates a population of perturbed networks WITHOUT receiving a
`(pop_size, dim)` population — it gets the tuple `(center_flat, seeds, sigma)` and
regenerates each individual's weight/bias noise on demand from its seed.
`virtual_problem.py` is a frozen config dataclass + plain module-level
`make_virtual_problem` + plain module-level `evaluate(config, problem_state,
payload)` (NO `@evoxc.defn` — DESIGN §4.3). The torch `VirtualProblem` class and the
distinct `virtual_lora_problem.py` low-rank forward are unified here behind the
`lora_rank` config field (torch's public `VirtualLoRAProblem` export is itself an
alias of `VirtualProblem`); the other torch neuroevolution problems (brax,
mujoco_playground, supervised_learning) are NOT ported (external-library
dependent — reported to root, see `../CONTEXT.md`).

## API Surface
- `VirtualProblemConfig` (`= VirtualProblem`, `= VirtualLoRAProblem`): frozen,
  all-plain-static-leaf dataclass. Fields: `param_shapes` (tuple of int-tuples in
  torch `named_parameters()` order), `layer_specs`
  (`("linear", weight_idx, bias_idx_or_None, activation)` per layer),
  `inputs`/`targets` (flat float tuples), `in_features`, `out_features`,
  `batch_size`, `reduction` (`"mean"`|`"sum"`), `loss` (`"mse"`|`"cross_entropy"`),
  `lora_rank` (`int | None`, default `None`).
- `make_virtual_problem(...)`: normalizes (`to_float_tuple(dtype=np.float32)`) +
  validates; raises `ValueError` for every inconsistency (see below).
- `evaluate(config, problem_state, payload) -> (fitness, problem_state)`.
- Module-level `__all__` also exposes `SUPPORTED_ACTIVATIONS` / `SUPPORTED_LOSSES`
  / `SUPPORTED_REDUCTIONS` (module only, NOT re-exported at package level).
- `__init__.py` exports `virtual_problem`, `VirtualProblemConfig`,
  `VirtualProblem`, `VirtualLoRAProblem`, `make_virtual_problem`, `evaluate`.

## Payload protocol (SHARED contract with VirtualES/VirtualLoRAES)
`payload = (center_flat, seeds, sigma)` — `center_flat` `(dim,)` f32, `seeds`
`(pop_size,)` int64, `sigma` a STATIC Python float (it may live in the payload
pytree as a static leaf — verified). Returns `fitness` `(pop_size,)` f32,
minimization semantics. `problem_state` is the shared empty
`evox_etl.problems.numerical.state.ProblemState` (reused, not redefined); the
problem is stateless and defines no `init`. `sigma` passed at `evoxc.build` MUST be
re-passed by value at `evoxc.run` (`../CONTEXT.md` issue 17).

## Forward + noise
Uses the SHARED `evox_etl.algorithms.so.es_variants.virtual_noise` stream
(splitmix64 + Box-Muller): `virtual_normal`, `lora_factors`, `compute_offsets`,
`compute_counter_offsets`. Per Linear layer: `base = h @ W_center^T` (shared) plus
a `sigma`-scaled per-individual noise matmul, so the `(out, in)` weight delta is
never added to the center weight. Activations stay at `(pop_size, batch,
features)`. Supported activations: `relu`, `tanh`, `sigmoid`, `gelu`
(`evoxc.gelu` == exact erf GELU, torch `nn.GELU` default), `identity`.

**Offsets (easy to get wrong):** center-vector slicing ALWAYS uses the full flat
parameter layout (`compute_offsets(param_shapes)`); only the NOISE stream switches
to `compute_counter_offsets(param_shapes, lora_rank)` in LoRA mode (counter
offsets can exceed `dim` — they index a separate padded stream).

**LoRA mode** (`lora_rank` not `None`): weight delta `= sigma * (h @ A^T) @ B^T`
with `A (pop, rank, in)`, `B (pop, out, rank)` from `lora_factors`; the bias delta
is direct 1-D noise (`lora_factors` on the `(out,)` shape), mirroring torch
`lora_delta_output`.

**Noise-stream parity:** full-noise `virtual_normal` is formula-identical to torch's
PyTorch fallback `_cpu_normal_noise` (same `seed ^ (idx*GAMMA)` hash input, splitmix64,
low32→u1 / high32→u2, Box-Muller cos branch, `u1` clamped at 1e-10) but NOT to torch's
Triton path (`tl.randn`), so bit-parity holds only against the CPU fallback. LoRA
factors reuse the same splitmix64 stream while torch's LoRA path uses `philox_normal`
→ statistically, not bit, equivalent.

**Bias noise offset:** the port draws bias noise from `compute_offsets(b_idx)` and
slices the center bias at its own index (bias need not follow its weight in flat
order), while torch's fused op hard-codes `weight_offset + out*in`. Equal for the
standard `named_parameters()` adjacency.

## Config construction / validation (`make_virtual_problem`)
`ValueError` (never bare assert) for: non-2-D weight shape, bias shape ≠
`(out,)`, out-of-range weight/bias index, non-`"linear"` kind, unsupported
activation, empty `param_shapes`/`layer_specs`, malformed (non-4-tuple) layer
spec, non-positive `in_features`/`out_features`/`batch_size`, the layer feature
chain (`in_features` → … → `out_features`), `inputs` length ≠
`batch_size * in_features`, `targets` length ≠ `batch_size * out_features` (mse)
or ≠ `batch_size` (cross_entropy), bad `reduction`/`loss`, and a non-positive /
non-int `lora_rank`.

## Torch parity map (frozen reference `../../../evox/problems/neuroevolution/`)
| Torch file (public class) | Status | Reason / lost capability |
|---|---|---|
| `virtual_problem.py` (`VirtualProblem`) | PORTED | full-noise mode (`lora_rank=None`); fused Triton kernel replaced by an explicit `h @ noise^T` matmul. |
| `virtual_lora_problem.py` (`VirtualLoRAProblem`) | PORTED | LoRA mode (`lora_rank>0`); non-fused `base + sigma*(h@A^T)@B^T`. |
| `utils.py` (`ModelStateForwardResult`, `get_vmap_model_state_forward`) | INTENTIONALLY OMITTED | built on `torch.func.stack_module_state` + `evox.core.vmap` + `use_state`; no state_dict/vmap in evoxc. NOTE this file has NO param-flatten helper and no `FlattenedState` — flattening lives in `evox.utils.ParamsAndVector` (`parameters_and_vector.py`), which is what the virtual problems used. |
| `supervised_learning.py` (`SupervisedLearningProblem`) | INTENTIONALLY OMITTED | needs `torch.nn.Module` + `DataLoader` + vmapped module state. Lost: train a population of torch models on an arbitrary dataset (the general neuroevolution entry point). |
| `brax.py` (`BraxProblem`) | INTENTIONALLY OMITTED | needs `jax`/`brax` envs, DLPack bridging, `torch.library.custom_op` + functorch unwrapping + `weakref`-keyed global registry + mutated `self.key`. Lost: policy search on Brax locomotion envs + `visualize()`. |
| `mujoco_playground.py` (`MujocoProblem`) | INTENTIONALLY OMITTED | needs `jax`/`brax`/`mujoco_playground` MJX + `imageio` rendering (signature also lacks `backend`/`compile_policy` vs `BraxProblem`). Lost: MJX env policy search + `visualize()`. |

## Migration notes (torch → this port)
- The `(center, seeds, sigma)` payload protocol is NOT new: torch's
  `VirtualES.step` already called `self.evaluate((center, seeds, sigma))` and the
  workflow already routed payloads, so `solution_transform` MUST stay `None` in both —
  no host-side wrapper is needed, `StdWorkflow` passes the payload straight to
  `problem.evaluate` (payload-aware monitor via the algorithm's `monitor_candidate`).
- There is NO `ParamsAndVector` analogue anywhere in `src/evox_etl` (no
  `to_vector`/`to_params`; not in `evox_etl.utils.__all__`): the host must derive
  `param_shapes` (torch `named_parameters()` order), the `layer_specs` 4-tuples and
  the flat f32 `center`/`center_init` by hand, and keep the algorithm config's
  `param_shapes` identical to the problem config's.
- `criterion: nn.Module` → the string `loss` (`mse`|`cross_entropy` only, no
  weights/label_smoothing/ignore_index); `device=` and `n_batch_per_eval` are gone.
- `sigma` is a static payload leaf, not a config field → one compiled step graph per
  distinct sigma value (`../CONTEXT.md` issue 17).
- No example driver exists (`examples/` has quickstart + HPO only); the only real
  wiring is `../../../unit_test/etl/algorithms/so/es_variants/test_virtual_e2e.py`
  (StdWorkflow + `make_virtual_es`/`make_virtual_lora_es` on a 4→8→1 net) — use it as
  the template.

## Deliberate differences from the torch reference
1. **EvoX Compiler trade-off (materialization).** Torch's fused Triton kernel never
   materializes `(out, in)` noise; evoxc has no fused matmul, so this port DOES
   materialize per-block `(pop_size, out, in)` noise (and `(pop_size, rank, k)` /
   `(pop_size, d, rank)` LoRA factors). Still far smaller than a full
   `(pop_size, dim)` population, so the O(dim) algorithm state (a `(dim,)` center
   + `(pop_size,)` seeds) is preserved.
2. **LoRA mode is an addition.** Torch's modern `VirtualProblem` has no LoRA mode;
   the unified `lora_rank` field makes the `VirtualLoRAProblem` alias usable with
   the (distinct) LoRA algorithm.
3. **Deterministic fixed batch.** Torch round-robins a `DataLoader` iterator
   (`n_batch_per_eval`) with mutable iterator state; here the full baked dataset is
   ONE deterministic `(batch_size, in_features)` batch, so `evaluate` is a pure
   function of its payload (`reduction` then aggregates over that batch only).
4. torch's `LeakyReLU`/`ELU`/`Softmax` activations are NOT ported (their
   non-default parameters / softmax axis have no plain-static-leaf slot).
5. `mse` per-sample loss is the mean over output features; identical to torch for
   `out_features == 1` (the only case torch's `MSELoss` path supports).

## Verified evoxc facts (do not re-investigate)
- Batched AND mixed-rank `evoxc.matmul` broadcast like numpy: `(batch,in) @
  (pop,in,out) -> (pop,batch,out)`, `(pop,batch,in) @ (pop,in,out) -> (pop,batch,out)`
  — so the first layer needs no `(pop, ...)` input expansion at all.
- `evoxc.gelu` is the EXACT erf GELU (matches `0.5*x*(1+erf(x/sqrt2))`); `evoxc.equal`
  exists (`!=` is still not overloaded — use `evoxc.not_equal`); `enp.zeros(shape)` is
  usable inside a trace.
- Static 1-D slicing `leaf[a:b]` with static ints works (unlike full-axis `:` over
  a dynamic dim — see `../numerical/CONTEXT.md`).
- `reshape`/`transpose` with static ints (incl. the pop dim from
  `int(seeds.shape[0])`, the shared `lora_factors` idiom) work.

## EvoX Compiler issues found (escalated to root — recorded as #11 in `../CONTEXT.md`)
1. **`evoxc` operators reject numpy scalars as operands.** `x / np.float32(2.0)`
   raises `TypeError: unsupported operand type float32: op operands must be
   SymbolicTensor or Python scalars (bool, int, float, complex); numpy
   scalars/arrays are not Python scalars`. Repro: trace
   `lambda x: x / np.float32(2.0)` → TypeError. Workaround: pass a Python float
   (`x / 2.0`). Minor/cosmetic (host constants, not state).

## Test suite
`../../../unit_test/etl/problems/test_virtual_problem.py` (canonical mirror
location — sibling, write access requires parent). 51 tests, all green on the
numpy backend. The problems gate also collects `parity/`, which imports `torch`,
so a torch (CPU) wheel must be installed alongside evoxc/pytest/numpy:

    BASE=$TMPDIR/evoxc_run
    uv venv --python 3.13 $BASE/venv
    uv pip install --python $BASE/venv/bin/python <path-to-evoxc-checkout> pytest numpy
    uv pip install --python $BASE/venv/bin/python torch \
        --index-url https://download.pytorch.org/whl/cpu
    PYTHONPATH="src" $BASE/venv/bin/python -m pytest unit_test/etl/problems -q

Coverage: `make_virtual_problem` validation (param_shapes/layer_specs mismatch,
bad lora_rank/reduction/loss, malformed specs, chain/length mismatches), alias +
frozen-config checks, forward parity against an INDEPENDENT pure-numpy reference
(all 5 activations × {mean,sum} × {full-noise, LoRA ranks 1/2/3}, plus 3-layer,
no-bias-layer and cross-entropy configs) and determinism. The reference re-derives
the splitmix64 noise itself (own offsets/forward/loss). Measured worst absolute
difference vs the reference: **0.0** (asserted tolerance 1e-5).

## Routing Table
| Area | Path | Notes |
|---|---|---|
| Virtual problem (config + make_* + evaluate) | `virtual_problem.py` | single module |
| Test suite | `../../../unit_test/etl/problems/test_virtual_problem.py` | sibling — write access requires parent |
| Shared noise contract | `../../algorithms/so/es_variants/virtual_noise.py` | sibling subtree — READ-ONLY, do NOT modify |
| Shared `ProblemState` | `../numerical/state.py` | sibling — READ-ONLY |
| Reference (torch) impl | `../../../../evox/problems/neuroevolution/` | READ-ONLY, never modify |
