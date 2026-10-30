# evox/problems/neuroevolution — virtual-population neuroevolution problems

## Intent
The virtual-population neuroevolution problem of the 2.0 package: a
Gaussian-noise ES problem that evaluates a population of perturbed networks
WITHOUT receiving a `(pop_size, dim)` population — it gets the tuple
`(center_flat, seeds, sigma)` and regenerates each individual's weight/bias noise
on demand from its seed.
`virtual_problem.py` is a frozen config dataclass + plain module-level
`make_virtual_problem` + plain module-level `evaluate(config, problem_state,
payload)` (NO `@evoxc.defn` — DESIGN §4.3). Full-noise and low-rank forward are
unified behind the `lora_rank` config field, and `VirtualLoRAProblem` is an alias
of `VirtualProblem`; the environment-backed neuroevolution problems (brax,
mujoco_playground, supervised_learning) are NOT implemented (external-library
dependent — reported to root, see `../CONTEXT.md`).

## API Surface
- `VirtualProblemConfig` (`= VirtualProblem`, `= VirtualLoRAProblem`): frozen,
  all-plain-static-leaf dataclass. Fields: `param_shapes` (tuple of int-tuples in
  parameter order), `layer_specs`
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
`evox.problems.numerical.state.ProblemState` (reused, not redefined); the
problem is stateless and defines no `init`. `sigma` passed at `evoxc.build` MUST be
re-passed by value at `evoxc.run` (`../CONTEXT.md` issue 17).

## Forward + noise
Uses the SHARED `evox.algorithms.so.es_variants.virtual_noise` stream
(splitmix64 + Box-Muller): `virtual_normal`, `lora_factors`, `compute_offsets`,
`compute_counter_offsets`. Per Linear layer: `base = h @ W_center^T` (shared) plus
a `sigma`-scaled per-individual noise matmul, so the `(out, in)` weight delta is
never added to the center weight. Activations stay at `(pop_size, batch,
features)`. Supported activations: `relu`, `tanh`, `sigmoid`, `gelu`
(`evoxc.gelu` == exact erf GELU), `identity`.

**Offsets (easy to get wrong):** center-vector slicing ALWAYS uses the full flat
parameter layout (`compute_offsets(param_shapes)`); only the NOISE stream switches
to `compute_counter_offsets(param_shapes, lora_rank)` in LoRA mode (counter
offsets can exceed `dim` — they index a separate padded stream).

**LoRA mode** (`lora_rank` not `None`): weight delta `= sigma * (h @ A^T) @ B^T`
with `A (pop, rank, in)`, `B (pop, out, rank)` from `lora_factors`; the bias delta
is direct 1-D noise (`lora_factors` on the `(out,)` shape).

**Noise stream:** full-noise `virtual_normal` is the deterministic splitmix64 +
Box-Muller stream (`seed ^ (idx*GAMMA)` hash input, low32→u1 / high32→u2, cos
branch, `u1` clamped at 1e-10), so a run is reproducible from the seeds alone.
LoRA factors reuse the same splitmix64 stream.

**Bias noise offset:** bias noise comes from `compute_offsets(b_idx)` and the center
bias is sliced at its own index (bias need not follow its weight in flat order), so
the layout stays correct for any parameter ordering.

## Config construction / validation (`make_virtual_problem`)
`ValueError` (never bare assert) for: non-2-D weight shape, bias shape ≠
`(out,)`, out-of-range weight/bias index, non-`"linear"` kind, unsupported
activation, empty `param_shapes`/`layer_specs`, malformed (non-4-tuple) layer
spec, non-positive `in_features`/`out_features`/`batch_size`, the layer feature
chain (`in_features` → … → `out_features`), `inputs` length ≠
`batch_size * in_features`, `targets` length ≠ `batch_size * out_features` (mse)
or ≠ `batch_size` (cross_entropy), bad `reduction`/`loss`, and a non-positive /
non-int `lora_rank`.

## Implemented scope
| Capability | Status | Notes |
|---|---|---|
| full-noise mode (`lora_rank=None`) | IMPLEMENTED | explicit `h @ noise^T` matmul per Linear layer. |
| LoRA mode (`lora_rank>0`) | IMPLEMENTED | non-fused `base + sigma*(h@A^T)@B^T`. |
| vmapped module-state forward (`ModelStateForwardResult`, `get_vmap_model_state_forward`) | NOT IMPLEMENTED | needs module state_dict + vmap; evoxc has neither. Parameter flattening is the host's job (see Migration notes). |
| supervised-learning problem (`SupervisedLearningProblem`) | NOT IMPLEMENTED | needs `torch.nn.Module` + `DataLoader` + vmapped module state. |
| Brax policy search (`BraxProblem`) | NOT IMPLEMENTED | needs `jax`/`brax` envs, DLPack bridging, a custom-op registry and `visualize()`. |
| MuJoCo Playground policy search (`MujocoProblem`) | NOT IMPLEMENTED | needs `jax`/`brax`/`mujoco_playground` MJX + `imageio` rendering. |

## Usage notes
- The `(center, seeds, sigma)` payload flows from the virtual ES algorithms
  (`VirtualES.step` calls `evaluate((center, seeds, sigma))`), so
  `solution_transform` MUST stay `None` — no host-side wrapper is needed,
  `StdWorkflow` passes the payload straight to `problem.evaluate` (payload-aware
  monitor via the algorithm's `monitor_candidate`).
- There is NO `ParamsAndVector` analogue anywhere in `src/evox` (no
  `to_vector`/`to_params`; not in `evox.utils.__all__`): the host must derive
  `param_shapes` (parameter order), the `layer_specs` 4-tuples and the flat f32
  `center`/`center_init` by hand, and keep the algorithm config's `param_shapes`
  identical to the problem config's.
- The loss is a string `loss` (`mse`|`cross_entropy` only, no
  weights/label_smoothing/ignore_index); there is no `device` field and no
  `n_batch_per_eval`.
- `sigma` is a static payload leaf, not a config field → one compiled step graph per
  distinct sigma value (`../CONTEXT.md` issue 17).
- No example driver exists (`examples/` has quickstart + HPO only); the only real
  wiring is `../../../unit_test/evox/algorithms/so/es_variants/test_virtual_e2e.py`
  (StdWorkflow + `make_virtual_es`/`make_virtual_lora_es` on a 4→8→1 net) — use it as
  the template.

## Deliberate design choices
1. **EvoX Compiler trade-off (materialization).** A fused kernel would never
   materialize `(out, in)` noise; evoxc has no fused matmul, so this module DOES
   materialize per-block `(pop_size, out, in)` noise (and `(pop_size, rank, k)` /
   `(pop_size, d, rank)` LoRA factors). Still far smaller than a full
   `(pop_size, dim)` population, so the O(dim) algorithm state (a `(dim,)` center
   + `(pop_size,)` seeds) is preserved.
2. **LoRA mode is an addition.** The unified `lora_rank` field makes the
   `VirtualLoRAProblem` alias usable with the (distinct) LoRA algorithm.
3. **Deterministic fixed batch.** The full baked dataset is ONE deterministic
   `(batch_size, in_features)` batch (no round-robin iterator, no mutable iterator
   state), so `evaluate` is a pure function of its payload (`reduction` aggregates
   over that batch only).
4. `LeakyReLU`/`ELU`/`Softmax` activations are NOT supported (their non-default
   parameters / softmax axis have no plain-static-leaf slot).
5. `mse` per-sample loss is the mean over output features (exact for
   `out_features == 1`).

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
`../../../unit_test/evox/problems/test_virtual_problem.py` (canonical mirror
location — sibling, write access requires parent), all green on the numpy backend.
Run it with the package venv (see `../../CONTEXT.md` "Environment"):

    PYTHONPATH="src" <venv>/bin/python -m pytest unit_test/evox/problems -q

Coverage: `make_virtual_problem` validation (param_shapes/layer_specs mismatch,
bad lora_rank/reduction/loss, malformed specs, chain/length mismatches), alias +
frozen-config checks, forward agreement against an INDEPENDENT pure-numpy reference
(all 5 activations × {mean,sum} × {full-noise, LoRA ranks 1/2/3}, plus 3-layer,
no-bias-layer and cross-entropy configs) and determinism. The reference re-derives
the splitmix64 noise itself (own offsets/forward/loss). Measured worst absolute
difference vs the reference: **0.0** (asserted tolerance 1e-5).

## Routing Table
| Area | Path | Notes |
|---|---|---|
| Virtual problem (config + make_* + evaluate) | `virtual_problem.py` | single module |
| Test suite | `../../../unit_test/evox/problems/test_virtual_problem.py` | sibling — write access requires parent |
| Shared noise contract | `../../algorithms/so/es_variants/virtual_noise.py` | sibling subtree — READ-ONLY, do NOT modify |
| Shared `ProblemState` | `../numerical/state.py` | sibling — READ-ONLY |
| Shared algorithm core | `../../algorithms/so/es_variants/virtual_es.py` | sibling subtree — the `(center, seeds, sigma)` producer |
