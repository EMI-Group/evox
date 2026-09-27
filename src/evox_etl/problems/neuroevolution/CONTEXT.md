# evox_etl/problems/neuroevolution — virtual-population neuroevolution problems

## Intent
Functional-ETL port of the torch virtual-population neuroevolution problem
(`../../../evox/problems/neuroevolution/`, read-only reference): a Gaussian-noise
ES problem that evaluates a population of perturbed networks WITHOUT receiving a
`(pop_size, dim)` population — it gets the tuple `(center_flat, seeds, sigma)` and
regenerates each individual's weight/bias noise on demand from its seed.
`virtual_problem.py` is a frozen config dataclass + plain module-level
`make_virtual_problem` + plain module-level `evaluate(config, problem_state,
payload)` (NO `@etl.defn` — DESIGN §4.3). The torch `VirtualProblem` and
`VirtualLoRAProblem` classes are unified here behind the `lora_rank` config field;
the other torch neuroevolution problems (brax, mujoco_playground,
supervised_learning, virtual_lora) are NOT ported (external-library dependent —
reported to root, see `../CONTEXT.md`).

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
- `__init__.py` exports `virtual_problem`, `VirtualProblemConfig`,
  `VirtualProblem`, `VirtualLoRAProblem`, `make_virtual_problem`, `evaluate`.

## Payload protocol (SHARED contract with VirtualES/VirtualLoRAES)
`payload = (center_flat, seeds, sigma)` — `center_flat` `(dim,)` f32, `seeds`
`(pop_size,)` int64, `sigma` a STATIC Python float (it may live in the payload
pytree as a static leaf — verified). Returns `fitness` `(pop_size,)` f32,
minimization semantics. `problem_state` is the shared empty
`evox_etl.problems.numerical.state.ProblemState` (reused, not redefined); the
problem is stateless and defines no `init`. `sigma` passed at `etl.build` MUST be
re-passed by value at `etl.run` (`../CONTEXT.md` issue 17).

## Forward + noise
Uses the SHARED `evox_etl.algorithms.so.es_variants.virtual_noise` stream
(splitmix64 + Box-Muller): `virtual_normal`, `lora_factors`, `compute_offsets`,
`compute_counter_offsets`. Per Linear layer: `base = h @ W_center^T` (shared) plus
a `sigma`-scaled per-individual noise matmul, so the `(out, in)` weight delta is
never added to the center weight. Activations stay at `(pop_size, batch,
features)`. Supported activations: `relu`, `tanh`, `sigmoid`, `gelu`
(`etl.gelu` == exact erf GELU, torch `nn.GELU` default), `identity`.

**Offsets (easy to get wrong):** center-vector slicing ALWAYS uses the full flat
parameter layout (`compute_offsets(param_shapes)`); only the NOISE stream switches
to `compute_counter_offsets(param_shapes, lora_rank)` in LoRA mode (counter
offsets can exceed `dim` — they index a separate padded stream).

**LoRA mode** (`lora_rank` not `None`): weight delta `= sigma * (h @ A^T) @ B^T`
with `A (pop, rank, in)`, `B (pop, out, rank)` from `lora_factors`; the bias delta
is direct 1-D noise (`lora_factors` on the `(out,)` shape), mirroring torch
`lora_delta_output`.

## Config construction / validation (`make_virtual_problem`)
`ValueError` (never bare assert) for: non-2-D weight shape, bias shape ≠
`(out,)`, out-of-range weight/bias index, non-`"linear"` kind, unsupported
activation, empty `param_shapes`/`layer_specs`, malformed (non-4-tuple) layer
spec, non-positive `in_features`/`out_features`/`batch_size`, the layer feature
chain (`in_features` → … → `out_features`), `inputs` length ≠
`batch_size * in_features`, `targets` length ≠ `batch_size * out_features` (mse)
or ≠ `batch_size` (cross_entropy), bad `reduction`/`loss`, and a non-positive /
non-int `lora_rank`.

## Deliberate differences from the torch reference
1. **ETL trade-off (materialization).** Torch's fused Triton kernel never
   materializes `(out, in)` noise; etl has no fused matmul, so this port DOES
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

## Verified etl facts (do not re-investigate)
- Batched AND mixed-rank `etl.matmul` broadcast like numpy: `(batch,in) @
  (pop,in,out) -> (pop,batch,out)`, `(pop,batch,in) @ (pop,in,out) -> (pop,batch,out)`
  — so the first layer needs no `(pop, ...)` input expansion at all.
- `etl.gelu` is the EXACT erf GELU (matches `0.5*x*(1+erf(x/sqrt2))`); `etl.equal`
  exists (`!=` is still not overloaded — use `etl.not_equal`); `enp.zeros(shape)` is
  usable inside a trace.
- Static 1-D slicing `leaf[a:b]` with static ints works (unlike full-axis `:` over
  a dynamic dim — see `../numerical/CONTEXT.md`).
- `reshape`/`transpose` with static ints (incl. the pop dim from
  `int(seeds.shape[0])`, the shared `lora_factors` idiom) work.

## ETL issues found (escalate to root — also needs adding to `../CONTEXT.md`)
1. **`etl` operators reject numpy scalars as operands.** `x / np.float32(2.0)`
   raises `TypeError: unsupported operand type float32: op operands must be
   SymbolicTensor or Python scalars (bool, int, float, complex); numpy
   scalars/arrays are not Python scalars`. Repro: trace
   `lambda x: x / np.float32(2.0)` → TypeError. Workaround: pass a Python float
   (`x / 2.0`). Minor/cosmetic (host constants, not state).

## Test suite
`tests/` (in-node, relocate-ready; canonical home after relocation is the sibling
`unit_test/etl/...`). `tests/conftest.py` is the same repo-root marker shim as
`../tests/conftest.py` (finds the `pyproject.toml` ancestor, so it survives
relocation). 51 tests, all green on the numpy backend:

    BASE=$TMPDIR/etl_run
    uv venv --python 3.13 $BASE/venv
    uv pip install --python $BASE/venv/bin/python $BASE/etl_src pytest numpy
    PYTHONPATH="src" $BASE/venv/bin/python -m pytest \
        src/evox_etl/problems/neuroevolution/tests -q

Coverage: `make_virtual_problem` validation (param_shapes/layer_specs mismatch,
bad lora_rank/reduction/loss, malformed specs, chain/length mismatches), alias +
frozen-config checks, forward parity against an INDEPENDENT pure-numpy reference
(all 5 activations × {mean,sum} × {full-noise, LoRA ranks 1/2/3}, plus 3-layer,
no-bias-layer and cross-entropy configs) and determinism. The reference re-derives
the splitmix64 noise itself (own offsets/forward/loss). Measured worst absolute
difference vs the reference: **0.0** (asserted tolerance 1e-5).

**.gitignore gotcha**: the repo-root `.gitignore` has a bare `tests` entry
(`.gitignore:168`) that matches `tests/` here — new files there must be added with
`git add -f`.

## Routing Table
| Area | Path | Notes |
|---|---|---|
| Virtual problem (config + make_* + evaluate) | `virtual_problem.py` | single module |
| Test suite (relocate-ready) | `tests/` | in-node copy; canonical home is the sibling unit_test dir — parent relocates |
| Shared noise contract | `../../algorithms/so/es_variants/virtual_noise.py` | sibling subtree — READ-ONLY, do NOT modify |
| Shared `ProblemState` | `../numerical/state.py` | sibling — READ-ONLY |
| Reference (torch) impl | `../../../../evox/problems/neuroevolution/` | READ-ONLY, never modify |
