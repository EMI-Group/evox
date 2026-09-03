# evox_etl/problems/numerical — benchmark problems in functional style

## Intent
Port of the numerical problems from torch evox (`../../../evox/problems/`, read-only):
basic.py (Sphere, Rastrigin, Rosenbrock, Ackley, Griewank, Schwefel...), dtlz.py
(DTLZ1-7 + helpers), cec2022.py (with shipped input data). `evaluate(config,
problem_state, pop) -> (fitness, problem_state)` as a PLAIN function (not
`@etl.defn` — see DESIGN §4.3).

NOT ported (external-library dependent — reported to root): neuroevolution problems
(brax, mujoco_playground, supervised_learning, virtual_lora), hpo_wrapper.

## Implemented
- `cec2022.py` — CEC2022 config dataclass + `evaluate` + all internal math as plain
  module-level functions (cec2022_f1..f12, shift/rotate/cut/sr_func_rate/cf_cal,
  levy/bent_cigar/hgbat/katsuura/modified_schwefel/schaffer_F7/escaffer6/happycat/
  grie_rosen/discus/ellips). Imports ackley/griewank/rastrigin/rosenbrock/zakharov
  from `basic` (sibling module). Verified: all 12 functions × dims 2/10/20 match
  torch within rel 1e-5 (tolerance 1e-3).

## Notes for Agents (verified etl facts — do not re-investigate)
- **`x[:, a:b]` slices FAIL when any axis keeps a full-axis `:` over a dynamic
  (`None`) dim.** Use `etl.gather(x, const_int32_arange, axis=1)` (np.take semantics;
  result shape `(n, len(idx))`). Static-int column indexing `x[:, 0]` / `x[:, -1]`
  works. Slicing CONSTANT tensors (fully static shapes) works normally.
- **Reductions directly over a trace-input leaf fail**: shape mismatch
  (`IR-inferred (None,)` vs frontend `(Dim('_dynamic_0_0'),)`). Any elementwise op
  first (e.g. `x * 1.0`) flips the tensor to the `None`-form that reductions accept.
- **`enp.expand_dims`/`reshape` cannot carry dynamic dims** (None entries rejected;
  Dim entries fail at lowering). Avoid expand_dims on batch-dim tensors — unroll
  static Python loops instead (see `katsuura_func`).
- No `zeros_like`/`ones_like`/`full_like` in etl.numpy — zero-init via `x * 0.0`.
- `!=` on symbolic tensors raises TraceError (only `== < > <= >=` overloaded) — use
  `etl.not_equal`. `enp.floor` doesn't exist — use top-level `etl.floor`.
- `etl.run(exe, ...)` needs ALL positional args (static config/state included) and
  returns etl `Tensor`s (`.numpy()` to inspect). `TensorSpec` shape is a flat tuple
  `(None, dim)`.
- `etl.build(fn, config, state, spec)` accepts frozen dataclasses (int fields) as
  static args; `etl.evaluate` rejects non-tensor args — don't use it here.

## Data location
cec2022 input data lives in `../../../evox/problems/numerical/cec2022_input_data/`
(torch source tree) — loaded as numpy with `DATA_DIR = Path(__file__).resolve()
.parents[4] / "src" / "evox" / "problems" / "numerical" / "cec2022_input_data"`.
numpy allowed ONLY for that loading; never in graph code (bake via
`etl.constant(etl.tensor(npy.astype(np.float32)))` inside the trace).
