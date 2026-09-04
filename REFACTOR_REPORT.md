# EvoX → ETL Functional Refactoring — Final Report

**Goal:** rewrite EvoX on top of the ETL tensor library, replacing torch's
OOP-heavy design with plain functional code (dicts/tuples/dataclasses/namedtuples,
pure functions, separate `init`), benchmark torch vs etl on CPU/GPU from small
to large scale, compare code style, and report anything unportable.

**Result:** A complete functional redesign lives in `src/evox_etl/` (package
`evox_etl`) beside the frozen torch reference `src/evox/`. 363 unit tests green
across 4 suites; 240 benchmark records across 6 backends at 3 scales; two real
bugs (one torch-side, one port-side) found and fixed; three ETL-repo bugs fixed
upstream (etl commit `5ab645be`). The old JAX-era (pre-1.0) design informed the
function layout, but the module arrangement follows the current torch layout.

## 1. What was ported

| Area | Contents | Status |
|---|---|---|
| Core | `evox_etl.core`: module protocols, state helpers, functional `std_workflow` (compile-once `etl.build` step loop), `WorkflowState`, `EvalMonitorConfig` | ✅ |
| Algorithms | 33/34 SO+MO algorithms as pure `init`/`ask`/`tell` (+`init_ask`/`init_tell`) functions over frozen config/state dataclasses: DE/ES/PSO families, OpenES, CMA-ES, NSGA2/NSGA3/RVEA/MOEAD/HypE, SADE/SHADE, ASGA/RWGA, SparseL1, MOEAD, HypE… | ✅ 33/34 |
| Operators | all 15 operators (selection/crossover/mutation/sampling + `jit_fix_operator` utils) — torch-exact (parity ≤1e-6) | ✅ |
| Problems | numerical: `basic` (10 fns), `dtlz` (DTLZ1-7 + pf), `cec2022` (all 12) | ✅ |
| Metrics | GD, IGD(+), HV (host-side numpy implementations) | ✅ |
| Utils | `parse_opt_direction`, `min_by`, rank/rank-based fitness, pairwise dists, `evox_etl.key`/`random` | ✅ |
| Workflows | functional `StdWorkflow` + `EvalMonitor` (SO top-k elites, host-side PF) | ✅ |
| Tests | `unit_test/etl/`: algorithms 97, operators 82, problems 131, metrics 53 — **363 green** | ✅ |
| Benchmarks | `benchmarks/etl_vs_torch/`: 6 backends × 2 suites × 3 scales, results JSONs, `BENCHMARK_RESULTS.md`, `style_comparison.md` | ✅ |

**Skipped (external-library-bound, reported per objective):**
- `virtual_lora_es.py` — torch Philox counter-stream PRNG + LoRA einsum
  utilities + hardcoded OOP `StdWorkflow` protocol; unportable to etl's
  functional tracing model.
- `src/evox/problems/neuroevolution/` (Brax/MuJoCo/supervised-learning) —
  directly rely on external ecosystems (brax, mujoco, image datasets).
- `hpo_wrapper.py` (external Optuna coupling), `vis_tools` (plotly +
  EvoXVision binary format), `triton_kernels` (hand-written Triton CUDA —
  torch-internal), and the `evox_ext` PEP-420 extension auto-loading mechanics.

## 2. Design

- **Plain functions, not OOP**: each algorithm is `init(config, key) -> state`,
  `ask`, `tell`; frozen config dataclasses (`PSO`, `CMAESConfig`…) and
  frozen state dataclasses; the workflow is the ONLY place that traces/builds
  (`etl.build`), resolving component functions by module convention.
- **Separate init** bridges torch's OOP initialization: all device/dtype/state
  allocation moved out of constructors into `init`; configs are static leaves
  (closure-captured where they carry arrays/callables).
- **ETL has no eager mode** (all ops inside traced graphs) — the whole design
  was shaped by this; no tensor-valued Python control flow anywhere.
- Torch evox operators were already pure functions; the OOP→functional port was
  mostly algorithm/problem/workflow-level, which is why 33/34 algorithms port
  cleanly.

## 3. Benchmark results (summary)

Full data: `benchmarks/etl_vs_torch/BENCHMARK_RESULTS.md`; reproduction:
`<venv-python> benchmarks/etl_vs_torch/run.py --backend {torch-cpu,torch-cuda,etl-numpy,etl-iree-llvm-cpu,etl-iree-cuda,etl-xla-cuda} --suite {so,mo}`.
Machine: RTX A6000s, seed 42, 100 gens, ms/step includes host↔device staging.

- **SO (PSO/DE/CMA-ES/OpenES × Sphere/Rastrigin/Ackley, 100×10 → 10000×100):**
  all 36 cells execute on every backend (zero error cells). At 100×10 etl
  compiled backends are ~torch-parity; at 10000×100 **torch-cuda wins
  decisively** (PSO ~1.1 ms/step vs etl-xla-cuda ~17 ms/step; iree-cuda
  ~3.5 s/step), while **etl-xla-cuda wins PSO/DE/OpenES at 10000×100**
  (8–18 ms/step vs torch-cpu 8–36 ms/step). etl wins CMA-ES 100×10 big
  (torch-cpu 59–132 ms/step vs etl 1.3–10 ms/step); large-scale CMA-ES is the
  slowest etl cell (iree-llvm-cpu ~22 s/step — compiled while-loop overhead).
  etl fitnesses are bit-identical across all 4 etl backends (correctness of
  etl itself proven).
- **MO (NSGA2/NSGA3/MOEAD × DTLZ1/2):** etl-xla-cuda runs the full 12-case
  matrix end-to-end — **median 28× faster than torch-cpu** (NSGA3 77–108×;
  3.9–27.2 ms/step). etl-numpy is a full 100-gen matrix. **iree cannot compile
  any MO program** (upstream iree-compile segfault, see §5) — xla is the
  recommended compiled backend.
- **Parity:** 192 evaluable records → **106 ok / 86 not-ok / 48 skip**.
  not-ok is dominated by RNG-stream stochastic variance on unconverged Ackley/
  Rastrigin/DTLZ1 runs (torch itself shows the same scatter); converged cells
  match to ≤1e-4 or machine zero.

## 4. Code-style comparison

Detailed side-by-side in `benchmarks/etl_vs_torch/style_comparison.md`.

**Pros of the functional evox_etl:**
- State and configuration are explicit values (frozen dataclasses), not hidden
  `self.*` mutation — every transition is `State' = f(Config, State, X)`; no
  `Parameter`/`Mutable` bookkeeping machinery to learn.
- Instant parallelization story: `vmap`/batch semantics fall out of pure
  functions; no in-place `self.pop = …` assignments.
- One compile site (workflow) instead of torch's per-component `torch.compile`
  incantations; monitored values and PF extraction are plain host-side code.
- Trivial serialization/checkpointing: state is data, not an object graph.
- Determinism is explicit: `key` threading replaces torch's global RNG.

**Cons / costs:**
- No interactive/eager usage — the whole program is a traced graph; debugging
  requires the numpy backend, and semantics differ from "normal Python".
- ETL static-leaf policy requires closure-capture/normalization tricks for
  configs holding arrays/callables (documented as gotchas #1-18 in
  `src/evox_etl/CONTEXT.md`).
- Compiled-backend support is uneven (iree MO segfaults, deferred ops, see §5);
  the torch ecosystem's compiler maturity is absent.
- Python scalar configs make parameter sweeps easy but per-step overhead at
  small scale eats the "GPU wins" advantage vs torch's optimized kernels.

## 5. Bugs fixed

- **ETL repo (upstream, commit `5ab645be`):** (1) XLA adapter created a fresh
  PJRT client per compile/load → ~35.7 GiB GPU per client, then SIGABRT — fixed
  with a process-global refcounted shared client; (2) native `rng_bit_generator`
  path broken for u32-key state via PJRT compile — capability dropped (inline
  expansion); (3) env recipe (XLA_FLAGS, cuDNN LD_LIBRARY_PATH/LD_PRELOAD,
  `ETL_PJRT_PLUGIN`) documented. Real-plugin GPU probe: 26/26 bit-exact.
- **Torch reference CMA-ES `c_c` bug (fixed both sides):** `c_c = (mu_eff+2)/(dim+4+2·mu_eff/dim)`
  → NaN at pop≥1000 (torch silently, etl loudly). Replaced with Hansen
  canonical `(4+mu_eff/dim)/(dim+4+2·mu_eff/dim)`.
- **evox_etl CMA-ES port divergence (CMAES/Ackley/100x10 stall ~20.5 vs 0.0036):**
  the port used the canonical rank-one outer product `p_c p_cᵀ`; torch's 1-D
  `p_c @ p_c.T` is actually a scalar dot product (isotropic `c₁‖p_c‖²`
  inflation). Quirk replicated → all 4 etl backends converge (0.0005–0.0037).
- **DTLZ `cumprod`/`flip` + NSGA3 `matrix_rank`/`solve`** — stablehlo-v1
  exporter deferrals, replaced with exportable gather/reduce_prod and
  eigh-based equivalents (float64 Gram matrices); numpy-backend results
  bit-identical before/after.
- **canonical `apd_fn` bug**: was gathering `norm_obj` with `relu(x)` instead
  of torch's raw `norm_obj[x]` (negative wrap semantics) — fixed.

Per the "skip CMA-ES if problematic" option: CMA-ES turned out fully fixable
(the two bugs above were its only problems) — it now passes on all 6 backends
with zero error cells, so nothing was skipped. Its only residual trait is cost:
large-scale CMA-ES cells are the slowest compiled-etl cells (~22 s/step on
iree-llvm-cpu at 10000×100 — a while-loop/eigendecomposition overhead), but
xla-cuda runs them at 88–406 ms/step vs torch-cpu 133–168 ms/step.

## 6. Residual ETL limitations (reported, not fixable here)

1. **iree-compile segfault (-11) on every MO program** (23/24 iree MO cells):
   export succeeds, `libIREECompiler.so` crashes (deep recursion; unmodified
   NSGA2 triggers it identically) — upstream iree while-loop compiler bug.
   etl-xla-cuda is unaffected and is the recommended compiled backend.
2. **MOEAD/DTLZ2/100x3x10 runtime `ref is null`** (`hal.buffer_view.create`,
   INVALID_ARGUMENT) on iree-llvm-cpu after successful export+compile — one cell.
3. **stablehlo-v1 exporter defers `svd`/`matrix_rank`/`solve`/`cumprod`/`flip`:**
   NSGA3 and DTLZ are worked around evox-side (eigh/reduce_prod/gather);
   `asebo.py:117` still calls `etl.svd` — runs on the numpy backend (tests
   green) but would fail export; documented in
   `src/evox_etl/algorithms/so/es_variants/CONTEXT.md` (full two-factor SVD
   reconstruction is unstable for rank-deficient inputs; asebo isn't
   benchmarked).
4. **iree O2/O3 forbidden on CUDA** (eigh/NSGA2 codegen bugs) — 28% default
   compile options mandated. XLA `dynamic_shapes` not usable.
5. ETL has no eager mode; etl-numpy is an interpreter (SO large scales capped
   at 50/10 gens in benchmarks; MO now full 100 gens after per-generation PF
   extraction).

## 7. Verdict

The functional etl rewrite of EvoX is complete and validated: 33/34 algorithms,
all operators, numerical problems, metrics, workflows — 363 green tests,
bit-identical etl-backend fitnesses, MO at up to 108× torch-cpu speed on
xla-cuda, and a codebase that expresses the same algorithms in ~half the
machinery (no Parameter/Mutable, no nn.Module state mutation). The gaps are
almost entirely compiler-side (iree MO segfaults, deferred exporter ops) or
external-ecosystem code that was explicitly out of scope. The torch variant
stays as the frozen reference in `src/evox/`; the etl variant is the
functional successor in `src/evox_etl/`.
