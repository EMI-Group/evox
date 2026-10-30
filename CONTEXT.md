# EvoX — Functional Evolutionary Computation on the EvoX Compiler (2.0 pre-release)

## Intent
EvoX (distribution `evox`, version `2.0.0a1`, experimental pre-release) is a GPU-accelerated framework for scalable evolutionary computation, rewritten as a purely functional library on the **EvoX Compiler** (`evoxc`) tensor library.
The framework is the package `src/evox_etl/`, imported as `evox_etl`; no module is importable as `evox`, and the 1.x PyTorch tree no longer exists in this repository.
It ships 32 distinct algorithm classes, exported as 27 single-objective names (one of them, `VirtualLoRAES`, aliases `VirtualES`) plus 6 multi-objective names, along with 15 operators, numerical and virtual problems, metrics, visualization, and a host-side HPO wrapper.

**Core goals:**
- **High performance**: one compile site (`StdWorkflow` traces a single step graph through `evoxc.build`) plus per-workflow backend selection — the pure-NumPy interpreter by default, compiled IREE / TVM / XLA execution for CPU and GPU by changing one argument.
- **All-in-one**: single- and multi-objective optimization, virtual neuroevolution problems, and hyperparameter optimization, with 140+ algorithms and 240+ benchmark problems/environments across the EvoX ecosystem.
- **Easy-to-use**: frozen config dataclasses, `make_*` constructors, plain `init`/`step` functions, tensor-only state, and stateless RNG keys for exact reproducibility.

**Project metadata:** GPL-3.0-or-later, Python ≥3.11, runtime dependencies `evoxc >= 0.1.0` and `numpy >= 2.0.0`. Version 2.0.0a1, single-sourced in `pyproject.toml`.

## Repository Structure

```
evox/
├── src/evox_etl/       ← the framework (import name `evox_etl`): core, algorithms, operators,
│                         problems, metrics, workflows, vis_tools, utils, DESIGN.md
├── src/evox_etl_ext/   ← PEP 420 namespace package for plugins, autoloaded at `import evox_etl`
├── unit_test/etl/      ← the test suite (pytest, mirrors src/evox_etl)
├── benchmarks/         ← `etl_vs_torch/` backend-comparison harness + committed result JSONs
├── examples/           ← runnable quickstart and HPO demo scripts
├── docs/               ← Sphinx documentation (bilingual EN / zh_CN, autodoc2 API pages)
├── REFACTOR_REPORT.md  ← historical 1.4 → 2.0 refactor report
├── README.md / README_ZH.md   ← the 2.0 pre-release landing pages (EN / ZH)
├── pyproject.toml / MANIFEST.in
├── conftest.py         ← pytest sys.path shim (repo root + src/)
├── flake.nix / flake.lock / dev_env_builder.nix
├── .pre-commit-config.yaml / .readthedocs.yaml
└── LICENSE
```

## Routing Table

| Area | Path | Description |
|---|---|---|
| **Framework** | `src/evox_etl/` | Functional EvoX on `evoxc`: core protocols + `StdWorkflow`, all SO/MO algorithms (incl. the training-free virtual ES family), pure operators, numerical (basic/DTLZ/CEC2022) and virtual neuroevolution problems, host-side HPO wrapper, GD/IGD/HV metrics, Plotly + `.exv` visualization, `make_*` constructors; binding spec `DESIGN.md` |
| **Extensions** | `src/evox_etl_ext/` | PEP 420 plugin namespace; `auto_load_extensions()` merges installed packages into the matching `evox_etl.*` modules at `import evox_etl` time |
| **Tests** | `unit_test/etl/` | The sole test suite (pytest), mirroring `src/evox_etl/`: algorithms, operators, problems, metrics, workflows, `vis_tools`, extension autoload, plus the root `test_no_torch.py` release gate |
| **Benchmarks** | `benchmarks/` | `etl_vs_torch/` harness (backend ids × SO/MO suites × scales), committed result JSONs, `BENCHMARK_RESULTS.md` and `style_comparison.md` |
| **Docs** | `docs/` | Sphinx + shibuya, autodoc2 API reference, MyST Markdown: `tutorial/` (7 parts incl. the 1.x migration page), `guide/install/`, `guide/developer/`, `guide/advanced/` (backends, operators, metrics, viz, HPO, extensions), `examples/`, `miscellaneous/`, `locale/zh_CN` |
| **Examples** | `examples/` | Runnable standalone scripts: `quickstart.py` (functional numpy backend) and `hpo/` (single-objective, multi-objective, repeated HPO) |

## Key Architectural Concepts

### Component Model
An algorithm module exposes `init(config, key) -> state` and `step(config, state, evaluate) -> state`, optionally with `init_step`/`final_step` variants.
A problem module exposes `evaluate(config, state, candidates)`; a monitor module exposes `init`/`step`/`record_*` and keeps its history host-side.
`StdWorkflow` resolves each component's plain functions from `type(config).__module__`, so a config and its functions always live in the same module.

### Configuration and State
Configs are frozen dataclasses holding only static leaves (Python scalars, strings, flat `float` tuples) and are built by `make_*` constructors that normalize bounds and validate arguments.
State is a pytree whose leaves are `evoxc` tensors only; `WorkflowState` carries the algorithm state, problem state, monitor state, a generation counter and the RNG key.
Randomness is stateless (`evox_etl.key(seed)`, `evox_etl.random.*`): keys live in the state, are split there, and make a whole run reproducible from one integer seed.

### Compile Site and Backends
`StdWorkflow` is the only compile site: it builds one executable per step variant through `evoxc.build` and runs it in a same-device loop (`run`, or `init`/`init_step`/`step`/`final_step`).
`backend=` defaults to `"numpy"` and `device=` to `"cpu"`; `"iree"`, `"tvm"` and `"xla"` require their opt-in extras (XLA additionally needs a PJRT plugin).
Unsupported graph shapes raise a loud `evoxc` `BackendError`/`TransformError`; there is never a silent fallback.

### Host-Side Surface
Monitors, metrics, visualization and HPO run on the host in NumPy: `EvalMonitor` reads state through a token-passing history channel, and `HPOProblemWrapper` rebuilds and runs an inner `StdWorkflow` per candidate.
That is why `HPOProblemWrapper` is not an `evox_etl` `Problem` and cannot be nested inside an outer workflow's `evaluate`.

### Extension System
External packages install into the `evox_etl_ext` namespace package and expose plain functions, config dataclasses and `make_*` constructors.
`evox_etl/__init__.py` calls `evox_etl_ext.autoload_ext.auto_load_extensions()`, which merges them into the matching `evox_etl.*` modules; a missing `evox_etl_ext` is a silent no-op.

## Constraints
- **Eager-mode-free traced code**: every `evoxc` tensor op must run inside an active trace, so framework code never calls tensor ops at import time or outside `init`/`step`/`evaluate`.
- **No torch, host-side NumPy only**: nothing in the framework imports torch; NumPy is permitted host-side (CEC2022 data loading, `make_*` preprocessing, metrics, HPO, visualization) and never inside traced code.
- **Compile-friendly**: no Python control flow depending on tensor values; batches are expressed as leading tensor dimensions, and `vmap` is not used in algorithm or operator code.
- **Static configs, tensor state**: config fields are plain static leaves baked into the trace; state leaves are `evoxc` tensors only.
- **Minimization semantics**: algorithms minimize internally; `StdWorkflow` applies the `opt_direction` sign transform for maximization.
- **Monitors outside the graph**: monitors, metrics and HPO run host-side, connected to the loop through token-passing history.
- **Compiler dependency is the `evoxc` distribution**: the compiler reads `EVOXC_*` environment variables (e.g. `EVOXC_PJRT_PLUGIN` for the XLA plugin) and exposes `EVOXCError` as its public error base class.
- **Port names stay unchanged**: `evox_etl`/`evox_etl_ext`, `unit_test/etl/`, `benchmarks/etl_vs_torch/` and the frozen `etl-*` benchmark backend ids (`etl-numpy`, `etl-iree-llvm-cpu`, `etl-iree-cuda`, `etl-iree-rocm`, `etl-xla-cuda`, `etl-xla-rocm`) keep the ETL naming even though the compiler is `evoxc`; the ids are frozen keys into the committed result JSONs.
- **Bilingual docs**: documentation is English with a Simplified Chinese translation (`docs/source/locale/zh_CN`) served through Read the Docs language switching.
- **Count claims kept in sync**: the ecosystem "140+ algorithms" / "240+ benchmark problems/environments" counts and the built-in "30+ algorithms" / "30+ benchmark problems" counts are stated in this file, `README.md` and `README_ZH.md` (EN + ZH); any change must be applied to all three.
- **Descriptions vs counts**: `pyproject.toml` `[project] description` is just `"evox"` and carries no count claims — do not look there for them.

## Release Process

The 2.0 line is released as pre-releases; `2.0.0a1` is the current version and is single-sourced in `pyproject.toml` under `[project] version`.
A release is: a commit bumping the version (message style `Bump version to vX.Y.Z`), a `vX.Y.Z` tag on that commit, and a GitHub Release with **"Set as a pre-release"** ticked.
Ticking pre-release matters twice: the PyPI classifier metadata derives from the version string (`2.0.0a1` is a PEP 440 pre-release), and the GitHub release page tells users what they are getting.
PyPI treats `2.0.0a1` as a pre-release, so `pip install evox` keeps resolving the 1.x line and only `pip install --pre evox==2.0.0a1` (or `pip install 'evox>=2.0.0a1'`) installs 2.0.
`.github/workflows/python-publish.yml` triggers on the GitHub `release` (published) event, builds with `python -m build`, and publishes to PyPI through trusted publishing (OIDC, no repository token).
Trusted publishing requires a PyPI-side trusted-publisher entry for repo `EMI-Group/evox` / workflow `python-publish.yml`; a failing publish is usually a missing or mismatched entry there.
Because the distribution name stays `evox`, both lines share one PyPI project page, and the 1.x wheels remain the default resolution.

## Related Projects (EMI-Group ecosystem)

EvoX is one of ~13 sibling libraries published under https://github.com/EMI-Group/ .
These are SEPARATE repos (not part of this source tree) and are the upstream homes of the `evox_etl_ext` plugin family.
Together with EvoX, these libraries form an ecosystem of 140+ algorithms and 240+ benchmark problems/environments.
- **evomo** — PyTorch GPU-accelerated evolutionary multi-objective optimization (EMO). 57 tensorized MOEAs + 131 numerical/constrained benchmark problems (ZDT/DTLZ/WFG/MAF/LSMOP + CTP/MW/DOC/SDC/LSCMOP/DASCMOP/FCP) + MoRobtrol multi-objective robot-control suite.
- **evorl** — JAX fully GPU-accelerated Evolutionary RL framework (RL/EA/ERL/PBT/multi-agent algorithms; wraps Brax/MuJoCo-Playground/gymnax/Jumanji/JaxMARL/EnvPool/Gymnasium).
- **evogp** — PyTorch + custom CUDA kernels for Tree-based Genetic Programming (symbolic regression, classification, transformation, Brax/MuJoCo control).
- **tensorneat** — JAX tensorized NEAT / HyperNEAT.
- **tensoraco** — JAX tensorized Ant Colony Optimization for TSP (bundles the TSPLIB).
- **evoxbench** — NAS-as-multi-objective-optimization benchmark platform (NAS-Bench-101/201, NATS, DARTS, MobileNetV3, ResNet50, Transformer, MoSegNAS); no GPU/PyTorch needed.
- **tensorrvea**, **metade**, **istratde**, **evogo**, **autopso**, **evonas**, **evococ** — additional tensorized EC libraries/frameworks (RVEA, DE meta-framework, DE variant, data-driven black-box opt, automated PSO, NAS, automatic tensorization).
- Research/paper repos (single-algorithm code, not general libraries): GMOEA, RelativeNAS, DenseNAS, nsganetv2, fr-nas, gpbt-pl, Incremental-Optimization, GMPB, BLVG.
- Non-EC repos in the org (unrelated to EvoX): FaPN, PDVC, SoloGAN, mixer-snn, spikefpn, snn-temporal-fusion, spikingedn, SFLM, mesa, genesis, evox-portal (website).
The official website https://evox.group/ is built from the SEPARATE repo `EMI-Group/evox-portal` (Astro).
Its algorithm/problem count claims live in `src/i18n/locales/*.json` (`features.allinone.item1`, `runtime.hero.desc`) and in `src/content/tutorials/introduction/en.md` — editing counts in THIS repository does not update the website, they must be synced separately.
Pages under `src/content/releases/*` are historical release announcements and intentionally keep the numbers of the release they describe.

## Known Issues

- `evox_etl.__version__` is a hand-written string that still reports the 1.x value and is not derived from `pyproject.toml`, so it is not a reliable 2.0 release check — verify a 2.0 install by importing the package and running a smoke test.
- `evoxc` has no eager mode, so any direct `evoxc` tensor call outside an active trace raises `TraceError`; ad-hoc scripting must go through `StdWorkflow` or an explicit `evoxc.build`/`evoxc.run`.
- The XLA backend has no pip extra and needs a user-supplied PJRT plugin found through `EVOXC_PJRT_PLUGIN`, so it is unavailable on a default install; the `plugin_path` compile option alone is not enough because loading and running rediscover the plugin from the environment.
- `HPOProblemWrapper` is host-side by design: it is not an `evox_etl` `Problem`, evaluates candidates one at a time, and re-traces the inner workflow per candidate because config values are baked in as static leaves.
- Not implemented in the 2.0 line: the distributed multi-rank workflow, the Brax / MuJoCo Playground / supervised-learning problems, `ParamsAndVector` / `torch.nn` interop, the Triton kernel package, and any `torch.compile`/`vmap` layer.
- `unit_test/etl/` exercises the `"numpy"` backend only; compiled backends and GPUs are covered solely by the `benchmarks/etl_vs_torch/` harness.
