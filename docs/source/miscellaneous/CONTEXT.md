# docs/source/miscellaneous/ — Miscellaneous Reference Pages

## Intent
This directory holds the reference material that fits neither the tutorial nor the guides: which hardware an EvoX 2.0 workflow runs on, what non-NVIDIA hardware is reachable, which Linux and GPU driver to install on a server, and how to port MATLAB habits.
Every page targets the **experimental EvoX 2.0 pre-release** (import name `evox_etl`, compiled by `evoxc`) and therefore has no EvoX 1.x torch content — no `import evox`, `torch.device`, `Parameter`/`Mutable`, `ModuleBase`, `#evox.compile` or `mps` device.

## API Surface
| File | Page title | Purpose |
|---|---|---|
| `index.md` | Miscellaneous | One-line directory intro plus the MyST toctree of the four pages (explicit titles, one per page) |
| `selecting_gpu.md` | Choosing a CPU or a GPU | Device selection through `StdWorkflow(device=..., backend=...)`: a link to the `_parse_device` table (owned by `../guide/advanced/backends`) plus the index and 0-based semantics, the `evoxc` device kinds (`cpu`/`cuda`/`rocm`/`vulkan`), `evoxc.devices()` enumeration, why `device=` alone does not accelerate (the placement failure in `init()`, described by observable behaviour), and `CUDA_VISIBLE_DEVICES` pinning for multi-GPU hosts |
| `other_gpus.md` | Non-NVIDIA Hardware: AMD, Vulkan and Apple Silicon | A link to the backend × device-kind table (owned by `../guide/advanced/backends`) with one-line shorthand, the CPU-only `numpy` path and its result-agreement caveats (IREE bit-identical, XLA TF32, TVM CPU-only), AMD ROCm through IREE (incl. `--iree-rocm-target` detection), the XLA PJRT-plugin hardware mapping (`EVOXC_PJRT_PLUGIN` / `plugin_path`), the IREE Vulkan path, and the absence of any Metal/MPS path on Apple Silicon |
| `linux_distribution.md` | Linux Distribution and GPU Driver | Host advice: prefer a newer distribution, install the distribution-provided GPU driver, and note that EvoX needs the driver but not the CUDA toolkit |
| `transfer-from-matlab.md` | Porting MATLAB Code to EvoX | MATLAB → EvoX 2.0 comparison (indexing, broadcasting, functions, control flow) plus a full PSO port: a MATLAB struct/function version, the built-in `make_pso` + `StdWorkflow` version, and a hand-written algorithm module (config/state dataclasses + `init`/`init_step`/`step`) |

## Constraints
- All pages are MyST Markdown with one H1 each, ```` ```python ```` fences and `{note}`/`{warning}`/`{tip}` admonitions; cross-page links use `{doc}` relative paths.
- Device and backend facts must come from `evoxc` and `src/evox_etl/core/workflow.py`, never from memory: kinds, aliases, option names (`target_backends`, `iree_compile_args`, `vulkan_target`, `plugin_path`, `opt_level`) and `EVOXC_*` environment variables are all verifiable in the installed `evoxc` package.
- Backend mechanics (extras, compile options, limitations, performance, the `device=`/`compile_options=` tables) are owned by `../guide/advanced/backends` and the extras story by `../guide/install/evox`; link there instead of duplicating tables, and keep any local restatement consistent with them.
- Compiled-backend extras are `evox[iree]` / `evox[tvm]` / `evox[compilers]`, which forward to `evoxc[iree]` / `evoxc[tvm]` / `evoxc[compilers]`; either spelling installs the same toolchain, and `../guide/install/evox` owns the full extras story.
- Exception classes quoted on a page must have been reproduced in this repository: state the timing (parse-time `TypeError`, placement-time failure during `init()`, build-time `BackendError`) and describe the generic device-failure case behaviourally rather than naming one stable class.
- Every `python` sample must run against `src/evox_etl` (numpy backend, CPU); samples that need a GPU or an optional extra only need to *construct*, and that must be said when it is not obvious. Shown `text` output must match what the sample prints, line for line.
- Hardware that was not verified in this repository (ROCm, Vulkan, Apple Silicon GPUs, real PJRT plugins) must be labelled as documented-from-code, not measured.
- Count claims stay exactly as in the rest of the docs: "30+ algorithms", "30+ benchmark problems", ecosystem "140+ algorithms", "240+ benchmark problems/environments".
- The directory is listed from `docs/source/index.md` under the "Additional Resources" caption; that file is owned elsewhere and must not be edited from here.

## Routing Table
This is a leaf directory — all content lives in the five `.md` files above; there are no child subdirectories.
