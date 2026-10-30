# install/ — Installation Guide

## Intent
This directory is the **Installation Guide** section of the EvoX documentation — the first step of the "Getting Started" journey.
It documents installing the **experimental EvoX 2.0 pre-release** only: the `evox` distribution whose documented import name is `evox_etl` and which runs on the `evoxc` compiler.
The 1.x torch API (`import evox`, `ModuleBase`, `Parameter`/`Mutable`, `torch.compile`/`vmap`, `src/evox/`) was removed in 2.0 and must not be reintroduced here.

## API Surface

| File | Purpose |
|---|---|
| `index.md` | Section landing page: `:maxdepth: 1` toctree over `python` and `evox`, a one-paragraph intro, and a `{warning}` marking the 2.0 pre-release scope. |
| `python.md` | "Python Installation Guide" — **Python ≥3.11 required** (`requires-python = ">=3.11"` in the 2.0 `pyproject.toml`), interpreter install per platform (Windows / macOS / Linux / `uv`), why an isolated environment matters, and `venv` + `pip` / `uv` workflows (with a pointer that plain `pip install evox` resolves the 1.x line, so the pre-release command in `evox.md` is the one to use). |
| `evox.md` | "EvoX Installation Guide" — the pre-release install (`pip install --pre evox==2.0.0a1` / `pip install "evox>=2.0.0a1"`) and the source checkout (`pip install -e .`, which pulls `evoxc` and `numpy` on its own), where `evox_etl.__version__` comes from, the extras table (`vis`, `all`, `iree`/`evoxc[iree]`, `tvm`/`evoxc[tvm]`, `compilers`/`evoxc[compilers]`), XLA's user-supplied PJRT plugin via `EVOXC_PJRT_PLUGIN`, `backend=`/`device=` selection, a PSO-on-Sphere smoke test with its real output, contributor checkout, and next steps. |

## Constraints
- Every code snippet must be verified against `src/evox_etl` (and `pyproject.toml` for packaging claims) before it is documented; the six-line smoke test runs 50 PSO generations on `Sphere` on the default `numpy` backend.
- Packaging facts these pages must match: `pyproject.toml` declares `version = "2.0.0a1"`, `requires-python = ">=3.11"`, `dependencies = ["evoxc >= 0.1.0", "numpy >= 2.0.0"]`, and the extras `iree`, `tvm`, `compilers`, `all`, `vis`, `default` (a 1.x alias of `vis`), `test` and `docs`; the compiled-backend extras are thin forwarders to `evoxc[iree]` / `evoxc[tvm]` / `evoxc[compilers]`.
- The pages document the compiled-backend extras in both spellings, `evox[iree]`/`evox[tvm]`/`evox[compilers]` and `evoxc[iree]`/`evoxc[tvm]`/`evoxc[compilers]`, and state that either spelling installs the same toolchain.
- `default` is described as the surviving 1.x alias of `vis`; it has no row of its own in the extras table.
- Python floor: 2.0 requires **3.11+**, declared by both the `evox` manifest and `evoxc`, and `import evox_etl` imports `evoxc` unconditionally.
- `pip install evox` resolves the **1.x stable line** (2.0.0a1 is a PEP 440 pre-release, and 1.4.0 is still the newest final release), whose tree has no `src/evox_etl`; 2.0 comes from `pip install --pre evox==2.0.0a1`, `pip install "evox>=2.0.0a1"`, or a source checkout.
- The XLA backend has no pip extra: it needs a user-provided PJRT plugin, named by `EVOXC_PJRT_PLUGIN` before the program runs; `compile_options={"plugin_path": …}` is documented as **not** a standalone alternative (evoxc's `build` forwards compile options to the compile stage only, so the load stage rediscovers the plugin from the environment / well-known paths).
- The `iree` extra ships no Vulkan loader or ICD, so the `vulkan-spirv` target needs a host Vulkan driver, and IREE's default profile caps workgroup shared memory at 16 KiB without a `vulkan_target`.
- `evox_etl.__version__` is derived from `importlib.metadata.version("evox")`: `2.0.0a1` once the distribution is installed, `0.0.0.dev0` in a bare source checkout; the verification step still leads with an import and the smoke-test number.
- Cross-links use the agreed paths `../../tutorial/tutorial_part2`, `../advanced/backends`, and `../developer/environment`.
- This is a leaf node: no child subdirectories, all content lives in the three `.md` files above.
- These pages are not manually translated; Chinese comes from the gettext catalog under `docs/source/locale/zh_CN/`, which is outside this node.

## Routing Table

| Page | Purpose |
|---|---|
| `index.md` | Section landing page and toctree |
| `python.md` | Interpreter and virtual-environment setup |
| `evox.md` | Installing, configuring and verifying EvoX 2.0 |
