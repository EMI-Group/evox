# install/ — Installation Guide

## Intent
This directory is the **Installation Guide** section of the EvoX documentation — the first step of the "Getting Started" journey.
It documents installing the **experimental EvoX 2.0 pre-release** only: the `evox` distribution whose documented import name is `evox_etl` and which runs on the `evoxc` compiler.
The 1.x torch API (`import evox`, `ModuleBase`, `Parameter`/`Mutable`, `torch.compile`/`vmap`) is deliberately absent from these pages and must not be reintroduced.

## API Surface

| File | Purpose |
|---|---|
| `index.md` | Section landing page: `:maxdepth: 1` toctree over `python` and `evox`, a one-paragraph intro, and a `{warning}` marking the 2.0 pre-release scope. |
| `python.md` | "Python Installation Guide" — **Python ≥3.11 required** (the 1.x `>=3.10` metadata floor is mentioned once as not applying to 2.0), interpreter install per platform (Windows / macOS / Linux / `uv`), why an isolated environment matters, and `venv` + `pip` / `uv` workflows. |
| `evox.md` | "EvoX Installation Guide" — source-checkout install (`pip install -e .` **and** `pip install evoxc`), the extras table (`vis`, `iree`/`evoxc[iree]`, `tvm`/`evoxc[tvm]`, `compilers`/`evoxc[compilers]`), XLA's user-supplied PJRT plugin via `EVOXC_PJRT_PLUGIN`, `backend=`/`device=` selection, a PSO-on-Sphere smoke test with its real output, contributor checkout, and next steps. |

## Constraints
- Every code snippet must be verified against `src/evox_etl` (and `pyproject.toml` for packaging claims) before it is documented; the six-line smoke test runs 50 PSO generations on `Sphere` on the default `numpy` backend.
- Packaging facts these pages must match: `pyproject.toml` declares `dependencies = ["torch >= 2.6.0", "numpy >= 2.0.0"]` (torch is a hard dependency the 2.0 code never imports) and does **not** declare `evoxc`; its extras are still the 1.x set (`vis`, `default` = a duplicate of `vis`, `neuroevolution`, `test`, `docs`, `triton`).
- The pages document the compiled-backend extras in both spellings, `evox[iree]`/`evox[tvm]`/`evox[compilers]` first and `evoxc[iree]`/`evoxc[tvm]`/`evoxc[compilers]` second, with a note that only the `evoxc[...]` spelling installs anything today (the `evox` manifest declares no compiled-backend extra) — the `evox[...]` form is the intended 2.0 surface.
- The 1.x extras (`default`, `neuroevolution`, `test`, `docs`, `triton`) are never advertised; `default` has no row in the extras table.
- Python floors: 2.0 requires **3.11+** (`evoxc` declares `>=3.11` and `import evox_etl` imports it unconditionally); `evox`'s `requires-python >= 3.10` is the 1.x metadata floor and must not be presented as usable.
- The only tagged release is `v1.4.0`, whose tree has no `src/evox_etl` (the package landed after the tag), so `pip install evox` from PyPI does not provide `import evox_etl`; only a source checkout does.
- The XLA backend has no pip extra: it needs a user-provided PJRT plugin, named by `EVOXC_PJRT_PLUGIN` before the program runs; `compile_options={"plugin_path": …}` is documented as **not** a standalone alternative (evoxc's `build` forwards compile options to the compile stage only, so the load stage rediscovers the plugin from the environment / well-known paths).
- The `iree` extra ships no Vulkan loader or ICD, so the `vulkan-spirv` target needs a host Vulkan driver, and IREE's default profile caps workgroup shared memory at 16 KiB without a `vulkan_target`.
- The verification step asserts an import and a smoke-test number only — never a version value (`evox_etl.__version__` still reports the 1.x version).
- Cross-links use the agreed paths `../../tutorial/tutorial_part2`, `../advanced/backends`, and `../developer/environment`.
- This is a leaf node: no child subdirectories, all content lives in the three `.md` files above.
- These pages are not manually translated; Chinese comes from the gettext catalog under `docs/source/locale/zh_CN/`, which is outside this node.

## Routing Table

| Page | Purpose |
|---|---|
| `index.md` | Section landing page and toctree |
| `python.md` | Interpreter and virtual-environment setup |
| `evox.md` | Installing, configuring and verifying EvoX 2.0 |
