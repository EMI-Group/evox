# install/ — Installation Guide

## Intent
This directory is the **Installation Guide** section of the EvoX documentation — the first step of the "Getting Started" journey.
It documents installing the **experimental EvoX 2.0 pre-release** only: the `evox` distribution whose documented import name is `evox_etl` and which runs on the `evoxc` compiler.
The 1.x torch API (`import evox`, `ModuleBase`, `Parameter`/`Mutable`, `torch.compile`/`vmap`) is deliberately absent from these pages and must not be reintroduced.

## API Surface

| File | Purpose |
|---|---|
| `index.md` | Section landing page: `:maxdepth: 1` toctree over `python` and `evox`, a one-paragraph intro, and a `{warning}` marking the 2.0 pre-release scope. |
| `python.md` | "Python Installation Guide" — Python ≥3.10 (3.11+ recommended), interpreter install per platform (Windows / macOS / Linux / `uv`), why an isolated environment matters, and `venv` + `pip` / `uv` workflows. |
| `evox.md` | "EvoX Installation Guide" — `pip install "evox"` and the extras table (`default`, `vis`, `iree`, `tvm`, `compilers`), XLA's user-supplied PJRT plugin via `EVOXC_PJRT_PLUGIN`, `backend=`/`device=` selection, a PSO-on-Sphere smoke test, dev-checkout install, and next steps. |

## Constraints
- Every code snippet must be verified against `src/evox_etl` (and `pyproject.toml` for packaging claims) before it is documented; the six-line smoke test runs 50 PSO generations on `Sphere` on the default `numpy` backend.
- Packaging facts these pages must match: `pyproject.toml` declares `dependencies = ["torch >= 2.6.0", "numpy >= 2.0.0"]` (torch is a hard dependency) and does **not** declare `evoxc`; its extras are exactly `vis`, `default` (identical to `vis`: plotly+pandas), `neuroevolution`, `test`, `docs`, `triton`.
- The `iree`, `tvm` and `compilers` extras belong to the `evoxc` distribution (`evoxc[iree]`/`evoxc[tvm]`/`evoxc[compilers]`), not to `evox`; XLA has no extra anywhere and needs a user-supplied PJRT plugin.
- Python floors: `evox` is `requires-python >= 3.10`, `evoxc` is `>= 3.11`, and 2.0 code imports `evoxc` unconditionally — so 3.11+ is the practical floor for the documented program.
- The only tagged release is `v1.4.0`, whose tree has no `src/evox_etl` (the package landed after the tag), so `pip install evox` from PyPI does not provide `import evox_etl`; only a source checkout does.
- The XLA backend has no pip extra: it needs a user-provided PJRT plugin selected by `EVOXC_PJRT_PLUGIN` (or the `plugin_path` compile option).
- Cross-links use the agreed paths `../../tutorial/tutorial_part2`, `../advanced/backends`, and `../developer/environment`.
- This is a leaf node: no child subdirectories, all content lives in the three `.md` files above.
- These pages are not manually translated; Chinese comes from the gettext catalog under `docs/source/locale/zh_CN/`, which is outside this node.

## Routing Table

| Page | Purpose |
|---|---|
| `index.md` | Section landing page and toctree |
| `python.md` | Interpreter and virtual-environment setup |
| `evox.md` | Installing, configuring and verifying EvoX 2.0 |
