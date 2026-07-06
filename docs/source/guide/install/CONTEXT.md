# install/ — Installation Guide

## Intent
This directory contains the **Installation Guide** section of the EvoX documentation — the first step in the "Getting Started" journey. It's linked from the root `source/index.md` as a grid-item-card and toctree entry. The audience spans from Python newcomers to experienced users setting up GPU-accelerated environments.

## API Surface

| File | Purpose |
|---|---|
| `index.md` | Section landing page. Renders a toctree (`:maxdepth: 1`) linking to `python` and `torch_and_evox`. No other content. |
| `python.md` | **"Python Installation Guide"** — aimed at Python newcomers. Covers installing the Python interpreter (Windows, Linux, via `uv`), managing virtual environments with `pip`+`venv` and `uv`, and basic package management commands. |
| `torch_and_evox.md` | **"EvoX Installation Guide"** — covers installing EvoX from PyPI (`pip install "evox[default]"` with optional extras: `vis`, `neuroevolution`, `test`, `docs`, `default`), PyTorch accelerator support (CPU / Nvidia CUDA / AMD ROCm), Windows-specific GPU setup (one-click batch script, manual install, WSL 2), AMD GPU via Docker (`rocm/pytorch`), and a verification step. Includes a Mermaid diagram of the `evox → torch → GPU/CPU` dependency chain. |

## Constraints

- This is a **leaf node** — no child subdirectories. All pages are single `.md` files.
- Both pages use MyST Markdown with Sphinx directives: `{tip}`, `{note}`, `{warning}`, `{seealso}`, `{tab-set}`/`{tab-item}`, `{mermaid}`.
- The `{mermaid}` diagram in `torch_and_evox.md` requires the `sphinxcontrib.mermaid` extension (configured in `conf.py`).
- The one-click Windows script referenced (`/_static/win-install.bat`) lives in `docs/source/_static/` — this page is a consumer, not the owner.
- `python.md` references specific `uv` version URLs (`0.6.16`) — these may need updates over time.
- These pages are **not** manually translated; they rely on the gettext `.po` catalog under `source/locale/zh_CN/` for Chinese localization (see `docs/CONTEXT.md` for translation workflow).

## Routing Table

_No child directories. This is a leaf node — all content lives in the three `.md` files above._
