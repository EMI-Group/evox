# docs/source/guide/install/ — Installation Guide

## Intent
This directory contains the **installation guide** for EvoX — a two-page walkthrough that helps users set up their Python environment and then install PyTorch + EvoX with appropriate accelerator support. It targets newcomers to Python as well as experienced developers who need GPU-accelerated setups.

## API Surface

Three files, linked via the toctree in `index.md`:

| File | Purpose |
|---|---|
| `index.md` | Landing page with `{toctree}` linking to both sub-pages (`:maxdepth: 1`) |
| `python.md` | Python installation & environment management guide |
| `torch_and_evox.md` | PyTorch + EvoX installation and accelerator configuration |

### TOC Structure
```
Installation Guide (index.md)
├── python.md          — "Python Installation Guide"
└── torch_and_evox.md  — "EvoX Installation Guide"
```
This toctree is pulled into `source/index.md` under the "Getting Started" section as `guide/install/index`.

### `python.md` Coverage
- **Python Installation**: Windows (python.org download + PATH), Linux (distro package managers), and cross-platform via `uv` (tabbed instructions for Windows/Linux/macOS)
- **Environment Management**: Two approaches — `pip` + `venv` (create, activate, deactivate, install, list, uninstall, upgrade) and `uv venv` + `uv pip`
- Recommends Python 3.10+ and `uv` as the preferred toolchain

### `torch_and_evox.md` Coverage
- **EvoX Installation**: `pip install "evox[default]"` from PyPI, with optional extras (`vis`, `neuroevolution`, `test`, `docs`)
- **Accelerator Architecture**: Mermaid diagram showing `evox → torch → {NVIDIA GPU, AMD GPU, CPU}`
- **NVIDIA GPU on Windows**: WSL 2 path vs. native Windows path; provides a one-click batch script (`win-install.bat`) that installs VSCode, Git, MiniForge3, and PyTorch
- **AMD GPU (ROCm)**: Docker-based workflow using `rocm/pytorch` image
- **Verification**: Python snippet using `torch.utils.collect_env.get_pretty_env_info()` and `import evox`

## Constraints
- All content is **hand-written MyST Markdown** (`.md`)
- Uses MyST directives: `{toctree}`, `{note}`, `{tip}`, `{warning}`, `{seealso}`, `{tab-set}`/`{tab-item}`, and `{mermaid}` for the architecture diagram
- The `win-install.bat` script referenced in `torch_and_evox.md` lives at `source/_static/win-install.bat`
- Internal cross-references to other doc sections use standard MyST `[label](#ref)` syntax
- No translations present — Chinese translations of these guides (if any) would live in `source/locale/zh_CN/`
