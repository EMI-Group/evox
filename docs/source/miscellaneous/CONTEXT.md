# docs/source/miscellaneous/ — Miscellaneous Reference Guides

## Intent
This directory holds **miscellaneous reference material** for EvoX users — topics that don't fit neatly into tutorials, installation, or developer guides. It covers GPU setup, Linux OS guidance, and a detailed MATLAB-to-EvoX migration guide.

## API Surface

| File | Purpose |
|---|---|
| `index.md` | MyST toctree hub linking to all four reference pages |
| `selecting_gpu.md` | Using `CUDA_VISIBLE_DEVICES` to target a specific GPU, multiple GPUs, or fall back to CPU |
| `other_gpus.md` | Non-NVIDIA GPU support: AMD (ROCm, treated as `cuda` device) and Apple Silicon (MPS, no `jit` support) |
| `linux_distribution.md` | Advice on picking a recent Linux distro for GPU servers and installing distribution-packaged GPU drivers (not NVIDIA `.run` installers) |
| `transfer-from-matlab.md` | Comprehensive MATLAB → PyTorch/EvoX migration guide: syntax comparison (indexing, matrix ops, control flow, functions), a full PSO algorithm ported from MATLAB to EvoX, and an explanation of EvoX's `Algorithm`/`StdWorkflow` design |

## Constraints
- All pages are plain MyST Markdown; no subdirectories exist
- The `index.md` toctree is the canonical navigation hub — new pages must be added there
- Pages assume the reader is already familiar with the core documentation (tutorials, install guide)

## Routing Table
No child directories. All content is in this flat directory.
