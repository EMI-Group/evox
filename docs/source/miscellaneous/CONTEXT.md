# docs/source/miscellaneous/ — Miscellaneous Reference Pages

## Intent
This directory houses **miscellaneous reference articles** that don't fit neatly into the main tutorial or developer guide categories. These pages cover environment setup advice (GPU selection, OS choice, driver installation) and a transitional guide for MATLAB users moving to PyTorch and EvoX.

## API Surface

| File | Purpose |
|---|---|
| `index.md` | MyST toctree listing all four child pages; rendered as the "Miscellaneous" landing page |
| `selecting_gpu.md` | How to select specific GPUs (or fallback to CPU) via `CUDA_VISIBLE_DEVICES` |
| `other_gpus.md` | Guide for using AMD GPUs (ROCm) and Apple Silicon GPUs (MPS) with PyTorch in EvoX; notes that MPS does **not** support `#evox.compile` |
| `linux_distribution.md` | Advice on picking a modern Linux distribution and GPU driver for GPU servers; recommends distro-packaged drivers over NVIDIA's direct installer |
| `transfer-from-matlab.md` | Comprehensive MATLAB → PyTorch/EvoX transition guide: side-by-side syntax comparison (1-based vs 0-based indexing, `*` vs `@`, broadcasting behavior) and a full PSO algorithm walkthrough rewritten from MATLAB struct/function style to an EvoX `Algorithm` subclass with a `StdWorkflow` |

## Constraints
- All pages use MyST Markdown (the same as the rest of the documentation).
- The `transfer-from-matlab.md` page contains inline code blocks annotated as `matlab` and `python`; the `other_gpus.md` page contains an `{note}` admonition.
- These pages are linked from `docs/source/index.md` under the "Additional Resources" toctree caption.

## Routing Table

This is a leaf directory — all content is in the four `.md` files listed above. There are no child subdirectories.
