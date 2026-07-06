# docs/source/guide/experimental/ — Experimental Features Guide

## Intent
This directory documents **experimental / unstable features** of EvoX. Content here is considered provisional — APIs, behaviors, and even the documented features themselves may change or be removed in future releases without the usual deprecation cycle. Currently the sole topic is the multi-GPU / distributed workflow.

## API Surface

| File | Purpose |
|---|---|
| `index.md` | TOC landing page. A single `{toctree}` directive with `:maxdepth: 1` that links to `distributed_workflow`. |
| `distributed_workflow.md` | **Multi-GPU and Distributed Workflow** guide. Covers: manually fixing seeds (`torch.manual_seed`, `np.random.seed`, `torch.use_deterministic_algorithms`) before any operations, and launching distributed scripts via `torchrun` (`--standalone --nnodes=1 --nproc-per-node=$NUM_GPUS`). Includes admonitions about ordering (set seeds *before* any torch/numpy operations) and a tip linking to the PyTorch `torchrun` docs. |

### Parent TOC Integration
The experimental section is wired into the documentation at two levels:
- **Root landing page** (`source/index.md`): The toctree under "Getting Started" includes `guide/experimental/index` directly.
- **Guide section** (`source/guide/CONTEXT.md` routing table): Lists this directory as the experimental features area.

## Constraints
- All pages are **hand-written MyST Markdown** (`.md`). No auto-generated or Jupyter notebook content remains (historical notebooks were removed).
- Content is **English only**; translations live under `source/locale/zh_CN/`.
- The experimental nature means pages should clearly warn users that features are **not stable** and may change.
- New experimental pages must be added to the `index.md` toctree to appear in the documentation navigation.

## Routing Table

No child subdirectories — this is a leaf node. All content lives in the two `.md` files listed above.

## History
- Originally contained `multidevice_algorithm.ipynb` (multi-device algorithm notebook, added ~Nov 2024) and `evoxvision.ipynb` (evoxvision notebook). Both were removed in commit `12fcd95b` ("Remove old docs and examples", Jan 2025).
- Current `distributed_workflow.md` was added in commit `510452f7` ("Add experimental document about distributed workflow") and later refined in `ea64a230` ("Include doc about distributed workflow in the experimental feature section").
