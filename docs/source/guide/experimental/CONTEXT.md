# experimental/ — Experimental Features Guide

## Intent
This directory hosts the **Experimental Features** section of the EvoX Getting Started guides. It documents features that are not yet considered stable or fully supported, currently limited to distributed/multi-GPU workflow support.

## API Surface

| File | Purpose |
|---|---|
| `index.md` | Section landing page titled "Experimental Features" with a `{toctree}` (`:maxdepth: 1`) linking to `distributed_workflow` |
| `distributed_workflow.md` | "Multi-GPU and Distributed Workflow" — explains how to use EvoX's experimental distributed support: manual seed fixing (`torch.manual_seed`, `np.random.seed`, `torch.use_deterministic_algorithms`) and launching via `torchrun` with `--standalone --nnodes=1 --nproc-per-node=$NUM_GPUS` |

## Constraints
- No child subdirectories — all content is in the two markdown files directly under this node
- Linked from the root `source/index.md` toctree only (not as a grid-item-card — only Tutorial, Installation Guide, and Developer Guide have cards on the landing page)
- Uses Sphinx/MyST markup (`{toctree}`, `{important}`, `{tip}` admonitions)
- This section is intentionally minimal; new experimental features should be added as new `.md` files and listed in the `index.md` toctree

## Routing Table

This directory is a leaf node — there are no child subdirectories.
