# docs/source/guide/ — EvoX User & Developer Guides

## Intent
This directory contains the **guide section** of the EvoX documentation — a collection of practical how-to pages organized by audience. The guides complement the tutorial series by offering focused, topic-oriented instructions for installation, development, and experimental features. All content is in English (translations live under `source/locale/`).

## API Surface
The guide section is integrated into the documentation via the top-level `source/index.md` toctree under the "Getting Started" caption. There is **no central index page** for guides — each child directory has its own `index.md` that serves as the landing page for that topic. The three guide areas are linked directly from the root landing page's toctree and card grid.

### TOC Integration (`source/index.md`)
```toctree
guide/install/index
guide/developer/index
guide/experimental/index
```

Each child index page uses `{toctree}` directives with `:maxdepth: 1` to link its own sub-pages.

## Constraints
- Guides use **MyST Markdown** (`.md`) and **Jupyter notebooks** (`.ipynb`) — the same formats as the rest of the documentation
- Notebooks must be **pre-executed and saved with output** before committing (CI has no GPU support)
- Internal cross-references use `[label](#ref)` syntax, images use `![Alt](path)` relative to `source/`
- No auto-generated content — all guide pages are hand-written

## Routing Table

| Area | Child Directory | Description |
|---|---|---|
| Installation Guide | `install/` | Python setup, PyTorch installation (with CUDA/ROCm accelerator support), EvoX installation via pip |
| Developer Guide | `developer/` | Development environment, ModuleBase design, custom algorithm & problem implementation, HPO deployment, documentation writing |
| Experimental Features | `experimental/` | Experimental features — primarily multi-GPU/distributed workflow support via `torchrun` |
