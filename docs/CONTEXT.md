# docs/ — EvoX Documentation

## Intent
This directory contains the **Sphinx-based documentation system** for EvoX, published on Read the Docs. It covers tutorials, installation guides, developer guides, API references (auto-generated from source code), Jupyter notebook examples, and miscellaneous reference material. The documentation is bilingual (English and Simplified Chinese).

## API Surface
The documentation build system exposes these entry points:

| File / Directory | Purpose |
|---|---|
| `Makefile` | Standard Sphinx Makefile; routes all targets (`html`, `clean`, etc.) to `sphinx-build` |
| `make.bat` | Windows batch equivalent of the Makefile |
| `source/conf.py` | Sphinx configuration — extensions, theme, autodoc setup, i18n, nav links |
| `source/index.md` | Root documentation landing page (MyST Markdown) |
| `fix_output.py` | Post-processing script that removes "Duplicate implicit target name" entries from the zh_CN `.po` translation file |
| `machine_translate.py` | Machine-translates `.po` entries from English → Chinese using a TGI-compatible LLM API (GPT-4o); skips code references, tutorials (already manually translated), SVGs, and fuzzy entries |
| `fix_notebook_translation.py` | Wraps translated notebook strings as JSON cell structures in the `.po` file to satisfy the Sphinx notebook builder |

### Build System
- **Builder**: Sphinx via `sphinx-build`, source directory is `source/`, output goes to `build/`
- **Theme**: `shibuya` (dark/light modes, GitHub integration, social links)
- **Key extensions**:
  - `autodoc2` — Auto-generates API docs from `src/evox` (rendered via MyST)
  - `myst_nb` — Executable MyST Markdown notebooks (execution mode: off)
  - `sphinxcontrib.mermaid` — Mermaid diagram support
  - `sphinx_copybutton` — Copy button on code blocks
  - `sphinx_design` — UI components (cards, grids)
  - `sphinx.ext.mathjax`, `sphinx.ext.napoleon`, `sphinx.ext.viewcode`
- **Mock imports**: `brax`, `mujoco_playground`, `torchvision` (allow building without them)
- **Autodoc packages**: `evox.core.*`, `evox.problems.*`, `evox.workflows.*` expose `__all__`

### i18n / Localization
- **Locale directory**: `source/locale/zh_CN/LC_MESSAGES/docs.po` (gettext translation catalog)
- Translated tutorial pages live under `source/locale/zh_CN/tutorial/`
- `conf.py` uses `READTHEDOCS_LANGUAGE` env var to switch between English and Chinese for nav labels and theme context

## Constraints
- `apidocs/` is **auto-generated** by autodoc2 at build time — it is NOT committed to git and should not be edited manually
- The `.po` file at `source/locale/zh_CN/LC_MESSAGES/docs.po` requires post-processing after machine translation: run `fix_output.py` to remove duplicate-anchor entries, and `fix_notebook_translation.py` to reconcile notebook vs. regular page translations
- Notebook execution is disabled at build time (`nb_execution_mode = "off"`, `nbsphinx_execute = "never"`) — notebooks must be pre-executed before committing
- The source tree root points to `../../src` (the `src/` directory at the repo root) for autodoc imports

## Routing Table

| Area | Child Directory | Description |
|---|---|---|
| Sphinx config & build | `source/conf.py`, `Makefile`, `make.bat` | Build system configuration and scripts |
| Static assets | `source/_static/` | Images, CSS, logos, favicons |
| Jinja2 templates | `source/_templates/` | Autosummary templates (class/module) and HTML partials (social links, webfonts) |
| Tutorials | `source/tutorial/` | 7-part English getting-started tutorial series |
| Chinese translations | `source/locale/zh_CN/` | Translated tutorial pages and gettext `.po` catalog |
| Installation guide | `source/guide/install/` | Python setup, PyTorch + EvoX installation |
| Developer guide | `source/guide/developer/` | Custom algorithms, HPO problems, module base design, environment setup, doc contribution |
| Experimental features | `source/guide/experimental/` | Distributed workflow documentation |
| Notebook examples | `source/examples/` | Jupyter notebooks: single/multi-objective optimization, neuroevolution (Brax), HPO, supervised learning |
| Miscellaneous | `source/miscellaneous/` | GPU selection, Linux distribution notes, MATLAB migration |
| Translation tooling | `machine_translate.py`, `fix_output.py`, `fix_notebook_translation.py` | Scripts for maintaining the zh_CN `.po` catalog |
