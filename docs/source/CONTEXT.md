# docs/source/ — Sphinx Documentation Source Root

## Intent
This directory is the **Sphinx documentation source root** for the EvoX project. It contains all source materials — configuration, content pages, templates, static assets, and i18n resources — that `sphinx-build` compiles into the published HTML documentation on Read the Docs. The documentation is **bilingual** (English and Simplified Chinese), with English as the default language and Chinese served at `/zh-cn/`.

## Core Files

| File | Purpose |
|---|---|
| `conf.py` | **Sphinx configuration hub**. Defines extensions, theme (`shibuya`), autodoc2 package scanning, i18n settings, mock imports, and nav bar links. All build-time behavior is configured here. |
| `index.md` | **Root landing page**. MyST Markdown with two hidden `{toctree}` directives (Getting Started + Additional Resources) and two `{eval-rst}` grid-card blocks for visual navigation. Also contains bilingual language-switching links. |

## Key Architectural Decisions

### Auto-generated API Docs
- `apidocs/` is **generated at build time** by the `autodoc2` extension — it is NOT committed to git
- `autodoc2_packages` points to `../../src/evox` (the repo's `src/` directory)
- `autodoc2_module_all_regexes` restricts `__all__`-based documentation to `evox.core.*`, `evox.problems.*`, and `evox.workflows.*`
- API docs are rendered via MyST (`autodoc2_render_plugin = "myst"`)

### Bilingual i18n Architecture
- Language switching driven by `READTHEDOCS_LANGUAGE` environment variable (set by Read the Docs)
- Nav labels and theme text conditionally rendered in English or Chinese in `conf.py`
- **Dual translation strategy** (see `locale/` routing): tutorials are full-page manual translations; everything else uses a gettext `.po` catalog
- `.readthedocs.yaml` pre-build job copies translated tutorial `.md` files into place before Sphinx runs

### Build System
- **Builder**: Sphinx via `sphinx-build` (invoked from parent `docs/Makefile`)
- **Source dir**: `source/`; **Output dir**: `build/` (in parent `docs/`)
- **Extensions**: `autodoc2`, `myst_nb`, `sphinx_design`, `sphinx_copybutton`, `sphinxcontrib.mermaid`, `sphinx_favicon`, `sphinx.ext.mathjax`, `sphinx.ext.napoleon`, `sphinx.ext.viewcode`
- **Theme**: `shibuya` with dark/light mode, GitHub integration, Discord link, and ecosystem nav children (EvoMO, EvoRL, EvoGP, TensorNEAT, TensorRVEA, TensorACO, EvoXBench)
- **Mock imports**: `brax`, `mujoco_playground`, `torchvision` — allows building docs without these optional dependencies
- **Notebook execution**: Disabled at build time (`nb_execution_mode = "off"`, `nbsphinx_execute = "never"`) — all notebooks must be pre-executed before committing

## Constraints
- `apidocs/` is auto-generated — do NOT create or edit files there; they are ephemeral
- Notebooks (`.ipynb`) must be **pre-executed** with outputs saved before committing, since build-time execution is disabled
- The `exclude_patterns = ["locale/**"]` in `conf.py` prevents the locale directory from being scanned as documentation source; the locale must still be present for gettext lookups at build time
- All content pages use **MyST Markdown** (`.md` with `{toctree}`, `{note}`, `{eval-rst}`, etc.), not reStructuredText (except for Jinja2 templates in `_templates/`)
- Source tree root (`sys.path`) points to `../../src` for autodoc imports — the `src/` directory at the repo root must contain `evox/` at build time

## Routing Table

| Area | Child | Description |
|---|---|---|
| Sphinx configuration | `conf.py` | Extension loading, theme, i18n, autodoc2 setup, nav links |
| Root landing page | `index.md` | Toctrees and grid-card navigation |
| Static assets | `_static/` | Images, CSS (`css/custom.css`), logos (light/dark/brand), favicons, SVGs, GIFs, AVIFs, AVIF images, Windows install batch script |
| Jinja2 templates | `_templates/` | Autosummary templates (`custom_class.rst`, `custom_module.rst`) and HTML partials (`nav-socials.html`, `foot-socials.html`, `webfonts.html`) |
| Tutorials | `tutorial/` | 7-part getting-started tutorial series (English, MyST Markdown) |
| Guides | `guide/` | Installation guide, developer guide, and experimental features |
| Example notebooks | `examples/` | 6 pre-executed Jupyter notebooks covering single/multi-objective optimization, neuroevolution, HPO, and custom algorithms |
| Miscellaneous | `miscellaneous/` | GPU selection, Linux distribution notes, MATLAB migration guide |
| i18n / Chinese translations | `locale/` | `zh_CN/` gettext `.po` catalog + manually translated tutorial pages |
