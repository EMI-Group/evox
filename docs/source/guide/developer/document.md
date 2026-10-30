# Document Writing Guide

This page covers how the EvoX 2.0 documentation is organised, how to write and cross-reference a
page, which docstring conventions the API reference expects, and how the bilingual (English /
Simplified Chinese) build is kept in sync.

```{note}
These pages describe the experimental EvoX 2.0 pre-release, whose import name is `evox_etl`.
Task-focused guides live under `docs/source/guide/`; the tutorials under `docs/source/tutorial/`
are written for end users and follow exactly the same conventions.
```

## Where the documentation lives

- The Sphinx project is `docs/`: `docs/Makefile` and `docs/source/conf.py` drive the build, the
  theme is `shibuya`, and all content is written in **MyST Markdown** (`.md`) and rendered by
  `sphinx-build`.
- English sources are the pages under `docs/source/`. The section landings — `docs/source/index.md`,
  `docs/source/guide/*/index.md`, `docs/source/tutorial/index.md` — own the `{toctree}`
  directives that order and include the child pages.
- The API reference is generated from `src/evox_etl` by the `autodoc2` extension at build time.
  `docs/source/apidocs/` is therefore **generated, git-ignored, and never edited by hand**.
  `autodoc2_module_all_regexes` restricts `__all__`-based filtering to `evox_etl.core.*`,
  `evox_etl.problems.*`, `evox_etl.utils.*` and `evox_etl.workflows.*`; every other module is
  documented by its public members.

## Writing a page

- Exactly **one H1 per page** (`# Title`), with sections at `##` and `###`.
- Every new page must be listed in a parent `{toctree}` (explicit or `:hidden:`). An unlisted page
  still builds, but emits a "not included in any toctree" warning.
- Cross-reference another page with the `doc` role, writing its path without the `.md` suffix:

  ````markdown
  {doc}`../advanced/backends`
  ````

  Generated API anchors use the label syntax instead, for example
  `[StdWorkflow](#evox_etl.core.workflow.StdWorkflow)`.
- Use `{note}`, `{tip}` and `{warning}` admonitions for asides, and fenced code blocks tagged
  `python` or `bash` (or `{code-block} python`) for code.
- No page may teach the removed 1.x object-oriented API; EvoX 2.0 uses configs, tensor states and
  plain functions.
- Run every snippet you publish — the NumPy backend needs no toolchain, so there is no excuse for
  an unverified example.
- Keep count claims byte-identical to the source pages: "30+ algorithms" and "30+ benchmark
  problems" for EvoX itself; "140+ algorithms" and "240+ benchmark problems/environments" on
  ecosystem pages.

For a page example image use real assets only, for instance:

```markdown
![EvoX logo](/_static/evox_logo_dark.svg)
```

## Docstrings

Docstrings feed the generated API reference, so they follow the house conventions:

- Document every public module, class and function; keep docstrings to 1-3 lines.
- Use **Sphinx style**, and do **not** repeat parameter types in the docstring — they belong in
  the signature's type hints.
- Use `:param <name>:`, `:return:` and `:raises <exception>:` fields.

```python
def add(a: int, b: int) -> int:
    """
    Add two integers.

    :param a: The first integer.
    :param b: The second integer.
    :return: The sum of the two integers.
    :raises ValueError: If either input is not an integer.
    """
    if not isinstance(a, int) or not isinstance(b, int):
        raise ValueError("Inputs must be integers.")
    return a + b
```

## Notebooks

```{warning}
`nb_execution_mode = "off"` and `nbsphinx_execute = "never"`: notebooks are **not executed at
build time**. If you add or change a `.ipynb`, execute all cells locally and commit the saved
outputs, otherwise the published page ships without results.
```

Markdown is preferred for narrative pages; reach for a notebook only when the content is an
interactive tutorial that benefits from stored cell output.

## Local build

```bash
pip install sphinx shibuya sphinx-autodoc2 myst-parser myst-nb sphinx-copybutton \
  sphinx-design sphinx-favicon sphinxcontrib-mermaid linkify-it-py
rm -rf docs/source/apidocs
PYTHONPATH=src python -m sphinx -b html docs/source docs/build/html
```

Treat new warnings as errors in the page you touched. A broken cross-reference into a page
another author is still writing is not yours to fix — note it and move on.

## Bilingual documentation

Two channels keep English and Simplified Chinese in step:

- English pages live in `docs/source/`; Chinese translations live under
  `docs/source/locale/zh_CN/`.
- Most pages are translated through a gettext catalog,
  `docs/source/locale/zh_CN/LC_MESSAGES/docs.po` (configured by `locale_dirs = ["locale/"]` and
  `gettext_compact = "docs"`). Every `msgid` must stay byte-identical to the English source
  sentence; only the `msgstr` is translated.
- The tutorial series is translated manually as whole pages under
  `docs/source/locale/zh_CN/tutorial/`. On Read the Docs, the `.readthedocs.yaml` pre-build job
  copies them over `docs/source/tutorial/` whenever `READTHEDOCS_LANGUAGE` is not `en`.
- `conf.py` reads `READTHEDOCS_LANGUAGE` to switch navigation labels and theme text, so a local
  Chinese preview needs `READTHEDOCS_LANGUAGE=zh-cn` (see {doc}`environment`).

## Updating translations

`sphinx-intl` is not part of the toolchain installed above, so install it once:

```bash
pip install sphinx-intl
```

```bash
cd docs
make gettext
sphinx-intl update -p build/gettext -l zh_CN
python fix_output.py
```

- `fix_output.py` opens the catalog by a relative path
  (`source/locale/zh_CN/LC_MESSAGES/docs.po`), so it must run **from `docs/`** — the sequence
  above stays in that directory throughout.
- Then translate the new or changed `msgstr` entries by hand in
  `docs/source/locale/zh_CN/LC_MESSAGES/docs.po`.
- `docs/fix_output.py` strips the duplicate-implicit-target entries that Sphinx would otherwise
  reject; `docs/fix_notebook_translation.py` reconciles notebook cell strings with the catalog;
  `docs/machine_translate.py` is an optional LLM-assisted helper (it skips code references,
  tutorials, SVGs and fuzzy entries) whose output still needs review.
- Stale catalog entries whose English source no longer exists are harmless — they are simply
  never used, and can be pruned when convenient.
- Keep the Chinese and English count claims numerically identical.

```{tip}
Run `make gettext` after any change to an English sentence, not only when adding a page; a changed
`msgid` silently orphans the existing translation until `sphinx-intl update` re-creates it.
```

## Next

- {doc}`environment` — set up the toolchain and run the build commands above.
- {doc}`../advanced/extensions` — expose an extension's modules through the API reference.
