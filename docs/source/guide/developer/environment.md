# Develop environment

This page walks through a development checkout of the experimental EvoX 2.0 pre-release: cloning
the repository, installing it editable, running the lint hooks and the test suite, and building
these documentation pages. The import name in 2.0 is `evox`.

## Clone and install

```bash
git clone https://github.com/EMI-Group/evox.git
cd evox
python -m venv .venv           # Python 3.11+
source .venv/bin/activate
pip install -e .               # `evox` plus its declared dependencies (`evoxc`, `numpy`)
```

- **Python 3.11 or newer** is required. `import evox` imports the
  [EvoX Compiler](https://github.com/EMI-Group/evoxc) (`evoxc`) unconditionally, and both `evoxc` and
  the `evox` manifest (2.0.0a1) declare `requires-python = ">=3.11"`.
- One `pip` command installs both halves: `pyproject.toml` declares
  `dependencies = ["evoxc >= 0.1.0", "numpy >= 2.0.0"]`, so the editable install brings the compiler
  and NumPy with it and installs no PyTorch. If `import evox` fails with
  `ModuleNotFoundError: No module named 'evoxc'`, the environment holds the 1.x `evox` (no `evoxc`
  dependency and no `evox`) rather than this checkout.
- If you are developing against a compiler checkout instead of the published distribution, install
  that path over the one `pip` resolved:

```bash
pip install /path/to/evoxc
```

The default **NumPy backend** then needs no compiler toolchain and no GPU driver: the editable install
alone is enough for everything on this page.

```{tip}
Editable mode (`-e`) keeps `evox` importable straight from `src/`, so edits take effect
without reinstalling. Add what the test suite needs with `pip install pytest numpy`.
```

Visualization tests import Plotly (`pip install plotly`, or `pip install -e ".[vis]"`). The compiled
backends are extras of the compiler, re-exported by the distribution: `evox[iree]`, `evox[tvm]` and
`evox[compilers]` are thin aliases of `evoxc[iree]`, `evoxc[tvm]` and `evoxc[compilers]`, so
`pip install -e ".[iree]"` really does pull the IREE toolchain.
{doc}`../install/evox` lists every extra and what it pulls in.

## Nix dev shell

If you prefer Nix over a hand-managed virtual environment, the repository flake provides a
ready-made shell:

```bash
nix develop .        # or: nix develop .#cuda / .#rocm / .#vulkan
```

- The flake declares `devShells.default` (an alias of `cpu`) plus `cpu`, `cuda`, `rocm` and
  `vulkan`.
- The shell brings its own Python environment, including a `.venv` directory, so you can skip the
  manual `python -m venv` step above.

```{note}
The flake is x86_64-linux only — that is the single system it declares.
```

## Style and pre-commit

Ruff is both the linter and the formatter; its configuration lives under `[tool.ruff]` in
`pyproject.toml` (line length 128, LF line endings, `F`/`E`/`W`/`I`/`NPY` rules). The remaining
whitespace rules — no trailing whitespace, a final newline — are enforced by pre-commit.

Install the hooks once, then run them before pushing:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

`.pre-commit-config.yaml` wires up trailing-whitespace, end-of-file-fixer, check-yaml,
check-added-large-files, `ruff --fix` and `ruff-format`.

## Run the test suite

The canonical suite is **`unit_test/evox/`**, a pytest suite that mirrors `src/evox/`.
`evox` is a regular installed package (`src/evox/__init__.py`); only the extension
namespace `evox_ext` is a PEP 420 namespace package. Either install the
project editable (see above) or prepend `src` to the import path; the repo-root `conftest.py`
already shims `sys.path` with both the repository root and `src`.

Per-suite commands:

```bash
pytest unit_test/evox/algorithms -q
pytest unit_test/evox/operators -q
pytest unit_test/evox/problems -q
pytest unit_test/evox/metrics -q
pytest unit_test/evox/vis_tools -q
pytest unit_test/evox/workflows -q
pytest unit_test/evox/ext -q
```

Whole suite:

```bash
pytest unit_test/evox -q
```

```{note}
Whole-suite runs share one collection; separate suite runs stay useful for per-suite numbers and
for isolating a failure faster.
```

- The suite needs **no PyTorch**: the old torch-parity runs were removed with the 1.x package, and
  `unit_test/evox/test_no_torch.py` is a release gate that imports `evox` in a fresh interpreter
  and asserts no torch module was loaded.
- The `vis_tools` and monitor-plot assertions need Plotly and are `skipif`-guarded otherwise.
- Tests use the `numpy` backend only, so no GPU is needed. Compiled backends and GPU execution
  are exercised by the comparison harness under `benchmarks/evox_vs_torch/`.

To narrow a failing run, point pytest at a single file, and append `::test_name` to run a single
test:

```bash
pytest unit_test/evox/workflows/test_aux_history.py -q
```

## Build the documentation

```bash
pip install sphinx shibuya sphinx-autodoc2 myst-parser myst-nb sphinx-copybutton \
  sphinx-design sphinx-favicon sphinxcontrib-mermaid linkify-it-py
rm -rf docs/source/apidocs
PYTHONPATH=src python -m sphinx -b html docs/source docs/build/html
```

- `docs/source/apidocs/` is generated by the `autodoc2` extension at build time and is
  git-ignored: delete it before a clean build, and never edit it by hand.
- On Windows use `docs\make.bat html`, or run the same `sphinx` command from a shell; from
  `docs/` the shortcut is `make html`. The output always lands in `docs/build/html`.
- Set `READTHEDOCS_LANGUAGE=zh-cn` to switch the navigation labels and theme text to Chinese;
  the tutorial translations are copied into place by the Read the Docs pre-build job, so a
  full Chinese preview also needs that copy step (described in {doc}`document`).

```{warning}
A stale `docs/source/apidocs/` tree produces confusing build warnings and missing members. If the
API reference looks wrong, remove the directory and rebuild before debugging anything else.
```

## Next

- {doc}`document` — documentation conventions, autodoc rules and the translation workflow.
- {doc}`architecture` — how the functional 2.0 framework fits together.
- {doc}`../install/evox` — install options, extras and backend selection.
