# .github — CI/CD for the EvoX 2.0 experimental pre-release

## Intent
Gate and ship the `evox` distribution for the 2.0 line, which is the functional `evox_etl` port on the EvoX Compiler (`evoxc`).
Nothing here installs or imports torch and nothing needs a GPU: the whole tree runs on the default `numpy` backend.
The single test suite is `unit_test/etl/` (pytest); the only supported interpreters are >= 3.11 because `evoxc` requires it.
CI proves three things: the built wheel works (installed, imported from outside the checkout), the user-facing examples run, and the Sphinx docs still build.
Publishing goes to PyPI only through OIDC trusted publishing — there is no API token secret anywhere.

## Workflow surface

| Workflow (name) | File | Trigger | Jobs | Proves |
|---|---|---|---|---|
| `Python package` | `./workflows/python-package.yml` | `push` / `pull_request` on `main`, `workflow_dispatch` | `build` (matrix 3.11/3.12/3.13), `examples`, `docs` | the wheel builds and the suite passes on every supported interpreter; the demo scripts and the docs build |
| `Upload Python Package` | `./workflows/python-publish.yml` | GitHub `release: types: [published]`, `workflow_dispatch` | `deploy` | the released artifact is built, smoke-tested, then uploaded to PyPI via OIDC |
| `Ruff` | `./workflows/ruff-check.yml` | `push` / `pull_request` on `main`, `workflow_dispatch` | `ruff` | `ruff check` passes with the `[tool.ruff]` config from `pyproject.toml` (line-length 128, `F/E/W/I/NPY`, per-file-ignores for `src/evox_etl/__init__.py` and `src/evox_etl_ext/autoload_ext.py`) |
| `Github Releases To Discord` | `./workflows/discord-bot.yml` | GitHub `release: types: [published]` | `github-releases-to-discord` | the release body is posted to the Discord changelog channel (pre-releases included) |

`build` job steps: checkout → setup-python → `pip install --upgrade pip` + `pip install build` → `python -m build` (sdist + wheel) → install the built wheel with the `test` extra → `python -m pytest` (picks up `testpaths = ["unit_test/etl"]`) → wheel-only smoke test.
The wheel-only smoke test runs `.github/scripts/wheel_smoke.py` with `cwd = $RUNNER_TEMP`, so `import evox_etl` must resolve to `site-packages`, not the checkout.
`python-publish.yml` `deploy` job steps: checkout → setup-python 3.12 → `python -m build` → install the fresh wheel and run the same smoke script → `pypa/gh-action-pypi-publish`.
The publish job is serialized by `concurrency: group: pypi-publish` (`cancel-in-progress: false`) and carries `id-token: write`; the workflow-level permission is `contents: read`.

## Local equivalents (what a contributor runs)
```
python -m build
pip install "dist/evox-2.0.0a1-py3-none-any.whl[test]"
pytest                                  # = the `build` job's test step
python examples/quickstart.py           # = the `examples` job
python examples/hpo/single_objective_hpo.py
pip install ".[docs]" && sphinx-build -b html docs/source /tmp/docs-out
ruff check .                            # = the `Ruff` job
```

## Constraints
- No torch, no GPU, no CUDA steps anywhere: anything that reintroduces `torch` in these workflows contradicts the 2.0 packaging.
- Python matrix is `3.11`–`3.13`; 3.10 is unsupported by `evoxc` and must not come back.
- Keep the action style: `actions/checkout@v6`, `actions/setup-python@v6`, `fail-fast: false` on the matrix; third-party actions stay pinned as they are.
- CI installs `evoxc` from PyPI as a plain dependency; compiled-backend extras (`iree`/`tvm`/`compilers`) are never installed here (numpy backend only).
- Do not add `-W` to the Sphinx build: pre-existing docstring warnings must not fail the job; RTD (`.readthedocs.yaml`) is the real docs publisher.
- Never add a PyPI API token/password: publishing is OIDC trusted publishing, which requires a PyPI-side trusted-publisher entry for repo `EMI-Group/evox` / workflow `python-publish.yml`.
- `workflow_dispatch` on the publish workflow rebuilds from the branch the run is started on, so start it on the release's tag/commit when retrying.
- `wheel_smoke.py` must stay runnable outside the checkout and keep asserting the import came from `site-packages`; the CEC2022 part is the package-data guard (`cec2022_input_data/*.txt`).
- Workflow changes are YAML: validate locally with `python -c "import yaml,sys;[yaml.safe_load(open(f)) for f in sys.argv[1:]]" .github/workflows/*.yml`.

## Release procedure
1. Bump `version` in `pyproject.toml` (single source of the version; no `__version__` gate) — experimental 2.0 builds use a PEP 440 pre-release such as `2.0.0a1`.
2. Commit with the message `Bump version to vX.Y.Z`.
3. `git tag vX.Y.Z` on that commit and push the tag.
4. Publish a GitHub release for the tag and tick **"pre-release"** for experimental builds; the release body is the changelog that `discord-bot.yml` posts.
5. Publishing is automatic (`python-publish.yml` on `release: published`): it rebuilds, smoke-tests the wheel, and uploads via OIDC.
6. If the publish failed, re-run it from the Actions UI (`workflow_dispatch`) on the tag's commit; a plain `v2.0.0a1` tag plus a "pre-release" release is enough for pip users, who are not auto-upgraded to a pre-release.
7. Keep the news entries in `README.md` / `README_ZH.md` in sync with the release, as the 1.x bumps did.

## Known issues
- `unit_test/etl/algorithms/parity/test_parity.py` still imports `torch` and `evox.algorithms` (the deleted 1.x code) and fails (`test_asebo_state_parity_with_torch`); the `build` job runs the whole suite, so it stays red until that file is removed/retired with `src/evox/` (owner: `unit_test/`, not this node).
- Repo-wide `ruff check .` is clean (the 2.0 lint cleanup fixed the former `unit_test/etl` / `src/evox_etl` / `examples` violations); `ruff format --check` would still reformat many files, so the `Ruff` workflow runs `ruff check` only (unchanged behaviour).
- The full suite takes ~8 minutes per matrix entry (numpy-backend tracing dominates), so a 3-version matrix is ~8 minutes wall-clock with parallel runners.
- The wheel currently still ships a legacy `evox/` package because `src/evox/` is mid-deletion in the 2.0 refactor; no workflow references that path.

## Routing Table
| Area | Path | Description |
|---|---|---|
| Workflows | `./workflows/` | The four GitHub Actions workflows above (package CI, PyPI publish, Ruff, Discord release bot) |
| CI scripts | `./scripts/` | `wheel_smoke.py` — the wheel-only smoke test shared by the CI and publish jobs |
| PR template | `./pull_request_template.md` | The checklist contributors see when opening a PR |
