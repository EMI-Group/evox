"""pytest bootstrap shim for the evox_etl vis_tools test suite.

``evox_etl`` is not pip-installed (no ``src/evox_etl/__init__.py`` exists — it
is a PEP 420 namespace package), so pytest needs the repository root and its
``src/`` directory on ``sys.path`` before the test modules are collected.

The shim is idempotent (skips entries already on ``sys.path``) and computes the
locations from ``__file__``: the repository root is the first ancestor of this
directory containing ``pyproject.toml`` (from ``unit_test/etl/vis_tools/``:
0 = vis_tools, 1 = etl, 2 = unit_test, 3 = repo root).
"""

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent

_REPO_ROOT: Path | None = None
for _candidate in (_TESTS_DIR, *_TESTS_DIR.parents):
    if (_candidate / "pyproject.toml").is_file():
        _REPO_ROOT = _candidate
        break
if _REPO_ROOT is None:  # pragma: no cover — fallback for the as-written layout
    _REPO_ROOT = _TESTS_DIR.parents[4]

for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
