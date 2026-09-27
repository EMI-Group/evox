"""pytest bootstrap shim for the evox_etl workflows test suite.

``evox_etl`` is not pip-installed (it is a PEP 420 namespace package under
``src/``), so pytest needs the repository root and its ``src/`` directory on
``sys.path`` before the test modules are collected. The shim is idempotent and
computes the locations from ``__file__`` (same pattern as
``../metrics/conftest.py``).
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
