"""pytest bootstrap shim for the evox_etl operators parity test suite.

``evox_etl`` is not pip-installed (it is a PEP 420 namespace package under
``src/``), so pytest needs the repository root and its ``src/`` directory on
``sys.path`` before the test modules are collected.  The repo-root
``conftest.py`` already does this for suites collected from the repository
root, but this subdirectory must also keep working if the suite is relocated
(e.g. moved into ``src/evox_etl/operators/tests/parity/``), so it re-applies
the shim idempotently: the repository root is the first ancestor of this
directory containing ``pyproject.toml`` (from the current location the plain
parents chain is: 0 = parity, 1 = operators, 2 = etl, 3 = unit_test, 4 =
repo root; the marker search generalizes that to the relocated layout).

No other pytest configuration is needed — pytest collects the test modules
as given.
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
