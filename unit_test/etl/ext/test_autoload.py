"""Root exports + the guarded ``evox_etl_ext`` autoload.

``evox_etl/__init__.py`` ends with a guarded
``from evox_etl_ext.autoload_ext import auto_load_extensions`` — the extension
package may or may not exist, so importing ``evox_etl`` must never fail.  These
tests pin the root surface and prove the guard is a genuine no-op without the
extension, in a FRESH interpreter (module caching cannot mask a broken guard).
"""

from __future__ import annotations

import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version as _dist_version
from pathlib import Path

import evox_etl
import evox_etl.vis_tools  # noqa: F401 — root export must stay importable

#: Version the package is required to report.  Never pinned here: the version is
#: single-sourced from ``pyproject.toml`` into the distribution metadata, so the
#: expectation is derived the same way ``evox_etl/__init__.py`` derives it --
#: the installed ``evox`` distribution when present, else the source-checkout
#: fallback.
_SOURCE_TREE_VERSION = "0.0.0.dev0"

#: Fresh-interpreter script: poison ``sys.modules`` so any ``evox_etl_ext`` import
#: raises ImportError (the exception the guard must swallow), then import
#: ``evox_etl`` and print its version.  Forcing absence this way keeps the test
#: meaningful whether or not ``src/evox_etl_ext`` ships with the checkout.
_BLOCK_EXTENSION = """
import sys
sys.modules["evox_etl_ext"] = None
import evox_etl
print(evox_etl.__version__)
"""


def _repo_root() -> Path:
    """Repository root: first ancestor of this file containing ``pyproject.toml``."""
    for candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parent.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError("could not locate the repository root (no pyproject.toml above)")


def _expected_version() -> str:
    """Version ``evox_etl`` must report, derived instead of pinned.

    Mirrors ``evox_etl/__init__.py``: the installed ``evox`` distribution
    metadata when the dist is importable, else the source-checkout fallback.
    """
    try:
        return _dist_version("evox")
    except PackageNotFoundError:
        return _SOURCE_TREE_VERSION


def test_root_exports_and_version() -> None:
    """The root package imports, exposes ``__version__`` and ``vis_tools``."""
    assert hasattr(evox_etl, "__version__"), "evox_etl must expose __version__"
    assert isinstance(evox_etl.__version__, str)
    assert evox_etl.__version__ == _expected_version()
    assert evox_etl.vis_tools.__name__ == "evox_etl.vis_tools"
    assert "vis_tools" in evox_etl.__all__


def test_autoload_guard_is_a_noop_without_extension() -> None:
    """Importing ``evox_etl`` without ``evox_etl_ext`` must succeed (guard swallows it)."""
    root = _repo_root()
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    result = subprocess.run(
        [sys.executable, "-c", _BLOCK_EXTENSION],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"import failed without the extension:\n{result.stderr}"
    assert result.stdout.strip() == _expected_version()
