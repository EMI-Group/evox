"""Shared pytest path setup for the unit_test/etl/algorithms tests.

Makes ``src/`` importable in every test collected under this directory, so the
``evox_etl`` package (the package under test, living under ``src/``) works
without any installation. The repository root is kept on ``sys.path`` as well so
repository-level helpers stay importable.
"""

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))
