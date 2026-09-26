"""Shared pytest path setup for the unit_test/etl/algorithms tests.

Makes both the repo root (``evox``) and ``src/`` (``evox_etl``) importable in
every test collected under this directory, so ``import evox`` (torch
reference, used by parity tests) and ``import evox_etl`` (the functional
package under test) work without any installation.
"""

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))
