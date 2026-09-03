"""Local path shim: make `import helpers` (unit_test/etl/algorithms/helpers.py) work for the es_variants smoke tests."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
