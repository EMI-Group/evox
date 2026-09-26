# evox_etl numerical problems — test suite

Relocate-ready pytest suite for the functional `evox_etl` numerical problems
(`basic`, `dtlz`, `cec2022`), running everything through `etl.build` +
`etl.run` on the numpy backend. The pure-etl tests (no torch) live at the top
level; `parity/` compares `evox_etl` against the pip-installed torch `evox`
reference and is the only place allowed to import torch. The canonical
destination of this directory is `unit_test/etl/problems/` — the parent agent
will move it there; the `conftest.py` sys.path shim locates the repository root
from `__file__`, so no path adjustments are needed after the move.

Run (from the repository root):

```
/mnt/local-ssd/bchuang/evox/.venv/bin/python -m pytest src/evox_etl/problems/tests -q
```
