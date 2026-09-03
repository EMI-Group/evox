# unit_test/etl — tests for the functional evox_etl package

## Intent
unittest/pytest suite mirroring `src/evox_etl/`. Two kinds:
1. Correctness tests (port of torch evox tests in `../` where applicable; no torch
   import in these).
2. `parity/` — tests importing BOTH torch `evox` and `evox_etl`, asserting the
   etl versions converge to fitness within tolerance of the torch reference
   (may import torch).
3. Cross-backend: numpy-backend determinism / parity where applicable.

Run: `/mnt/local-ssd/bchuang/evox/.venv/bin/python -m pytest unit_test/etl -q`
See `../../src/evox_etl/DESIGN.md` §6.
