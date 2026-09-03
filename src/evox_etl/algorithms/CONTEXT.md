# evox_etl/algorithms — SO and MO algorithms in functional style

## Intent
Port of all 34 torch evox algorithms (read-only reference in `../../../evox/
algorithms/`) to `init/ask/tell` `@etl.defn` functions + frozen config dataclasses.
See `../../DESIGN.md` §4-5.

## Routing Table
| Area | Path |
|---|---|
| DE variants (code, de, jade, ode, sade, shade) | `so/de_variants/` |
| ES variants (adam_step, ars, asebo, cma_es, des, esmc, guided_es, nes, noise_reuse_es, open_es, persistent_es, snes, sort_utils, virtual_lora_es) | `so/es_variants/` |
| PSO variants (clpso, cso, dms_pso_el, fs_pso, pso, sl_pso_gs, sl_pso_us, utils) | `so/pso_variants/` |
| MO algorithms (nsga2, nsga3, moead, rvea, rveaa, hype) | `mo/` |
