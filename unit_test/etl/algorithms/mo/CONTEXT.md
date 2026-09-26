# unit_test/etl/algorithms/mo — MO algorithm smoke tests (etl-only)

## Intent
Step-protocol smoke tests for the six functional evox_etl MO ports
(`src/evox_etl/algorithms/mo/`): nsga2, nsga3, rvea, rveaa, moead, hype.
No torch imports here — torch parity tests live in `../parity/`.

## Files
Every file follows the same three-part pattern: (1) a gen-0/step contract
test, (2) a 3-generation DTLZ1 run via `helpers.run_generations`, (3) a
seed-determinism check.

| File | Family-specific checks |
|---|---|
| `test_nsga2.py` | plain pop_size (20); `state.offspring` fused-step contract |
| `test_nsga3.py` | plain pop_size (20); `state.off` fused-step contract |
| `test_rvea.py` | N_EFF=15 (Das-Dennis overwrite), N_OFFSPRING=14 (SBX odd-row drop); NaN survivor rows → nanmin assertions |
| `test_rveaa.py` | N_EFF=15; pop grows to (2·N_EFF, dim) after the first step |
| `test_moead.py` | N_EFF=15; init_step sets ideal point z = min(fit, axis=0); `state.next_generation` fused-step contract |
| `test_hype.py` | init_step derives ref = 1.2 · max(all fit entries); `state.offspring` fused-step contract |

## Step-protocol mechanics (shared `_trace_step` helper, one copy per file)
- `_trace_step(step_fn, cfg, state)` builds one generation function whose
  `evaluate` closure evaluates DTLZ1 AND records the candidate batch it
  received in a closure cell; the traced fn returns `(state, batch)` so the
  batch is an exact graph output of the same trace (no host approximation).
- Gen-0 assertions: exactly ONE evaluate call of the FULL pop shape; the
  recorded batch equals the pre-step pop bit-for-bit.
- Step assertions: exactly ONE evaluate call whose batch differs from the
  parents and equals the offspring leaf stored in the returned state
  (`offspring`/`off`/`next_generation`) — the candidates-in-state contract
  of the fused step protocol.
- Full runs use `helpers.run_generations` (dispatch mirrors StdWorkflow:
  init_step at gen 0 when present, plain step after).

## Known Issues (behaviors the assertions must respect)
- MOEAD is NOT elitist: per-objective bests of the FINAL population can
  regress vs the run history. Final-pop sanity bounds stay loose (100 on
  DTLZ1); convergence-quality assertions live in `../parity/test_moead.py`
  with the history-min metric and MOEAD_ABS_TOL=0.20.
- RVEA/RVEAa selection keeps one survivor row per reference vector,
  NaN-filled for unmatched vectors (torch-faithful `ref_vec_guided`):
  use `np.nanmin`, and compare determinism with `equal_nan=True`.
- NSGA3 with odd pop_size raises ShapeError — pre-existing upstream quirk;
  the smoke test uses even pop_size (20) and does not pin it.
- RVEAa state grows to `(2·n_v, dim)` rows after the first step and only
  truncates at the final generation (`gen == max_gen`); intermediate-shape
  assertions must use 2·N_EFF.

## Constraints
- ETL only (numpy backend): everything inside `etl.build`/`etl.run`.
- No thresholds beyond the loose sanity bounds above; convergence-quality
  parity vs torch is `../parity/`'s job.
