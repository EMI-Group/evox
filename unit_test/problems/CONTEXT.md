# unit_test/problems/ — Problem Tests

## Intent
Tests for benchmark problems under `evox.problems`, including numerical benchmarks (CEC2022), neuroevolution problems, and HPO wrappers.

## Routing Table

| Area | Path | Description |
|---|---|---|
| Neuroevolution | `neuroevolution/` | Tests for neuroevolution problems (e.g. `VirtualLoRAProblem`) using inline MLP + synthetic DataLoader fixtures |
| HPO wrapper | `test_hpo_wrapper.py` | Tests for `HPOProblemWrapper` / `HPOFitnessMonitor` / `HPOMonitor`: init params, `evaluate`, the `num_repeats>1` fit-aggregation path (SO + MO), and SO/MO outer `StdWorkflow` optimization loops (incl. multi-step) |
