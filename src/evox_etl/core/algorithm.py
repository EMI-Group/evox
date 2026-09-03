"""Documentation-only functional protocol for evox_etl algorithm modules."""

from __future__ import annotations

from typing import Any, Protocol

__all__ = [
    "Algorithm",
    "AlgorithmConfig",
    "AlgorithmState",
    "Candidates",
    "Fitness",
    "KeyArray",
]

# Type aliases — the concrete types are defined by each algorithm module.
AlgorithmConfig = Any
AlgorithmState = Any
Candidates = Any
Fitness = Any
KeyArray = Any


class Algorithm(Protocol):
    """Functional contract of an evox_etl algorithm module.

    **This protocol is documentation only**: algorithms are duck-typed and do NOT
    inherit from it. An algorithm module is a plain Python module that defines:

    - a frozen config dataclass named after the algorithm (e.g. ``PSO``),
    - PLAIN module-level functions (NOT ``@etl.defn`` — defn objects raise when
      called, even inside a trace) ``init``/``ask``/``tell`` (and optionally
      ``init_ask``/``init_tell``) living in the SAME module as the config.

    The workflow resolves them via
    ``importlib.import_module(type(config).__module__)``. All functions must be
    called inside an active etl trace (``etl.build``/``etl.run``/``etl.cond``);
    calling them outside a trace raises etl's ``TraceError``.

    Key convention: functions that need randomness split the key FROM the state
    (``key, subkey = etl.random.split(state.key)``), use subkeys, and store the
    parent key back, so ask/tell stay deterministic given the state.
    """

    def init(self, config: AlgorithmConfig, key: KeyArray) -> AlgorithmState:
        """Create the initial state from the config and a random key."""
        ...

    def ask(self, config: AlgorithmConfig, state: AlgorithmState) -> tuple[Candidates, AlgorithmState]:
        """Produce the next candidate population and the updated state."""
        ...

    def tell(self, config: AlgorithmConfig, state: AlgorithmState, fitness: Fitness) -> AlgorithmState:
        """Update the state with the evaluated fitness (minimization semantics)."""
        ...

    def init_ask(self, config: AlgorithmConfig, state: AlgorithmState) -> tuple[Candidates, AlgorithmState]:
        """Optional: ask variant for the first generation (e.g. NSGA-style batch sizes).

        Used by the workflow only if BOTH `init_ask` and `init_tell` exist.
        """
        ...

    def init_tell(self, config: AlgorithmConfig, state: AlgorithmState, fitness: Fitness) -> AlgorithmState:
        """Optional: tell variant paired with `init_ask`."""
        ...
