"""Documentation-only functional protocol for evox_etl algorithm modules."""

from __future__ import annotations

from typing import Any, Callable, Protocol

__all__ = [
    "Algorithm",
    "AlgorithmConfig",
    "AlgorithmState",
    "Candidates",
    "Evaluate",
    "Fitness",
    "KeyArray",
]

# Type aliases — the concrete types are defined by each algorithm module.
AlgorithmConfig = Any
AlgorithmState = Any
Candidates = Any
Fitness = Any
KeyArray = Any
Evaluate = Callable[[Candidates], Fitness]


class Algorithm(Protocol):
    """Functional contract of an evox_etl algorithm module (step protocol).

    **This protocol is documentation only**: algorithms are duck-typed and do NOT
    inherit from it. An algorithm module is a plain Python module that defines:

    - a frozen config dataclass named after the algorithm (e.g. ``PSO``),
    - PLAIN module-level functions (NOT ``@etl.defn`` — defn objects raise when
      called, even inside a trace) ``init`` and ``step`` (and optionally
      ``init_step``/``final_step``) living in the SAME module as the config.

    The workflow resolves them via
    ``importlib.import_module(type(config).__module__)``. All functions must be
    called inside an active etl trace (``etl.build``/``etl.run``); calling them
    outside a trace raises etl's ``TraceError``.

    Key convention: functions that need randomness split the key FROM the state
    (``key, subkey = etl.random.split(state.key)``), use subkeys, and store the
    parent key back, so every step stays deterministic given the state.

    The pre-1.0 ``ask``/``tell``/``init_ask``/``init_tell`` protocol is GONE:
    a step function owns the WHOLE generation (generation → evaluation →
    state update), mirroring torch ``Algorithm.step``.
    """

    def init(self, config: AlgorithmConfig, key: KeyArray) -> AlgorithmState:
        """Create the initial state from the config and a random key."""
        ...

    def step(self, config: AlgorithmConfig, state: AlgorithmState, evaluate: Evaluate) -> AlgorithmState:
        """Run ONE full generation and return the updated state (REQUIRED).

        Owns the whole generation: generate candidates, obtain their fitness
        via ``fitness = evaluate(candidates)`` (may call it several times with
        different candidate sets), update the state, return it.

        ``evaluate`` is a traced closure created by the workflow. It applies,
        in order: solution_transform → ``problem.evaluate`` → opt-direction
        scaling (min semantics) → fitness_transform → monitor update, and
        returns the TRANSFORMED fitness. Treat it as fully opaque:

        - pass whatever tensor/pytree the candidates are, get fitness back;
        - do NOT store it in the state or re-thread it through returns — the
          workflow owns the problem/monitor state and keeps the state of the
          LAST ``evaluate`` call (torch's stateful ``Problem`` semantics,
          reified). The workflow also handles the generation counter and key.
        """
        ...

    def init_step(
        self, config: AlgorithmConfig, state: AlgorithmState, evaluate: Evaluate
    ) -> AlgorithmState:
        """Optional: first-generation variant of `step` (e.g. NSGA-style batch sizes).

        Used by the workflow only if the module defines it; otherwise the
        workflow calls ``step`` for generation 0.
        """
        ...

    def final_step(
        self, config: AlgorithmConfig, state: AlgorithmState, evaluate: Evaluate
    ) -> AlgorithmState:
        """Optional: last-generation variant of `step`.

        Used by the workflow only if the module defines it; otherwise the
        workflow calls ``step`` for the last generation.
        """
        ...
