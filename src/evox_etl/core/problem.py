"""Documentation-only functional protocol for evox_etl problem modules."""

from __future__ import annotations

from typing import Any, Protocol

__all__ = ["Problem", "ProblemConfig", "ProblemState", "Fitness", "KeyArray"]

# Type aliases — the concrete types are defined by each problem module.
ProblemConfig = Any
ProblemState = Any
Fitness = Any
KeyArray = Any


class Problem(Protocol):
    """Functional contract of an evox_etl problem module.

    **This protocol is documentation only**: problems are duck-typed and do NOT
    inherit from it. A problem module is a plain Python module that defines:

    - a frozen config dataclass named after the problem (e.g. ``Sphere``),
    - a PLAIN module-level function (NOT ``@etl.defn``) ``evaluate`` living in
      the SAME module as the config, plus optionally ``init`` for stateful
      problems (numerical problems are stateless and simply omit it).

    The workflow resolves them via
    ``importlib.import_module(type(config).__module__)``. All functions must be
    called inside an active etl trace; calling them outside a trace raises etl's
    ``TraceError``.
    """

    def evaluate(
        self,
        config: ProblemConfig,
        state: ProblemState,
        pop: Any,
    ) -> tuple[Fitness, ProblemState]:
        """Evaluate a population `pop` of shape (n, dim) and return the updated state.

        Returns `(fitness, state)` where fitness has shape (n,) or (n, n_obj).
        """
        ...

    def init(self, config: ProblemConfig, key: KeyArray) -> ProblemState:
        """Optional: create the initial problem state from the config and a random key."""
        ...
