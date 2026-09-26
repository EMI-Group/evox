"""Documentation-only functional protocol for evox_etl monitor modules."""

from __future__ import annotations

from typing import Any, Protocol

__all__ = ["Monitor", "MonitorConfig", "MonitorState", "Candidates", "Fitness", "KeyArray"]

# Type aliases — the concrete types are defined by each monitor module.
MonitorConfig = Any
MonitorState = Any
Candidates = Any
Fitness = Any
KeyArray = Any


class Monitor(Protocol):
    """Functional contract of an evox_etl monitor module.

    **This protocol is documentation only**: monitors are duck-typed and do NOT
    inherit from it. A monitor module is a plain Python module that defines:

    - a frozen config dataclass named after the monitor, whose host-side history
      lists (e.g. ``fit_history``/``sol_history``/``pop_history``) live on the
      config object (plain Python lists, appended by the workflow),
    - a PLAIN module-level function (NOT ``@etl.defn``) ``monitor_update`` living
      in the SAME module as the config, plus optionally ``init``,
    - optionally a host-side convenience wrapper class named ``Monitor``
      (constructor ``Monitor(config=..., state=...)``) that exposes accessors
      such as ``get_best_fitness``.

    The workflow resolves the functions via
    ``importlib.import_module(type(config).__module__)``. All graph functions must
    be called inside an active etl trace; calling them outside a trace raises
    etl's ``TraceError``.
    """

    def monitor_update(
        self,
        config: MonitorConfig,
        state: MonitorState,
        candidate: Candidates,
        fitness: Fitness,
    ) -> MonitorState:
        """Update the monitor state with the raw candidates and the (transformed) fitness."""
        ...

    def init(self, config: MonitorConfig, key: KeyArray) -> MonitorState:
        """Optional: create the initial monitor state from the config and a random key."""
        ...
