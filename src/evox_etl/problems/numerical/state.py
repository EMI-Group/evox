"""Problem state for evox_etl numerical problems."""

import dataclasses

__all__ = ["ProblemState"]


@dataclasses.dataclass(frozen=True)
class ProblemState:
    """Stateless problem state shared by all numerical problems."""
