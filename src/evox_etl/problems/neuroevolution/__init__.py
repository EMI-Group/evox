"""Functional ETL neuroevolution problems (virtual Gaussian-noise ES).

Port of the torch ``evox.problems.neuroevolution`` virtual-population problems
onto ETL.  :mod:`~evox_etl.problems.neuroevolution.virtual_problem` implements
the ``(center_flat, seeds, sigma)`` payload problem whose forward pass regenerates
per-individual noise from seeds (full-noise or LoRA mode) instead of receiving a
``(pop_size, dim)`` population.  External-library problems (brax, mujoco,
supervised learning) are not ported.
"""

__all__ = [
    "virtual_problem",
    "VirtualProblemConfig",
    "VirtualProblem",
    "VirtualLoRAProblem",
    "make_virtual_problem",
    "evaluate",
]

from . import virtual_problem
from .virtual_problem import (
    VirtualLoRAProblem,
    VirtualProblem,
    VirtualProblemConfig,
    evaluate,
    make_virtual_problem,
)
