__all__ = ["VectorMetricProblem"]

import torch

from evox.core import Problem
from evox.triton_kernels import virtual_reduce_metric


class VectorMetricProblem(Problem):
    """Flat-vector virtual-metric problem.

    This is a flat-vector "virtual metric" experiment: unlike
    :class:`VirtualProblem`, it does **not** perform any model forwarding.
    Instead it treats all parameters as a single flat vector of length
    ``n_params`` and computes a normalized metric over it directly.

    Because the noise is never materialized — the fused kernel
    :func:`virtual_reduce_metric` applies the perturbations on the fly
    during the reduction — this problem demonstrates virtual populations at
    huge scales (``pop_size`` up to 16384+).

    The :meth:`evaluate` method supports two mutually-exclusive paths, selected
    automatically based on the type of ``payload``:

    - **Virtual path**: when ``payload`` is the tuple
      ``(center_flat, seeds, sigma)``, the fused
      :func:`virtual_reduce_metric` kernel computes the per-individual metric
      ``mean_k(|center[k] + sigma * noise[i, k]|)`` without ever materializing
      the ``(pop_size, n_params)`` perturbed population.
    - **Naive / materialized path**: when ``payload`` is a full
      ``(pop_size, n_params)`` 2D population tensor, the metric is computed
      directly as ``population.abs().mean(dim=-1)``. This serves as a baseline
      against the virtual path.

    ```{warning}
    This problem does NOT support HPO wrapper
    (``problems.hpo_wrapper.HPOProblemWrapper``).
    ```
    """

    def __init__(self, n_params: int, device: torch.device | None = None):
        """Initialize the ``VectorMetricProblem``.

        :param n_params: The length of the flat parameter vector. Both the
            virtual ``center_flat`` vector and the materialized population's
            second dimension must equal this value.
        :param device: The device to run the computations on. Defaults to the
            current default device.
        """
        super().__init__()
        self.n_params = n_params
        self.device = torch.get_default_device() if device is None else device

    def evaluate(self, payload) -> torch.Tensor:
        """Evaluate the flat-vector virtual metric for a population.

        The path is selected automatically based on the type of ``payload``:

        - If ``payload`` is a length-3 tuple ``(center_flat, seeds, sigma)``,
          the **virtual path** is taken. The fused
          :func:`virtual_reduce_metric` kernel applies the perturbations on the
          fly and computes the per-individual metric
          ``mean_k(|center[k] + sigma * noise[i, k]|)`` without ever
          materializing the perturbed population.
        - Otherwise, the **naive / materialized path** is taken. ``payload`` is
          treated as a full ``(pop_size, n_params)`` 2D population tensor and
          the metric is computed directly as
          ``population.abs().mean(dim=-1)``.

        :param payload: Either a tuple ``(center_flat, seeds, sigma)`` for the
            virtual path, where:

            - ``center_flat``: ``(n_params,)`` float tensor — flat center vector.
            - ``seeds``: ``(pop_size,)`` integer tensor — per-individual seeds.
            - ``sigma``: Python float — noise standard deviation.

            Or a ``(pop_size, n_params)`` 2D population tensor for the naive /
            materialized path.
        :return: A tensor of shape ``(pop_size,)`` containing the per-individual
            metric (fitness).
        """
        if isinstance(payload, tuple):
            center_flat, seeds, sigma = payload
            return virtual_reduce_metric(center_flat, seeds, sigma, self.n_params, offset=0)
        else:
            with torch.no_grad():
                return payload.abs().mean(dim=-1)
