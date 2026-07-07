import unittest

import torch
import torch.nn as nn

from evox.core import Algorithm, Monitor, Mutable, Problem
from evox.workflows import EvalMonitor, StdWorkflow


class _SphereProblem(Problem):
    """Minimal tensor-based problem used for the standard-path test."""

    def __init__(self):
        super().__init__()

    def evaluate(self, pop: torch.Tensor) -> torch.Tensor:
        return (pop**2).sum(dim=-1)


class _TensorAlgorithm(Algorithm):
    """Minimal tensor-based algorithm that evaluates a random population."""

    def __init__(self, dim: int, pop_size: int):
        super().__init__()
        self.dim = dim
        self.pop_size = pop_size
        self.pop = Mutable(torch.empty(pop_size, dim))
        self.fit = Mutable(torch.empty(pop_size))

    def step(self):
        pop = torch.rand(self.pop_size, self.dim)
        self.pop = pop
        self.fit = self.evaluate(pop)


class RecordingProblem(Problem):
    """Records whatever payload it receives and computes a simple fitness.

    For a plain tensor the fitness is the sum of squares per row.
    For a virtual-population tuple ``(center, seeds, sigma)`` the fitness is
    broadcast to ``pop_size`` (``seeds.shape[0]``).
    """

    def __init__(self):
        super().__init__()
        self.received = None

    def evaluate(self, pop):
        self.received = pop
        if isinstance(pop, torch.Tensor):
            return (pop**2).sum(dim=-1)
        center, seeds, sigma = pop
        # fitness based on center, broadcast to pop_size
        return center.sum().expand(seeds.shape[0])


class VirtualPayloadAlgorithm(Algorithm):
    """Passes a tuple ``(center, seeds, sigma)`` through ``self.evaluate()``."""

    def __init__(self, dim, pop_size):
        super().__init__()
        self.dim = dim
        self.pop_size = pop_size
        self.pop = Mutable(torch.empty(pop_size, dim))
        self.fit = Mutable(torch.empty(pop_size))

    def step(self):
        center = torch.zeros(self.dim)
        seeds = torch.zeros(self.pop_size, dtype=torch.int64)
        sigma = 0.1
        payload = (center, seeds, sigma)
        self.fit = self.evaluate(payload)
        self.pop = center.unsqueeze(0).expand(self.pop_size, self.dim)

    def record_step(self):
        return {"center": self.pop[0]}


class RecordingMonitor(Monitor):
    """A lightweight monitor that records the payloads of every hook.

    The default :class:`EvalMonitor` cannot be used with the virtual-population
    path because its ``pre_tell`` implementation assumes the number of candidate
    solutions recorded in ``post_ask`` matches the fitness length. With a virtual
    population only the (single) ``center`` is passed to ``post_ask`` while the
    fitness has ``pop_size`` entries, which would raise an ``IndexError`` in
    ``EvalMonitor.pre_tell``. This monitor simply captures the raw payloads so the
    virtual path can be exercised end-to-end.
    """

    def __init__(self):
        super().__init__()
        self.post_ask_data = None
        self.pre_eval_data = None
        self.post_eval_data = None
        self.pre_tell_data = None

    def post_ask(self, candidate_solution):
        self.post_ask_data = candidate_solution

    def pre_eval(self, transformed_candidate_solution):
        self.pre_eval_data = transformed_candidate_solution

    def post_eval(self, fitness):
        self.post_eval_data = fitness

    def pre_tell(self, transformed_fitness):
        self.pre_tell_data = transformed_fitness


class TestStdWorkflowVirtual(unittest.TestCase):
    def setUp(self):
        torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
        self.dim = 3
        self.pop_size = 5

    def test_standard_tensor_path_backward_compatibility(self):
        """The standard tensor path should keep working and record fitness history."""
        algo = _TensorAlgorithm(self.dim, self.pop_size)
        prob = _SphereProblem()
        monitor = EvalMonitor(full_sol_history=True)
        workflow = StdWorkflow(algo, prob, monitor=monitor)

        workflow.init_step()
        for _ in range(2):
            workflow.step()

        # fitness history should have one entry per step
        self.assertEqual(len(monitor.fitness_history), 3)
        for fit in monitor.fitness_history:
            # each recorded fitness corresponds to the evaluated population
            self.assertEqual(fit.shape[0], self.pop_size)
        # topk fitness must have been populated
        self.assertIsNotNone(monitor.topk_fitness)

    def test_tuple_payload_routing(self):
        """A tuple payload should be routed to the problem as-is (no transformation)."""
        algo = VirtualPayloadAlgorithm(self.dim, self.pop_size)
        prob = RecordingProblem()
        monitor = RecordingMonitor()
        workflow = StdWorkflow(algo, prob, monitor=monitor)

        workflow.step()

        # the problem received the full tuple unchanged
        self.assertIsInstance(prob.received, tuple)
        self.assertEqual(len(prob.received), 3)
        center, seeds, sigma = prob.received
        self.assertIsInstance(center, torch.Tensor)
        self.assertEqual(center.shape, (self.dim,))
        self.assertIsInstance(seeds, torch.Tensor)
        self.assertEqual(seeds.dtype, torch.int64)
        self.assertEqual(seeds.shape, (self.pop_size,))
        self.assertEqual(sigma, 0.1)

        # the evaluated fitness should match the population size
        self.assertEqual(monitor.post_eval_data.shape, (self.pop_size,))
        self.assertEqual(algo.fit.shape, (self.pop_size,))

    def test_solution_transform_skipped(self):
        """``solution_transform`` must NOT be applied on the virtual-population path."""

        class FlagTransform(nn.Module):
            def __init__(self):
                super().__init__()
                self.called = False

            def forward(self, x):
                self.called = True
                return x

        algo = VirtualPayloadAlgorithm(self.dim, self.pop_size)
        prob = RecordingProblem()
        solution_transform = FlagTransform()
        workflow = StdWorkflow(
            algo,
            prob,
            monitor=RecordingMonitor(),
            solution_transform=solution_transform,
        )

        workflow.step()

        # the solution transform should never have been invoked
        self.assertFalse(solution_transform.called)
        # the same module is held by the workflow; assert there as well
        self.assertFalse(workflow.solution_transform.called)
        # confirm the tuple still reached the problem untouched
        self.assertIsInstance(prob.received, tuple)
        self.assertEqual(prob.received[1].dtype, torch.int64)


if __name__ == "__main__":
    unittest.main()
