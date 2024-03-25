from typing import Tuple

import jax.numpy as jnp
import numpy as np
from evoxbench.test_suites import c10mop, citysegmop, in1kmop
from jax import random, ShapeDtypeStruct
from jax.experimental import io_callback
from functools import partial

from evox import Problem, State, jit_class


def evaluate_with_seed(benchmark, seed, pop):
    np.random.seed(seed)
    fitness = benchmark.evaluate(pop)
    fitness = fitness.astype(np.float32)
    return fitness


@jit_class
class EvoXBenchProblem(Problem):
    def __init__(self, benchmark):
        super().__init__()
        self.benchmark = benchmark
        self.n_objs = self.benchmark.evaluator.n_objs
        self.lb = self.benchmark.search_space.lb
        self.ub = self.benchmark.search_space.ub
        self._evaluate = partial(evaluate_with_seed, self.benchmark)

    def setup(self, key):
        return State(key=key)

    def evaluate(self, state, pop):
        # use io_callback instead of pure_callback
        # because the evoxbench's `evaluate` is noisy
        key, subkey = random.split(state.key)
        seed = random.randint(subkey, (1,), 0, 2**31 - 1)
        pop_size = pop.shape[0]
        fitness = io_callback(
            self._evaluate,
            ShapeDtypeStruct((pop_size, self.n_objs), dtype=jnp.float32),
            seed,
            pop,
        )
        return fitness, state.update(key=key)


@jit_class
class C10MOP(EvoXBenchProblem):
    def __init__(self, problem_id) -> None:
        assert (
            isinstance(problem_id, int) and 1 <= problem_id and problem_id <= 9
        ), "For c10mop, problem_id must be an integer between 1 and 9"
        benchmark = c10mop(problem_id)
        super().__init__(benchmark)


@jit_class
class CitySegMOP(EvoXBenchProblem):
    def __init__(self, problem_id) -> None:
        assert (
            isinstance(problem_id, int) and 1 <= problem_id and problem_id <= 15
        ), "For citysegmop, problem_id must be an integer between 1 and 9"
        benchmark = citysegmop(problem_id)
        super().__init__(benchmark)


@jit_class
class IN1kMOP(EvoXBenchProblem):
    def __init__(self, problem_id) -> None:
        assert (
            isinstance(problem_id, int) and 1 <= problem_id and problem_id <= 9
        ), "For in1kmop, problem_id must be an integer between 1 and 9"
        benchmark = in1kmop(problem_id)
        super().__init__(benchmark)
