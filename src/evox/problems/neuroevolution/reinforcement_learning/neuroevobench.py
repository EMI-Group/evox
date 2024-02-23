from typing import List, Callable, Any
import jax
from jax import jit, vmap
import jax.numpy as jnp
from jax.tree_util import tree_leaves
from evox import Problem, State, jit_method
from evosax.problems import BBOBFitness

bbob_fns = [
    "Sphere",
    "RosenbrockRotated",
    "Discus",
    "RastriginRotated",
    "Schwefel",
    "BuecheRastrigin",
    "AttractiveSector",
    "Weierstrass",
    "SchaffersF7",
    "GriewankRosenbrock",
    # Part 1: Separable functions
    "EllipsoidalOriginal",
    "RastriginOriginal",
    "LinearSlope",
    # Part 2: Functions with low or moderate conditions
    "StepEllipsoidal",
    "RosenbrockOriginal",
    # Part 3: Functions with high conditioning and unimodal
    "EllipsoidalRotated",
    "BentCigar",
    "SharpRidge",
    "DifferentPowers",
    # Part 4: Multi-modal functions with adequate global structure
    "SchaffersF7IllConditioned",
    # Part 5: Multi-modal functions with weak global structure
    "Lunacek",
    "Gallagher101Me",
    "Gallagher21Hi",
]


class NeuroEvoBench(Problem):
    def __init__(
        self,
        policy: Callable,
        evaluator: str,
        eval_param: dict,
        seed_id: int = 0,
    ):
        """Contruct a neuroevobench-based problem

        Parameters
        ----------
        policy
            A function that accept two arguments
            the first one is the parameter and the second is the input.
        evaluator
            The evaluator name that you use.
        eval_param
            The evaluator's param that user should set. More information about it will add later.
        """
        self.batched_policy = jit(vmap(policy))
        self.policy = policy
        self.evaluator = evaluator
        self.eval_param = eval_param
        self.seed_id = seed_id

    def setup(self, key):
        return State(key=key)

    def evaluate(self, state, x):
        if self.evaluator == "bbob":
            # total_reward = jnp.zeros(25)
            total_reward = 0
            i = 0
            rng = jax.random.PRNGKey(0)
            if "eval_fn" in self.eval_param:
                self.eval_fn = self.eval_param["eval_fn"]
            else:
                self.eval_fn = bbob_fns
            for fn in self.eval_fn:
                rng, rng_eval = jax.random.split(rng)
                evaluator = BBOBFitness(fn, self.eval_param["dim"])
                fitness = evaluator.rollout(rng_eval, x)
                # total_reward = total_reward.at[i].set(fitness)
                total_reward += fitness
                i += 1
            return total_reward, state
        else:
            raise NameError(
                f'No "{self.evaluator}" exist. Please check the evaluator you choose is right'
            )
