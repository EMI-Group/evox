import types
from typing import Tuple

import jax
from jaxtyping import Array

from .module import *
from .state import State
from evox import Algorithm, has_init_ask, has_init_tell


class SurrogateAlgorithm(Algorithm):
    """Base class for all surrogate-based algorithms"""

    def get_surrogate_samples(
        self, state: State
    ) -> Tuple[jax.Array | None, jax.Array | None, State]:
        """Obtain the training samples (or None) to train a new surrogate model.
        Returns `None` if the surrogate model is not needed to be trained.

        Parameters
        ----------
        state
            The state of this algorithm.

        Returns
        -------
        samples_pop
            The training samples (or None) to train a new surrogate model.
        samples_fit
            The training samples (or None) to train a new surrogate model.
        state
            The new state of the algorithm.
        """
        return None, None, state

    def ask(self, state: State) -> Tuple[jax.Array, jax.Array, State]:
        """Ask the algorithm for points to be evaluated by real function and surrogate model

        Parameters
        ----------
        state
            The state of this algorithm

        Returns
        -------
        real_points
            The exploring points to be evaluated by real function
        surrogate_points
            The exploring points to be evaluated by surrogate model
        state
            The new state of the algorithm
        """
        return jnp.zeros(0), jnp.zeros(0), state

    def tell(
        self, state: State, fitness_real: jax.Array, fitness_surrogate: jax.Array
    ) -> State:
        """Tell the algorithm more information.

        Tell the algorithm about the points evaluated by real function and surrogate model.

        Parameters
        ----------
        state
            The state of this algorithm.
        fitness_real
            The fitness evaluated by real function.
        fitness_surrogate
            The fitness evaluated by surrogate model.

        Returns
        -------
        state
            The new state of the algorithm.
        """
        return state


class SurrogateAlgorithmWrapper(SurrogateAlgorithm):
    """
    A wrapper class for wrapping a base algorithm with surrogate model capabilities.

    This class extends the `SurrogateAlgorithm` and provides a flexible interface for users
    to leverage existing optimization algorithms while controlling when to train the surrogate model
    and how to evaluate populations using the surrogate model, which is specified for `SurrogateWorkflow`.

    It is acceptable to imitate this class to create a new wrapper class for a specific surrogate-based strategy.
    """

    def __init__(
        self,
        base_algorithm: Algorithm,
        begin_training_iter: int,
        retrain_per_iter: int,
        real_eval_portion: float,
    ):
        """
        Parameters
        ----------
        base_algorithm
            The base algorithm to be wrapped.
        begin_training_iter
            The iteration to start training the surrogate model.
        retrain_per_iter
            The iteration to restart the training process of the surrogate model.
        real_eval_portion
            The portion of the population to be evaluated by the real function.
        """
        super().__init__()
        self.base_algorithm = base_algorithm
        self.begin_training_iter = begin_training_iter
        self.retrain_per_iter = retrain_per_iter
        self.real_eval_portion = real_eval_portion

    def setup(self, key: jax.Array) -> State:
        """
        Set up the initial state of the algorithm.

        Parameters
        ----------
        key
            The random key for the algorithm.

        Returns
        -------
        state
            The initial state of the algorithm.
        """
        sub_state = self.base_algorithm.setup(key)
        sub_state = sub_state.replace(
            iter=1,
            training_xs=jnp.zeros((0, sub_state.population.shape[1])),
            training_ys=None,
        )
        return sub_state

    def init_ask(self, state: State) -> Tuple[jax.Array, jax.Array, State]:
        """
        The first ask behavior of the algorithm (if it exists).

        Parameters
        ----------
        state
            The state of this algorithm.

        Returns
        -------
        ask_real
            The initial population to be evaluated by the real function.
        ask_surrogate
            The initial population to be evaluated by the surrogate model.
        state
            The new state of the algorithm.
        """
        if has_init_ask(self.base_algorithm):
            ask_real, state = self.base_algorithm.init_ask(state)
            state = state.replace(training_xs=ask_real)
            ask_surrogate = None
        else:
            ask_real, ask_surrogate, state = self.ask(state)
        return ask_real, ask_surrogate, state

    def init_tell(self, state: State, fitness_real, fitness_surrogate) -> State:
        """
        The first tell behavior of the algorithm (if it exists).

        Parameters
        ----------
        state
            The state of this algorithm.
        fitness_real
            The fitness evaluated by the real function.
        fitness_surrogate
            The fitness evaluated by the surrogate model.

        Returns
        -------
        state
            The new state of the algorithm.
        """
        if fitness_surrogate is not None:
            fitness = jnp.concatenate([fitness_real, fitness_surrogate])
        else:
            fitness = fitness_real

        if has_init_tell(self.base_algorithm):
            state = self.base_algorithm.init_tell(state, fitness)
        else:
            state = self.tell(state, fitness, None)

        state = state.replace(training_ys=fitness, iter=state.iter + 1)
        return state

    def ask(self, state: State):
        """
        Ask the algorithm for points to explore.
        Returns different populations distributed to the real function and the surrogate model according to specified arguments.
        Control the training samples of the surrogate model.

        Parameters
        ----------
        state
            The state of this algorithm.

        Returns
        -------
        ask_real
            The exploring points to be evaluated by the real function.
        ask_surrogate
            The exploring points to be evaluated by the surrogate model.
        state
            The new state of the algorithm.
        """
        ask_real = jnp.zeros((0, state.population.shape[1]))
        ask_surrogate = jnp.zeros((0, state.population.shape[1]))
        ask_base, state = self.base_algorithm.ask(state)

        if (not has_init_ask(self.base_algorithm)) and state.iter == 1:
            state = state.replace(training_xs=ask_base)
            return ask_base, None, state

        if state.iter % self.retrain_per_iter == 0:
            state = state.replace(
                training_xs=jnp.zeros((0, state.population.shape[1])),
            )
            ask_real = ask_base
            ask_surrogate = None
        else:
            if state.iter % self.retrain_per_iter <= self.begin_training_iter:
                state = state.replace(
                    training_xs=jnp.concatenate([state.training_xs, ask_base])
                )
                ask_real = ask_base
                ask_surrogate = None
            else:
                n_real = int(self.real_eval_portion * ask_base.shape[0])
                ask_real = ask_base[:n_real]
                ask_surrogate = ask_base[n_real:]
        return ask_real, ask_surrogate, state

    def tell(
        self, state: State, fitness_real: jax.Array, fitness_surrogate: jax.Array
    ) -> State:
        """Tell the algorithm more information.
        Tell the algorithm about the points evaluated by real function and surrogate model.
        Control the training targets of the surrogate model.

        Parameters
        ----------
        state
            The state of this algorithm.
        fitness_real
            The fitness evaluated by real function.
        fitness_surrogate
            The fitness evaluated by surrogate model.

        Returns
        -------
        state
            The new state of the algorithm.
        """
        if (not has_init_tell(self.base_algorithm)) and state.iter == 1:
            state = state.replace(training_ys=fitness_real, iter=state.iter + 1)
            state = self.base_algorithm.tell(state, fitness_real)
            return state

        if fitness_surrogate is not None:
            fitness = jnp.concatenate([fitness_real, fitness_surrogate])
        else:
            fitness = fitness_real
        state = self.base_algorithm.tell(state, fitness)
        if state.iter % self.retrain_per_iter == 0:
            state = state.replace(
                training_ys=jnp.zeros(
                    (0, fitness.shape[1]) if fitness.ndim > 1 else (0,)
                ),
                iter=state.iter + 1,
            )
        else:
            if state.iter % self.retrain_per_iter <= self.begin_training_iter:
                state = state.replace(
                    training_ys=jnp.concatenate([state.training_ys, fitness_real]),
                    iter=state.iter + 1,
                )
            else:
                state = state.replace(iter=state.iter + 1)
        return state

    def get_surrogate_samples(
        self, state: State
    ) -> Tuple[jax.Array | None, jax.Array | None, State]:
        """
        Obtain the training samples (or None) to train a new surrogate model.
        Returns `None` if the surrogate model is not needed to be trained.

        Parameters
        ----------
        state
            The state of the algorithm.

        Returns
        -------
        samples_pop
            The training samples (or None) to train a new surrogate model.
        samples_fit
            The training samples (or None) to train a new surrogate model.
        state
            The new state of the algorithm.
        """
        if state.iter % self.retrain_per_iter == self.begin_training_iter:
            return (
                state.training_xs[: state.training_ys.shape[0]],
                state.training_ys,
                state,
            )
        else:
            return None, None, state
