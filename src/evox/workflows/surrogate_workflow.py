from typing import Optional, Union
from collections.abc import Callable, Sequence
from typing import Tuple
import jax
from flax import linen as nn
import optax
import jax.numpy as jnp
import jax.tree_util as jtu

from evox import (
    SurrogateAlgorithm,
    Problem,
    State,
    Workflow,
    Monitor,
    use_state,
    dataclass,
    pytree_field,
    has_init_ask,
    has_init_tell,
)
from evox.core.distributed import POP_AXIS_NAME, all_gather, get_process_id
from evox.utils import parse_opt_direction


@dataclass
class SurrogateWorkflowState:
    generation: int
    first_step: bool = pytree_field(static=True)
    rank: int
    world_size: int = pytree_field(static=True)


class SurrogateWorkflow(Workflow):
    """Experimental unified surrogate workflow,
    designed to provide surrogate model support for EC workflow.
    """

    def __init__(
        self,
        surrogate_model: nn.Module,
        optimizer: optax.GradientTransformation,
        surrogate_algorithm: SurrogateAlgorithm,
        problem: Problem,
        monitors: Sequence[Monitor] = (),
        opt_direction: Union[str, Sequence] = "min",
        candidate_transforms: Sequence[Callable[[jax.Array], jax.Array]] = (),
        fitness_transforms: Sequence[Callable[[jax.Array], jax.Array]] = (),
        external_problem: bool = False,
        num_objectives: Optional[int] = None,
        train_epochs: int = 100,
    ):
        """
        Parameters
        ----------
        surrogate_model
            The surrogate model used in the EC workflow.
        optimizer
            The optimizer used in the training process of the surrogate model provided by Optax.
        surrogate_algorithm
            The EC algorithm using surrogate model.
        problem
            The problem.
        monitor
            Optional monitor(s).
            Configure a single monitor or a list of monitors.
            The monitors will be called in the order of the list.
        opt_direction
            The optimization direction, can be either "min" or "max"
            or a list of "min"/"max" to specific the direction for each objective.
        candidate_transforms
            Optional candidate solution transform function,
            usually used to decode the candidate solution
            into the format that can be understood by the problem.
            Should be a list of functions,
            and the functions will be applied in the order of the list.
        fitness_transforms
            Optional fitness transform function.
            usually used to apply fitness shaping.
            Should be a list of functions,
            and the functions will be applied in the order of the list.
        external_problem
            Tell workflow whether the problem is external that cannot be jitted.
            Default to False.
        num_objectives
            Number of objectives. Used when external_problem=True.
            When the problem cannot be jitted, JAX cannot infer the shape, and
            this field should be manually set.
        train_epochs
            Number of the surrogate model training epochs.
        """
        self.surrogate_model = surrogate_model
        self.optimizer = optimizer
        self.surrogate_algorithm = surrogate_algorithm
        self.problem = problem
        self.monitors = monitors

        self.registered_hooks = {
            "pre_step": [],
            "pre_ask": [],
            "post_ask": [],
            "pre_eval": [],
            "post_eval": [],
            "pre_tell": [],
            "post_tell": [],
            "post_step": [],
        }

        for monitor in self.monitors:
            hooks = monitor.hooks()
            for hook in hooks:
                self.registered_hooks[hook].append(monitor)

        self.opt_direction = parse_opt_direction(opt_direction)
        for monitor in self.monitors:
            monitor.set_opt_direction(self.opt_direction)

        self.candidate_transforms = candidate_transforms
        self.fitness_transforms = fitness_transforms
        self.external_problem = external_problem
        self.num_objectives = num_objectives
        self.train_epochs = train_epochs
        self.params = None
        self.key = jax.random.PRNGKey(42)
        self.training_xs = jnp.array([])

        if self.num_objectives == 1:
            self.training_ys = jnp.array([])
        else:
            self.training_ys = jnp.array([]).reshape([0, self.num_objectives])

        if self.external_problem is True and self.num_objectives is None:
            raise ValueError(("Using external problem, but num_objectives isn't set."))

        def _ask(self, state):
            if (
                has_init_ask(self.surrogate_algorithm.base_algorithm)
                and state.first_step
            ):
                ask = self.surrogate_algorithm.init_ask
            else:
                ask = self.surrogate_algorithm.ask

            # candidate: individuals that need to be evaluated (may differ from population)
            # Note: num_cands can be different from init_ask() and ask()
            cands_real, cands_surrogate, state = use_state(ask)(state)

            return cands_real, cands_surrogate, state

        def _evaluate(self, state, transformed_cands):
            num_cands = jtu.tree_leaves(transformed_cands)[0].shape[0]

            # if the function is jitted
            if not self.external_problem:
                fitness, state = use_state(self.problem.evaluate)(
                    state, transformed_cands
                )
            else:
                if self.num_objectives == 1:
                    fit_shape = (num_cands,)
                else:
                    fit_shape = (num_cands, self.num_objectives)
                fitness, state = jax.pure_callback(
                    use_state(self.problem.evaluate),
                    (
                        jax.ShapeDtypeStruct(fit_shape, dtype=jnp.float32),
                        state,
                    ),
                    state,
                    transformed_cands,
                )

            fitness = all_gather(fitness, self.pmap_axis_name, axis=0, tiled=True)
            fitness = fitness * self.opt_direction

            return fitness, state

        def _train_surrogate_model(surrogate_model, optimizer, X, y, key, epochs):
            def train_step(
                surrogate_model, params, x, y, opt_state, loss_fn, optimizer
            ):
                def loss(params):
                    pred = surrogate_model.apply(params, x)
                    return loss_fn(y, pred)

                grad = jax.grad(loss)(params)
                updates, opt_state = optimizer.update(grad, opt_state)
                new_params = optax.apply_updates(params, updates)
                return new_params, opt_state

            def mse_loss(y_true, y_pred):
                return jnp.mean((y_true - y_pred) ** 2)

            params = surrogate_model.init(key, jnp.zeros((1, X.shape[1])))
            opt_state = optimizer.init(params)

            normalize = lambda X: (X - jnp.mean(X, axis=0)) / jnp.std(X, axis=0)
            X = normalize(X)
            print("======Begin training the surrogate model======")
            for epoch in range(epochs):
                params, opt_state = train_step(
                    surrogate_model, params, X, y, opt_state, mse_loss, optimizer
                )
                if epoch % 10 == 0:
                    pred = surrogate_model.apply(params, X)
                    loss_val = mse_loss(y, pred)
                    print(f"Epoch {epoch}, Loss: {loss_val}")
            print("============Training process ends=============")

            return params

        def _predict(surrogate_model, params, pop):
            return surrogate_model.apply(params, pop)

        def _evaluate_via_surrogate(
            self, state, transformed_cands, surrogate_model, params
        ):
            if transformed_cands is None or params is None:
                return None, state
            else:
                num_cands = jtu.tree_leaves(transformed_cands)[0].shape[0]

                if not self.external_problem:
                    fitness = _predict(surrogate_model, params, transformed_cands)
                else:
                    if self.num_objectives == 1:
                        fit_shape = (num_cands,)
                    else:
                        fit_shape = (num_cands, self.num_objectives)
                    fitness, state = jax.pure_callback(
                        use_state(self.problem.evaluate),
                        (
                            jax.ShapeDtypeStruct(fit_shape, dtype=jnp.float32),
                            state,
                        ),
                        state,
                        transformed_cands,
                    )

                fitness = all_gather(fitness, self.pmap_axis_name, axis=0, tiled=True)
                fitness = fitness * self.opt_direction

                return fitness, state

        def _tell(self, state, transformed_fitness_real, transformed_fitness_surrogate):
            if (
                has_init_tell(self.surrogate_algorithm.base_algorithm)
                and state.first_step
            ):
                tell = self.surrogate_algorithm.init_tell
            else:
                tell = self.surrogate_algorithm.tell

            state = use_state(tell)(
                state, transformed_fitness_real, transformed_fitness_surrogate
            )

            return state

        def _step(self, state: State) -> Tuple[dict, State]:
            for monitor in self.registered_hooks["pre_step"]:
                monitor.pre_step(state)

            for monitor in self.registered_hooks["pre_ask"]:
                monitor.pre_ask(state)

            cands_real, cands_surrogate, state = _ask(self, state)
            if cands_surrogate is not None:
                cands = jnp.concatenate([cands_real, cands_surrogate])
            else:
                cands = cands_real

            for monitor in self.registered_hooks["post_ask"]:
                monitor.post_ask(state, cands)

            num_cands = jtu.tree_leaves(cands)[0].shape[0]
            # in multi-device|host mode, each device only evaluates a slice of the population
            if num_cands % state.world_size != 0:
                raise ValueError(
                    f"#Candidates ({num_cands}) should be divisible by the number of devices ({state.world_size})"
                )
            # Note: slice_size is static
            slice_size = num_cands // state.world_size
            cands = jtu.tree_map(
                lambda x: jax.lax.dynamic_slice_in_dim(
                    x, state.rank * slice_size, slice_size, axis=0
                ),
                cands,
            )

            transformed_cands = cands
            transformed_cands_real = cands_real
            transformed_cands_surrogate = cands_surrogate

            for transform in self.candidate_transforms:
                transformed_cands = transform(transformed_cands)
                transformed_cands_real = transform(transformed_cands_real)
                transformed_cands_surrogate = transform(transformed_cands_surrogate)

            for monitor in self.registered_hooks["pre_eval"]:
                monitor.pre_eval(state, cands, transformed_cands)

            fitness_real, state = _evaluate(self, state, transformed_cands_real)

            get_surrogate_samples = self.surrogate_algorithm.get_surrogate_samples
            training_xs, training_ys, state = use_state(get_surrogate_samples)(state)
            if training_xs is not None and training_ys is not None:
                self.params = _train_surrogate_model(
                    self.surrogate_model,
                    self.optimizer,
                    training_xs,
                    training_ys,
                    self.key,
                    self.train_epochs,
                )

            fitness_surrogate, state = _evaluate_via_surrogate(
                self,
                state,
                transformed_cands_surrogate,
                self.surrogate_model,
                self.params,
            )

            if fitness_surrogate is not None:
                fitness = jnp.concatenate([fitness_real, fitness_surrogate])
            else:
                fitness = fitness_real

            for monitor in self.registered_hooks["post_eval"]:
                monitor.post_eval(state, cands, transformed_cands, fitness)

            transformed_fitness = fitness
            transformed_fitness_real = fitness_real
            transformed_fitness_surrogate = fitness_surrogate
            for transform in self.fitness_transforms:
                transformed_fitness = transform(transformed_fitness)
                transformed_fitness_real = transform(transformed_fitness_real)
                transformed_fitness_surrogate = transform(transformed_fitness_surrogate)

            for monitor in self.registered_hooks["pre_tell"]:
                monitor.pre_tell(
                    state, cands, transformed_cands, fitness, transformed_fitness
                )

            state = _tell(
                self, state, transformed_fitness_real, transformed_fitness_surrogate
            )

            for monitor in self.registered_hooks["post_tell"]:
                monitor.post_tell(state)

            train_info = dict(fitness=fitness, transformed_fitness=transformed_fitness)

            if has_init_ask(self.surrogate_algorithm) and state.first_step:
                # this ensures that _step() will be re-jitted
                state = state.replace(generation=state.generation + 1, first_step=False)
            else:
                state = state.replace(generation=state.generation + 1)

            for monitor in self.registered_hooks["post_step"]:
                monitor.post_step(state)

            return train_info, state

        self._step = _step

        # by default, use the first device
        self.devices = jax.local_devices()[:1]
        self.pmap_axis_name = None

    def setup(self, key) -> State:
        return State(
            SurrogateWorkflowState(generation=0, first_step=True, rank=0, world_size=1)
        )

    def step(self, state: State) -> Tuple[dict, State]:
        return self._step(self, state)
