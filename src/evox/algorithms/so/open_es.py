import jax
import jax.numpy as jnp
import optax

import evox


@evox.jit_class
class OpenES(evox.Algorithm):
    def __init__(
        self, init_params, pop_size, learning_rate, noise_std, optimizer="sgd", mirrored_sampling=True, utility=True, l2coeff=None, lr_decay=None
    ):
        """
        Implement the algorithm described in "Evolution Strategies as a Scalable Alternative to Reinforcement Learning"
        from https://arxiv.org/abs/1703.03864
        """
        assert noise_stdev > 0
        assert learning_rate > 0
        assert pop_size > 0

        if mirrored_sampling is True:
            assert (
                pop_size % 2 == 0
            ), "When mirrored_sampling is True, pop_size must be a multiple of 2."

        self.dim = init_params.shape[0]
        self.init_params = init_params
        self.pop_size = pop_size
        self.learning_rate = learning_rate
        self.noise_stdev = noise_stdev
        self.mirrored_sampling = mirrored_sampling
        self.utility = utility
        self.lr_decay = lr_decay
        self.l2coeff = l2coeff


        if optimizer == "adam":
            self.optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=l2coeff)
        elif optimizer == "sgd":
            self.optimizer = optax.sgd(learning_rate=learning_rate)
        else:
            raise TypeError(f"{optimizer} is not supported right now")

        if self.utility:
            rank = jnp.arange(1, pop_size + 1)
            util_ = jnp.maximum(0, jnp.log(pop_size / 2 + 1) - jnp.log(rank))
            utility = util_ / util_.sum() - 1 / pop_size
            self.utility_score = utility

        if optimizer == "adam":
            self.optimizer = evox.utils.OptaxWrapper(
                optax.adam(learning_rate=learning_rate), init_params
            )
        else:
            self.optimizer = None

    def setup(self, key):
        population = jnp.tile(self.init_params, (self.pop_size, 1))
        noise = jnp.tile(self.init_params, (self.pop_size, 1))

        opt_state = self.optimizer.init(self.init_params)

        return ex.State(
            population=population, params=self.init_params, noise=noise, key=key, opt_state=opt_state
        )

    def ask(self, state):
        key, noise_key = jax.random.split(state.key)
        if self.mirrored_sampling:
            noise = jax.random.normal(noise_key, shape=(self.pop_size // 2, self.dim))
            noise = jnp.concatenate([noise, -noise], axis=0)
        else:
            noise = jax.random.normal(noise_key, shape=(self.pop_size, self.dim))
        population = state.center[jnp.newaxis, :] + self.noise_stdev * noise

        return (
            state.update(population=population, key=key, noise=noise),
            population,
        )

    def tell(self, state, fitness):
        if self.utility:
            cumulative_update = jnp.zeros_like(state.params)
            fitness_rank = jnp.argsort(fitness)[::-1]
            for ui, kid in enumerate(fitness_rank):
                cumulative_update += self.utility_score[ui] * state.noise[kid]
            cumulative_update /= self.pop_size * self.noise_std
        else:
            cumulative_update = jnp.mean(fitness[:, jnp.newaxis] * state.noise, axis=0) / self.noise_std

        updates, opt_state = self.optimizer.update(cumulative_update, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        return state.update(params=params, opt_state=opt_state)
