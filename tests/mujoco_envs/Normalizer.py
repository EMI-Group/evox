import evox as ex
import jax
import jax.numpy as jnp
from evox import Stateful


def main():
    normalizer = Normalizer()
    state = normalizer.init()
    print(state)
    mean, std = -100, 10

    key = jax.random.PRNGKey(0)

    for i in range(100):
        key, _ = jax.random.split(key)
        x = jax.random.normal(key, (376, 10)) * std + mean
        x_, state = normalizer.normalize_obvs2(state, x)
        print(x, x_)

    mean, state = normalizer.mean(state)
    std, state = normalizer.std(state)

    print("mean", mean, "std", std)

if __name__ == '__main__':
    main()