import evox as ex
import jax
import jax.numpy as jnp
from evox import Stateful

@ex.jit_class
class Normalizer(Stateful):
    def __init__(self):
        self.sum = 0
        self.sumOfSquares = 0
        self.count = 0

    def setup(self, key):
        return ex.State(sum=0, sumOfSquares=0, count=0)

    def normalize(self, state, x):
        newCount = state.count + 1
        newSum = state.sum + x
        newSumOfSquares = state.sumOfSquares + x ** 2
        state = state.update(count=newCount, sum=newSum, sumOfSquares=newSumOfSquares)
        
        state, mean = self.mean(state)
        state, std = self.std(state)
        return state, (x - mean) / std

    def mean(self, state):
        mean = state.sum / state.count
        return state.update(mean=mean), mean

    def std(self, state):
        return state, jnp.sqrt(jnp.maximum(state.sumOfSquares / state.count - state.mean ** 2, 1e-2))

    def normalize_obvs1(self, state, obvs):
        newCount = state.count + len(obvs)
        newSum = state.sum + jnp.sum(obvs, axis=0)
        newSumOFsquares = state.sumOfSquares + jnp.sum(obvs ** 2, axis=0)
        state = state.update(count=newCount, sum=newSum, sumOfSquares=newSumOFsquares)

        state, mean = self.mean(state)
        state, std = self.std(state)
        return state, (obvs - mean) / std

def main():
    normalizer = Normalizer()
    state = normalizer.init()
    print(state)
    mean, std = -100, 10

    key = jax.random.PRNGKey(0)

    for i in range(100):
        key, _ = jax.random.split(key)
        x = jax.random.normal(key, (376, 10)) * std + mean
        state, x_ = normalizer.normalize_obvs2(state, x)
        print(x, x_)

    state, mean = normalizer.mean(state)
    state, std = normalizer.std(state)

    print("mean", mean, "std", std)

if __name__ == '__main__':
    main()