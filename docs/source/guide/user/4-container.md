# Container Algorithms

Container algorithms are a special type of algorithms that work by containing other algorithms and cannot work on their own.
Container algorithms can be used to compose a series of normal algorithms together.

## Cooperative Coevolution

We offer Cooperative Coevolution (CC) framework for all algorithms.
Currently, there are two types of CC container in EvoX, {class}`evox.algorithms.Coevolution` and {class}`evox.algorithms.VectorizedCoevolution`.
The difference is that `VectorizedCoevolution` updates all sub-populations at the same time in each generation,
but `Coevolution` follows the traditional approach that updates a single sub-populations at each generation.

The difference can be illustrated as follows, where $pop1$ and $pop2$ are two sub-populations, and together they form the whole population.

Coevolution:
```
| $pop1_t$ | $pop2_t$ | -> | $pop1_{t+1}$ | $pop2_t$ | -> | $pop1_{t+1}$ | $pop2_{t+1}$ |
```

VectorizedCoevolution:
```
| $pop1_t$ | $pop2_t$ | -> | $pop1_{t+1}$ | $pop2_{t+1}$ |
```

Coevolution will update each sub-population one by one, so later sub-population can utilize the most up-to-date information from the previous sub-populations. Vectorized Coevolution on the other hand, updates all sub-populations at the same time, thus gives a more parallelized computation.
In conclusion `VectorizedCoevolution` is faster, but `Coevolution` could be better in terms of optimization result with a limited number of evaluations.


### Code Example

Coevolution with 2 CSO algorithms.

```python
monitor = EvalMonitor()
# create a base algorithms
base_algorithm = algorithms.CSO(
    lb=jnp.full(shape=(20,), fill_value=-32),
    ub=jnp.full(shape=(20,), fill_value=32),
    pop_size=100,
)
# here we use [base_algorithm] * 2 to create two copies of the base algorithm
base_algorithms = Stateful.stack([base_algorithm] * 2)

# apply the container algorithm
algorithm = algorithms.VectorizedCoevolution(
    base_algorithms=base_algorithms,
    dim=40,
    num_subpops=2,
    random_subpop=False,
)

# run the workflow as normal
workflow = workflows.StdWorkflow(
    algorithm,
    problem=problems.numerical.Ackley(),
    monitors=[monitor],
)
# init the workflow
key = jax.random.PRNGKey(42)
state = workflow.init(key)

for i in range(200):
    state = workflow.step(state)
```

In summary to apply the container algorithm, you need to:

1. Create a list of base algorithms.
2. Use `Stateful.stack` to stack the base algorithms.
3. Create a container algorithm with the stacked base algorithms.

Please notice that the coevolution implementation in EvoX is still under development, so there might be some rough edges, and does not reflect the latest research results in the coevolution field.