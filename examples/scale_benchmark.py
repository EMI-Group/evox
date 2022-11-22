import evox
from evox import algorithms, problems, pipelines
import jax.numpy as jnp
import jax
import time
import json


num_rounds = 1


def run_benchmark(dim, pop_size):
    init_mean = jnp.zeros((dim,))
    algorithm = algorithms.CMA_ES(init_mean=init_mean, init_stdvar=1, pop_size=pop_size)
    problem = problems.classic.Sphere()
    pipeline = pipelines.StdPipeline(algorithm, problem)
    key = jax.random.PRNGKey(42)
    state = pipeline.init(key)

    # warm up
    for i in range(10):
        pipeline.step(state)
    # real benchmark
    start = time.perf_counter()
    for i in range(1000):
        state = pipeline.step(state)
    end = time.perf_counter()
    return end - start


dim_scale_result = {}
for dim in [10, 100, 1000, 10_000]:
    dim_scale_result[dim] = [run_benchmark(dim, 100) for i in range(num_rounds)]

with open("log/dim_scale_result.json", "w") as f:
    json.dump(dim_scale_result, f)

pop_scale_result = {}
for pop_size in [10, 100, 1000, 10_000, 100_000]:
    pop_scale_result[pop_size] = [run_benchmark(100, pop_size) for i in range(num_rounds)]

with open("log/pop_scale_result.json", "w") as f:
    json.dump(pop_scale_result, f)
