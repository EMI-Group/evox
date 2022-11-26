import evox
from evox import algorithms, problems, pipelines
import jax.numpy as jnp
import jax
import time
import json


backend = jax.default_backend()
num_rounds = 1
dim_lists = jnp.arange(0, 4 + 1/4, 1/4)
dim_lists = jnp.round(10 ** dim_lists).astype(jnp.int32)
dim_lists = dim_lists.tolist()

pop_lists = jnp.arange(0, 5 + 1/4, 1/4)
pop_lists = jnp.round(10 ** pop_lists).astype(jnp.int32)
pop_lists = pop_lists.tolist()

print(dim_lists)
print(pop_lists)
print(f"backend: {backend}")

def run_benchmark(dim, pop_size):
    init_mean = jnp.zeros((dim,))
    algorithm = algorithms.CMA_ES(init_mean=init_mean, init_stdvar=1, pop_size=pop_size)
    problem = problems.classic.Sphere()
    pipeline = pipelines.StdPipeline(algorithm, problem)
    key = jax.random.PRNGKey(42)
    state = pipeline.init(key)

    # warm up
    for i in range(3):
        pipeline.step(state)
    # real benchmark
    start = time.perf_counter()
    for i in range(100):
        state = pipeline.step(state)
    end = time.perf_counter()
    return end - start


dim_scale_result = {}
for dim in dim_lists:
    print(dim)
    dim_scale_result[dim] = [run_benchmark(dim, 100) for i in range(num_rounds)]

with open(f"exp/{backend}_dim_scale_result.json", "w") as f:
    json.dump(dim_scale_result, f)

pop_scale_result = {}
for pop_size in pop_lists:
    print(pop_size)
    pop_scale_result[pop_size] = [run_benchmark(100, pop_size) for i in range(num_rounds)]

with open(f"exp/{backend}_pop_scale_result.json", "w") as f:
    json.dump(pop_scale_result, f)
