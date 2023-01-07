from evox import pipelines, algorithms, problems
from evox.monitors import FitnessMonitor
from evox.utils import TreeAndVector
import jax
import jax.numpy as jnp
from flax import linen as nn

# change here only
gym_name = "Pendulum-v1"

def tanh2(x):
    return 2*nn.tanh(x)

classic_params = {
    "Acrobot-v1":(3, (6,), jnp.argmax), 
    "CartPole-v1":(2, (4,), jnp.argmax),
    "MountainCarContinuous-v0":(1, (2,), nn.tanh),
    "MountainCar-v0":(3, (2,), jnp.argmax),
    "Pendulum-v1":(1, (3,), tanh2)
}

class ClassicPolicy(nn.Module):
    """A simple model for Classic Control problem"""

    @nn.compact
    def __call__(self, x):
        x = x.at[1].multiply(10)  # normalization
        x = nn.Dense(16)(x)
        x = nn.relu(x)
        x = nn.Dense(classic_params[gym_name][0])(x)

        return classic_params[gym_name][2](x)

key = jax.random.PRNGKey(42)
model_key, pipeline_key = jax.random.split(key)

model = ClassicPolicy()
params = model.init(model_key, jnp.zeros(classic_params[gym_name][1]))
adapter = TreeAndVector(params)
monitor = FitnessMonitor()
problem = problems.neuroevolution.Gym(
    env_name=gym_name,
    policy=jax.jit(model.apply),
    num_workers=16,
    env_per_worker=4,
    controller_options={
        "num_cpus": 0,
        "num_gpus": 0,
    },
    worker_options={"num_cpus": 1, "num_gpus": 1 / 16},
    batch_policy=False,
)
center = adapter.to_vector(params)
# create a pipeline
pipeline = pipelines.StdPipeline(
    algorithm=algorithms.CMAES(
        init_mean=center,
        init_stdev=1,
        pop_size=64
    ),
    problem=problem,
    pop_transform=adapter.batched_to_tree,
    fitness_transform=monitor.update,
)
# init the pipeline
state = pipeline.init(pipeline_key)
# run the pipeline for 100 steps
for i in range(100):
    print(monitor.get_min_fitness())
    state = pipeline.step(state)

state, sample_pop = pipeline.sample(state)
# problem._render(state.get_child_state("problem"), adapter.to_tree(sample_pop[0]))

min_fitness = monitor.get_min_fitness()
print(min_fitness)
print(monitor.history)
