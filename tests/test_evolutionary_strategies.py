import jax
from jax import jit, vmap, numpy as jnp
from flax import linen as nn

from src import evox
from src.evox import algorithms
from src.evox.problems.rl.gym_no_distribution import Gym
from src.evox.utils import TreeAndVector


class ClassicPolicy(nn.Module):
    """A simple model for Classic Control problem"""

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(16)(x)
        x = nn.relu(x)
        x = nn.Dense(2)(x)

        return jnp.argmax(x)


def cartpole(seed, al: type, **kwargs):
    key = jax.random.PRNGKey(seed)
    model_key, pipeline_key = jax.random.split(key)

    model = ClassicPolicy()
    params = model.init(model_key, jnp.zeros((4,)))
    adapter = TreeAndVector(params)
    center = adapter.to_vector(params)

    monitor = evox.monitors.StdSOMonitor()

    problem = Gym(
        policy=jit(vmap(model.apply)),
        env_name="CartPole-v1",
        env_options={"new_step_api": True},
        pop_size=64,
    )

    # create a pipeline
    pipeline = evox.pipelines.StdPipeline(
        algorithm=al(
            center_init=center,
            pop_size=64,
            **kwargs,
        ),
        problem=problem,
        pop_transform=adapter.batched_to_tree,
        fitness_transform=monitor.record_fit,
    )
    # init the pipeline
    state = pipeline.init(key)

    # run the pipeline for 10 steps
    for i in range(50):
        state = pipeline.step(state)
        if monitor.get_min_fitness() == -500:
            break
        # print(monitor.get_min_fitness())

    # obtain 500
    min_fitness = monitor.get_min_fitness()
    # cartpole is simple. expect to obtain max score(500) in each algorithm
    return min_fitness == -500


def test_cma_es_0():
    assert cartpole(0, algorithms.CMAES, init_stdev=1)


def test_cma_es_1():
    assert cartpole(1, algorithms.CMAES, init_stdev=1)


def test_cma_es_2():
    assert cartpole(2, algorithms.CMAES, init_stdev=1)


def test_pgpe_0():
    assert cartpole(0, algorithms.PGPE, optimizer="adam")


def test_pgpe_1():
    assert cartpole(1, algorithms.PGPE, optimizer="adam")


def test_pgpe_2():
    assert cartpole(2, algorithms.PGPE, optimizer="adam")


def test_open_es_0():
    assert cartpole(0, algorithms.OpenES, learning_rate=0.2, noise_stdev=0.2, mirrored_sampling=True)


def test_open_es_1():
    assert cartpole(1, algorithms.OpenES, learning_rate=0.2, noise_stdev=0.2, mirrored_sampling=True)


def test_open_es_2():
    assert cartpole(2, algorithms.OpenES, learning_rate=0.2, noise_stdev=0.2, mirrored_sampling=True)


def test_ars_0():
    assert cartpole(0, algorithms.ARS)


def test_ars_1():
    assert cartpole(1, algorithms.ARS)


def test_ars_2():
    assert cartpole(2, algorithms.ARS)
    
def test_sepcma_0():
    assert cartpole(0, algorithms.SepCMAES,init_stdev=1)

def test_sepcma_1():
    assert cartpole(1, algorithms.SepCMAES,init_stdev=1)

def test_sepcma_2():
    assert cartpole(2, algorithms.SepCMAES,init_stdev=1)

def test_ipopcma_0():
    assert cartpole(0, algorithms.IPOPCMAES,init_stdev=1)

def test_ipopcma_1():     
    assert cartpole(1, algorithms.IPOPCMAES,init_stdev=1)

def test_ipopcma_2():          
    assert cartpole(2, algorithms.IPOPCMAES,init_stdev=1)

def test_bipopcma_0():               
    assert cartpole(0, algorithms.BIPOPCMAES,init_stdev=1)

def test_bipopcma_1():                    
    assert cartpole(1, algorithms.BIPOPCMAES,init_stdev=1)

def test_bipopcma_2():                         
    assert cartpole(2, algorithms.BIPOPCMAES,init_stdev=1)

def test_amalgam_0():
    assert cartpole(0, algorithms.AMaLGaM,init_stdev=1)

def test_amalgam_1():     
    assert cartpole(1, algorithms.AMaLGaM,init_stdev=1)

def test_amalgam_2():          
    assert cartpole(2, algorithms.AMaLGaM,init_stdev=1)

def test_independent_amalgam_0():               
    assert cartpole(0, algorithms.IndependentAMaLGaM,init_stdev=1) 

def test_independent_amalgam_1():
    assert cartpole(1, algorithms.IndependentAMaLGaM,init_stdev=1)

def test_independent_amalgam_2():
    assert cartpole(2, algorithms.IndependentAMaLGaM,init_stdev=1)     
