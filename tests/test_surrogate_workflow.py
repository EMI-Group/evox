from evox import workflows, algorithms, problems
from evox.monitors import StdSOMonitor
from evox.core.surrogate_algorithm import SurrogateAlgorithmWrapper
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

def run_surrogate_workflow_with_non_jit_problem():
    monitor = StdSOMonitor()
    #creat a surrogate model
    class MLP(nn.Module):
        @nn.compact
        def __call__(self, x: jax.Array):
            x = nn.Dense(features=128)(x)
            x = nn.relu(x)
            x = nn.Dense(features=64)(x)
            x = nn.relu(x)
            x = nn.Dense(features=1)(x)
            x = x.flatten()
            return x
    #creat a surrogate-based algorithm
    surrogate_algorithm = SurrogateAlgorithmWrapper(
        base_algorithm=algorithms.pso(
            lb=jnp.full(shape=(2,), fill_value=-32),
            ub=jnp.full(shape=(2,), fill_value=32),
            pop_size=20
        ),
        begin_training_iter=10,
        retrain_per_iter=20,
        real_eval_portion=0.5
    )
    #create a surrogate workflow
    workflow = workflows.SurrogateWorkflow(
        surrogate_model=MLP(),
        optimizer=optax.adam(learning_rate=0.01),
        surrogate_algorithm=surrogate_algorithm,
        problem=problems.numerical.Ackley(),
        monitors=[monitor],
        num_objectives=1
    )
    #init the surrogate workflow
    key = jax.random.PRNGKey(42)
    state = workflow.init(key)
    
    #run the workflow for 100 steps
    for i in range(100):
        _, state = workflow.step(state)
        
    monitor.close()
    min_fitness = monitor.get_best_fitness()
    assert min_fitness < 1e-2
    return min_fitness