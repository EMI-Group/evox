from evox import algorithms, problems, pipelines
from evox.monitors import FitnessMonitor, PopulationMonitor
import jax
import jax.numpy as jnp
#import tqdm

EPOCH_NUM = 100
POP_SIZE = 100
SEED = 42

fit_monitor = FitnessMonitor()
pop_monitor = PopulationMonitor(2)

# create a pipeline
pipeline = pipelines.StdPipeline(
    algorithms.PSO(
        lb=jnp.full(shape=(2,), fill_value=-32),
        ub=jnp.full(shape=(2,), fill_value=32),
        pop_size=POP_SIZE,
    ),
    problems.classic.Rastrigin(),
    fitness_transform=fit_monitor.update,
    pop_transform=pop_monitor.update,
)

# init the pipeline
key = jax.random.PRNGKey(SEED)
state = pipeline.init(key)

# run the pipeline for Epoch_num steps
#pbar = tqdm(range(EPOCH_NUM))

#for i in pbar:
for i in range(EPOCH_NUM):
    # pbar.set_postfix_str(' Fitness = ' + str(monitor.get_min_fitness()))
    state = pipeline.step(state)
    print(fit_monitor.get_min_fitness())

pop_monitor.save('PSO_Rast_pop_information.html')
fit_monitor.save('PSO_Rast_fit_information.html')