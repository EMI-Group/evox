from evox import workflows, algorithms, problems
from evox.monitors import StdMOMonitor
from evox.metrics import IGD
import jax
import jax.numpy as jnp


N = 12
M = 3
POP_SIZE = 100
LB = 0
UB = 1
ITER = 10


def rm_meda():
    algorithm = algorithms.RMMEDA(
        lb=jnp.full(shape=(N,), fill_value=LB),
        ub=jnp.full(shape=(N,), fill_value=UB),
        n_objs=M,
        pop_size=POP_SIZE,
    )
    run_moea(algorithm)

def im_moea():
    algorithm = algorithms.IMMOEA(
        lb=jnp.full(shape=(N,), fill_value=LB),
        ub=jnp.full(shape=(N,), fill_value=UB),
        n_objs=M,
        pop_size=POP_SIZE,
    )
    run_moea(algorithm)
def run_moea(algorithm, problem=problems.numerical.DTLZ1(m=M)):
    key = jax.random.PRNGKey(42)
    monitor = StdMOMonitor(record_pf=False)
    workflow = workflows.StdWorkflow(
        algorithm=algorithm,
        problem=problem,
        monitors=[monitor],
    )
    state = workflow.init(key)
    true_pf, state = problem.pf(state)

    for i in range(ITER):
        state = workflow.step(state)
        # obj = state.get_child_state("algorithm").fitness
        # print(obj)

    objs = monitor.get_last()
    print(objs)

if __name__ == "__main__":
    rm_meda()
    im_moea()
    print("done")