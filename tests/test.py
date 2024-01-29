from evox import workflows, algorithms, problems
from evox.monitors import StdMOMonitor
from evox.metrics import IGD
import jax
import jax.numpy as jnp
from evox.operators.gaussian_process.regression import GPRegression
import gpjax as gpx
from gpjax.kernels import Linear
from gpjax.likelihoods import Gaussian
from gpjax.mean_functions import Zero
import optax as ox
import tqdm
import logging
# 设置JAX的日志级别
logging.disable(logging.WARNING)
def tqdm_replacement(*args,**kwargs):
    if args:
        return args[0]  # 返回第一个位置参数，即迭代对象
    return None
tqdm_copy = tqdm.tqdm # store it if you want to use it later
tqdm.tqdm = tqdm_replacement

# import any other module you need after this line
N = 12
M = 3
POP_SIZE = 100
LB = 0
UB = 1
ITER = 100

def nsga2():
    algorithm = algorithms.NSGA2(
        lb=jnp.full(shape=(N,), fill_value=LB),
        ub=jnp.full(shape=(N,), fill_value=UB),
        n_objs=M,
        pop_size=POP_SIZE,
    )
    run_moea(algorithm)

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

def run_moea(algorithm, problem=problems.numerical.DTLZ2(m=M)):
    key = jax.random.PRNGKey(42)
    monitor = StdMOMonitor(record_pf=False)
    workflow = workflows.StdWorkflow(
        algorithm=algorithm,
        problem=problem,
        monitors=[monitor],
    )
    state = workflow.init(key)
    true_pf, state = problem.pf(state)
    ind = IGD(true_pf)
    with open("./log.txt",'w') as file:
        for i in range(ITER):
            state = workflow.step(state)
            obj = state.get_child_state("algorithm").fitness
            file.write(str(ind(obj))+"\n")
            # print(ind(obj))

    # objs = monitor.get_last()
    # print(objs)
@jax.jit
def run_gp(i):
    # 假设 matrix 是一个 N x M 的矩阵
    x = jnp.arange(i + 5)[:, jnp.newaxis]
    pre_x = jnp.array([4, 5, 6, jnp.inf])[:, jnp.newaxis]
    y = (jnp.arange(i + 5) * 6)[:, jnp.newaxis]
    # 假设 mask 是一个长度为 N 的布尔数组
    likelihood = Gaussian(num_datapoints=len(x))
    model = GPRegression(likelihood=likelihood)
    model.fit(x, y, optimzer=ox.sgd(0.001,  nesterov=True))
    _, mu, std = model.predict(pre_x)
    return mu

def f():
    # 这里只是一个示例，您可以根据实际需求定义 f
    x = jnp.array([1, 2, 3, 12])[:, jnp.newaxis]
    pre_x = jnp.array([4, 5, 6, jnp.inf])[:, jnp.newaxis]
    y = jnp.array([4, 5, 6, 13])[:, jnp.newaxis]
    mus = jax.vmap(lambda i:run_gp(i,x,y,pre_x))(jnp.arange(3))

def rr():

    # 假设 arr 是一个二维数组
    arr = jnp.array([[1, 2, 3], [4, 5, 6], [1, 2, 3], [4, 5, 6]])

    # 使用 unique 按行筛选
    unique_rows = jnp.unique(arr, axis=0,size=2)

    # result2 = vmap_f(jnp.array([0, 1, 2]), out_axes=1)
    print(unique_rows.shape)
    # print(result2.shape)
    # print(jnp.vstack(result).shape)
    # print(b)
if __name__ == "__main__":
    # rm_meda()
    im_moea()
    # nsga2()
    # print("done")
    # rr()
    # run_gp()
    # f()