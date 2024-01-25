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
    for i in range(ITER):
        state = workflow.step(state)
        obj = state.get_child_state("algorithm").fitness
        print(ind(obj))

    # objs = monitor.get_last()
    # print(objs)

def test():
    # 假设 matrix 是一个 N x M 的矩阵
    matrix = jnp.array([[1, 2, 3], [4, 0, 6], [7, 8, 9], [10, 11, 12]])
    # 假设 mask 是一个长度为 N 的布尔数组
    mask = jnp.array([False, True, True, False])

    # 获取 mask 为 True 的行的索引
    true_indices = jnp.where(mask)[0]

    # 计算 mask 为 True 的行的数量
    true_count = jnp.sum(mask)

    # 初始化一个与 matrix 相同形状的新矩阵，用于存放 mask 为 False 的行
    # 这里我们可以使用任意内容填充，因为我们只关心前 A 行
    false_rows = jnp.zeros((mask.size - true_count, matrix.shape[1]), dtype=matrix.dtype)

    # 使用 PRNGKey 生成随机数
    key = jax.random.PRNGKey(0)

    # 随机选择 mask 为 True 的行的索引
    sampled_true_indices = jnp.random.choice(true_indices, size=2, replace=False)

    # 使用这些索引来选择 mask 为 True 的行的内容，并填充到 false_rows 中
    # 这里我们使用一个简单的随机填充策略：从 0 到 9 之间随机选择一个数字
    random_values = jnp.random.randint(0, 10, size=(2,))
    false_rows = false_rows.at[:, jnp.newaxis].set(random_values)

    # 使用这些索引来选择 matrix 中的相应行，并填充到 false_rows 中
    false_rows = jnp.take_along_axis(matrix, sampled_true_indices[:, jnp.newaxis], axis=0)

    # 将 false_rows 和 original_matrix 拼接在一起，得到最终的矩阵
    result_matrix = jnp.concatenate((matrix, false_rows), axis=0)

    return result_matrix


if __name__ == "__main__":
    # rm_meda()
    # nsga2()
    # print("done")
    test()