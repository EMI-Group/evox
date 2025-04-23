# 7. 实战案例

本章将通过几个完整的实战案例，来演示如何将前面章节所学应用于具体场景。我们将从零开始搭建一个优化项目，并展示如何将 EvoX 和其他工具结合使用。这些案例涵盖不同类型的问题，帮助您举一反三，在真实需求中使用 EvoX。

---

## 案例一：单目标优化

**问题描述**：优化一个经典的测试函数——[Rastrigin 函数](https://www.sfu.ca/~ssurjano/rastr.html)。该函数具有大量局部极值点，是测试算法全局优化能力的常用基准。定义如下：

```{math}
f(\mathbf{x}) = 10 d + \sum_{i=1}^{d}[x_i^2 - 10 \cos{(2\pi x_i)}],
```

其中$\mathbf{x} \in \mathbb{R}^d$, $d$为该函数的维数。Rastrigin 函数的全局最优值为$0$，出现在原点。为了清晰地展示该函数具有多个局部极值点的性质，我们绘制了二维 Rastrigin 函数的图像。

```{figure} /_static/rastrigin_function.svg
:alt: A plot of the Rastrigin function
:figwidth: 70%
:align: center

Rastrigin 函数
```

在本案例中，我们将使用粒子群优化（PSO）算法在十维 Rastrigin 函数上寻优。

**步骤1：配置环境**

假设您已经按照教程第2章配置好了 EvoX 运行环境（Python 环境、安装 EvoX 等）。该问题不需要特殊依赖。

**步骤2：搭建流程**

编写 Python 脚本`opt_rastrigin_10.py`，内容如下：

```python
# 导入必要的模块
import torch
from evox.algorithms.so.pso_variants import PSO
from evox.problems.numerical.basic import Rastrigin
from evox.workflows import StdWorkflow, EvalMonitor
```

首先，我们将定义需要使用的算法（Algorithm）PSO。

```python
# 定义问题维度
dim = 10

# 初始化算法和问题
algo = PSO(
    pop_size = 50,
    lb = -32 * torch.ones(dim),
    ub = 32 * torch.ones(dim)
)
```

在实例化一个 PSO 算法类的时候，我们需要明确其参数的含义：

- `pop_size`：粒子种群的大小。
- `lb`和`ub`：搜索空间的下界和上界。
- 其他参数均为默认值，请查阅具体的API。

由于各种常见的测试函数（包括 Ackley 函数、Rosenbrock 函数等）在 EvoX 中均已实现，在这里我们仅需直接调用 Rastrigin 函数作为问题（Problem）即可。当然，如果您想实现其他的测试函数，请您参考本教程的第五章“自定义问题（Problem）”部分的内容。同时，我们将创建一个监视器（EvalMonitor），并使用算法，问题和监视器共同创建一个工作流（StdWorkflow）。

```python
# 定义问题
prob = Rastrigin()

# 定义监视器
monitor = EvalMonitor()

# 定义工作流
workflow = StdWorkflow(
    algorithm = algo,
    problem = prob,
    monitor = monitor
)
```

**步骤3：算法迭代**

接下来，我们需要通过`init_step()`和`step()`函数来进行算法的迭代。在迭代结束后，可以通过`get_best_solution()`和`get_best_fitness()`函数来分别获取最优解和其对应的适应度（在本案例中指函数值）。

```python
workflow.init_step()
for iter in range(501):
    workflow.step()
    if iter % 100 == 0:
        current_best_fitness = monitor.get_best_fitness().item()
        print(f"Iter {iter}, Best Fitness: {current_best_fitness}")

print(f"Final Best Solution: {monitor.get_best_solution()}")
```

代码的运行结果如下：

```
Iter 0, Best Fitness: 1398.625
Iter 100, Best Fitness: 11.608497619628906
Iter 200, Best Fitness: 2.5700759887695312
Iter 300, Best Fitness: 1.9909820556640625
Iter 400, Best Fitness: 1.9899139404296875
Iter 500, Best Fitness: 0.9976348876953125
Final Best Solution: tensor([-6.8931e-04, -2.0245e-04,  7.1968e-04,  1.9589e-04, -1.0042e-03,
         1.2888e-04, -2.6531e-03, -9.9485e-01,  2.0368e-03,  4.3372e-04])
```

我们可以看到，使用 PSO 算法可以将十维 Rastrigin 函数的值优化至`0.99`左右，最优值在原点附近，这正是我们想看到的结果！

---

## 案例二：多目标优化

**问题描述**：考虑一个简单的双目标优化问题：同时最小化两个目标函数$f_1(x)$和$f_2(x)$，它们分别是两个单峰函数：

```{math}
f_1(x) = x^2,\\
f_2(x) = (x-2)^2.
```

这里$x$是决策变量，令其取值范围为$[-5, 5]$。这实际上是一个很简单的优化，有一个帕累托最优解集在两个目标之间（当$x=0$完全最优$f_1$，当$x=2$完全最优$f_2$，中间权衡）。我们将使用经典的多目标演化算法 NSGA-II 来求解它的帕累托前沿。

**步骤1：配置环境**

需要确保安装了 EvoX 的多目标算法支持，默认安装已包含 NSGA-II 算法。

**步骤2：实现自定义问题**

EvoX 内置很多多目标测试问题，但在这个案例中，我们需要参考第五章内容，自定义一个 Problem，来优化本案例中的双目标问题：

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

from evox.algorithms import NSGA2
from evox.workflows import StdWorkflow, EvalMonitor
# 导入EvoX的核心类，具体请查阅本教程第五章
from evox.core import Problem

class TwoObjectiveProblem(Problem):
    def __init__(
        self,
        d: int = 1,
        m: int = 2,
    ):
        super().__init__()
        self.d = d
        self.m = m

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        x = X[:, 0]
        f_1 = x ** 2
        f_2 = (x - 2) ** 2
        return torch.stack([f_1, f_2], dim=1)

    # 可以定义pf函数，返回真实的Pareto前沿
    def pf(self) -> torch.Tensor:
        pass
```

**步骤3：搭建流程**

类似与案例一，我们会先后定义算法（Algorithm），问题（Problem）和监视器（EvalMonitor），并将其组成工作流（StdWorkflow）。

```python
prob = TwoObjectiveProblem()

torch.set_default_device("cuda:0")

algo = NSGA2(
    pop_size=50,
    n_objs=2,
    lb=-5 * torch.ones(1),
    ub=5 * torch.ones(1),
    device=torch.device("cuda"),
)

monitor = EvalMonitor()
workflow = StdWorkflow(algo, prob, monitor)
```

**步骤4：算法迭代**

接下来，我们需要通过`init_step()`和`step()`函数来进行算法的迭代。我们可以可视化经过NSGA-II算法优化后的结果，代码如下：

```python
workflow.init_step()
for i in range(100):
    workflow.step()

data = algo.fit.cpu().numpy()

def f1(x):
    return x ** 2

def f2(x):
    return (x - 2) ** 2

f1_values = data[:, 0]
f2_values = data[:, 1]

x_vals = np.linspace(0, 2, 400)
pf_f1 = f1(x_vals)
pf_f2 = f2(x_vals)
# 或者利用等价关系计算：pf_f2 = 4 - 4*np.sqrt(pf_f1) + pf_f1

# 绘制图形
plt.figure(figsize=(8, 6))
plt.scatter(f1_values, f2_values, c='blue', alpha=0.7)
plt.plot(pf_f1, pf_f2, 'r-', linewidth=2)
plt.xlabel("f1")
plt.ylabel("f2")
plt.legend()
plt.grid(True)
plt.show()
```

我们可以得到多次迭代后的种群分布情况，实验结果同样符合我们的预期：

```{figure} /_static/example_nsga2_result.svg
:alt: A plot of the NSGA-II population
:figwidth: 70%
:align: center

NSGA-II算法优化后的种群分布
```

同时，在 JupyterNotebook 中，您还可以通过 EvoX 的可视化模块直接得到动态的实验结果，您可以直观地看到种群是如何随着算法的迭代更新的。为了实现这一目标，您只需要运行一行代码：

```python
monitor.plot()
```

---

## 案例三：超参数优化

**问题描述**：在机器学习领域，算法的超参数对于模型的性能具有重要影响，然而，如何选择合适的超参数往往需要借助经验以及多次手动尝试。如今我们可以通过 EvoX 来简化这一流程。我们以逻辑回归(Logistic Regression)模型在乳腺癌数据集上的分类准确率为例，优化两个超参数：正则化强度 `C`（或正则项系数的倒数）和训练迭代次数 `max_iter`。我们的目标是最大化验证集准确率。

**步骤1：数据与模型准备**

使用`scikit-learn`加载乳腺癌数据集，并设定一个逻辑回归模型。由于 EvoX 优化目标默认是**最小化**，我们可以优化**错误率**来等价于最大化准确率。或者在 Problem 返回的时候取负的准确率作为需要最小化的值。这里选择优化验证集错误率（$1 - \text{Accuracy}$）。

**步骤2：定义问题**

问题（Problem）的输入为两个超参数 `[C, max_iter]`，我们在 evaluate 中训练模型并评估：

```python
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from evox.core import Problem
from evox.algorithms.so.es_variants import CMAES
from evox.workflows import EvalMonitor, StdWorkflow
from evox.problems.numerical.basic import Ackley

# 准备数据
data = load_breast_cancer()
X_all, y_all = data.data, data.target
# 划分训练集验证集
X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
# 标准化
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)

class HyperParamOptProblem(Problem):
    def __init__(self):
        super().__init__()
    def evaluate(self, pop):
        # params: (N,2)张量
        # 因为sklearn不是GPU计算，这里将tensor转回numpy
        pop = pop.detach().cpu().numpy()
        objs = []
        for C_val, max_iter_val in pop:
            # 注意：C是正则强度倒数，取值范围正数，max_iter范围正整数
            # 需要对候选值进行合法变换，例如C用10**x范围变换更合理，这里简单处理
            C_val = float(max(1e-3, C_val))  # 确保为正
            max_iter_val = int(max(50, max_iter_val))  # 至少50次迭代
            # 训练模型
            model = LogisticRegression(C=C_val, max_iter=max_iter_val, solver='liblinear')
            model.fit(X_train, y_train)
            acc = model.score(X_val, y_val)  # 验证集准确率
            error_rate = 1 - acc            # 错误率作为优化目标
            objs.append(error_rate)
        objs = torch.tensor(objs)
        return objs
```

**步骤3：搭建流程**

我们使用CMA-ES算法来优化超参数。这里我们初始化CMA-ES的种群均值为[1.0, 100]，即初始认为`C=1`, `max_iter=100`。`sigma=1.0`是初始步长。这个问题两个决策一个取正实数，一个取正整数，但我们暂且当作连续优化，让算法自行探索。不过在 evaluate 中我们做了截断，保证含义有效。

```python
prob = HyperParamOptProblem()

init_params = torch.tensor([1.0, 100.0])
init_error_rate = prob.evaluate(init_params.unsqueeze(0)).item()
print(f"模型初始错误率为：{init_error_rate}")
algo = CMAES(
    mean_init=init_params,
    sigma=1.0,
)
monitor = EvalMonitor()
workflow = StdWorkflow(algo, prob, monitor=monitor)
```

我们可以看到，使用初始状态的超参数会带来较高的错误率：

```
模型初始错误率为：0.02631578966975212
```

**步骤4：算法迭代**

```python
workflow.init_step()
for _ in range(100):
    workflow.step()

optimized_params = monitor.get_best_solution()
optimized_error_rate = prob.evaluate(optimized_params.unsqueeze(0)).item()
print(f"使用优化后的超参数的模型错误率为：{optimized_error_rate}")
```

经过100次迭代后，我们可以获得更优异的超参数配置，代码运行结果如下：

```
使用优化后的超参数的模型错误率为：0.008771929889917374
```

我们可以看到，只需要灵活地根据需求定义问题，就可以使用EvoX使繁琐的调参过程完全自动化！

---

以下实用示例展示了 EvoX 如何在各个领域中高效应用，从数学测试函数到机器学习流程。一旦你熟悉了基本结构 —— **算法 + 问题 + 监控器 + 工作流** —— 就可以将 EvoX 灵活地适用于几乎任何优化任务。
