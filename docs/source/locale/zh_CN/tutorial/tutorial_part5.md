# 5. 开发与扩展

&emsp;&emsp;EvoX 不仅提供了开箱即用的功能，还为开发者和高级用户提供了丰富的接口，支持自定义功能开发和扩展集成。本章将详细介绍如何编写自定义算法和问题，如何利用 EvoX 的 API 更加深入地控制流程，以及如何将 EvoX 与其他工具集成，构建更复杂的应用。

## 5.1 开发自定义功能

&emsp;&emsp;在某些情况下，您可能需要解决的问题或算法并未包含在 EvoX 的内置库中。此时，您可以通过 EvoX 提供的接口开发自定义模块。

### 5.1.1 自定义问题（MyProblem）

&emsp;&emsp;如果您需要优化的目标函数在 `evox.problems` 中没有现成实现，可以通过继承 EvoX 提供的基类（`evox.core.Problem`）或按照问题接口的要求实现必要的方法，来定义自己的问题类。在自定义问题之前，我们首先了解一下 EvoX 中 `Problem` 基类的结构。一个典型的问题类需要实现<u>评价函数</u>（`evaluate`），该函数的功能是接收一组解（`pop`）并返回对应的适应度/目标值。为了充分利用并行计算，EvoX 要求评价函数能够处理<u>种群批量输入</u>。

  ```python
  import torch
  from abc import ABC
  from typing import Any, Dict
  from evox.core.module import ModuleBase

  """Problem 基类"""
  class Problem(ModuleBase, ABC):
      def __init__(self):
          super().__init__()

      def evaluate(self, pop: torch.Tensor) -> torch.Tensor:
          """评估给定点上的适应度"""
          # 输入: 种群 (pop)/解
          # 输出: 适应度 (fitness)/目标值
          return torch.empty(0)
  ```

  &emsp;&emsp;例如，假设我们的目标是使决策向量各元素的立方和最小，即：

$$
\min {f(x)} = \sum_{i=1}^{n} x_i^3 \quad
$$

  &emsp;&emsp;对应在自定义的 **MyProblem** 类中，您需要实现 `evaluate` 方法，计算决策向量各元素的立方和并返回结果。以下是实现示例：

  ```python
  import torch
  from evox.core import Problem # 导入基类

  """自定义问题"""
  class MyProblem(Problem):
      def __init__(self):
          super().__init__()

      def evaluate(self, pop: torch.Tensor):
          # pop 是形状为 (种群大小, dim) 的张量
          fitness = torch.sum(pop**3, dim=1)
          # 返回适应度值
          return fitness
  ```

  &emsp;&emsp;在上述代码中，`pop` 是一个形状为 (种群大小, 维度) 的张量，表示待求解问题的一组解。`evaluate` 方法用于计算这些解的适应度值，并返回结果。为了兼容多目标优化场景（可能涉及多个目标函数），您可以通过返回一个字典来组织适应度值，其中每个键对应一个目标值。EvoX 中的问题定义能够灵活地支持不同类型的优化问题。

  &emsp;&emsp;假设您将自定义问题类存储在 `MyProblems.py` 文件中，导入 `MyProblem` 类后，您可以随机初始化一个种群，并直接调用 `MyProblem` 的 `evaluate` 方法计算种群的适应度值：

  ```python
import torch
from MyProblems import MyProblem

popsize = 10
dim = 2
initial_pop = torch.rand(popsize, dim)
problem = MyProblem()
initial_fitness = problem.evaluate(initial_pop)
  ```

  &emsp;&emsp;定义好问题类后，您也可以像使用内置问题一样使用它，EvoX 的工作流会自动调用您实现的 `evaluate` 方法来计算适应度并进行优化。

### 5.1.2 自定义算法（MyAlgorithm）

&emsp;&emsp;相较于自定义问题类，自定义算法的实现通常更为复杂，因为演化算法通常包含初始化、繁衍（生成新解）、选择（筛选解）等一系列过程。在自定义算法时，您需要逐一实现这些模块。在 EvoX 中，实现新算法通常需要继承 `evox.core.Algorithm` 基类，并至少实现以下方法：

- `__init__`：初始化算法所需的算子和参数。
- `step`：定义算法的优化迭代过程。

&emsp;&emsp;下面以粒子群优化算法（PSO）为例，展示如何在 EvoX 中定义算法。PSO 是一种基于种群的搜索算法，其核心思想是模拟鸟群或鱼群的社会行为，通过粒子之间的协作和信息共享逐步逼近最优解。算法中每个粒子根据自身的历史最优解和群体的历史最优解调整自己的位置和速度。下面是 EvoX 中实现的PSO算法类：

  ```python
  import torch
  from evox.core import Algorithm, Mutable, Parameter
  from evox.utils import clamp
  from evox.algorithms.so.pso_variants.utils import min_by

  class PSO(Algorithm): # 继承 `evox.core.Algorithm` 基类
     """
      粒子群优化算法（PSO）实现。

      参数：
      - pop_size: 种群大小
      - lb: 搜索空间下界
      - ub: 搜索空间上界
      - w: 惯性权重
      - phi_p: 个体学习因子
      - phi_g: 社会学习因子
      - device: 计算设备（如 CPU 或 GPU）
    """
    def __init__(
        self,
        pop_size: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        w: float = 0.6,
        phi_p: float = 2.5,
        phi_g: float = 0.8,
        device: torch.device | None = None,
    ):
        super().__init__()
        device = torch.get_default_device() if device is None else device
        assert lb.shape == ub.shape and lb.ndim == 1 and ub.ndim == 1 and lb.dtype == ub.dtype
        self.pop_size = pop_size
        self.dim = lb.shape[0]

        # 初始化搜索空间上下界(lb, ub)，粒子位置(pop)，粒子速度(velocity)
        lb = lb[None, :].to(device=device)
        ub = ub[None, :].to(device=device)
        length = ub - lb
        pop = torch.rand(self.pop_size, self.dim, device=device)
        pop = length * pop + lb
        velocity = torch.rand(self.pop_size, self.dim, device=device)
        velocity = 2 * length * velocity - length
        self.lb = lb
        self.ub = ub

        # 使用Parameter 包装超参数，以便跟踪和向量化
        self.w = Parameter(w, device=device)
        self.phi_p = Parameter(phi_p, device=device)
        self.phi_g = Parameter(phi_g, device=device)

        # 使用 Mutable 包装可变张量
        self.pop = Mutable(pop)
        self.velocity = Mutable(velocity)
        self.fit = Mutable(torch.full((self.pop_size,), torch.inf, device=device))
        self.local_best_location = Mutable(pop)
        self.local_best_fit = Mutable(torch.full((self.pop_size,), torch.inf, device=device))
        self.global_best_location = Mutable(pop[0])
        self.global_best_fit = Mutable(torch.tensor(torch.inf, device=device))

    def step(self):
        """执行一次优化迭代"""
        # 更新个体历史最优信息(local_best)和全局历史最优信息(global_best)
        compare = self.local_best_fit > self.fit
        self.local_best_location = torch.where(compare[:, None], self.pop, self.local_best_location)
        self.local_best_fit = torch.where(compare, self.fit, self.local_best_fit)
        self.global_best_location, self.global_best_fit = min_by(
            [self.global_best_location.unsqueeze(0), self.pop],
            [self.global_best_fit.unsqueeze(0), self.fit],
        )
        rg = torch.rand(self.pop_size, self.dim, device=self.fit.device)
        rp = torch.rand(self.pop_size, self.dim, device=self.fit.device)

        # 更新粒子速度(velocity)和位置(pop)
        velocity = (
            self.w * self.velocity
            + self.phi_p * rp * (self.local_best_location - self.pop)
            + self.phi_g * rg * (self.global_best_location - self.pop)
        )
        pop = self.pop + velocity
        self.pop = clamp(pop, self.lb, self.ub)
        self.velocity = clamp(velocity, self.lb, self.ub)
        self.fit = self.evaluate(self.pop)

    def init_step(self):
        """初始化状态"""
        self.fit = self.evaluate(self.pop)
        self.local_best_fit = self.fit
        self.global_best_fit = torch.min(self.fit)
  ```

&emsp;&emsp;以上算法实现主要包含了三个部分：

**（1）初始化**（`__init__`）

  - 定义了算法的核心参数，包括种群大小、搜索空间上下界、惯性权重、学习因子等。
  - 使用 `Parameter` 包装超参数，以便 EvoX 能够正确跟踪和向量化。
  - 使用 `Mutable` 包装可变张量（如粒子位置、速度、适应度值等），以便在算法迭代过程中更新。

**（2）优化迭代**（`step`）

  - 更新每个粒子的个体历史最优解和全局历史最优解。
  - 根据粒子群优化公式更新粒子的速度和位置。
  - 使用 `clamp` 函数确保粒子位置和速度在搜索空间范围内。

**（3）初始化状态**（`init_step`）

  - 在算法首次迭代前，初始化粒子的适应度值、个体历史最优解和全局历史最优解。

&emsp;&emsp;在完成算法实现后，您可以将其与前面定义的 `MyProblem` 结合，进行问题的求解。以下是一个完整的示例代码，展示了如何将自定义算法与问题集成到 EvoX 的工作流中：

   ```python
  import torch
  from MyProblems import MyProblem
  from evox.workflows import EvalMonitor, StdWorkflow
  from evox.algorithms import PSO

  problem = MyProblem()
  algorithm = PSO(
      pop_size=100,
      lb=torch.tensor([-10.0]),
      ub=torch.tensor([10.0])
      )
  monitor = EvalMonitor()
  workflow = StdWorkflow(algorithm, problem, monitor)
  for i in range(10):
      workflow.step()
   ```

&emsp;&emsp;在自定义算法时，请务必仔细验证其正确性，特别是 `step` 方法中算法逻辑的实现是否准确。此外，确保算法的超参数（如种群大小、学习因子等）设置合理，以获得良好的优化性能。通过以上方式，您可以灵活地定义和扩展 EvoX 的算法模块，从而满足特定的优化需求。

### 5.1.3 自定义其他模块

&emsp;&emsp;除去关键的算法和问题类，EvoX 的模块化设计允许用户灵活扩展 `Monitor`、`Operator` 等核心组件，以满足更复杂的功能需求或特定场景的优化目标。例如，您可以自定义一个 `MyMonitor`，在每代优化中不仅记录最优适应度值，还可以计算并记录种群的多样性指标。实现时只需继承 `evox.core.Monitor` 基类并重写相应方法即可。同样，您可以通过自定义 `MyOperator` 实现特定的交叉（`crossover`）或变异（`mutation`）功能，只需编写一个函数或类并在自定义算法中调用即可。在具体实现过程中，您可以参考前面自定义问题和算法的步骤：首先观察 EvoX 中对应模块的基类，了解需要重写的方法，然后逐步实现目标功能。

&emsp;&emsp;编写自定义功能需要对演化算法的基本原理有一定了解，同时熟悉 EvoX 已有模块的实现风格。建议您阅读 EvoX 源码或开发者文档，了解内部类的接口要求。在调试过程中，可以通过小种群规模和低维问题快速验证算法行为是否符合预期。此外，建议您充分利用 EvoX 的现有组件，尝试通过组合已有算子实现目标功能，而不是从零开始编写每一步。

```
  总之，EvoX 为高级用户提供了广阔的扩展空间，鼓励您根据实际需求创造新的功能模块，从而构建更强大的优化算法和应用。
```

## 5.2 API 使用

&emsp;&emsp;通过前面的内容，您已经初步接触了 EvoX 中的部分API。本节将系统性地梳理 EvoX 提供的关键 API，并介绍如何利用这些接口实现更复杂的控制逻辑。

### 5.2.1 算法与问题库 API

&emsp;&emsp;EvoX 将算法和问题按类别进行了模块化划分，便于用户快速调用和扩展。

**（1）算法模块**  (`evox.algorithms`)

&emsp;&emsp;EvoX 中的算法模块分为单目标优化 (`evox.algorithms.so`) 和多目标优化 (`evox.algorithms.mo`) 两类。例如，您可以通过以下方式调用单目标 PSO 算法和多目标 RVEA 算法：

```python
import torch
from evox.algorithms.so import PSO
from evox.algorithms.mo import RVEA

so_algorithm = PSO(
      pop_size=100,
      lb=torch.tensor([-10.0]),
      ub=torch.tensor([10.0])
    )

mo_algorithm = RVEA(
      pop_size=100,
      n_objs=2,
      lb=torch.tensor([-10.0]),
      ub=torch.tensor([10.0])
    )
```

&emsp;&emsp;使用时如果想查看某个算法的具体参数，可以使用 Python 的 `help` 函数，例如：

```python
import evox
help(evox.algorithms.so.PSO)
```

&emsp;&emsp;这将显示 PSO 算法的文档字符串和参数说明。此外，您也可以查阅 EvoX 在线文档中的对应部分查看这些信息 ([API Reference/algorithms - Evox · Documentation](#apidocs))。

**（2）问题模块**  (`evox.problems`)
&emsp;&emsp;EvoX 的问题模块同样分类别进行了组织，涵盖了数值优化、神经进化、超参数优化等多种场景。

  - `evox.problems.numerical` 包含经典的数值优化基准函数（如 Ackley、CEC 测试集等）。以下代码展示了如何调用 CEC2022 测试集中的 F1 函数：

  ```python
  from evox.problems.numerical import CEC2022

  # 定义一个问题维度为10的CEC2022 F1函数
  numerical_problem = CEC2022(problem_number=1, dimension=10)
  ```

  - `evox.problems.neuroevolution` 包含Brax、Gym等环境的封装。以下代码定义了一个 Brax中的 `swimmer` 环境：

```python
import torch
import torch.nn as nn
from evox.problems.neuroevolution.brax import BraxProblem

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.features = nn.Sequential(
          nn.Linear(8, 4),
          nn.Tanh(),
          nn.Linear(4, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return torch.tanh(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleMLP().to(device)
POP_SIZE = 1024

# 初始化一个 Brax 问题，使用的环境为 swimmer
neuroevolution_problem = BraxProblem(
    policy=model,
    env_name="swimmer",
    max_episode_length=1000,
    num_episodes=3,
    pop_size=POP_SIZE,
    device=device,
)
```

  - `evox.problems.hpo_wrapper` 则提供了对机器学习训练过程的封装，适用于超参数优化任务（完整实例可参考 EvoX 文档 （[HPO example - EvoX Documentation](#Deploy HPO with Custom Algorithms)）。

```python
import torch
from evox.algorithms.pso_variants.pso import PSO
from evox.problems import Sphere
from evox.core import Problem
from evox.problems.hpo_wrapper import HPOFitnessMonitor, HPOProblemWrapper
from evox.workflows import EvalMonitor, StdWorkflow

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

inner_algo = PSO(
    pop_size=100,
    lb= -10 * torch.ones(10),
    ub= 10 * torch.ones(10),
)
inner_prob = Sphere()
inner_monitor = HPOFitnessMonitor()
inner_workflow = StdWorkflow(
    inner_algo,
    inner_prob,
    monitor=inner_monitor,
)

# 将内层算法工作流转换成一个HPO问题，该问题的目标就是优化内层算法的参数，得到内层问题的最优结果
hpo_problem = HPOProblemWrapper(
    iterations=30,
    num_instances=128,
    workflow=inner_workflow,
    copy_init_state=True,
)
```

&emsp;&emsp;使用这些模块时，建议配合官方文档来了解对应问题类的构造参数。有的复杂问题可能需要提供额外数据，例如某些工程问题需要读入数据集或者设定实例规模。EvoX的API部分（[API Reference/problems - EvoX Documentation](#apidocs)）有详细说明。利用好这些现有API，可以大大减少自己编码的工作量。

### 5.2.2 工作流与工具 API

**（1）工作流模块**  (`evox.workflows`)

&emsp;&emsp;该模块提供了标准工作流（`evox.workflows.StdWorkflow`）和评估监控器（`evox.workflows.EvalMonitor`）等工具。前面的例子中我们已经使用过这两个模块，其中 `EvalMonitor` 可以在优化过程中记录适应度信息，`StdWorkflow` 用于将算法、问题、监控器进行封装。
以下代码展示了如何使用 EvoX 中的 `StdWorkflow` 和 `EvalMonitor`：

```python
import torch
from MyProblems import MyProblem
from evox.algorithms import PSO
from evox.workflows import EvalMonitor, StdWorkflow

problem = MyProblem()
algorithm = PSO(
    pop_size=100,
    lb=torch.tensor([-10.0]),
    ub=torch.tensor([10.0])
    )
monitor = EvalMonitor()
workflow = StdWorkflow(algorithm, problem, monitor)

compiled_step = torch.compile(workflow.step)
for i in range(10):
    compiled_step()
    top_fitness = monitor.topk_fitness
    print(f"The top fitness is {top_fitness}")
```

**（2）评价指标模块** (`evox.metrics`)

&emsp;&emsp;该模块包含常用的解集评价指标，如反世代距离（IGD）和超体积（Hypervolume）等。这些指标可以直接调用，用于评估当前种群的质量，将当前种群目标值与问题的真实 Pareto 前沿传入，即可得到指标值。以下代码展示了如何计算 IGD：

```python
import torch
from evox.algorithms import RVEA
from evox.metrics import igd
from evox.problems.numerical import DTLZ2
from evox.workflows import StdWorkflow, EvalMonitor

problem = DTLZ2(m=3)
pf = problem.pf() # 得到问题对应的 True Pareto Front
algorithm = RVEA(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
monitor = EvalMonitor()
workflow = StdWorkflow(algorithm, problem, monitor)
workflow.init_step()

for i in range(10):
    compiled_step()
    fit = workflow.algorithm.fit
    fit = fit[~torch.isnan(fit).any(dim=1)]
    # 传入种群目标值和 True Pareto Front
    print(igd(fit, pf))
```

&emsp;&emsp;通过结合这些 API，您可以在优化过程中插入自定义检查或记录。例如，每隔 10 代计算一次 IGD 并打印，以监控多目标优化的进展。

**（3）与PyTorch的交互**

&emsp;&emsp;EvoX 的许多对象（如张量和模块）与 PyTorch 兼容，因此您可以无缝对接 PyTorch 的 API。例如，在自定义问题中构建神经网络模型时，您可以直接使用 `torch.nn` 定义模型结构，并在 `evaluate`方法中调用模型的前向传播计算输出（在之前章节介绍 `evox.problems.neuroevolution` 时有相关使用案例）。
EvoX 不限制您使用任何外部库，只要最终能返回所需的适应度值即可。
这种开放性使得 EvoX 可以轻松融入其他框架。例如，您可以将 EvoX 用作调优器，结合 `sklearn` 训练模型或使用 `matplotlib` 绘制优化结果。
在使用 EvoX 时，建议充分利用 PyTorch 和 Python 生态提供的丰富 API，共同完成任务。

**（4）查询和帮助**

&emsp;&emsp;文档里没有完整列出 EvoX 中的所有模块，当您忘记某个类名或函数名时，可以利用 Python 的属性查询功能进行快速查找。例如，以下代码可以列出 `evox.algorithms` 模块下的所有对象：

```python
import evox
print(dir(evox.algorithms))
```

如果您安装了 IPython，还可以使用其强大的自动补全功能，通过输入模块名后按 `Tab` 键浏览可用的方法和类。

&emsp;&emsp;当您在使用过程中需要寻求帮助时，EvoX 也提供了丰富的资源支持：

- **在线文档**：EvoX 提供详细的用户指南和 API 参考文档（[EvoX  Documentation](#apidocs)），涵盖了从基础到高级的各类主题。如果您遇到问题，可以查阅文档中的 `Developer Guide` 或 `API Reference` 等部分。
- **社区支持**：您可以通过 EvoX 的 GitHub 仓库（[EvoX Github](https://github.com/EMI-Group/evox)）提交 Issues，或在相关论坛和 QQ 讨论群 （ID: 297969717）中寻求帮助。
- **示例代码**：对于初学者，多参考官方示例是快速掌握 API 用法的有效途径。EvoX 官方文档的 `examples` 目录中包含了许多示例脚本（[Examples - EvoX Documentation](#examples)），您可以克隆仓库并运行这些脚本，边看代码边理解。

```
  掌握EvoX API的使用将使您更自如地实现复杂的功能。例如，您可以通过API组合来实现多层优化：外层用EvoX优化内层算法的参数，实现AutoML中的自动算法配置。这种高级应用需要对EvoX API非常熟练，但从技术上完全可行。我们鼓励您在逐步深入学习后尝试创新性地运用这些接口，打造出适合自己场景的优化方案。
```

## 5.3 与其他工具集成

&emsp;&emsp;EvoX 的设计目标之一是易于与外部工具集成，包括机器学习库、仿真环境、数据处理工具甚至企业级应用程序。以下是几种典型场景，展示如何将 EvoX 嵌入到更大的工作流程中：

### 5.3.1 与机器学习框架集成

&emsp;&emsp;假设您有一个基于 PyTorch 的深度学习模型，需要在某数据集上进行训练，但不确定最佳超参数配置。您可以使用 EvoX 来<u>自动调整超参数</u>：

- **定义问题**：将模型训练和验证过程封装为一个`Problem`。输入是一个超参数向量 (如学习率、正则项系数等)，`evaluate` 方法根据这些超参数训练模型若干epoch，并在验证集上计算损失，返回验证损失作为适应度值。
- **选择算法**：选择一个合适的演化算法（如 CMA-ES 或 PSO），设置合理的种群大小和迭代次数。
- **迭代优化**：运行 EvoX 进行迭代优化，目标是最小化验证损失。每个候选解对应一组超参数配置，EvoX 会自动尝试不同的超参数配置，并逐步趋向更优解。
- **模型训练**：在获得最优超参数后，使用全训练集训练最终模型。

&emsp;&emsp;这种集成方式将 EvoX 作为外层优化器，包裹训练过程。由于 EvoX 本身基于PyTorch，实现此类 `Problem` 较为便利，可以直接调用 PyTorch 的张量操作和模型接口。在实际操作中，建议控制训练 epoch 数量，以避免整体耗时过长，可采用逐步增加等方式。

### 5.3.2 与强化学习环境集成

&emsp;&emsp;演化算法在强化学习中特别适合优化无梯度的策略（如直接优化策略网络参数或强化学习算法的超参数）。EvoX 已支持多种强化学习环境（如 Brax），您可以按以下步骤集成：

- **定义问题**：使用 `evox.problems.neuroevolution.BraxProblem` 将 Brax 环境包装为优化问题，评估时返回智能体在环境中运行若干步的累计奖励。
- **选择算法**：选择一个演化算法（如 CMA-ES 或 GA）来优化策略参数。策略可以是一个神经网络控制器，通过 `ParamsAndVector` 将其参数展开为优化向量。
- **迭代优化**：EvoX 将尝试不同的策略参数，在环境中模拟并获得奖励作为适应度，逐步改进策略。
- **应用策略**：将优化后的策略用于实际环境，或作为初始化提供给梯度学习算法（如进化初步找到较好策略，然后交给RL算法精调，得到进化算法和强化学习的混合方法）。

&emsp;&emsp;这种集成需要注意的是：仿真环境通常可以并行运行多个实例，同时评估多个个体将得到更高的效率。因此建议在 `Problem` 的 `evaluate` 中并行运行多个环境实例，充分利用CPU/GPU算力。EvoX 的并行框架非常适合此类任务。

---

EvoX 的集成之所以简单，是因为它遵循 Python 科学计算生态的习惯，没有封闭自己的运行环境。对于初学者，建议从简单的机器学习任务入手，例如用 EvoX 优化一个小型 MLP 在简单数据集上的超参数。一旦掌握了这种思路，您就能将 EvoX 应用于各种需要优化的场景，为现有工具增加智能搜索的能力。
