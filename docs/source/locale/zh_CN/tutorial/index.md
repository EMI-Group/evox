# EvoX 教程

```{warning} EvoX 2.0 预览版（实验性）
本教程针对 **EvoX 2.0 预览版** 重写：导入名称为 **`evox_etl`**，基于 **EvoX Compiler**（`evoxc`）张量库的函数式 API。
1.x 版本的 PyTorch 面向对象 API（`import evox`、`ModuleBase`、`Parameter`/`Mutable`、`torch.compile`/`vmap`）已不再提供，也不再记录在本教程中。
```

**中文翻译状态：** 教程正在针对 2.0 预览版整体重写，中文译文尚未完成。
下列条目暂时指向英文原文；译文完成前请直接阅读英文页面，英文页面与 `evox_etl` 代码保持同步。

```{toctree}
:maxdepth: 1

tutorial_part1
tutorial_part2
tutorial_part3
tutorial_part4
tutorial_part5
tutorial_part6
tutorial_part7
```

| 章节 | 英文标题 | 内容 |
|---|---|---|
| 1 | EvoX 2.0 and installation | 2.0 的新变化、安装与后端扩展、冒烟测试 |
| 2 | Your first optimization: PSO on Sphere | 第一个完整程序：PSO 求解 Sphere |
| 3 | Workflow and monitor | `StdWorkflow` 与 `EvalMonitor`：状态传递、随机数键、历史记录 |
| 4 | Algorithms | 27 个单目标 + 6 个多目标算法一览 |
| 5 | Problems | 数值问题与虚拟（神经进化）问题一览 |
| 6 | Multi-objective optimization | NSGA-II、指标（GD/IGD/HV）与可视化 |
| 7 | Migrating from EvoX 1.x | 从 1.x 迁移的对照表 |

如果是从 1.x 迁移，请先阅读第 7 章（英文）。

英文原版教程入口：[EvoX Tutorial](https://evox.readthedocs.io/en/latest/tutorial/index.html)
