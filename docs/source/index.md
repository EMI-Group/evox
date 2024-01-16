# Welcome to EvoX's documentation!


```{toctree}
:caption: 'User Guide'
:maxdepth: 1
:hidden:

guide/install
guide/basics/index
guide/advanced/index
```

```{toctree}
:caption: 'API Reference'
:maxdepth: 1
:hidden:
api/core/index
api/algorithms/index
api/problems/index
api/workflows/index
api/monitors/index
api/metrics/index
```

```{toctree}
:caption: 'Additional Resources'
:maxdepth: 2
:hidden:

Examples <example/index>
```

[[English Version]](https://evox.readthedocs.io/en/latest/)   [[中文版本]](https://evox.readthedocs.io/zh/latest/)

EvoX is a distributed GPU-accelerated framework for scalable evolutionary computation.

---

## Key Features

- 🚀 **Fast Performance**:
  - Experience **GPU-Accelerated** optimization, achieving speeds 100x faster than traditional methods.
  - Leverage the power of {class}`Distributed Workflows <evox.workflows.RayDistributedWorkflow>` for even more rapid optimization.

- 🌐 **Versatile Optimization Suite**:
  - Cater to all your needs with both {doc}`Single-objective <api/algorithms/so/index>` and {doc}`Multi-objective <api/algorithms/mo/index>` optimization capabilities.
  - Dive into a comprehensive library of {doc}`Benchmark Problems <api/problems/numerical/index>`, ensuring robust testing and evaluation.
  - Explore the frontier of AI with extensive tools for {doc}`Neuroevolution <api/problems/neuroevolution/index>` tasks.

- 🛠️ **Designed for Simplicity**:
  - Embrace the elegance of **Functional Programming**, simplifying complex algorithmic compositions.
  - Benefit from **Hierarchical State Management**, ensuring modular and clean programming.

---
<br></br>

```{eval-rst}
.. grid:: 3
   :gutter: 1 2 3 5
   :padding: 1

   .. grid-item-card:: :octicon:`desktop-download;1.5em` Installation Guide
      :padding: 3
      :margin: 0
      :link: guide/install
      :link-type: doc

   .. grid-item-card:: :octicon:`people;1.5em` Getting Started
      :padding: 3
      :margin: 0
      :link: guide/basics/index
      :link-type: doc

   .. grid-item-card:: :octicon:`mortar-board;1.5em` Advanced Guide
      :padding: 3
      :margin: 0
      :link: guide/advanced/index
      :link-type: doc

.. grid:: 3
   :gutter: 1 2 3 5
   :padding: 1

   .. grid-item-card:: :octicon:`list-unordered;1.5em` Algorithms
      :padding: 3
      :margin: 0
      :link: api/algorithms/index
      :link-type: doc

   .. grid-item-card:: :octicon:`list-unordered;1.5em` Problems
      :padding: 3
      :margin: 0
      :link: api/problems/index
      :link-type: doc


   .. grid-item-card:: :octicon:`list-unordered;1.5em` Metrics
      :padding: 3
      :margin: 0
      :link: api/metrics/index
      :link-type: doc
```
