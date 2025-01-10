# Welcome to EvoX's documentation!


```{toctree}
:caption: 'User Guide'
:maxdepth: 1
:hidden:

guide/install/index
guide/user/index
guide/developer/index
guide/experimental/index
```

```{toctree}
:caption: 'API Reference'
:maxdepth: 1
:hidden:
apidocs/index
```

```{toctree}
:caption: 'Additional Resources'
:maxdepth: 2
:hidden:

Examples <example/index>
Miscellaneous <miscellaneous/index>
```

[[English Version]](https://evox.readthedocs.io/en/latest/)   [[中文版本]](https://evox.readthedocs.io/zh/latest/)

EvoX is a distributed GPU-accelerated framework for scalable evolutionary computation.

---

## Key Features

- 🚀 **Fast Performance**:
  - Experience **GPU-Accelerated** optimization, achieving speeds over 100x faster than traditional methods.
  - Leverage the power of Distributed Workflows for even more rapid optimization.

- 🌐 **Versatile Optimization Suite**:
  - Cater to all your needs with both {doc}`Single-objective and Multi-objective <apidocs/evox/evox.algorithms>` optimization capabilities.
  - Dive into a comprehensive library of {doc}`Benchmark Problems <apidocs/evox/evox.problems>`, ensuring robust testing and evaluation.
  - Explore the frontier of AI with extensive tools for {doc}`Neuroevolution <apidocs/evox/evox.problems.neuroevolution>` tasks.

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
      :link: guide/install/index
      :link-type: doc

   .. grid-item-card:: :octicon:`people;1.5em` User Guide
      :padding: 3
      :margin: 0
      :link: guide/user/index
      :link-type: doc

   .. grid-item-card:: :octicon:`mortar-board;1.5em` Developer Guide
      :padding: 3
      :margin: 0
      :link: guide/developer/index
      :link-type: doc

.. grid:: 3
   :gutter: 1 2 3 5
   :padding: 1

   .. grid-item-card:: :octicon:`list-unordered;1.5em` Algorithms
      :padding: 3
      :margin: 0
      :link: apidocs/evox/evox.algorithms
      :link-type: doc

   .. grid-item-card:: :octicon:`list-unordered;1.5em` Problems
      :padding: 3
      :margin: 0
      :link: apidocs/evox/evox.problems
      :link-type: doc


   .. grid-item-card:: :octicon:`list-unordered;1.5em` Metrics
      :padding: 3
      :margin: 0
      :link: apidocs/evox/evox.metrics
      :link-type: doc
```
