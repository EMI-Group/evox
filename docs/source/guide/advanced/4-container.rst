====================
Container Algorithms
====================

Container algorithms are a special type of algorithms that works by containing other algorithms and cannot work on its own.
Container algorithms can be used to compose a series of normal algorithms together.

Working with expensive algorithms
=================================

Many algorithms are expensive in term of space or time. For example, CMA-ES requires :math:`O(N^2)` space.
Thus, it is costly to run CMA-ES on high-dimension problems.
Sep-CMA-ES scales better, but sacrifice the performance.
That's where container algorithm comes in.
With it, we can easily construct a variant of CMA-ES that uses :math:`O((\frac{N}{M})^2)` space, where :math:`M` is the number of block.
This variant is a balance between the normal CMA-ES and Sep-CMA-ES.

Working with PyTree
===================

Usually, algorithms expect the decision variables to be in the form of a 1D-vector.
PyTrees are tree-like structures that are not directly compatible with normal algorithms.
So, there are two solutions out there:

1. Flatten the PyTree to 1D-vector.
2. Use a specialized algorithm that work with PyTree directly.

Solution 1 is called ``adapter`` in EvoX, which is quite simple, but we are not talking about this here.
Solution 2 seems more complicated, but the advantage is that the structural information is preserved,
meaning the algorithm could see the tree structure and apply some type of heuristic here.

Cooperative Coevolution
=======================

We offer Cooperative Coevolution (CC) framework for all algorithms.
Currently, there are two types of CC container in EvoX, :class:`evox.algorithms.Coevolution` and :class:`evox.algorithms.VectorizedCoevolution`.
The difference is that ``VectorizedCoevolution`` update all sub-populations at the same time in each iteration,
but ``Coevolution`` follows traditional approach that update a single sub-populations at each iteration.
Thus ``VectorizedCoevolution`` is faster, but ``Coevolution`` could be better in terms of best fitness with a limited number of evaluations.
