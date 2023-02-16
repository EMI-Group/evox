===================
Jit-able components
===================

A common pitfall in jit
=======================

In JAX, it's hard to jump out of a jit-compiled function, meaning if you jit-compile one function,
then all other functions used within this function must also be jit-compiled.

For example, the follow code will result in compilation error

.. code-block:: python

    @jax.jit
    def foo(x):
        return bar(x)

    def bar(x):
        return x[x > 0] # dynamic index, not jit-able

Even though ``bar`` is not marked with ``jax.jit``, it is still compiled as ``foo`` calls ``bar``.
And since bar uses dynamic index, which is not compatible with ``jax.jit``, an error will occur.

Solution
========

To solve is problem, it is common practice to jit-compile low level components, thus give high level components more freedom.
In EvoX, we have some general rules on whether a function should be jit-able or not.

+-----------+----------+
| Component | jit-able |
+===========+==========+
| Pipeline  | Optional |
+-----------+----------+
| Algorithm | Yes      |
+-----------+----------+
| Problem   | Optional |
+-----------+----------+
| Operators | Yes      |
+-----------+----------+
| Monitor   | No       |
+-----------+----------+

For standard pipeline, one can jit compile when not using monitors and working with jit-able problems.
But even though the pipeline can be compiled, there isn't much performance gain.
For problems, it depends on the task.