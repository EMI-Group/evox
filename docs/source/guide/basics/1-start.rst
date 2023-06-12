===========
Quick Start
===========

Algorithm, Problem & Pipeline
=============================

To start with, import ``evox``

.. code-block:: python

    import evox
    from evox import algorithms, problems, pipelines

Then, create an ``algorithm`` and a ``problem``.
The list of available algorithms and problems can be found in :mod:`here <evox.algorithms>` and :mod:`here <evox.problems>`.

.. code-block:: python

    pso = algorithms.PSO(
        lb=jnp.full(shape=(2,), fill_value=-32),
        ub=jnp.full(shape=(2,), fill_value=32),
        pop_size=100,
    )
    ackley = problems.classic.Ackley()

The algorithm and the problem are composed together using ``pipeline``:

.. code-block:: python

    pipeline = pipelines.StdPipeline(pso, ackley)


To initialize the whole pipeline, call ``init`` on the pipeline object with a PRNGKey.
Calling ``init`` will recursively initialize a tree of objects, meaning the algorithm pso and problem ackley are automatically initialize as well.

.. code-block:: python

    key = jax.random.PRNGKey(42)
    state = pipeline.init(key)

To run the pipeline, call ``step`` on the pipeline.

.. code-block:: python

    # run the pipeline for 100 steps
    for i in range(100):
        state = pipeline.step(state)

Monitor
=======

Usually, we don't care about the final state of the pipeline, instead, we are more interested in things like the fitness or the final solution.

The ``monitor`` is the way to record these values in EvoX.

First, import monitors and create a monitor

.. code-block:: python

    from evox.monitors import StdSOMonitor
    monitor = StdSOMonitor()

Then set this monitor as the fitness transform for the pipeline

.. code-block:: python

    pipeline = pipelines.StdPipeline(
        pso,
        ackley,
        fitness_transform=monitor.record_fit,
    )

Then continue to run the pipeline as ususal. now at each iteration, the pipeline will call ``monitor.record_fit`` with the fitness at that iteration.

.. code-block:: python
    # init the pipeline
    state = pipeline.init(key)
    # run the pipeline for 100 steps
    for i in range(100):
        state = pipeline.step(state)

To get the minimum fitness of all time, call the ``get_min_fitness`` method on the monitor.

.. code-block:: python

    # print the min fitness
    print(monitor.get_min_fitness())
