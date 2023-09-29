===========
Quick Start
===========

Algorithm, Problem & Workflow
=============================

To start with, import ``evox``

.. code-block:: python

    import evox
    from evox import algorithms, problems, workflows

Then, create an ``algorithm`` and a ``problem``.
The list of available algorithms and problems can be found in :mod:`here <evox.algorithms>` and :mod:`here <evox.problems>`.

.. code-block:: python

    pso = algorithms.PSO(
        lb=jnp.full(shape=(2,), fill_value=-32),
        ub=jnp.full(shape=(2,), fill_value=32),
        pop_size=100,
    )
    ackley = problems.numerical.Ackley()

The algorithm and the problem are composed together using ``workflow``:

.. code-block:: python

    workflow = workflows.StdWorkflow(pso, ackley)


To initialize the whole workflow, call ``init`` on the workflow object with a PRNGKey.
Calling ``init`` will recursively initialize a tree of objects, meaning the algorithm pso and problem ackley are automatically initialize as well.

.. code-block:: python

    key = jax.random.PRNGKey(42)
    state = workflow.init(key)

To run the workflow, call ``step`` on the workflow.

.. code-block:: python

    # run the workflow for 100 steps
    for i in range(100):
        state = workflow.step(state)

Monitor
=======

Usually, we don't care about the final state of the workflow, instead, we are more interested in things like the fitness or the final solution.

The ``monitor`` is the way to record these values in EvoX.

First, import monitors and create a monitor

.. code-block:: python

    from evox.monitors import StdSOMonitor
    monitor = StdSOMonitor()

Then configure the workflow to use the monitor.

.. code-block:: python

    workflow = workflows.StdWorkflow(
        pso,
        ackley,
        monitor,
    )

Then continue to run the workflow as ususal. Now at each iteration, the workflow will call ``monitor.record_fit`` with the fitness at that iteration.

.. code-block:: python

    # init the workflow
    state = workflow.init(key)
    # run the workflow for 100 steps
    for i in range(100):
        state = workflow.step(state)

To get the minimum fitness of all time, call the ``get_min_fitness`` method on the monitor.

.. code-block:: python

    # print the min fitness
    print(monitor.get_min_fitness())
