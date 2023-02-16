======================================
Custom algorithms and problems in EvoX
======================================

In this chapter, we will introduce how to implement your own algorithm in EvoX.

The Algorithm Class
===================

The :class:`Algorithm <evox.Algorithm>` class is inherented from :class:`Stateful <evox.Stateful>`.
Besides the things in ``Stateful``, your should also implement a ``ask`` and a ``tell`` method.
In total, there are four methods one need to implement.

+----------+------------------------------------------------------------------------------------+
| Method   | Usage                                                                              |
+==========+====================================================================================+
| __init__ | Initialize hyperparameters that are fixed though out the optimization process,     |
|          | for example, the ``population size``.                                              |
+----------+------------------------------------------------------------------------------------+
| setup    | Initialize mutable state, for example the ``momentum``.                            |
+----------+------------------------------------------------------------------------------------+
| ask      | Gives a candidate population for evaluation.                                       |
+----------+------------------------------------------------------------------------------------+
| tell     | Receive the fitness for the candidate population and update the algorithm's state. |
+----------+------------------------------------------------------------------------------------+


Migrate from traditional EC library
-----------------------------------

In traditional EC library, algorithm usually calls the objective function internally, which gives the following layout

.. code-block::

    Algorithm
    |
    +--Problem

But in EvoX, we have a flat layout

.. code-block::

    Algorithm.ask - Problem.evaluate - Algorithm.tell


Here is a pseudocode of a genetic algorithm.

.. code-block:: python

    Set hyperparameters
    Generate the initial population
    Do
        Generate Offspring
            Selection
            Crossover
            Mutation
        Compute fitness
        Replace the population
    Until stopping criterion

And Here is what each part of the algorithm correspond to in EvoX.

.. code-block:: python

    Set hyperparameters # __init__
    Generate the initial population # setup
    Do
        # ask
        Generate Offspring
            Selection
            Crossover
            Mutation

        # problem.evaluate (not part of the algorithm)
        Compute fitness

        # tell
        Replace the population
    Until stopping criterion

The Problem Class
=================