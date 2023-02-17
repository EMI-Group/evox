.. role:: python(code)
  :language: python
  :class: highlight

======================================
Custom algorithms and problems in EvoX
======================================

In this chapter, we will introduce how to implement your own algorithm in EvoX.

The Algorithm Class
===================

The :class:`Algorithm <evox.Algorithm>` class is inherented from :class:`Stateful <evox.Stateful>`.
Besides the things in ``Stateful``, your should also implement a ``ask`` and a ``tell`` method.
In total, there are four methods one need to implement.

+----------+-----------------------------------------+------------------------------------------------------------------------------------+
| Method   | Signature                               | Usage                                                                              |
+==========+=========================================+====================================================================================+
| __init__ | :python:`(self, ...)`                   | Initialize hyperparameters that are fixed though out the optimization process,     |
|          |                                         | for example, the ``population size``.                                              |
+----------+-----------------------------------------+------------------------------------------------------------------------------------+
| setup    | :python:`(self, RRNGKey) -> State`      | Initialize mutable state, for example the ``momentum``.                            |
+----------+-----------------------------------------+------------------------------------------------------------------------------------+
| ask      | :python:`(self, State) -> Array, State` | Gives a candidate population for evaluation.                                       |
+----------+-----------------------------------------+------------------------------------------------------------------------------------+
| tell     | :python:`(self, State, Array) -> State` | Receive the fitness for the candidate population and update the algorithm's state. |
+----------+-----------------------------------------+------------------------------------------------------------------------------------+


Migrate from traditional EC library
-----------------------------------

In traditional EC library, algorithm usually calls the objective function internally, which gives the following layout

.. code-block::

    Algorithm
    |
    +--Problem

But in EvoX, we have a flat layout

.. code-block::

    Algorithm.ask -- Problem.evaluate -- Algorithm.tell


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

The Problem class is quite simple, beside ``__init__`` and ``setup``, the only required the method is ``evaluate``.

Migrate from traditional EC library
-----------------------------------

There is one thing to notice here, ``evaluate`` is a stateful function, meaning it should accept a state and return a new state.
So, if you are working with numerical benchmark functions, which don't need to statefule,
you can simply ignore the state, but remember that you still have to use this stateful interface.

+----------+------------------------------------------------+-------------------------------------------------------+
| Method   | Signature                                      | Usage                                                 |
+----------+------------------------------------------------+-------------------------------------------------------+
| __init__ | :python:`(self, ...)`                          | Initialize the settings of the problem.               |
+----------+------------------------------------------------+-------------------------------------------------------+
| setup    | :python:`(self, RRNGKey) -> State`             | Initialize mutable state of this problem.             |
+----------+------------------------------------------------+-------------------------------------------------------+
| evaluate | :python:`(self, State, Array) -> Array, State` | Evaluate the fitness of the given candidate solution. |
+----------+------------------------------------------------+-------------------------------------------------------+

More on the problem's state
---------------------------

If you still wonders what the problem's state actually do, here are the explanations.

Unlike numerical benchmark functions, real-life problems are more complex, and may require stateful computations.
Here are some examples:

* When dealing with ANN training, we often have training, validation and testing phase.
  This implies that the same solution could have different fitness values during different phases.
  So clearly, we can't model the `evaluate` as a stateless pure function any more.
  To implement this mechanism, simple put an value in the state to indicate the phase.
* Virtual batch norm is a effective trick especially when dealing with RL tasks.
  To implement this mechanism, the problem must be stateful,
  as the problem have to remember the initial batch norm parameters during the first run.