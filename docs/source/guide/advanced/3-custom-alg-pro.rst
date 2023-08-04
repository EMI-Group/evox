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
            Mating Selection
            Crossover
            Mutation

        # problem.evaluate (not part of the algorithm)
        Compute fitness

        # tell
        Survivor Selection
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


Example
=======

Here we give an exmaple of implementing the OneMax problem, along with a genetic algorithm that solves this problem.
The problem itself is straight forward, the fitness is defined as the sum of every digits in a fixed-length bitstring.
For example, "100111" gives 4 and "000101" gives 2.

Let's starts with implementing the OneMax problem.
In JAX a bitstring can be easily represented with a tensor of type bool.

.. code-block:: python

    import jax.numpy as jnp
    from evox import Problem, jit_class


    @jit_class
    class OneMax(Problem):
        def __init__(self, neg_fitness=True) -> None:
            super().__init__()
            self.neg_fitess = neg_fitness

        def evaluate(self, state, bitstrings):
            # bitstrings has shape (pop_size, num_bits)
            # so sum along the axis 1.
            fitness = jnp.sum(bitstrings, axis=1)
            # Since in EvoX, algorithms try to minimize the fitness
            # so return the negitive value.
            if self.neg_fitess:
                fitness = -fitness
            return fitness, state


Then we implement a genetic algorithm that uses bitflip mutation and one-point crossover.

.. code-block:: python

    @jit_class
    class ExampleGA(Algorithm):
        def __init__(self, pop_size, ndim, flip_prob):
            super().__init__()
            # those are hyperparameters that stay fixed.
            self.pop_size = pop_size
            self.ndim = ndim
            # the probability of fliping each bit
            self.flip_prob = flip_prob

        def setup(self, key):
            # initialize the state
            # state are mutable data like the population, offsprings
            # the population is randomly initialized.
            # we don't have any offspring now, but initialize it as a placeholder
            # because jax want static shaped arrays.
            key, subkey = random.split(key)
            pop = random.uniform(subkey, (self.pop_size, self.ndim)) < 0.5
            return State(
                pop=pop,
                offsprings=jnp.empty((self.pop_size * 2, self.ndim)),
                fit=jnp.full((self.pop_size,), jnp.inf),
                key=key,
            )

        def ask(self, state):
            key, mut_key, x_key = random.split(state.key, 3)
            # here we do mutation and crossover (reproduction)
            # for simplicity, we didn't use any mating selections
            # so the offspring is twice as large as the population
            offsprings = jnp.concatenate(
                (
                    mutation.bitflip(mut_key, state.pop, self.flip_prob),
                    crossover.one_point(x_key, state.pop),
                ),
                axis=0,
            )
            # return the candidate solution and update the state
            return offsprings, state.update(offsprings=offsprings, key=key)

        def tell(self, state, fitness):
            # here we do selection
            merged_pop = jnp.concatenate([state.pop, state.offsprings])
            merged_fit = jnp.concatenate([state.fit, fitness])
            new_pop, new_fit = selection.topk_fit(merged_pop, merged_fit, self.pop_size)
            # replace the old population
            return state.update(pop=new_pop, fit=new_fit)

Now, you can assemble a pipeline and run it.