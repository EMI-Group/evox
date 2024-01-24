# Working with state in EvoX

EvoX is designed around stateful computation.

There are two most fundamental classes, namely {class}`Stateful <evox.Stateful>` and {class}`State <evox.State>`.

All class that involves stateful computation are inherited from `Stateful`. In EvoX, `Algorithm`, `Problem`, `Operator` and workflows are all stateful.

## The idea behind the design

```{image} /_static/hierarchical_state.svg
:alt: hierarchical state
:width: 400px
```

Here we have five different objects, and notice that they have a hierarchical structure.
To work with such structures, at each level, we must "lift the state" by managing the states of child components.
So, the state at the `workflow` level must contain the state of both `algorithm` and `problem`,
and since the state at the `algorithm` level must contain the state of both operators,
the state `workflow` level actually needs to handle states from all 5 components.

However, it is frustrating to manage the hierarchy manually, and it is not good for modular design.
To solve this problem, we introduce `Stateful` and `State`.

## An Overview of Stateful

In a `Stateful` class,
all immutable data are initialized in `__init__`,
the initial mutable state is generated in `setup`,
besides these two methods and private methods(start with "\_"),
all other methods are wrapped with `use_state`.

```python
class Foo(Stateful):
    def __init__(self,): # required
        pass

    def setup(self, key) -> State: # optional
        pass

    def stateful_func(self, state, args) -> State: # wrapped with use_state
        pass

    def _normal_func(self, args) -> vals: # not wrapped
        pass
```

will be wrapped with `use_state` decorator. This decorator requires the method have the following signature:

```python
def func(self, state: State, ...) -> Tuple[..., State]
```

which is a common pattern in stateful computation.

:::{warning}
Currently, for all user-defined private methods, the name of the method should start with `_``.
:::

## An overview of State

In EvoX `State` represents a tree of states, which stores the state of the current object and all child objects.

## Combined together

When combined together,
they will automatically go 1 level down in the tree of states,
and merge the subtree back to the current level.

So you could write code like this.

```python
class FooWorkflow(Stateful):
    ...
    def step(self, state):
        population, state = self.algorithm.ask(state)
        fitness, state = self.problem.evaluate(state, population)
        ...
```

Notice that, when calling the method `step`,
`state` is the state of the workflow,
but when calling `self.algorithm.ask`,
`state` behaves like the state of the algorithm,
and after the call, the state of the algorithm is automatically merged back into the state of the workflow.
