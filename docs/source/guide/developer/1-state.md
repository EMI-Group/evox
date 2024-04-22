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
all immutable data are initialized in `__init__`, or with dataclass,
the initial mutable state is generated in `setup`.

```python
class Foo(Stateful):
    def __init__(self,): # required
        pass

    def setup(self, key) -> State: # optional
        pass

    def stateful_func(self, state, args) -> State: # user-defined
        pass

    def init(self, key) -> State: # provided by Stateful
        pass
```

Notice that all stateful functions should have the following signature:
```python
def func(self, state: State, ...) -> Tuple[..., State]
```
which is a common pattern in stateful computation.

Stateful defines the standard for stateful modules in EvoX.
All other classes, such as `Algorithm`, `Problem`, and `Workflow`, are inherited from `Stateful`.
Then Stateful will provide a `init` method, which we will discuss later.

## An overview of State

In EvoX `State` represents a tree of states, which stores the state of the current object and all child objects.

The structure of `State` is roughly like this:

```python
{
    "self_state": Dict[str, Any],
    "child_states": Dict[str, State]
}
```
Where `self_state` stores the state of the current object (the one returned by `setup`), and `child_states` stores the states of all child objects.
Besides normal dicts as `self_state`, we also support the use of `dataclass` to define the state of the object.

## Combined together

### Initialization

To initialize a hierarchy of Stateful objects, and initialize the state of each object, you could write code like this.
You can to call `init` method of the top module.
`init` will recursively call the `setup` method of each module, and construct the complete state.


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
