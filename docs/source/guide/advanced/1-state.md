# Working with state in EvoX

EvoX is designed around the stateful computation.

There are two most fundamental classes, namely `Stateful` and `State`.

All class that involves stateful computation are inherented from `Stateful`. In EvoX, `Algorithm`, `Problem`, `Operator` and pipelines are all stateful.

In a `Stateful` class, all public method except `setup`, will be wrapped with `use_state` decorator. This decorator requires the method have the following signature `(self, State, ...) -> ..., State`, which is common pattern in stateful computation.

In EvoX `State` represents a tree of states, which stores the state of the current object and all child objects.