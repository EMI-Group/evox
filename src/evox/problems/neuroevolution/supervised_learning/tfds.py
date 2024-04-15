from typing import Any, Callable, Tuple, Any
from dataclasses import field
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

from evox.core.module import Any, Tuple
from evox.core.state import State
import grain.python as pygrain
import tensorflow_datasets as tfds
from evox.utils.io import x32_func_call

from evox import Problem, Static, dataclass


def get_dtype_shape(data):
    def to_dtype_struct(x):
        if hasattr(x, "shape") and hasattr(x, "dtype"):
            return jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
        elif isinstance(x, int):
            return jax.ShapeDtypeStruct(shape=(), dtype=jnp.int32)
        elif isinstance(x, float):
            return jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32)

    return tree_map(to_dtype_struct, data)


@dataclass
class TensorflowDataset(Problem):
    """Wrap a tensorflow dataset as a problem

    The details of the dataset can be found at https://www.tensorflow.org/datasets/catalog/overview

    Parameters
    ----------
    dataset
        The dataset name.
    batch_size
        The batch size.
    loss_func
        The loss function.
        The function signature is loss(weights, data) -> loss_value, and it should be jittable.
        The `weight` is the weight of the neural network, and the `data` is the data from TFDS, which is a dictionary.
    seed
        The random seed used to seed the dataloader.
        Given the same seed, the dataloader should data in the same order.
        Default to 0.
    """

    dataset: Static[str]
    batch_size: Static[int]
    loss_func: Static[Callable]
    seed: Static[int] = field(default=0)
    iterator: Static[pygrain.PyGrainDatasetIterator] = field(init=False)
    data_shape_dtypes: Static[Any] = field(init=False)

    def __post_init__(self):
        data_source = tfds.data_source(self.dataset, split="train")
        sampler = pygrain.IndexSampler(
            num_records=len(data_source),
            shard_options=pygrain.NoSharding(),
            shuffle=True,
            seed=self.seed,
        )
        loader = pygrain.DataLoader(
            data_source=data_source,
            operations=[pygrain.Batch(batch_size=self.batch_size, drop_remainder=True)],
            sampler=sampler,
            worker_count=0,
        )
        object.__setattr__(self, "iterator", iter(loader))
        data_shape_dtypes = get_dtype_shape(self._next_data())
        object.__setattr__(self, "data_shape_dtypes", data_shape_dtypes)

    @x32_func_call
    def _next_data(self):
        return next(self.iterator)

    def evaluate(self, state, pop):
        data = jax.experimental.io_callback(self._next_data, self.data_shape_dtypes)
        loss = jax.vmap(self.loss_func, in_axes=(0, None))(pop, data)
        return loss, state
