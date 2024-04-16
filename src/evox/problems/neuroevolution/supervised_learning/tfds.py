from dataclasses import field
from typing import Any, Callable, List

import grain.python as pygrain
import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds
from jax.tree_util import tree_map

from evox import Problem, Static, dataclass, jit_class
from evox.utils.io import x32_func_call


def get_dtype_shape(data):
    def to_dtype_struct(x):
        if hasattr(x, "shape") and hasattr(x, "dtype"):
            return jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
        elif isinstance(x, int):
            return jax.ShapeDtypeStruct(shape=(), dtype=jnp.int32)
        elif isinstance(x, float):
            return jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32)

    return tree_map(to_dtype_struct, data)


@jit_class
@dataclass
class TensorflowDataset(Problem):
    """Wrap a tensorflow dataset as a problem.

    TensorFlow Datasets (TFDS) directly depends on the package `tensorflow-datasets` and `grain`.
    Additionally, when downloading the dataset for the first time, it requires `tensorflow` to be installed and a active internet connection.
    If you want to avoid installing `tensorflow`, you can prepare the dataset beforehand in another environment with `tensorflow` installed,
    run:

    .. code-block:: python

        import tensorflow_datasets as tfds
        tfds.data_source(self.dataset)

    and then copy the dataset to the target machine.
    The default location is`~/tensorflow_datasets`. `~/` means the home directory of the user.

    Please notice that the data is loaded under JAX's jit context, so the data should be valid JAX data type,
    namely JAX or Numpy arrays, or Python's int, float, list, and dict.
    If the data contains other types like strings, you should convert them into arrays using the `operations` parameter.

    The details of the dataset can be found at https://www.tensorflow.org/datasets/catalog/overview
    The details about operations/transformations can be found at https://github.com/google/grain/blob/main/docs/transformations.md

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
    operations
        The list of transformations to apply to the data.
        Default to [].
        After the transformations, we will always apply a batch operation to create a batch of data.
    seed
        The random seed used to seed the dataloader.
        Given the same seed, the dataloader should data in the same order.
        Default to 0.
    """

    dataset: Static[str]
    batch_size: Static[int]
    loss_func: Static[Callable]
    operations: Static[List[Any]] = field(default_factory=list)
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

        operations = self.operations + [
            pygrain.Batch(batch_size=self.batch_size, drop_remainder=True)
        ]

        loader = pygrain.DataLoader(
            data_source=data_source,
            operations=operations,
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
