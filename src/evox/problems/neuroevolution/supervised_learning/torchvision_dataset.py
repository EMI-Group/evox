import warnings
from functools import partial
from typing import Callable, NamedTuple, Optional, Union

import evox
import jax
import jax.numpy as jnp
import numpy as np
import optax
from evox import Problem, State, Stateful, jit_class, jit_method
from jax import jit, lax, vmap
from jax.tree_util import tree_leaves
from torch.utils.data import DataLoader, Dataset, Sampler, Subset, random_split
from torchvision import datasets


def np_collate_fn(batch: list):
    data, labels = list(zip(*batch))
    return np.stack(data), np.array(labels)


def jnp_collate_fn(batch: list):
    data, labels = list(zip(*batch))
    return jnp.stack(data), jnp.array(labels)


class InMemoryDataset:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, indices):
        return self.data[indices], self.labels[indices]

    @classmethod
    def from_pytorch_dataset(cls, dataset):
        dataset = [dataset[i] for i in range(len(dataset))]
        data, labels = list(zip(*dataset))
        data, labels = jnp.array(data), jnp.array(labels)
        return cls(data, labels)


class DeterministicRandomSampler(Sampler):
    def __init__(self, key, max_len):
        self.key = key
        self.max_len = max_len

    def reset(self, indices):
        self.indices = indices

    def __iter__(self):
        self.key, subkey = jax.random.split(self.key)
        return iter(jax.random.permutation(subkey, self.max_len).tolist())

    def __len__(self):
        return self.max_len


@jit_class
class TorchvisionDataset(Problem):
    def __init__(
        self,
        root: str,
        forward_func: Callable,
        batch_size: int,
        num_passes: int = 1,
        dataset_name: Optional[str] = None,
        train_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        valid_percent: float = 0.2,
        num_workers: int = 0,
        in_memory: bool = False,
        loss_func: Callable = optax.softmax_cross_entropy_with_integer_labels,
    ):
        self.num_passes = num_passes

        if batch_size % num_passes != 0:
            self.batch_size = int(round(batch_size / num_passes))
            warn_msg = f"batch_size isn't evenly divisible by num_passes, the actual batch size will be rounded"
            warnings.warn(warn_msg)
        else:
            self.batch_size = batch_size // num_passes

        self.forward_func = forward_func
        self.loss_func = loss_func
        self.valid_percent = valid_percent
        self.num_workers = num_workers
        self.in_memory = in_memory
        self.collate_fn = np_collate_fn

        if train_dataset is not None and isinstance(train_dataset, Dataset):
            if dataset_name is not None:
                warnings.warn(
                    "When train_dataset and test_dataset are specified, dataset_name is ignored"
                )

            self.train_dataset = train_dataset
            self.test_dataset = test_dataset
        elif dataset_name == "mnist":
            self.train_dataset = datasets.MNIST(
                root, train=True, download=True, transform=np.array
            )
            self.test_dataset = datasets.MNIST(
                root, train=False, download=True, transform=np.array
            )
        elif dataset_name == "cifar10":
            self.train_dataset = datasets.CIFAR10(
                root, train=True, download=True, transform=np.array
            )
            self.test_dataset = datasets.CIFAR10(
                root, train=False, download=True, transform=np.array
            )
        elif dataset_name == "imagenet":
            self.train_dataset = datasets.ImageNet(
                root, train=True, download=True, transform=np.array
            )
            self.test_dataset = datasets.ImageNet(
                root, train=False, download=True, transform=np.array
            )
        else:
            raise ValueError(f"Not supported dataset: {dataset_name}")

        if in_memory:
            if self.num_workers != 0:
                warnings.warn("When in_memory is True, num_workers is ignored")
            self.train_dataset = InMemoryDataset.from_pytorch_dataset(
                self.train_dataset
            )
            self.test_dataset = InMemoryDataset.from_pytorch_dataset(self.test_dataset)

    @jit_method
    def _new_permutation(self, key):
        num_batches = len(self.train_dataset) // self.batch_size
        permutation = jax.random.permutation(key, len(self.train_dataset))
        permutation = permutation[: num_batches * self.batch_size].reshape(
            (-1, self.batch_size)
        )
        return permutation

    def _in_memory_setup(self, key):
        key, subset_key, perm_key = jax.random.split(key, num=3)
        indices = jax.random.permutation(subset_key, len(self.train_dataset))
        valid_len = int(len(self.train_dataset) * self.valid_percent)
        train_len = len(self.train_dataset) - valid_len

        self.valid_dataset = InMemoryDataset(*self.train_dataset[indices[train_len:]])
        self.train_dataset = InMemoryDataset(*self.train_dataset[indices[:train_len]])

        permutation = self._new_permutation(perm_key)

        return State(key=key, permutation=permutation, iter=0, mode=0, metric_func=0)

    def _setup(self, key):
        key, subset_key, sampler_key = jax.random.split(key, num=3)
        indices = jax.random.permutation(subset_key, len(self.train_dataset))
        valid_len = int(len(self.train_dataset) * self.valid_percent)
        train_len = len(self.train_dataset) - valid_len

        self.valid_dataset = Subset(self.train_dataset, indices[train_len:].tolist())
        self.train_dataset = Subset(self.train_dataset, indices[:train_len].tolist())
        self.sampler = DeterministicRandomSampler(sampler_key, len(self.train_dataset))

        self.train_dataloader = DataLoader(
            self.train_dataset,
            sampler=self.sampler,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            drop_last=True,
        )

        self.train_iter = iter(self.train_dataloader)
        # to work with jit, mode and metric are integer type
        # 0 - train, 1 - valid, 2 - test
        return State(mode=0, metric=0)

    def setup(self, key):
        if self.in_memory:
            return self._in_memory_setup(key)
        else:
            return self._setup(key)

    def train(self, state, metric=0):
        return state.update(mode=0, metric_func=metric)

    def valid(self, state, metric=0):
        return state.update(mode=1, metric_func=metric)

    def test(self, state, metric=0):
        return state.update(mode=2, metric_func=metric)

    @jit_method
    def _evaluate_in_memory_train(self, state, batch_params):
        def new_epoch(state):
            key, subkey = jax.random.split(state.key)
            permutation = self._new_permutation(subkey)
            return state.update(key=key, permutation=permutation, iter=0)

        def batch_evaluate(i, state_and_acc):
            state, accumulator = state_and_acc

            state = lax.cond(
                state.iter >= state.permutation.shape[0],
                new_epoch,
                lambda x: x,  # identity
                state,
            )
            data, labels = self.train_dataset[state.permutation[state.iter]]
            losses = self._metric_func(state, data, labels, batch_params)
            return state.update(iter=state.iter + 1), accumulator + losses

        pop_size = tree_leaves(batch_params)[0].shape[0]
        if self.num_passes > 1:
            state, total_loss = lax.fori_loop(
                0, self.num_passes, batch_evaluate, (state, jnp.zeros((pop_size,)))
            )
        else:
            state, total_loss = batch_evaluate(0, (state, jnp.zeros((pop_size,))))

        return total_loss / self.batch_size / self.num_passes, state

    def _evaluate_in_memory_valid(self, state, batch_params):
        num_batches = len(self.valid_dataset) // self.batch_size
        permutation = jnp.arange(num_batches * self.batch_size).reshape(
            num_batches, self.batch_size
        )

        def batch_evaluate(i, accumulated_metric):
            data, labels = self.valid_dataset[permutation[i]]
            return accumulated_metric + self._metric_func(
                state, data, labels, batch_params
            )

        pop_size = tree_leaves(batch_params)[0].shape[0]
        metric = lax.fori_loop(0, num_batches, batch_evaluate, jnp.zeros((pop_size,)))
        return metric / (num_batches * self.batch_size), state

    def _evaluate_train(self, state, batch_params):
        try:
            data, labels = next(self.train_iter)
            # data, labels = jnp.asarray(data), jnp.asarray(labels)
        except StopIteration:
            self.train_iter = iter(self.train_dataloader)
            data, labels = next(self.train_iter)
            # data, labels = jnp.asarray(data), jnp.asarray(labels)

        pop_size = tree_leaves(batch_params)[0].shape[0]
        total_loss = jnp.zeros((pop_size,))
        for _ in range(self.num_passes):
            total_loss += self._calculate_loss(data, labels, batch_params)
        return total_loss / self.batch_size / self.num_passes, state

    def _evaluate_valid(self, state, batch_params):
        valid_dataloader = DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            drop_last=True,
        )

        accumulated_metric = 0
        for data, labels in valid_dataloader:
            accumulated_metric += self._metric_func(state, data, labels, batch_params)
        return accumulated_metric / (len(valid_dataloader) * self.batch_size), state

    def evaluate(self, state, batch_params):
        if self.in_memory:
            return lax.switch(
                state.mode,
                [self._evaluate_in_memory_train, self._evaluate_in_memory_valid],
                state,
                batch_params,
            )
        else:
            return lax.switch(
                state.mode,
                [self._evaluate_train_mode, self._evaluate_valid_mode],
                state,
                batch_params,
            )

    def _metric_func(self, state, data, labels, batch_params):
        return lax.switch(
            state.metric_func,
            [self._calculate_loss, self._calculate_accuracy],
            data,
            labels,
            batch_params,
        )

    @jit_method
    def _calculate_accuracy(self, data, labels, batch_params):
        output = vmap(self.forward_func, in_axes=(0, None))(
            batch_params, data
        )  # (pop_size, batch_size, out_dim)
        output = jnp.argmax(output, axis=2)  # (pop_size, batch_size)
        num_correct = jnp.sum((output == labels), axis=1)  # don't reduce here
        num_correct = num_correct.astype(jnp.float32)

        return num_correct

    @jit_method
    def _calculate_loss(self, data, labels, batch_params):
        output = vmap(self.forward_func, in_axes=(0, None))(batch_params, data)
        loss = jnp.sum(vmap(self.loss_func, in_axes=(0, None))(output, labels), axis=1)
        loss = loss.astype(jnp.float32)

        return loss
