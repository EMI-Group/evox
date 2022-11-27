from functools import partial
from typing import Callable, Union, Optional

import evox
import jax
import jax.numpy as jnp
import numpy as np
import optax
from evox import Problem, State, Stateful
from jax import jit, vmap
from torch.utils.data import DataLoader, Dataset, Sampler, random_split, Subset
from torchvision import datasets


def np_collate_fn(batch: list):
    data, labels = list(zip(*batch))
    return np.stack(data), np.array(labels)


def jnp_collate_fn(batch: list):
    data, labels = list(zip(*batch))
    return jnp.stack(data), jnp.array(labels)


class InMemoryDataset(Dataset):
    def __init__(self, dataset):
        self.in_memory_dataset = [dataset[i] for i in range(len(dataset))]

    def __len__(self):
        return len(self.in_memory_dataset)

    def __getitem__(self, index):
        return self.in_memory_dataset[index]


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
        return max_len


class TorchvisionDataset(Problem):
    def __init__(
        self,
        root: str,
        batch_size: int,
        forward_func: Callable,
        dataset_name: Optional[str] = None,
        train_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        valid_percent: float = 0.2,
        num_workers: int = 0,
        in_memory: bool = False,
        loss_func: Callable = optax.softmax_cross_entropy_with_integer_labels,
    ):
        self.batch_size = batch_size
        self.forward_func = forward_func
        self.loss_func = loss_func
        self.valid_percent = valid_percent
        self.num_workers = num_workers
        self.in_memory = in_memory
        self.collate_fn = np_collate_fn

        if train_dataset is not None and isinstance(dataset, Dataset):
            if dataset_name is not None:
                raise ValueError(
                    f"dataset_name should not be specified when using train_dataset and test_dataset"
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
            assert (
                self.num_workers == 0
            ), "When in_memory is True, num_workers should be 0 to avoid multi-processing"
            self.train_dataset = InMemoryDataset(self.train_dataset)
            self.test_dataset = InMemoryDataset(self.test_dataset)

    def setup(self, key):
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
        return State(key=key, mode=0, metric=0)

    def train(self, state, metric="loss"):
        return state.update(mode=0, metric=(0 if metric == "loss" else 1))

    def valid(self, state, metric="loss"):
        return state.update(mode=1, metric=(0 if metric == "loss" else 1))

    def test(self, state, metric="loss"):
        return state.update(mode=2, metric=(0 if metric == "loss" else 1))

    def _evaluate_train_mode(self, state, batch_params):
        try:
            data, labels = next(self.train_iter)
            # data, labels = jnp.asarray(data), jnp.asarray(labels)
        except StopIteration:
            self.train_iter = iter(self.train_dataloader)
            data, labels = next(self.train_iter)
            # data, labels = jnp.asarray(data), jnp.asarray(labels)

        losses = self._calculate_loss(data, labels, batch_params)
        return state, losses

    def _evaluate_valid_mode(self, state, batch_params):
        valid_dataloader = DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            drop_last=True,
        )
        if state.metric == 0:
            metric_func = self._calculate_loss
        else:
            metric_func = self._calculate_accuracy

        accumulated_metric = 0
        for data, labels in valid_dataloader:
            accumulated_metric += metric_func(data, labels, batch_params)
        return state, accumulated_metric / len(valid_dataloader)

    def _evaluate_test_mode(self, state, batch_params):
        breakpoint()
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            drop_last=True,
        )
        if state.metric == 0:
            metric_func = self._calculate_loss
        else:
            metric_func = self._calculate_accuracy

        accumulated_metric = 0
        for data, labels in test_dataloader:
            accumulated_metric += metric_func(data, labels, batch_params)
        return state, accumulated_metric / len(test_dataloader)

    def evaluate(self, state, batch_params):
        if state.mode == 0:
            return self._evaluate_train_mode(state, batch_params)
        elif state.mode == 1:
            return self._evaluate_valid_mode(state, batch_params)
        elif state.mode == 2:
            return self._evaluate_test_mode(state, batch_params)
        else:
            raise ValueError(f"Unknown mode: {state.mode}")

    @evox.jit_method
    def _calculate_accuracy(self, data, labels, batch_params):
        output = vmap(self.forward_func, in_axes=(0, None))(
            batch_params, data
        )  # (pop_size, batch_size, out_dim)
        output = jnp.argmax(output, axis=2)  # (pop_size, batch_size)
        acc = jnp.mean(output == labels, axis=1)

        return acc

    @evox.jit_method
    def _calculate_loss(self, data, labels, batch_params):
        output = vmap(self.forward_func, in_axes=(0, None))(batch_params, data)
        loss = jnp.mean(vmap(self.loss_func, in_axes=(0, None))(output, labels), axis=1)

        return loss
