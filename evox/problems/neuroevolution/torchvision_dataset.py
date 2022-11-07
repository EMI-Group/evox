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

from .models import LeNet


def collate_fn(batch: list):
    data, labels = list(zip(*batch))
    return jnp.array(np.stack(data)), jnp.array(labels)


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
        loss_func: Callable = optax.softmax_cross_entropy_with_integer_labels,
    ):
        self.batch_size = batch_size
        self.forward_func = forward_func
        self.loss_func = loss_func
        self.valid_percent = valid_percent
        self.num_workers = num_workers

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
            raise ValueError(f"Not supported dataset: {dataset}")

    def setup(self, key):
        key, subset_key, sampler_key = jax.random.split(key, num=3)
        indices = jax.random.permutation(subset_key, len(self.train_dataset))
        valid_len = int(len(self.train_dataset) * self.valid_percent)
        train_len = len(self.train_dataset) - valid_len

        self.train_dataset = Subset(self.train_dataset, indices[:train_len].tolist())
        self.valid_dataset = Subset(self.train_dataset, indices[train_len:].tolist())

        self.sampler = DeterministicRandomSampler(sampler_key, len(self.train_dataset))

        self.train_dataloader = DataLoader(
            self.train_dataset,
            sampler=self.sampler,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            drop_last=True,
        )

        self.train_iter = iter(self.train_dataloader)
        # to work with jit, mode is an integer
        # 0 - train, 1 - valid, 2 - test
        return State(key=key, mode=0)

    def train(self, state):
        return state.update(mode=0)

    def valid(self, state):
        return state.update(mode=1)

    def test(self, state):
        return state.update(mode=2)

    def _evaluate_train_mode(self, state, batch_params):
        try:
            data, labels = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_dataloader)
            data, labels = next(self.train_iter)

        losses = self._calculate_loss(data, labels, batch_params)
        return state, losses

    def _evaluate_valid_mode(self, state, batch_params):
        valid_dataset = DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            drop_last=True,
        )
        accumulated_loss = 0
        for data, labels in valid_dataset:
            accumulated_loss += self._calculate_loss(data, labels, batch_params)
        return accumulated_loss / len(valid_dataset)

    def _evaluate_test_mode(self, state, batch_params):
        test_dataset = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            drop_last=True,
        )
        accumulated_loss = 0
        for data, labels in test_dataloader:
            accumulated_loss += self._calculate_loss(data, labels, batch_params)
        return accumulated_loss / len(test_dataloader)

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
    def _calculate_loss(self, data, labels, batch_params):
        output = vmap(self.forward_func, in_axes=(0, None))(batch_params, data)
        loss = jnp.mean(vmap(self.loss_func, in_axes=(0, None))(output, labels), axis=1)

        return loss
