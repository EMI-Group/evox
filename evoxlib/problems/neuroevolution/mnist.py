import numpy as np
import jax
from jax import jit, vmap
import jax.numpy as jnp
from torchvision import datasets
from torch.utils.data import DataLoader
import optax
from functools import partial

import evoxlib as exl
from evoxlib import Problem
from .models import LeNet


def numpy_collate_fn(batch: list):
    data, labels = list(zip(*batch))
    return np.stack(data), np.array(labels)


def new_permutation(key, ds_size, batch_size):
    perms = jax.random.permutation(key, ds_size)
    perms = perms.reshape(-1, batch_size)
    return perms


class MNIST(Problem):
    def __init__(
        self,
        root,
        batch_size,
        forward_func,
        num_workers=0,
        loss_func=optax.softmax_cross_entropy_with_integer_labels
    ):
        self.batch_size = batch_size
        self.forward_func = forward_func
        self.loss_func = loss_func

        mnist_dataset = datasets.MNIST(root, download=True, transform=np.array)
        self.dataloader = DataLoader(
            mnist_dataset,
            batch_size=batch_size,
            collate_fn=numpy_collate_fn,
            num_workers=num_workers,
        )
        self.iter = iter(self.dataloader)

    def evaluate(self, state, X):
        try:
            data, labels = next(self.iter)
        except StopIteration:
            self.iter = iter(self.dataloader)
            data, labels = next(self.iter)

        data, labels = jnp.array(data), jnp.array(labels)
        losses = jit(vmap(partial(self._calculate_loss, data, labels)))(X)
        return state, losses

    @exl.jit_method
    def _calculate_loss(self, data, labels, X):
        # add channel dim
        data = data[:, :, :, jnp.newaxis]
        output = self.forward_func(X, data)
        loss = jnp.mean(self.loss_func(output, labels))
        return loss

