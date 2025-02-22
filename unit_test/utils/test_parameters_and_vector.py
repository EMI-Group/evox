import unittest

import torch
import torch.nn as nn

from evox.core import vmap
from evox.utils.parameters_and_vector import ParamsAndVector


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(3, 3, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(108, 32), nn.ReLU(), nn.Linear(32, 10))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class Dim0_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cat = nn.Parameter(torch.rand(1))
        self.mouse = nn.Parameter(torch.rand(1))
        self.dog = nn.Parameter(torch.rand(1))


class Dim1_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cat = nn.Parameter(torch.rand(1, 2))
        self.mouse = nn.Parameter(torch.rand(1, 1))
        self.dog = nn.Parameter(torch.rand(1, 3))


class Dim2_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cat = nn.Parameter(torch.rand(1, 2, 3))
        self.mouse = nn.Parameter(torch.rand(1, 1, 4))
        self.dog = nn.Parameter(torch.rand(1, 3, 2))


class TestParamsAndVector(unittest.TestCase):
    def setUp(self):
        self.model = SimpleCNN()
        self.adapter = ParamsAndVector(dummy_model=self.model)
        self.model_params = dict(self.model.named_parameters())

    def params_all_close(self, params1, params2, atol=1e-5):
        # Check if two dictionaries of parameters are equal
        keys = set(params1.keys())
        self.assertEqual(keys, set(params2.keys()))
        for key in keys:
            self.assertTrue(torch.allclose(params1[key], params2[key], atol=atol))

    def test_to_vector_and_to_params(self):
        flat_params = self.adapter.to_vector(self.model_params)
        restored_params = self.adapter.to_params(flat_params)
        self.params_all_close(self.model_params, restored_params)

    def test_batched_to_vector_and_batched_to_params(self):
        model_groups = [Dim0_Model(), Dim1_Model(), Dim2_Model()]
        BATCH_SIZE = 5

        for test_model in model_groups:
            batched_params = {
                key: torch.stack([value] + [torch.randn_like(value) for _ in range(BATCH_SIZE - 1)])
                for key, value in dict(test_model.named_parameters()).items()
            }
            temp_adapter = ParamsAndVector(dummy_model=test_model)
            batched_flat_params = temp_adapter.batched_to_vector(batched_params)
            batched_restored_params = temp_adapter.batched_to_params(batched_flat_params)

            self.params_all_close(batched_params, batched_restored_params)

    def test_vmap(self):
        BATCH_SIZE = 5
        VMAP_SIZE = 3
        model = SimpleCNN()
        adapter = ParamsAndVector(dummy_model=model)
        model_params = dict(model.named_parameters())
        batched_params = {
            key: torch.stack([value] + [torch.randn_like(value) for _ in range(BATCH_SIZE - 1)])
            for key, value in model_params.items()
        }
        batched_params = {
            key: torch.stack([value] + [torch.randn_like(value) for _ in range(VMAP_SIZE - 1)])
            for key, value in batched_params.items()
        }
        vmap_batch_to_vector = torch.compile(vmap(adapter.batched_to_vector))
        vmap_batch_to_params = torch.compile(vmap(adapter.batched_to_params))
        batched_flat_params = vmap_batch_to_vector(batched_params)
        print(batched_flat_params.shape)
        batched_restored_params = vmap_batch_to_params(batched_flat_params)

        self.params_all_close(batched_params, batched_restored_params)
