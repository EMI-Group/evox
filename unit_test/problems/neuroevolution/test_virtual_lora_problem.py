"""Tests for ``evox.problems.neuroevolution.VirtualLoRAProblem``.

These tests validate the virtual LoRA-based neuroevolution problem: that
``evaluate`` returns correctly shaped fitness, that a ``sigma=0`` perturbation
reproduces the unperturbed center-model loss, that the seeded perturbations are
deterministic and seed-dependent, and that both weight (2-D) and bias (1-D)
parameters are perturbed.
"""

import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from evox.problems.neuroevolution import VirtualLoRAProblem
from evox.utils import ParamsAndVector


def make_simple_mlp():
    """A small 2-layer MLP: Linear(10,20) -> ReLU -> Linear(20,5)."""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )


def make_data_loader(n_samples=64, in_features=10, n_classes=5):
    """A deterministic classification ``DataLoader`` (no shuffle)."""
    X = torch.randn(n_samples, in_features)
    y = torch.randint(0, n_classes, (n_samples,))
    return DataLoader(TensorDataset(X, y), batch_size=16, shuffle=False)


def count_params(model):
    """Total number of scalar parameters in ``model``."""
    return sum(p.numel() for p in model.parameters())


class TestVirtualLoRAProblem(unittest.TestCase):
    def setUp(self):
        torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)

    def test_basic_evaluation(self):
        """evaluate returns a (pop_size,) fitness tensor."""
        model = make_simple_mlp()
        data_loader = make_data_loader()
        criterion = nn.CrossEntropyLoss(reduction="none")
        problem = VirtualLoRAProblem(
            model=model,
            data_loader=data_loader,
            criterion=criterion,
            lora_rank=4,
        )

        dim = count_params(model)
        pop_size = 8
        center_flat = torch.zeros(dim)
        seeds = torch.arange(1, pop_size + 1, dtype=torch.int64)
        sigma = 0.1

        fitness = problem.evaluate((center_flat, seeds, sigma))
        self.assertEqual(fitness.shape, (pop_size,))

    def test_sigma_zero_matches_center_loss(self):
        """With sigma=0 every individual equals the center-model loss."""
        model = make_simple_mlp()
        data_loader = make_data_loader()
        criterion = nn.CrossEntropyLoss(reduction="none")
        problem = VirtualLoRAProblem(
            model=model,
            data_loader=data_loader,
            criterion=criterion,
            lora_rank=4,
        )

        # Build a center matching the model's actual parameters.
        pv = ParamsAndVector(model)
        center_flat = pv.to_vector(dict(model.named_parameters()))

        pop_size = 8
        seeds = torch.arange(1, pop_size + 1, dtype=torch.int64)

        fitness = problem.evaluate((center_flat, seeds, 0.0))

        # All individuals must be identical: sigma=0 => no perturbation.
        self.assertTrue(torch.allclose(fitness, fitness[0].expand_as(fitness)))

        # Manually compute the center-model loss on the first batch the problem
        # consumed (data_loader has shuffle=False, so the first batch is the
        # first yielded batch). Default n_batch_per_eval=1 and reduction='mean'.
        inputs, labels = next(iter(data_loader))
        with torch.no_grad():
            logits = model(inputs)
            per_sample_loss = criterion(logits, labels)
        expected = per_sample_loss.mean()

        self.assertTrue(torch.allclose(fitness, expected.expand_as(fitness)))

    def test_determinism(self):
        """Identical seeds produce identical fitness across calls.

        We use ``n_batch_per_eval=-1`` (whole dataset) so that each ``evaluate``
        call iterates the dataset from scratch and operates on identical data;
        otherwise the internal ``data_loader_iter`` advances between calls.
        """
        model = make_simple_mlp()
        data_loader = make_data_loader()
        criterion = nn.CrossEntropyLoss(reduction="none")
        problem = VirtualLoRAProblem(
            model=model,
            data_loader=data_loader,
            criterion=criterion,
            lora_rank=4,
            n_batch_per_eval=-1,
        )

        dim = count_params(model)
        pop_size = 8
        center_flat = torch.zeros(dim)
        seeds = torch.arange(1, pop_size + 1, dtype=torch.int64)
        sigma = 0.1

        result1 = problem.evaluate((center_flat, seeds, sigma))
        result2 = problem.evaluate((center_flat, seeds, sigma))
        self.assertTrue(torch.equal(result1, result2))

    def test_different_seeds_different_fitness(self):
        """Different seeds (with sigma>0) yield different fitness."""
        model = make_simple_mlp()
        data_loader = make_data_loader()
        criterion = nn.CrossEntropyLoss(reduction="none")
        problem = VirtualLoRAProblem(
            model=model,
            data_loader=data_loader,
            criterion=criterion,
            lora_rank=4,
        )

        dim = count_params(model)
        pop_size = 8
        center_flat = torch.zeros(dim)
        sigma = 0.1

        seeds_a = torch.arange(1, pop_size + 1, dtype=torch.int64)
        seeds_b = torch.arange(100, 100 + pop_size, dtype=torch.int64)

        fit_a = problem.evaluate((center_flat, seeds_a, sigma))
        fit_b = problem.evaluate((center_flat, seeds_b, sigma))
        self.assertFalse(torch.equal(fit_a, fit_b))

    def test_multi_layer_mlp(self):
        """A 3-layer MLP with mixed activations forward-passes correctly."""
        model = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
        )
        data_loader = make_data_loader(n_samples=64, in_features=10, n_classes=4)
        criterion = nn.CrossEntropyLoss(reduction="none")
        problem = VirtualLoRAProblem(
            model=model,
            data_loader=data_loader,
            criterion=criterion,
            lora_rank=4,
        )

        dim = count_params(model)
        pop_size = 6
        center_flat = torch.zeros(dim)
        seeds = torch.arange(1, pop_size + 1, dtype=torch.int64)
        sigma = 0.1

        fitness = problem.evaluate((center_flat, seeds, sigma))
        self.assertEqual(fitness.shape, (pop_size,))

    def test_1d_and_2d_perturbation(self):
        """With sigma>0 both weight (2-D) and bias (1-D) are perturbed."""
        model = make_simple_mlp()
        data_loader = make_data_loader()
        criterion = nn.CrossEntropyLoss(reduction="none")
        problem = VirtualLoRAProblem(
            model=model,
            data_loader=data_loader,
            criterion=criterion,
            lora_rank=4,
        )

        dim = count_params(model)
        pop_size = 8
        center_flat = torch.zeros(dim)
        seeds = torch.arange(1, pop_size + 1, dtype=torch.int64)

        fit_sigma_zero = problem.evaluate((center_flat, seeds, 0.0))
        fit_sigma_pos = problem.evaluate((center_flat, seeds, 0.1))
        self.assertFalse(torch.allclose(fit_sigma_pos, fit_sigma_zero))


if __name__ == "__main__":
    unittest.main()
