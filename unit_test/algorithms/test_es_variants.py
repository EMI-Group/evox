import torch

from evox.algorithms import ARS, ASEBO, DES, ESMC, SNES, XNES, GuidedES, Noise_reuse_es, OpenES, PersistentES, SeparableNES, CMAES

from .test_base import TestBase


class TestESVariants(TestBase):
    def setUp(self):
        pop_size = 10
        dim = 4
        lb = -10 * torch.ones(dim)
        ub = 10 * torch.ones(dim)
        self.algos = [
            OpenES(
                pop_size=pop_size,
                center_init=torch.rand(dim) * (ub - lb) + lb,
                learning_rate=1e-2,
                noise_stdev=5,
                optimizer=None,
            ),
            OpenES(
                pop_size=pop_size,
                center_init=torch.rand(dim) * (ub - lb) + lb,
                learning_rate=1e-2,
                noise_stdev=5,
                optimizer="adam",
            ),
            XNES(
                pop_size=pop_size,
                init_mean=torch.rand(dim) * (ub - lb) + lb,
                init_covar=torch.eye(dim),
            ),
            SeparableNES(
                pop_size=pop_size,
                init_mean=torch.rand(dim) * (ub - lb) + lb,
                init_std=torch.full((dim,), 1),
            ),
            DES(
                pop_size=pop_size,
                center_init=torch.rand(dim) * (ub - lb) + lb,
            ),
            ESMC(
                pop_size=pop_size | 1,
                center_init=torch.rand(dim) * (ub - lb) + lb,
            ),
            ESMC(
                pop_size=pop_size | 1,
                center_init=torch.rand(dim) * (ub - lb) + lb,
                optimizer="adam",
            ),
            SNES(
                pop_size=pop_size,
                center_init=torch.rand(dim) * (ub - lb) + lb,
                weight_type="recomb",
            ),
            SNES(
                pop_size=pop_size,
                center_init=torch.rand(dim) * (ub - lb) + lb,
                weight_type="temp",
            ),
            PersistentES(
                pop_size=pop_size,
                center_init=torch.rand(dim) * (ub - lb) + lb,
            ),
            PersistentES(
                pop_size=pop_size,
                center_init=torch.rand(dim) * (ub - lb) + lb,
                optimizer="adam",
            ),
            GuidedES(
                pop_size=pop_size,
                center_init=torch.rand(dim) * (ub - lb) + lb,
            ),
            GuidedES(
                pop_size=pop_size,
                center_init=torch.rand(dim) * (ub - lb) + lb,
                optimizer="adam",
            ),
            Noise_reuse_es(
                pop_size=pop_size,
                center_init=torch.rand(dim) * (ub - lb) + lb,
            ),
            Noise_reuse_es(
                pop_size=pop_size,
                center_init=torch.rand(dim) * (ub - lb) + lb,
                optimizer="adam",
            ),
            ARS(
                pop_size=pop_size,
                center_init=torch.rand(dim) * (ub - lb) + lb,
            ),
            ARS(
                pop_size=pop_size,
                center_init=torch.rand(dim) * (ub - lb) + lb,
                optimizer="adam",
            ),
            ASEBO(
                pop_size=pop_size,
                center_init=torch.rand(dim) * (ub - lb) + lb,
            ),
            ASEBO(
                pop_size=pop_size,
                center_init=torch.rand(dim) * (ub - lb) + lb,
                optimizer="adam",
            ),
            CMAES(
                mean_init=torch.rand(dim) * (ub - lb) + lb,
                sigma=5,
            ),
        ]

    def test_es_variants(self):
        for algo in self.algos:
            self.run_algorithm(algo)
            self.run_trace_algorithm(algo)
            self.run_vmap_algorithm(algo)
