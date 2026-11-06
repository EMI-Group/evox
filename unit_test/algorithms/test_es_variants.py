import torch

from evox.algorithms import (
    ARS,
    ASEBO,
    CMAES,
    DES,
    ESMC,
    SNES,
    XNES,
    GuidedES,
    NoiseReuseES,
    OpenES,
    PersistentES,
    PlusLambdaES,
    SeparableNES,
)

from .test_base import TestBase


class TestESVariants(TestBase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
        self.pop_size = 10
        self.dim = 4
        self.lb = -10 * torch.ones(self.dim)
        self.ub = 10 * torch.ones(self.dim)
        self.center = torch.rand(self.dim) * (self.ub - self.lb) + self.lb

    def test_openes(self):
        algo = OpenES(
            pop_size=self.pop_size,
            center_init=self.center,
            learning_rate=1e-2,
            noise_stdev=5,
            optimizer=None,
        )
        self.run_all(algo)

    def test_openes_adam(self):
        algo = OpenES(
            pop_size=self.pop_size,
            center_init=self.center,
            learning_rate=1e-2,
            noise_stdev=5,
            optimizer="adam",
        )
        self.run_all(algo)

    def test_xnes(self):
        algo = XNES(
            pop_size=self.pop_size,
            init_mean=self.center,
            init_covar=torch.eye(self.dim),
        )
        self.run_all(algo)

    def test_sepnes(self):
        algo = SeparableNES(
            pop_size=self.pop_size,
            init_mean=self.center,
            init_std=torch.full((self.dim,), 1),
        )
        self.run_all(algo)

    def test_des(self):
        algo = DES(
            pop_size=self.pop_size,
            center_init=self.center,
        )
        self.run_all(algo)

    def test_esmc(self):
        algo = ESMC(
            pop_size=self.pop_size | 1,
            center_init=self.center,
        )
        self.run_all(algo)

    def test_esmc_adam(self):
        algo = ESMC(
            pop_size=self.pop_size | 1,
            center_init=self.center,
            optimizer="adam",
        )
        self.run_all(algo)

    def test_snes_recomb(self):
        algo = SNES(
            pop_size=self.pop_size,
            center_init=self.center,
            weight_type="recomb",
        )
        self.run_all(algo)

    def test_snes_temp(self):
        algo = SNES(
            pop_size=self.pop_size,
            center_init=self.center,
            weight_type="temp",
        )
        self.run_all(algo)

    def test_plus_lambda_es(self):
        algo = PlusLambdaES(
            center_init=self.center,
            lambda_offspring=2,
            sigma=0.5,
            delta_accept=0.0,
            lb=self.lb,
            ub=self.ub,
        )
        self.run_all(algo)

    def test_plus_lambda_es_schedule(self):
        algo = PlusLambdaES(
            center_init=self.center,
            lambda_offspring=2,
            sigma=0.5,
            delta_accept=0.1,
            lb=self.lb,
            ub=self.ub,
            sigma_schedule=torch.tensor([0.5, 0.25]),
        )
        self.run_all(algo)

    def test_persistentes(self):
        algo = PersistentES(
            pop_size=self.pop_size,
            center_init=self.center,
        )
        self.run_all(algo)

    def test_persistentes_adam(self):
        algo = PersistentES(
            pop_size=self.pop_size,
            center_init=self.center,
            optimizer="adam",
        )
        self.run_all(algo)

    def test_guided_es(self):
        algo = GuidedES(
            pop_size=self.pop_size,
            center_init=self.center,
        )
        self.run_all(algo)

    def test_guided_es_adam(self):
        algo = GuidedES(
            pop_size=self.pop_size,
            center_init=self.center,
            optimizer="adam",
        )
        self.run_all(algo)

    def test_noisereusees(self):
        algo = NoiseReuseES(
            pop_size=self.pop_size,
            center_init=self.center,
        )
        self.run_all(algo)

    def test_noisereusees_adam(self):
        algo = NoiseReuseES(
            pop_size=self.pop_size,
            center_init=self.center,
            optimizer="adam",
        )
        self.run_all(algo)

    def test_ars(self):
        algo = ARS(
            pop_size=self.pop_size,
            center_init=self.center,
        )
        self.run_all(algo)

    def test_ars_adam(self):
        algo = ARS(
            pop_size=self.pop_size,
            center_init=self.center,
            optimizer="adam",
        )
        self.run_all(algo)

    def test_asebo(self):
        algo = ASEBO(
            pop_size=self.pop_size,
            center_init=self.center,
        )
        self.run_all(algo)

    def test_asebo_adam(self):
        algo = ASEBO(
            pop_size=self.pop_size,
            center_init=self.center,
            optimizer="adam",
        )
        self.run_all(algo)

    def test_cmaes(self):
        return
        algo = CMAES(
            mean_init=self.center,
            sigma=5,
        )
        self.run_all(algo)
