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
    SeparableNES,
)
from evox.workflows import StdWorkflow

from .test_base import Sphere, TestBase


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

    def test_asebo_after_subspace_activates(self):
        # run_all stops after a few generations; the gradient subspace only takes over once more than
        # subspace_dims gradients are buffered, including when subspace_dims < dim
        for subspace_dims in (None, 2):
            algo = ASEBO(pop_size=self.pop_size, center_init=self.center, subspace_dims=subspace_dims)
            workflow = StdWorkflow(algo, Sphere())
            workflow.init_step()
            for _ in range(3 * self.dim):
                workflow.step()
            self.assertTrue(torch.isfinite(algo.center).all())
            self.assertTrue(0 <= float(algo.alpha) <= 1)
            # UUT is an orthogonal projector onto at most subspace_dims directions
            UUT = algo.UUT
            torch.testing.assert_close(UUT @ UUT, UUT, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(UUT, UUT.T)
            self.assertLessEqual(round(float(torch.trace(UUT))), algo.subspace_dims)
            self.assertGreater(float(torch.trace(UUT)), 0.5)

    def test_cmaes(self):
        algo = CMAES(
            mean_init=self.center,
            sigma=5,
        )
        self.run_all(algo)

    def test_cmaes_cumulation_constant(self):
        # c_c = (4 + mu_eff / n) / (n + 4 + 2 mu_eff / n), eq. (56) of the CMA-ES tutorial; it must stay
        # in (0, 1] so that sqrt(c_c * (2 - c_c) * mu_eff) in the p_c update is real
        for dim, pop_size in [(10, 10), (10, 100), (10, 1000), (100, 1000)]:
            algo = CMAES(mean_init=torch.zeros(dim), sigma=1, pop_size=pop_size)
            mu_eff = float(algo.mu_eff)
            expected = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            self.assertAlmostEqual(float(algo.c_c), expected, places=5)
            self.assertLessEqual(float(algo.c_c), 1)
            p_c = algo._update_path_c(torch.zeros(dim), torch.tensor(1.0), torch.ones(dim))
            self.assertTrue(torch.isfinite(p_c).all())

    def test_cmaes_covariance_rank_one(self):
        algo = CMAES(mean_init=torch.zeros(3), sigma=1, pop_size=4)
        C = torch.eye(3)
        p_c = torch.tensor([1.0, 2.0, -1.0])
        old_mean = torch.zeros(1, 3)
        population = old_mean.expand(algo.mu, -1)
        h_sigma = torch.tensor(1.0)

        rank_one = torch.tensor([[1.0, 2.0, -1.0], [2.0, 4.0, -2.0], [-1.0, -2.0, 1.0]])
        expected = (1 - algo.c_1 - algo.c_mu) * C + algo.c_1 * rank_one
        actual = algo._update_covariance_matrix(C, p_c, population, old_mean, h_sigma)

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual, actual.T)

    def test_cmaes_covariance_rank_one_compiled_vmap(self):
        algo = CMAES(mean_init=torch.zeros(3), sigma=1, pop_size=4)
        C = torch.eye(3)
        paths = torch.tensor([[1.0, 2.0, -1.0], [-2.0, 1.0, 3.0]])
        old_mean = torch.zeros(1, 3)
        population = old_mean.expand(algo.mu, -1)
        h_sigma = torch.tensor(1.0)

        expected = (1 - algo.c_1 - algo.c_mu) * C + algo.c_1 * paths.unsqueeze(-1) * paths.unsqueeze(-2)
        update = torch.vmap(algo._update_covariance_matrix, in_dims=(None, 0, None, None, None))
        torch.testing.assert_close(update(C, paths, population, old_mean, h_sigma), expected)
        actual = torch.compile(update, fullgraph=True)(C, paths, population, old_mean, h_sigma)

        torch.testing.assert_close(actual, expected)
