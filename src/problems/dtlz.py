import torch
from ..operators import uniform_sampling, grid_sampling
from ..core import vmap, Problem, jit_class


class DTLZTestSuit(Problem):
    """DTLZ Test Suite"""

    def __init__(self, d: int = None, m: int = None, ref_num: int = 1000):
        """Override the setup method to initialize the parameters"""
        super().__init__()
        self.d = d
        self.m = m
        self.ref_num = ref_num
        self.sample = UniformSampling(
            self.ref_num * self.m, self.m
        )  # Assuming UniformSampling is defined
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def pf(self):
        f = self.sample()[0] / 2
        return f


class DTLZ1(DTLZTestSuit):
    def __init__(self, d: int = None, m: int = None, ref_num: int = 1000):
        if m is None:
            m = 3
        if d is None:
            d = m + 4
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        m = self.m
        n, d = X.shape
        g = 100 * (
            d
            - m
            + 1
            + torch.sum(
                (X[:, m - 1 :] - 0.5) ** 2
                - torch.cos(20 * torch.pi * (X[:, m - 1 :] - 0.5)),
                dim=1,
                keepdim=True,
            )
        )
        flip_cumprod = torch.flip(
            torch.cumprod(
                torch.cat([torch.ones((n, 1), device=X.device), X[:, : m - 1]], dim=1),
                dim=1,
            ),
            dims=[1],
        )
        rest_part = torch.cat(
            [
                torch.ones((n, 1), device=X.device),
                1 - torch.flip(X[:, : m - 1], dims=[1]),
            ],
            dim=1,
        )
        f = 0.5 * (1 + g) * flip_cumprod * rest_part
        return f


class DTLZ2(DTLZTestSuit):
    def __init__(self, d: int = None, m: int = None, ref_num: int = 1000):
        if m is None:
            m = 3
        if d is None:
            d = m + 9
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        m = self.m
        g = torch.sum((X[:, m - 1 :] - 0.5) ** 2, dim=1, keepdim=True)
        f = (
            (1 + g)
            * torch.flip(
                torch.cumprod(
                    torch.cat(
                        [
                            torch.ones((X.shape[0], 1), device=X.device),
                            torch.maximum(
                                torch.cos(X[:, : m - 1] * torch.pi / 2),
                                torch.tensor(0.0, device=X.device),
                            ),
                        ],
                        dim=1,
                    ),
                    dim=1,
                ),
                dims=[1],
            )
            * torch.cat(
                [
                    torch.ones((X.shape[0], 1), device=X.device),
                    torch.sin(torch.flip(X[:, : m - 1], dims=[1]) * torch.pi / 2),
                ],
                dim=1,
            )
        )

        return f

    def pf(self):
        f = self.sample()[0]
        f /= torch.sqrt(f.pow(2).sum(dim=1, keepdim=True))
        return f


class DTLZ3(DTLZ2):
    def __init__(self, d: int = None, m: int = None, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        n, d = X.shape
        m = self.m
        g = 100 * (
            d
            - m
            + 1
            + torch.sum(
                (X[:, m - 1 :] - 0.5) ** 2
                - torch.cos(20 * torch.pi * (X[:, m - 1 :] - 0.5)),
                dim=1,
                keepdim=True,
            )
        )
        f = (
            (1 + g)
            * torch.flip(
                torch.cumprod(
                    torch.cat(
                        [
                            torch.ones((n, 1), device=X.device),
                            torch.maximum(
                                torch.cos(X[:, : m - 1] * torch.pi / 2),
                                torch.tensor(0.0, device=X.device),
                            ),
                        ],
                        dim=1,
                    ),
                    dim=1,
                ),
                dims=[1],
            )
            * torch.cat(
                [
                    torch.ones((n, 1), device=X.device),
                    torch.sin(torch.flip(X[:, : m - 1], dims=[1]) * torch.pi / 2),
                ],
                dim=1,
            )
        )
        return f


class DTLZ4(DTLZ2):
    def __init__(self, d: int = None, m: int = None, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        m = self.m

        X[:, : m - 1] = X[:, : m - 1].pow(100)

        g = torch.sum((X[:, m - 1 :] - 0.5) ** 2, dim=1, keepdim=True)

        f = (
            (1 + g)
            * torch.flip(
                torch.cumprod(
                    torch.cat(
                        [
                            torch.ones((g.shape[0], 1), device=X.device),
                            torch.maximum(
                                torch.cos(X[:, : m - 1] * torch.pi / 2),
                                torch.tensor(0.0, device=X.device),
                            ),
                        ],
                        dim=1,
                    ),
                    dim=1,
                ),
                dims=[1],
            )
            * torch.cat(
                [
                    torch.ones((g.shape[0], 1), device=X.device),
                    torch.sin(torch.flip(X[:, : m - 1], dims=[1]) * torch.pi / 2),
                ],
                dim=1,
            )
        )

        return f


class DTLZ5(DTLZTestSuit):
    def __init__(self, d: int = None, m: int = None, ref_num: int = 1000):

        if m is None:
            m = 3
        if d is None:
            d = m + 9
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        m = self.m

        g = torch.sum((X[:, m - 1 :] - 0.5) ** 2, dim=1, keepdim=True)
        temp = g.repeat(1, m - 2)

        X[:, 1 : m - 1] = (1 + 2 * temp * X[:, 1 : m - 1]) / (2 + 2 * temp)

        f = (
            (1 + g)
            * torch.flip(
                torch.cumprod(
                    torch.cat(
                        [
                            torch.ones((g.shape[0], 1), device=X.device),
                            torch.maximum(
                                torch.cos(X[:, : m - 1] * torch.pi / 2),
                                torch.tensor(0.0, device=X.device),
                            ),
                        ],
                        dim=1,
                    ),
                    dim=1,
                ),
                dims=[1],
            )
            * torch.cat(
                [
                    torch.ones((g.shape[0], 1), device=X.device),
                    torch.sin(torch.flip(X[:, : m - 1], dims=[1]) * torch.pi / 2),
                ],
                dim=1,
            )
        )

        return f

    def pf(self):
        n = self.ref_num * self.m

        f = torch.vstack(
            (
                torch.hstack(
                    (
                        torch.arange(0, 1, 1.0 / (n - 1), device=self.device),
                        torch.tensor(1.0, device=self.device),
                    )
                ),
                torch.hstack(
                    (
                        torch.arange(1, 0, -1.0 / (n - 1), device=self.device),
                        torch.tensor(0.0, device=self.device),
                    )
                ),
            )
        ).T

        f /= torch.tile(
            torch.sqrt(torch.sum(f**2, dim=1, keepdim=True)), (1, f.shape[1])
        )

        for i in range(self.m - 2):
            f = torch.cat((f[:, 0:1], f), dim=1)

        f = f / torch.sqrt(torch.tensor(2.0, device=self.device)) ** torch.tile(
            torch.hstack(
                (
                    torch.tensor(self.m - 2, device=self.device),
                    torch.arange(self.m - 2, -1, -1, device=self.device),
                )
            ),
            (f.shape[0], 1),
        )
        return f


class DTLZ6(DTLZTestSuit):
    def __init__(self, d: int = None, m: int = None, ref_num: int = 1000):

        if m is None:
            m = 3
        if d is None:
            d = m + 9
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        m = self.m
        g = torch.sum((X[:, m - 1 :] ** 0.1), dim=1, keepdim=True)
        temp = torch.tile(g, (1, m - 2))
        X[:, 1 : m - 1] = (1 + 2 * temp * X[:, 1 : m - 1]) / (2 + 2 * temp)

        f = (
            torch.tile(1 + g, (1, m))
            * torch.flip(
                torch.cumprod(
                    torch.cat(
                        [
                            torch.ones((X.shape[0], 1), device=X.device),
                            torch.maximum(
                                torch.cos(X[:, : m - 1] * torch.pi / 2),
                                torch.tensor(0.0, device=X.device),
                            ),
                        ],
                        dim=1,
                    ),
                    dim=1,
                ),
                dims=[1],
            )
            * torch.cat(
                [
                    torch.ones((X.shape[0], 1), device=X.device),
                    torch.sin(torch.flip(X[:, : m - 1], dims=[1]) * torch.pi / 2),
                ],
                dim=1,
            )
        )
        return f

    def pf(self):
        n = self.ref_num * self.m

        # Ensure the tensor is created on the same device (use X.device if needed)
        f = torch.vstack(
            (
                torch.hstack(
                    (
                        torch.arange(0, 1, 1.0 / (n - 1), device=self.device),
                        torch.tensor(1.0, device=self.device),
                    )
                ),
                torch.hstack(
                    (
                        torch.arange(1, 0, -1.0 / (n - 1), device=self.device),
                        torch.tensor(0.0, device=self.device),
                    )
                ),
            )
        ).T

        f /= torch.tile(
            torch.sqrt(torch.sum(f**2, dim=1, keepdim=True)), (1, f.shape[1])
        )

        for i in range(self.m - 2):
            f = torch.cat((f[:, 0:1], f), dim=1)

        f = f / torch.sqrt(torch.tensor(2.0, device=self.device)) ** torch.tile(
            torch.hstack(
                (
                    torch.tensor(self.m - 2, device=self.device),
                    torch.arange(self.m - 2, -1, -1, device=self.device),
                )
            ),
            (f.shape[0], 1),
        )
        return f


class DTLZ7(DTLZTestSuit):
    def __init__(self, d: int = None, m: int = None, ref_num: int = 1000):

        if m is None:
            m = 3
        if d is None:
            d = m + 19
        super().__init__(d, m, ref_num)
        self.sample, _ = grid_sampling(self.ref_num * self.m, self.m - 1)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        n, d = X.shape
        m = self.m
        f = torch.zeros((n, m), device=X.device)
        g = 1 + 9 * torch.mean(X[:, m - 1 :], dim=1, keepdim=True)

        f[:, : m - 1] = X[:, : m - 1]

        term = torch.sum(
            f[:, : m - 1]
            / (1 + torch.tile(g, (1, m - 1)))
            * (1 + torch.sin(3 * torch.pi * f[:, : m - 1])),
            dim=1,
            keepdim=True,
        )
        f[:, m - 1 :] = (1 + g) * (m - term)

        return f

    def pf(self):
        interval = torch.tensor([0, 0.251412, 0.631627, 0.859401], device=self.device)
        median = (interval[1] - interval[0]) / (
            interval[3] - interval[2] + interval[1] - interval[0]
        ).to(self.device)

        x = self.sample()[0].to(self.device)

        mask_less_equal_median = x <= median
        mask_greater_median = x > median

        x = torch.where(
            mask_less_equal_median,
            x * (interval[1] - interval[0]) / median + interval[0],
            x,
        )
        x = torch.where(
            mask_greater_median,
            (x - median) * (interval[3] - interval[2]) / (1 - median) + interval[2],
            x,
        )

        last_col = 2 * (
            self.m
            - torch.sum(x / 2 * (1 + torch.sin(3 * torch.pi * x)), dim=1, keepdim=True)
        )

        pf = torch.cat([x, last_col], dim=1)
        return pf


if __name__ == "__main__":
    # Problem dimensions and objectives
    d = 12
    m = 3
    ref_num = 1000

    # Create an instance of the DTLZ1 problem
    problem = DTLZ1(d=d, m=m, ref_num=ref_num)

    # Generate a random population (100 individuals, each with d features)
    # population = torch.rand(100, d)
    population = torch.tensor(
        [
            [
                0.1,
                0.5,
                0.2,
                0.1,
                0.5,
                0.2,
                0.1,
                0.5,
                0.2,
                0.1,
                0.5,
                0.2,
            ],
            [
                0.8,
                0.8,
                0.9,
                0.8,
                0.8,
                0.9,
                0.8,
                0.8,
                0.9,
                0.8,
                0.8,
                0.9,
            ],
        ]
    )

    # Evaluate the population
    fitness = problem.evaluate(population)

    print("Fitness of the population:")
    print(fitness)

    # Get the Pareto front for DTLZ1
    pf = problem.pf()
    print("Pareto front:")
    print(pf)
