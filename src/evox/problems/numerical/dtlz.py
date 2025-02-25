from typing import Optional

import torch

from ...core import Problem
from ...operators.sampling import grid_sampling, uniform_sampling


class DTLZ(Problem):
    """
    Base class for DTLZ test suite problems in multi-objective optimization.

    Inherit this class to implement specific DTLZ problem variants.

    :param d: Number of decision variables.
    :param m: Number of objectives.
    :param ref_num: Number of reference points used in the problem.
    """

    def __init__(self, d: int = None, m: int = None, ref_num: int = 1000, device: Optional[torch.device] = None):
        """Override the setup method to initialize the parameters"""
        super().__init__()
        device = device or torch.get_default_device()
        self.d = d
        self.m = m
        self.ref_num = ref_num
        sample, _ = uniform_sampling(self.ref_num * self.m, self.m)  # Assuming UniformSampling is defined
        self.sample = sample.to(device=device)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to evaluate the objective values for given decision variables.

        :param X: A tensor of shape (n, d), where n is the number of solutions and d is the number of decision variables.
        :return: A tensor of shape (n, m) representing the objective values for each solution.
        """
        raise NotImplementedError()

    def pf(self):
        """
        Return the Pareto front for the problem.

        :return: A tensor representing the Pareto front.
        """
        f = self.sample / 2
        return f


class DTLZ1(DTLZ):
    def __init__(self, d: int = 7, m: int = 3, ref_num: int = 1000, device: Optional[torch.device] = None):
        super().__init__(d, m, ref_num, device)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        m = self.m
        n, d = X.size()
        g = 100 * (
            d
            - m
            + 1
            + torch.sum(
                (X[:, m - 1 :] - 0.5) ** 2 - torch.cos(20 * torch.pi * (X[:, m - 1 :] - 0.5)),
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


class DTLZ2(DTLZ):
    def __init__(self, d: int = 12, m: int = 3, ref_num: int = 1000, device: Optional[torch.device] = None):
        super().__init__(d, m, ref_num, device)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        m = self.m
        g = torch.sum((X[:, m - 1 :] - 0.5) ** 2, dim=1, keepdim=True)
        f = (
            (1 + g)
            * torch.flip(
                torch.cumprod(
                    torch.cat(
                        [
                            torch.ones((X.size(0), 1), device=X.device),
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
                    torch.ones((X.size(0), 1), device=X.device),
                    torch.sin(torch.flip(X[:, : m - 1], dims=[1]) * torch.pi / 2),
                ],
                dim=1,
            )
        )

        return f

    def pf(self):
        f = self.sample
        f = f / torch.sqrt(f.pow(2).sum(dim=1, keepdim=True))
        return f


class DTLZ3(DTLZ2):
    def __init__(self, d: int = 12, m: int = 3, ref_num: int = 1000, device: Optional[torch.device] = None):
        super().__init__(d, m, ref_num, device)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        n, d = X.size()
        m = self.m
        g = 100 * (
            d
            - m
            + 1
            + torch.sum(
                (X[:, m - 1 :] - 0.5) ** 2 - torch.cos(20 * torch.pi * (X[:, m - 1 :] - 0.5)),
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
    def __init__(self, d: int = 12, m: int = 3, ref_num: int = 1000, device: Optional[torch.device] = None):
        super().__init__(d, m, ref_num, device)

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
                            torch.ones((g.size(0), 1), device=X.device),
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
                    torch.ones((g.size(0), 1), device=X.device),
                    torch.sin(torch.flip(X[:, : m - 1], dims=[1]) * torch.pi / 2),
                ],
                dim=1,
            )
        )

        return f


class DTLZ5(DTLZ):
    def __init__(self, d: int = 12, m: int = 3, ref_num: int = 1000, device: Optional[torch.device] = None):
        super().__init__(d, m, ref_num, device)

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
                            torch.ones((g.size(0), 1), device=X.device),
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
                    torch.ones((g.size(0), 1), device=X.device),
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
                        torch.arange(0, 1, 1.0 / (n - 1), device=self.sample.device),
                        torch.tensor(1.0, device=self.sample.device),
                    )
                ),
                torch.hstack(
                    (
                        torch.arange(1, 0, -1.0 / (n - 1), device=self.sample.device),
                        torch.tensor(0.0, device=self.sample.device),
                    )
                ),
            )
        ).T

        f = f / torch.tile(torch.sqrt(torch.sum(f**2, dim=1, keepdim=True)), (1, f.size(1)))

        for i in range(self.m - 2):
            f = torch.cat((f[:, 0:1], f), dim=1)

        f = f / torch.sqrt(torch.tensor(2.0, device=self.sample.device)) ** torch.tile(
            torch.hstack(
                (
                    torch.tensor(self.m - 2, device=self.sample.device),
                    torch.arange(self.m - 2, -1, -1, device=self.sample.device),
                )
            ),
            (f.size(0), 1),
        )
        return f


class DTLZ6(DTLZ):
    def __init__(self, d: int = 12, m: int = 3, ref_num: int = 1000, device: Optional[torch.device] = None):
        super().__init__(d, m, ref_num, device)

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
                            torch.ones((X.size(0), 1), device=X.device),
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
                    torch.ones((X.size(0), 1), device=X.device),
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
                        torch.arange(0, 1, 1.0 / (n - 1), device=self.sample.device),
                        torch.tensor(1.0, device=self.sample.device),
                    )
                ),
                torch.hstack(
                    (
                        torch.arange(1, 0, -1.0 / (n - 1), device=self.sample.device),
                        torch.tensor(0.0, device=self.sample.device),
                    )
                ),
            )
        ).T

        f = f / torch.tile(torch.sqrt(torch.sum(f**2, dim=1, keepdim=True)), (1, f.size(1)))

        for i in range(self.m - 2):
            f = torch.cat((f[:, 0:1], f), dim=1)

        f = f / torch.sqrt(torch.tensor(2.0, device=self.sample.device)) ** torch.tile(
            torch.hstack(
                (
                    torch.tensor(self.m - 2, device=self.sample.device),
                    torch.arange(self.m - 2, -1, -1, device=self.sample.device),
                )
            ),
            (f.size(0), 1),
        )
        return f


class DTLZ7(DTLZ):
    def __init__(self, d: int = 21, m: int = 3, ref_num: int = 1000, device: Optional[torch.device] = None):
        super().__init__(d, m, ref_num, device)
        sample, _ = grid_sampling(self.ref_num * self.m, self.m - 1)
        self.sample = sample.to(device or torch.get_default_device())

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        n, d = X.size()
        m = self.m
        f = torch.zeros((n, m), device=X.device)
        g = 1 + 9 * torch.mean(X[:, m - 1 :], dim=1, keepdim=True)

        first_part = X[:, : m - 1]

        term = torch.sum(
            first_part / (1 + torch.tile(g, (1, m - 1))) * (1 + torch.sin(3 * torch.pi * first_part)),
            dim=1,
            keepdim=True,
        )
        f = torch.cat([first_part, (1 + g) * (m - term)], dim=1)

        return f

    def pf(self):
        interval = torch.tensor([0.0, 0.251412, 0.631627, 0.859401], dtype=torch.float, device=self.sample.device)
        median = (interval[1] - interval[0]) / (interval[3] - interval[2] + interval[1] - interval[0]).to(self.sample.device)

        x = self.sample.to(self.sample.device)

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

        last_col = 2 * (self.m - torch.sum(x / 2 * (1 + torch.sin(3 * torch.pi * x)), dim=1, keepdim=True))

        pf = torch.cat([x, last_col], dim=1)
        return pf
