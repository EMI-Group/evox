from functools import partial

import torch

from ...core import Problem, jit_class


def _generic_zdt(f1, g, h, x):
    f1_x = f1(x)
    g_x = g(x)
    return torch.stack([f1_x, g_x * h(f1_x, g_x)],dim=1)


class ZDTTestSuit(Problem):
    def __init__(self, n: int, ref_num: int = 100):
        super().__init__()
        self.n = n
        self._zdt = None
        self.ref_num = ref_num

    def evaluate(self, X: torch.Tensor):
        return self._zdt(X)

    def pf(self):
        x = torch.linspace(0, 1, self.ref_num)
        return torch.stack([x, 1 - torch.sqrt(x)], dim=1)


@jit_class
class ZDT1(ZDTTestSuit):
    def __init__(self, n):
        super().__init__(n)
        f1 = lambda x: x[:,0]
        g = lambda x: 1 + 9 * torch.mean(x[:,1:])
        h = lambda f1, g: 1 - torch.sqrt(f1 / g)
        self._zdt = partial(_generic_zdt, f1, g, h)


@jit_class
class ZDT2(ZDTTestSuit):
    def __init__(self, n):
        super().__init__(n)
        f1 = lambda x: x[:,0]
        g = lambda x: 1 + 9 * torch.mean(x[:,1:])
        h = lambda f1_val, g_val: 1 - (f1_val / g_val) ** 2
        self._zdt = partial(_generic_zdt, f1, g, h)

    def pf(self):
        x = torch.linspace(0, 1, self.ref_num)
        return torch.stack([x, 1 - x**2], dim=1)


@jit_class
class ZDT3(ZDTTestSuit):
    def __init__(self, n):
        super().__init__(n)
        f1 = lambda x: x[:,0]
        g = lambda x: 1 + 9 * torch.mean(x[:,1:])
        h = lambda f1, g: 1 - torch.sqrt(f1 / g) - (f1 / g) * torch.sin(10 * torch.pi * f1)
        self._zdt = partial(_generic_zdt, f1, g, h)

    def pf(self):
        r = torch.tensor(
            [
                [0.0000, 0.0830],
                [0.1822, 0.2577],
                [0.4093, 0.4538],
                [0.6183, 0.6525],
                [0.8233, 0.8518],
            ]
        )

        pf_points = []
        segment_size = self.ref_num // len(r)
        for row in r:
            x_vals = torch.linspace(row[0].item(), row[1].item(), segment_size)
            f2_vals = 1 - torch.sqrt(x_vals) - x_vals * torch.sin(10 * torch.pi * x_vals)
            pf_points.append(torch.stack([x_vals, f2_vals], dim=1))
        pf = torch.cat(pf_points, dim=0)
        return pf


@jit_class
class ZDT4(ZDTTestSuit):
    def __init__(self, n):
        super().__init__(n)
        f1 = lambda x: x[:,0]
        g= lambda x: 1 + 10 * (self.n - 1) + torch.sum(x[:,1:] ** 2 - 10.0 * torch.cos(4.0 * torch.pi * x[:,1:]))
        h = lambda f1_val, g_val: 1 - torch.sqrt(f1_val / g_val)
        self._zdt = partial(_generic_zdt, f1, g, h)


@jit_class
class ZDT6(ZDTTestSuit):
    def __init__(self, n):
        super().__init__(n)
        f1 = lambda x: 1 - torch.exp(-4.0 * x[:,0]) * torch.sin(6.0 * torch.pi * x[:,0]) ** 6
        g = lambda x: 1 + 9.0 * (torch.sum(x[:,1:]) / 9.0) ** 0.25
        h = lambda f1_val, g_val: 1 - (f1_val / g_val) ** 2
        self._zdt = partial(_generic_zdt, f1, g, h)

    def pf(self):
        min_f1 = 0.280775
        f1_vals = torch.linspace(min_f1, 1.0, self.ref_num)
        f2_vals = 1.0 - f1_vals**2
        return torch.stack([f1_vals, f2_vals], dim=1)
