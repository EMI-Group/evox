import torch
import itertools
from math import comb


class UniformSampling:
    """
    Uniform sampling using Das and Dennis's method, Deb and Jain's method.
    Inspired by PlatEMO's NBI algorithm.
    """

    def __init__(self, n=None, m=None):
        self.n = n
        self.m = m

    def __call__(self):
        h1 = 1
        while comb(h1 + self.m, self.m - 1) <= self.n:
            h1 += 1

        w = torch.tensor(list(itertools.combinations(range(1, h1 + self.m), self.m - 1))) - \
            torch.tile(torch.tensor(range(self.m - 1)), (comb(h1 + self.m - 1, self.m - 1), 1)) - 1
        w = (torch.cat([w, torch.zeros((w.shape[0], 1), dtype=w.dtype) + h1], dim=1)
                - torch.cat([torch.zeros((w.shape[0], 1), dtype=w.dtype), w], dim=1)
            ) / h1

        if h1 < self.m:
            h2 = 0
            while comb(h1 + self.m - 1, self.m - 1) + comb(h2 + self.m, self.m - 1) <= self.n:
                h2 += 1
            if h2 > 0:
                w2 = torch.tensor(list(itertools.combinations(range(1, h2 + self.m), self.m - 1))) - \
                     torch.tile(torch.tensor(range(self.m - 1)), (comb(h2 + self.m - 1, self.m - 1), 1)) - 1
                w2 = (
                             torch.cat([w2, torch.zeros((w2.shape[0], 1), dtype=w2.dtype) + h2], dim=1)
                             - torch.cat([torch.zeros((w2.shape[0], 1), dtype=w2.dtype), w2], dim=1)
                     ) / h2

                w = torch.cat([w, w2 / 2.0 + 1.0 / (2.0 * self.m)], dim=0)

        w = torch.maximum(w, torch.tensor(1e-6))
        n = w.shape[0]
        return w, n


# 测试样例
if __name__ == "__main__":
    # 测试类
    n = 105
    m = 3
    uniform_sampling = UniformSampling(n=n, m=m)
    w, n_samples = uniform_sampling()
    print("Generated sample matrix w:\n", w)
    print("Number of samples: ", n_samples)
