from typing import Optional

import torch

from ...core import Problem
from ...operators.sampling import uniform_sampling
from ...operators.selection import non_dominate_rank
from ...problems.numerical.basic import griewank_func, rastrigin_func, rosenbrock_func, sphere_func


class MAF(Problem):
    """
    ------------------------------- Reference --------------------------------
    R. Cheng, M. Li, Y. Tian, X. Zhang, S. Yang, Y. Jin, and X. Yao, A benchmark test suite for evolutionary many-objective optimization, Complex & Intelligent Systems, 2017, 3(1): 67-81.
    """

    def __init__(self, d: int = None, m: int = 3, ref_num: int = 1000, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.get_default_device()
        self.m = m
        self.d = self.m + 9 if d is None else d
        self.ref_num = ref_num
        self._calc_pf()

    def _calc_pf(self):
        r, n = uniform_sampling(self.ref_num * self.m, self.m)
        self._pf_value = 1 - r

    def evaluate(self, X: torch.Tensor):
        pass

    def pf(self):
        return self._pf_value


class MAF1(MAF):
    def __init__(self, d=None, m=3, ref_num=1000, device: Optional[torch.device] = None):
        super().__init__(d, m, ref_num, device)

    def evaluate(self, X: torch.Tensor):
        m = self.m
        n = X.size(0)
        g = torch.sum(torch.pow(X[:, m - 1 :] - 0.5, 2), dim=1)

        ones_col = torch.ones(n, 1, device=X.device)
        cumprod_term = torch.flip(torch.cumprod(torch.cat([ones_col, X[:, : m - 1]], dim=1), dim=1), [1])
        reversed_term = torch.cat([ones_col, 1 - torch.flip(X[:, : m - 1], [1])], dim=1)  # Reversed slice for last term
        repeat_g = (1 + g).unsqueeze(1)

        f = repeat_g - repeat_g * cumprod_term * reversed_term
        return f


class MAF2(MAF):
    def __init__(self, d=None, m=3, ref_num=1000, device: Optional[torch.device] = None):
        super().__init__(d, m, ref_num, device)

    def evaluate(self, X: torch.Tensor):
        m = self.m
        n = X.size(0)
        d = self.d
        g = torch.zeros(n, m, device=X.device)
        interval = int((d - m + 1) / m)
        for i in range(m):
            if i < m - 1:
                start = m + i * interval - 1
                temp = X[:, start : start + interval].clone()
            else:
                start = m + (m - 1) * interval - 1
                temp = X[:, start:].clone()
            temp = torch.where(temp == 0, 0.5, temp)
            g[:, i] = torch.sum((temp - 0.5) ** 2, dim=1)

        f1 = torch.flip(
            torch.cumprod(
                torch.cat([torch.ones((n, 1), device=X.device), torch.cos((X[:, : m - 1] / 2 + 0.25) * torch.pi / 2)], dim=1),
                dim=1,
            ),
            [1],
        )
        f2 = torch.cat(
            [torch.ones(n, 1, device=X.device), torch.sin((torch.flip(X[:, : m - 1], [1]) / 2 + 0.25) * torch.pi / 2)], dim=1
        )
        f = (1 + g) * f1 * f2
        return f

    @torch.jit.ignore
    def _calc_pf(self):
        m = self.m
        r, n = uniform_sampling(self.ref_num * self.m, self.m)
        c = torch.zeros(n, m - 1, device=r.device)

        for i in range(n):
            for j in range(1, m):
                temp = r[i, j] / r[i, 0] * torch.prod(c[i, m - j : m - 1])
                c[i, m - j - 1] = torch.sqrt(1 / (1 + temp**2))

        if m > 5:
            c = c * (torch.cos(torch.pi / 8) - torch.cos(3 * torch.pi / 8)) + torch.cos(3 * torch.pi / 8)
        else:
            c = c[
                torch.all(
                    (c >= torch.cos(torch.tensor(torch.pi * 3 / 8, device=r.device)))
                    & (c <= torch.cos(torch.tensor(torch.pi / 8, device=r.device))),
                    dim=1,
                ),
                :,
            ]
        f = torch.flip(
            torch.cumprod(torch.cat([torch.ones(c.size(0), 1, device=r.device), c[:, : m - 1]], dim=1), dim=1), [1]
        ) * torch.cat([torch.ones(c.size(0), 1, device=r.device), torch.sqrt(1 - torch.flip(c[:, : m - 1], [1]) ** 2)], dim=1)
        self._pf_value = f


class MAF3(MAF):
    def __init__(self, d=None, m=3, ref_num=1000, device: Optional[torch.device] = None):
        super().__init__(d, m, ref_num, device)

    def evaluate(self, X: torch.Tensor):
        m = self.m
        n = X.size(0)
        d = self.d
        g = 100 * (d - m + 1 + torch.sum((X[:, m - 1 :] - 0.5) ** 2 - torch.cos(20 * torch.pi * (X[:, m - 1 :] - 0.5)), dim=1))
        f1 = (
            (1 + g).unsqueeze(1)
            * torch.flip(
                torch.cumprod(
                    torch.cat([torch.ones((n, 1)), torch.cos(X[:, : m - 1] * torch.pi / 2)], dim=1),
                    dim=1,
                ),
                [1],
            )
            * torch.cat([torch.ones(n, 1, device=X.device), torch.sin(torch.flip(X[:, : m - 1], [1]) * torch.pi / 2)], dim=1)
        )
        f = torch.cat([f1[:, : m - 1] ** 4, (f1[:, m - 1] ** 2).view(-1, 1)], dim=1)
        return f

    @torch.jit.ignore
    def _cal_pf(self):
        m = self.m
        r, n = uniform_sampling(self.ref_num * self.m, self.m)

        temp = (torch.sum(torch.sqrt(r[:, :-1]), dim=1) + r[:, -1]).view(-1, 1)
        f = r / torch.cat([(temp**2).repeat(1, m - 1), temp], dim=1)
        self._pf_value = f


class MAF4(MAF):
    def __init__(self, d=None, m=3, ref_num=1000, device: Optional[torch.device] = None):
        super().__init__(d, m, ref_num, device)

    def evaluate(self, X: torch.Tensor):
        m = self.m
        n = X.size(0)
        d = self.d
        g = 100 * (d - m + 1 + torch.sum((X[:, m - 1 :] - 0.5) ** 2 - torch.cos(20 * torch.pi * (X[:, m - 1 :] - 0.5)), dim=1))
        f1 = (1 + g).unsqueeze(1) - (1 + g).unsqueeze(1) * torch.flip(
            torch.cumprod(
                torch.cat([torch.ones(n, 1, device=X.device), torch.cos(X[:, : m - 1] * torch.pi / 2)], dim=1), dim=1
            ),
            [1],
        ) * torch.cat([torch.ones(n, 1, device=X.device), torch.sin(torch.flip(X[:, : m - 1], [1]) * torch.pi / 2)], dim=1)
        f = f1 * torch.pow(2, torch.arange(1, m + 1))
        return f

    @torch.jit.ignore
    def _cal_pf(self):
        m = self.m
        r, n = uniform_sampling(self.ref_num * self.m, self.m)
        r1 = r / torch.sqrt(torch.sum(r**2, dim=1))
        f = (1 - r1) * torch.pow(2, torch.arange(1, m + 1, device=r.device))
        self._pf_value = f


class MAF5(MAF):
    def __init__(self, d=None, m=3, ref_num=1000, device: Optional[torch.device] = None):
        super().__init__(d, m, ref_num, device)

    def evaluate(self, X: torch.Tensor):
        m = self.m
        n = X.size(0)
        alpha = 100
        Xfront = X[:, : m - 1] ** alpha
        g = torch.sum((X[:, m - 1 :] - 0.5) ** 2, dim=1)
        f1 = (
            (1 + g).unsqueeze(1)
            * torch.flip(
                torch.cumprod(
                    torch.cat([torch.ones(n, 1, device=X.device), torch.cos(Xfront * torch.pi / 2)], dim=1),
                    dim=1,
                ),
                [1],
            )
            * torch.cat([torch.ones(n, 1, device=X.device), torch.sin(torch.flip(Xfront, [1]) * torch.pi / 2)], dim=1)
        )
        f = f1 * torch.pow(2, torch.arange(m, 0, -1, device=X.device))
        return f

    @torch.jit.ignore
    def _cal_pf(self):
        m = self.m
        r, n = uniform_sampling(self.ref_num * self.m, self.m)
        r1 = r / torch.sqrt(torch.sum(r**2, dim=1))
        f = r1 * torch.pow(2, torch.arange(m, 0, -1, device=r.device))
        self._pf_value = f


class MAF6(MAF):
    def __init__(self, d=None, m=3, ref_num=1000, device: Optional[torch.device] = None):
        super().__init__(d, m, ref_num, device)

    def evaluate(self, X: torch.Tensor):
        m = self.m
        n = X.size(0)
        g = torch.sum((X[:, m - 1 :] - 0.5) ** 2, dim=1).unsqueeze(1)
        temp = g.repeat(1, m - 2)
        Xfront = X[:, : m - 1].clone()
        Xfront[:, 1:] = (1 + 2 * temp * X[:, 1 : m - 1]) / (2 + 2 * temp)
        f = (
            (1 + 100 * g)
            * torch.flip(
                torch.cumprod(
                    torch.cat([torch.ones((n, 1)), torch.cos(Xfront * torch.pi / 2)], dim=1),
                    dim=1,
                ),
                [1],
            )
            * torch.cat([torch.ones((n, 1)), torch.sin(torch.flip(Xfront, [1]) * torch.pi / 2)], dim=1)
        )
        return f

    @torch.jit.ignore
    def _cal_pf(self):
        m = self.m
        r, n = uniform_sampling(self.ref_num * self.m, 2)
        r1 = (r / torch.sqrt(torch.sum(r**2, dim=1))).repeat(1, r.size(1))
        if r1.size(1) < m:
            r1 = torch.cat((r1[:, torch.zeros(m - 4, device=r.device)], r1), dim=1)
        f = r1 / torch.pow(torch.sqrt(2), torch.maximum(self.m - 2, 0).repeat(n, 1))
        self._pf_value = f


class MAF7(MAF):
    def __init__(self, d=None, m=3, ref_num=1000, device: Optional[torch.device] = None):
        d = m + 19 if d is None else d
        super().__init__(d, m, ref_num, device)

    def evaluate(self, X: torch.Tensor):
        m = self.m
        n = X.size(0)
        f = torch.zeros(n, m, device=X.device)
        g = 1 + 9 * torch.mean(X[:, m - 1 :], dim=1)
        f[:, : m - 1] = X[:, : m - 1]
        f[:, m - 1] = (1 + g) * (
            m
            - torch.sum(
                f[:, : self.m - 1] / (1 + g) * (1 + torch.sin(3 * torch.pi * f[:, : m - 1])),
                dim=1,
            )
        )
        return f

    @torch.jit.ignore
    def _cal_pf(self):
        m = self.m
        interval = torch.tensor([0, 0.251412, 0.631627, 0.859401], device=self.device)
        median = (interval[1] - interval[0]) / (interval[3] - interval[2] + interval[1] - interval[0])
        X = self._grid(self.ref_num * self.m, m - 1)
        X = torch.where(X <= median, X * (interval[1] - interval[0]) / median + interval[0], X)
        X = torch.where(
            X > median,
            (X - median) * (interval[3] - interval[2]) / (1 - median) + interval[2],
            X,
        )
        f = torch.cat(
            [
                X,
                2 * (self.m - torch.sum(X / 2 * (1 + torch.sin(3 * torch.pi * X)), dim=1)).view(-1, 1),
            ],
            dim=1,
        )
        self._pf_value = f

    def _grid(self, N: int, M: int):
        gap = torch.linspace(0, 1, steps=int(N ** (1 / M)), device=self.device)
        mesh = torch.meshgrid(*([gap] * M), indexing="ij")
        W = torch.cat([x.view(-1, 1) for x in mesh], dim=1)
        return W


class MAF8(MAF):
    def __init__(self, d=2, m=10, ref_num=1000, device: Optional[torch.device] = None):
        assert d == 2, f"{self.__class__.__name__} is only defined for D = 2, got {d}."
        assert m >= 3, f"{self.__class__.__name__} is only defined for M >= 3, got {m}."
        super().__init__(d, m, ref_num, device)
        self.points = self._getPoints()

    def evaluate(self, X: torch.Tensor):
        f = self._eucl_dis(X[:, :2], self.points)
        return f

    @torch.jit.ignore
    def _cal_pf(self):
        temp = torch.linspace(
            -1, 1, steps=int(torch.sqrt(torch.tensor(self.ref_num * self.m, device=self.device))), device=self.device
        )
        y, x = torch.meshgrid(temp, temp, indexing="ij")
        x = x.flatten()
        y = y.flatten()
        _points = torch.stack([x, y], dim=-1)
        ND = torch.stack([self._point_in_polygon(self.points, p) for p in _points])
        f = self._eucl_dis(torch.stack([x[ND], y[ND]], dim=-1), self.points)
        self._pf_value = f

    def _eucl_dis(self, X: torch.Tensor, Y: torch.Tensor):
        return torch.cdist(X, Y)

    def _getPoints(self):
        m = self.m
        theta, rho = self._cart2pol(torch.tensor(0, device=self.device), torch.tensor(1, device=self.device))
        temp = torch.arange(1, m + 1, device=self.device).view(-1, 1)
        x, y = self._pol2cart(theta - temp * 2 * torch.pi / m, rho)
        return torch.cat([x, y], dim=1)

    def _cart2pol(self, x: torch.Tensor, y: torch.Tensor):
        rho = torch.sqrt(x**2 + y**2)
        theta = torch.atan2(y, x)
        return theta, rho

    def _pol2cart(self, theta: torch.Tensor, rho: torch.Tensor):
        x = rho * torch.cos(theta)
        y = rho * torch.sin(theta)
        return x, y

    def _inside(self, x, a, b):
        return (torch.minimum(a, b) <= x) & (x < torch.maximum(a, b))

    def _ray_intersect_segment(self, point, seg_init, seg_term):
        y_dist = seg_term[1] - seg_init[1]
        # special case: y_dist == 0, check P_y == seg_init_y and P_x inside the segment
        judge_1 = (point[1] == seg_init[1]) & self._inside(point[0], seg_init[0], seg_term[0])
        # check intersection_x >= P_x.
        LHS = seg_init[0] * y_dist + (point[1] - seg_init[1]) * (seg_term[0] - seg_init[0])
        RHS = point[0] * y_dist
        # since it's an inequation, reverse the inequation if y_dist is negative.
        judge_2 = ((y_dist > 0) & (LHS >= RHS)) | ((y_dist < 0) & (LHS <= RHS))
        # check intersection_y, which is P_y is inside the segment
        judge_3 = self._inside(point[1], seg_init[1], seg_term[1])
        return ((y_dist == 0) & judge_1) | ((y_dist != 0) & judge_2 & judge_3)

    def _point_in_polygon(self, polygon, point):
        seg_term = torch.roll(polygon, 1, dims=0)
        is_intersect = torch.cat(
            [self._ray_intersect_segment(point, polygon[i], seg_term[i]) for i in range(polygon.size(0))], dim=0
        )
        is_vertex = torch.any(torch.all(polygon == point, dim=1))
        return (torch.sum(is_intersect) % 2 == 1) | is_vertex


class MAF9(MAF8):
    def __init__(self, d=2, m=10, ref_num=1000, device: Optional[torch.device] = None):
        super().__init__(d, m, ref_num, device)

    def evaluate(self, X: torch.Tensor):
        f = torch.zeros(X.size(0), self.points.size(0), device=X.device)
        for i in range(self.points.size(0) - 1):
            f[:, i] = self._Point2Line(X, self.points[i : i + 2, :])
        f[:, -1] = self._Point2Line(X, torch.cat([self.points[-1, :].unsqueeze(0), self.points[0, :].unsqueeze(0)], dim=0))
        return f

    def _Point2Line(self, pop_dec: torch.Tensor, line: torch.Tensor):
        Distance = torch.abs(
            (line[0, 0] - pop_dec[:, 0]) * (line[1, 1] - pop_dec[:, 1])
            - (line[1, 0] - pop_dec[:, 0]) * (line[0, 1] - pop_dec[:, 1])
        ) / torch.sqrt((line[0, 0] - line[1, 0]) ** 2 + (line[0, 1] - line[1, 1]) ** 2)
        return Distance


class MAF10(MAF):
    def __init__(self, d=None, m=3, ref_num=1000, device: Optional[torch.device] = None):
        super().__init__(d, m, ref_num, device)

    def evaluate(self, X: torch.Tensor):
        m = self.m
        n = X.size(0)
        d = self.d
        s = torch.arange(2, 2 * m + 1, 2, device=X.device)
        z01 = X / 2 * torch.arange(1, d + 1, device=X.device)
        t0 = torch.zeros(n, d, device=X.device)
        t0[:, : m - 1] = z01[:, : m - 1]
        t0[:, m - 1 :] = self._s_linear(z01[:, m - 1 :], 0.35)

        t = self._evaluate(t0, X)

        x = torch.zeros(n, m, device=X.device)
        for i in range(m - 1):
            x[:, i] = torch.maximum(t[:, m - 1], torch.ones_like(t[:, m - 1], device=X.device)) * (t[:, i] - 0.5) + 0.5
        x[:, m - 1] = t[:, m - 1]

        h = self._convex(x)
        h[:, m - 1] = self._mixed(x)
        f = x[:, m - 1].unsqueeze(1) + s * h
        return f

    @torch.jit.ignore
    def _cal_pf(self):
        m = self.m
        x, temp, a = self._pf_a()
        e = torch.abs(
            temp.unsqueeze(1) * (1 - torch.cos(torch.pi / 2 * a))
            - 1
            + (a + torch.cos(10 * torch.pi * a + torch.pi / 2) / 10 / torch.pi)
        )
        rank = torch.argsort(e, dim=1)

        x[:, 0] = a[0, torch.min(rank[:, :10], dim=1)]
        f = self._convex(x)
        f[:, m - 1] = self._mixed(x)
        f = f * torch.arange(2, 2 * m + 1, 2, device=self.device)
        self._pf_value = f

    def _evaluate(self, t1: torch.Tensor, X: torch.Tensor):
        m = self.m
        n = X.size(0)
        d = self.d
        t2 = t1.clone()
        t2[:, m - 1 :] = self._b_flat(t1[:, m - 1 :], 0.8, 0.75, 0.85)

        t3 = t2**0.02

        t = torch.zeros(n, m, device=X.device)

        for i in range(m - 1):
            temp1 = t3[:, i : i + 2]
            temp2 = torch.arange(2 * i, 2 * (i + 1) + 1, 2, device=X.device)
            t[:, i] = self._r_sum(temp1, temp2)
        t[:, m - 1] = self._r_sum(t3[:, m - 1 : d], torch.arange(2 * m, 2 * d + 1, 2, device=X.device))

        return t

    def _s_linear(self, Y: torch.Tensor, a):
        return torch.abs(Y - a) / torch.abs(torch.floor(a - Y) + a)

    def _b_flat(self, Y: torch.Tensor, a, b, c):
        output = (
            a
            + torch.minimum(torch.zeros_like(Y - b, device=Y.device), torch.floor(Y - b)) * a * (b - Y) / b
            - torch.minimum(torch.zeros_like(c - Y, device=Y.device), torch.floor(c - Y)) * (1 - a) * (Y - c) / (1 - c)
        )
        return torch.round(output * 1e4) / 1e4

    def _r_sum(self, Y: torch.Tensor, W: torch.Tensor):
        return torch.sum(Y * W, dim=1) / torch.sum(W)

    def _convex(self, x: torch.Tensor):
        ones = torch.ones(x.size(0), 1, device=x.device)
        return torch.flip(
            torch.cumprod(torch.cat([ones, 1 - torch.cos(x[:, :-1] * torch.pi / 2)], dim=1), dim=1), [1]
        ) * torch.cat([ones, 1 - torch.sin(torch.flip(x[:, :-1], [1]) * torch.pi / 2)], dim=1)

    def _mixed(self, x: torch.Tensor):
        return 1 - x[:, 0] - torch.cos(10 * torch.pi * x[:, 0] + torch.pi / 2) / (10 * torch.pi)

    def _pf_a(self):
        m = self.m
        r, n = uniform_sampling(self.ref_num * self.m, self.m)
        c = torch.ones(n, m, device=r.device)

        for i in range(n):
            for j in range(1, m):
                temp = r[i, j] / r[i, 0] * torch.prod(1 - c[i, m - j : m - 1])
                c[i, m - j - 1] = (temp**2 - temp + torch.sqrt(2 * temp)) / (temp**2 + 1)

        x = torch.arccos(c) * 2 / torch.pi
        temp = (1 - torch.sin(torch.pi / 2 * x[:, 1])) * r[:, m - 1] / r[:, m - 2]
        a = torch.arange(0, 1.0001, 0.0001, device=r.device).unsqueeze(0)
        return x, temp, a


class MAF11(MAF10):
    def __init__(self, d=None, m=3, ref_num=1000, device: Optional[torch.device] = None):
        d = m + 9 if d is None else d
        d = int((d - m + 1) / 2) * 2 + m - 1
        super().__init__(d, m, ref_num, device)

    @torch.jit.ignore
    def _cal_pf(self):
        m = self.m
        x, temp, a = self._pf_a()
        e = torch.abs(temp.unsqueeze(1) * (1 - torch.cos(torch.pi / 2 * a)) - 1 + a * torch.cos(5 * torch.pi * a) ** 2)
        rank = torch.argsort(e, dim=1)
        x[:, 0] = a[0, torch.min(rank[:, :10], dim=1)]
        f = self._convex(x)
        f[:, m - 1] = self._mixed(x)
        non_dominated_rank = non_dominate_rank(f)
        f = f[non_dominated_rank == 0, :]
        f = f * torch.arange(2, 2 * m + 1, 2, device=self.sample.device)
        self._pf_value = f

    def _evaluate(self, t1: torch.Tensor, X: torch.Tensor):
        m = self.m
        n = X.size(0)
        d = self.d
        L = d - (m - 1)

        t2 = torch.zeros(n, m - 1 + L // 2, device=X.device)
        t2[:, : m - 1] = t1[:, : m - 1]
        t2[:, m - 1 : m - 1 + L // 2] = (t1[:, m - 1 :: 2] + t1[:, m::2] + 2 * torch.abs(t1[:, m - 1 :: 2] - t1[:, m::2])) / 3

        t = torch.zeros(n, m, device=X.device)

        for i in range(m - 1):
            temp = t2[:, i : i + 2]
            t[:, i - 1] = self._r_sum(temp, torch.ones(1, device=X.device))

        t[:, m - 1] = self._r_sum(t2[:, m - 1 : m - 1 + L // 2], torch.ones(L // 2, device=X.device))
        return t

    def _mixed(self, x: torch.Tensor):
        return self._disc(x)

    def _disc(self, x):
        return 1 - x[:, 0] * (torch.cos(5 * torch.pi * x[:, 0])) ** 2


class MAF12(MAF):
    def __init__(self, d=None, m=3, ref_num=1000, device: Optional[torch.device] = None):
        super().__init__(d, m, ref_num, device)

    def evaluate(self, X: torch.Tensor):
        m = self.m
        n = X.size(0)
        d = self.d
        L = d - (m - 1)
        S = torch.arange(2, 2 * m + 1, 2)

        z01 = X / torch.arange(2, d * 2 + 1, 2)

        t1 = torch.zeros(n, d, device=X.device)
        Y = (torch.flip(torch.cumsum(torch.flip(z01, [1]), dim=1), [1]) - z01) / torch.arange(d - 1, -1, -1, device=X.device)
        t1[:, : d - 1] = z01[:, : d - 1] ** (
            0.02
            + (50 - 0.02)
            * (0.98 / 49.98 - (1 - 2 * Y[:, : d - 1]) * torch.abs(torch.floor(0.5 - Y[:, : d - 1]) + 0.98 / 49.98))
        )
        t1[:, d - 1] = z01[:, d - 1]

        t2 = torch.zeros(n, d, device=X.device)
        t2[:, : m - 1] = self._s_decept(t1[:, : m - 1], 0.35, 0.001, 0.05)
        t2[:, m - 1 :] = self._s_multi(t1[:, m - 1 :], 30, 95, 0.35)

        t3 = torch.zeros(n, m, device=X.device)

        for i in range(m - 1):
            temp = t2[:, i : i + 2]
            t3[:, i - 1] = self._r_nonsep(temp, 1)

        SUM = torch.zeros(n, device=X.device)

        for i in range(m - 1, d - 1):
            for j in range(i + 1, d):
                SUM += torch.abs(t2[:, i] - t2[:, j])

        t3[:, m - 1] = (
            (torch.sum(t2[:, m - 1 :], dim=1) + SUM * 2)
            / torch.tensor(L / 2, device=X.device)
            / (1 + 2 * L - 2 * torch.tensor(L / 2, device=X.device))
        )

        x = torch.zeros(n, m, device=X.device)

        for i in range(m - 1):
            x[:, i] = torch.maximum(t3[:, m - 1], torch.ones_like(t3[:, m - 1], device=X.device)) * (t3[:, i] - 0.5) + 0.5

        x[:, m - 1] = t3[:, m - 1]

        h = self._concave(x)
        f = x[:, m - 1].view(-1, 1) + S * h
        return f

    def pf(self):
        m = self.m
        r, n = uniform_sampling(self.ref_num * self.m, self.m)
        r = r / torch.sqrt(torch.sum(r**2, dim=1)).reshape(-1, 1)
        f = torch.arange(2, 2 * m + 1, 2) * r
        return f

    def _s_decept(self, Y: torch.Tensor, a, b, c):
        return 1 + (torch.abs(Y - a) - b) * (
            torch.floor(Y - a + b) * (1 - c + (a - b) / b) / (a - b)
            + torch.floor(a + b - Y) * (1 - c + (1 - a - b) / b) / (1 - a - b)
            + 1 / b
        )

    def _s_multi(self, Y: torch.Tensor, a, b, c):
        return (
            1
            + torch.cos((4 * a + 2) * torch.pi * (0.5 - torch.abs(Y - c) / 2 / (torch.floor(c - Y) + c)))
            + 4 * b * (torch.abs(Y - c) / 2 / (torch.floor(c - Y) + c)) ** 2
        ) / (b + 2)

    def _r_nonsep(self, Y: torch.Tensor, a):
        Output = torch.zeros(Y.size(0))
        for j in range(Y.size(1)):
            Temp = torch.zeros(Y.size(0))
            for k in range(a - 1):
                Temp += torch.abs(Y[:, j] - Y[:, (j + 1 + k) % Y.size(1)])
            Output += Y[:, j] + Temp
        return Output / (Y.size(1) / a) / int(a / 2) / (1 + 2 * a - 2 * int(a / 2))

    def _concave(self, X: torch.Tensor):
        return torch.flip(
            torch.cumprod(
                torch.cat([torch.ones(X.shape[0], 1, device=X.device), torch.sin(X[:, :-1] * torch.pi / 2)], dim=1),
                dim=1,
            ),
            [1],
        ) * torch.cat(
            [
                torch.ones((X.size(0), 1)),
                torch.cos(torch.flip(X[:, :-1], [1]) * torch.pi / 2),
            ],
            dim=1,
        )


class MAF13(MAF):
    def __init__(self, d=5, m=3, ref_num=1000, device: Optional[torch.device] = None):
        assert m >= 3, f"{self.__class__.__name__} is only defined for M >= 3, got {m}."
        super().__init__(d, m, ref_num, device)

    def evaluate(self, X: torch.Tensor):
        m = self.m
        n = X.size(0)
        d = self.d
        Y = X - 2 * X[:, 1].view(-1, 1) * torch.sin(2 * torch.pi * X[:, 0].view(-1, 1) + torch.arange(1, d + 1) * torch.pi / d)
        f = torch.zeros(n, m, device=X.device)
        f[:, 0] = torch.sin(X[:, 0] * torch.pi / 2) + 2 * torch.mean(Y[:, 3:d:3] ** 2, dim=1)
        f[:, 1] = torch.cos(X[:, 0] * torch.pi / 2) * torch.sin(X[:, 1] * torch.pi / 2) + 2 * torch.mean(
            Y[:, 4:d:3] ** 2, dim=1
        )
        f[:, 2] = torch.cos(X[:, 0] * torch.pi / 2) * torch.cos(X[:, 1] * torch.pi / 2) + 2 * torch.mean(
            Y[:, 2:d:3] ** 2, dim=1
        )
        f[:, 3:] = (
            (f[:, 0] ** 2 + f[:, 1] ** 10 + f[:, 2] ** 10 + 2 * torch.mean(Y[:, 3:d] ** 2, dim=1))
            .unsqueeze(1)
            .repeat(1, self.m - 3)
        )
        return f

    @torch.jit.ignore
    def _cal_pf(self):
        m = self.m
        r, n = uniform_sampling(self.ref_num * self.m, 3)
        r = r / torch.sqrt(torch.sum(r**2, dim=1))
        f = torch.cat([r, (r[:, 0] ** 2 + r[:, 1] ** 10 + r[:, 2] ** 10).unsqueeze(1).repeat(1, m - 3)], dim=1)
        return f


class MAF14(MAF):
    def __init__(self, d=None, m=3, ref_num=1000, device: Optional[torch.device] = None):
        d = 20 * m if d is None else d
        super().__init__(d, m, ref_num, device)
        nk = 2
        c = torch.zeros(self.m, device=device)
        c[0] = 3.8 * 0.1 * (1 - 0.1)
        for i in range(1, self.m):
            c[i] = 3.8 * c[i - 1] * (1 - c[i - 1])

        self.sublen = torch.floor(c / torch.sum(c) * (self.d - self.m + 1) / nk)
        self.len = torch.cat([torch.tensor([0]), torch.cumsum(self.sublen * nk, dim=0)], dim=0)
        self.sublen = tuple(map(int, self.sublen))
        self.len = tuple(map(int, self.len))

    def evaluate(self, X: torch.Tensor):
        m = self.m
        n = X.size(0)
        g = self._evaluate(X)
        f = (
            (1 + g)
            * torch.flip(torch.cumprod(torch.cat([torch.ones(n, 1, device=X.device), X[:, : m - 1]], dim=1), dim=1), [1])
            * torch.cat([torch.ones(n, 1, device=X.device), 1 - torch.flip(X[:, : m - 1], [1])], dim=1)
        )
        return f

    @torch.jit.ignore
    def _cal_pf(self):
        self._pf_value = uniform_sampling(self.ref_num * self.m, self.m)

    def _evaluate(self, X: torch.Tensor):
        m = self.m
        n = X.size(0)
        nk = 2
        X = self._modify_X(X)
        g = torch.zeros(n, m, device=X.device)

        for i in range(0, m, 2):
            g = self._inner_loop(i, self._func1, g, nk, X)
        for i in range(1, m, 2):
            g = self._inner_loop(i, self._func2, g, nk, X)
        g = g / torch.tensor(self.sublen).unsqueeze(0) * nk
        return g

    def _modify_X(self, X: torch.Tensor):
        X[:, self.m - 1 :] = (1 + torch.arange(self.m, self.d + 1, device=X.device) / self.d) * X[:, self.m - 1 :] - (
            X[:, 0] * 10
        ).unsqueeze(-1)
        return X

    def _inner_loop(self, i, inner_fun, g, nk, X: torch.Tensor):
        for j in range(0, nk):
            start = self.len[i] + self.m - 1 + j * self.sublen[i]
            end = start + self.sublen[i]
            temp = X[:, start:end]
            g[:, i] = g[:, i] + inner_fun(temp)
        return g

    def _func1(self, X):
        return rastrigin_func(X)

    def _func2(self, X):
        return rosenbrock_func(X)


class MAF15(MAF14):
    def __init__(self, d=None, m=3, ref_num=1000, device: Optional[torch.device] = None):
        super().__init__(d, m, ref_num, device)

    def evaluate(self, X: torch.Tensor):
        m = self.m
        n = X.size(0)
        g = self._evaluate(X)
        f = (1 + g + torch.cat([g[:, 1:], torch.zeros(n, 1, device=X.device)], dim=1)) * (
            1
            - torch.flip(
                torch.cumprod(
                    torch.cat([torch.ones(n, 1, device=X.device), torch.cos(X[:, : m - 1] * torch.pi / 2)], dim=1),
                    dim=1,
                ),
                [1],
            )
            * torch.cat([torch.ones(n, 1, device=X.device), torch.sin(torch.flip(X[:, : m - 1], [1]) * torch.pi / 2)], dim=1)
        )
        return f

    @torch.jit.ignore
    def _cal_pf(self):
        r, n = uniform_sampling(self.ref_num * self.m, self.m)
        r = 1 - r / torch.sqrt(torch.sum(r**2, axis=1)).reshape(-1, 1)
        self._pf_value = r

    def _modify_X(self, X: torch.Tensor):
        X[:, self.m - 1 :] = (1 + torch.cos(torch.arange(self.m, self.d + 1, device=X.device) / self.d * torch.pi / 2)) * X[
            :, self.m - 1 :
        ] - (X[:, 0] * 10).unsqueeze(-1)
        return X

    def _func1(self, X):
        return griewank_func(X)

    def _func2(self, X):
        return sphere_func(X)
