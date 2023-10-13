import evox
import jax
from jax import lax, jit, vmap
import jax.numpy as jnp
from evox import Problem, State
from evox.operators.sampling import UniformSampling
from evox.operators.non_dominated_sort import non_dominated_sort
import math
from evox.problems.numerical import Sphere, Griewank


@jit
def inside(x, a, b):
    """check if x is in [a, b) or [b, a)"""
    return (jnp.minimum(a, b) <= x) & (x < jnp.maximum(a, b))


@jit
def ray_intersect_segment(point, seg_init, seg_term):
    """
    point is a 2d point, representing a horizontal ray casting from the point.
    segment is segment represented by two 2d points in shape (2, ),
    where seg_init is the initial point, and seg_term is the terminal point.
    Thus check if the intersection_x >= P_x.
    """
    y_dist = seg_term[1] - seg_init[1]
    # special case: y_dist == 0, check P_y == seg_init_y and P_x inside the segment
    judge_1 = (point[1] == seg_init[1]) & inside(point[0], seg_init[0], seg_term[0])
    # check intersection_x >= P_x.
    LHS = seg_init[0] * y_dist + (point[1] - seg_init[1]) * (seg_term[0] - seg_init[0])
    RHS = point[0] * y_dist
    # since it's an inequation, reverse the inequation if y_dist is negative.
    judge_2 = ((y_dist > 0) & (LHS >= RHS)) | ((y_dist < 0) & (LHS <= RHS))
    # check intersection_y, which is P_y is inside the segment
    judge_3 = inside(point[1], seg_init[1], seg_term[1])
    return ((y_dist == 0) & judge_1) | ((y_dist != 0) & judge_2 & judge_3)


@jit
def point_in_polygon(polygon, point):
    """
    Determine whether a point is within a regular polygon by ray method
    Args:
        polygon (jnp.array): Vertex coordinates of polygons, shape is (n, 2)
        point (jnp.array): The coordinates of the points that need to be determined, shape is (2,)
    Returns:
        bool: If the point is within the polygon, return True; Otherwise, return False
    """

    seg_term = jnp.roll(polygon, 1, axis=0)
    is_intersect = vmap(ray_intersect_segment, in_axes=(None, 0, 0))(
        point, polygon, seg_term
    )
    is_vertex = jnp.any(jnp.all(polygon == point, axis=1), axis=0)
    return (jnp.sum(is_intersect) % 2 == 1) | is_vertex


@evox.jit_class
class MaF(Problem):
    """
    ------------------------------- Reference --------------------------------
    R. Cheng, M. Li, Y. Tian, X. Zhang, S. Yang, Y. Jin, and X. Yao, A benchmark test suite for evolutionary many-objective optimization, Complex & Intelligent Systems, 2017, 3(1): 67-81.
    """

    def __init__(self, d=None, m=None, ref_num=1000):
        # ref_num is as n
        if m is None:
            self.m = 3
        else:
            self.m = m
        if d is None:
            self.d = self.m + 9
        else:
            self.d = d
        self._maf = None
        self.ref_num = ref_num

    def setup(self, key):
        return State(key=key)

    def evaluate(self, state, X):
        pass
        return jax.jit(jax.vmap(self._maf))(X), state

    def pf(self, state):
        f = 1 - UniformSampling(self.ref_num * self.m, self.m)()[0]
        return f, state


@evox.jit_class
class MaF1(MaF):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, state, X):
        m = self.m
        n, d = jnp.shape(X)
        g = jnp.sum((X[:, m - 1 :] - 0.5).__pow__(2), axis=1).reshape((-1, 1))
        ones_col = jnp.ones((n, 1))
        cumprod_term = jnp.fliplr(
            jnp.cumprod(jnp.hstack([ones_col, X[:, : m - 1]]), axis=1)
        )
        reversed_term = jnp.hstack(
            [ones_col, 1 - X[:, m - 2 :: -1]]
        )  # Reversed slice for last term
        repeat_g = jnp.tile(1 + g, (1, m))
        f = repeat_g - repeat_g * cumprod_term * reversed_term
        return f, state

    def pf(self, state):
        n = self.ref_num * self.m
        f = 1 - UniformSampling(n, self.m)()[0]
        return f, state


class MaF2(MaF):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)

    @evox.jit_method
    def evaluate(self, state, X):
        m = self.m
        n, d = jnp.shape(X)
        g = jnp.zeros((n, m))
        interval = int((d - m + 1) / m)
        mask_ones = jnp.ones((X.shape[0], interval), dtype=bool)
        mask_zeros = jnp.zeros_like(X, dtype=bool)

        def body_fun(i, g):
            def true_fun():
                start = (m + i * jnp.floor((d - m + 1) / m) - 1).astype(int)
                mask = lax.dynamic_update_slice(mask_zeros, mask_ones, (0, start))
                temp = jnp.where(mask, X, 0.5)
                return temp / 2 + 1 / 4

            def false_fun():
                start = m + (m - 1) * int((d - m + 1) / m) - 1
                mask_ones = jnp.ones((X.shape[0], d - start), dtype=bool)
                mask = lax.dynamic_update_slice(mask_zeros, mask_ones, (0, start))
                temp = jnp.where(mask, X, 0.5)
                return temp / 2 + 1 / 4

            temp1 = lax.cond(i < m - 1, true_fun, false_fun)
            return g.at[:, i].set(jnp.sum((temp1 - 0.5) ** 2, axis=1))

        g = lax.fori_loop(0, m, body_fun, g)

        f1 = jnp.fliplr(
            jnp.cumprod(
                jnp.hstack(
                    [
                        jnp.ones((n, 1)),
                        jnp.cos((X[:, : m - 1] / 2 + 1 / 4) * jnp.pi / 2),
                    ]
                ),
                axis=1,
            )
        )
        f2 = jnp.hstack(
            [jnp.ones((n, 1)), jnp.sin(((X[:, m - 2 :: -1]) / 2 + 1 / 4) * jnp.pi / 2)]
        )
        f = (1 + g) * f1 * f2
        return f, state

    def pf(self, state):
        n = self.ref_num * self.m
        r = UniformSampling(n, self.m)()[0]
        c = jnp.zeros((r.shape[0], self.m - 1))

        def inner_fun(i, c):
            for j in range(2, self.m + 1):
                temp = (
                    r[i - 1, j - 1]
                    / r[i - 1, 0]
                    * jnp.prod(c[i - 1, self.m - j + 1 : self.m - 1])
                )
                c = c.at[i - 1, self.m - j].set(jnp.sqrt(1 / (1 + temp**2)))
            return c

        c = lax.fori_loop(1, r.shape[0] + 1, inner_fun, c)

        if self.m > 5:
            c = c * (jnp.cos(jnp.pi / 8) - jnp.cos(3 * jnp.pi / 8)) + jnp.cos(
                3 * jnp.pi / 8
            )
        else:
            c = c[
                jnp.all(
                    (c >= jnp.cos(3 * jnp.pi / 8)) & (c <= jnp.cos(jnp.pi / 8)), axis=1
                ),
                :,
            ]

        n, _ = jnp.shape(c)
        f = jnp.fliplr(
            jnp.cumprod(jnp.hstack([jnp.ones((n, 1)), c[:, : self.m - 1]]), axis=1)
        ) * jnp.hstack([jnp.ones((n, 1)), jnp.sqrt(1 - c[:, self.m - 2 :: -1] ** 2)])
        return f, state


@evox.jit_class
class MaF3(MaF):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, state, X):
        n, d = jnp.shape(X)
        g = 100 * (
            d
            - self.m
            + 1
            + jnp.sum(
                (X[:, self.m - 1 :] - 0.5) ** 2
                - jnp.cos(20 * jnp.pi * (X[:, self.m - 1 :] - 0.5)),
                axis=1,
            )
        ).reshape((-1, 1))
        f1 = (
            jnp.tile(1 + g, (1, self.m))
            * jnp.fliplr(
                jnp.cumprod(
                    jnp.hstack(
                        [jnp.ones((n, 1)), jnp.cos(X[:, : self.m - 1] * jnp.pi / 2)]
                    ),
                    axis=1,
                )
            )
            * jnp.hstack(
                [jnp.ones((n, 1)), jnp.sin(X[:, self.m - 2 :: -1] * jnp.pi / 2)]
            )
        )
        f = jnp.hstack(
            [f1[:, : self.m - 1] ** 4, (f1[:, self.m - 1] ** 2).reshape((-1, 1))]
        )
        return f, state

    def pf(self, state):
        n = self.ref_num * self.m
        r = UniformSampling(n, self.m)()[0] ** 2
        temp = (jnp.sum(jnp.sqrt(r[:, :-1]), axis=1) + r[:, -1]).reshape((-1, 1))
        f = r / jnp.hstack([jnp.tile(temp**2, (1, r.shape[1] - 1)), temp])
        return f, state


@evox.jit_class
class MaF4(MaF):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, state, X):
        n, d = jnp.shape(X)
        g = (
            100
            * (
                d
                - self.m
                + 1
                + jnp.sum(
                    (X[:, self.m - 1 :] - 0.5) ** 2
                    - jnp.cos(20 * jnp.pi * (X[:, self.m - 1 :] - 0.5)),
                    axis=1,
                )
            )[:, jnp.newaxis]
        )
        f1 = jnp.tile(1 + g, (1, self.m)) - jnp.tile(1 + g, (1, self.m)) * jnp.fliplr(
            jnp.cumprod(
                jnp.hstack(
                    [jnp.ones((n, 1)), jnp.cos(X[:, : self.m - 1] * jnp.pi / 2)]
                ),
                axis=1,
            )
        ) * jnp.hstack([jnp.ones((n, 1)), jnp.sin(X[:, self.m - 2 :: -1] * jnp.pi / 2)])
        f = f1 * jnp.tile(jnp.power(2, jnp.arange(1, self.m + 1)), (n, 1))
        return f, state

    def pf(self, state):
        n = self.ref_num * self.m
        r = UniformSampling(n, self.m)()[0]
        r1 = r / jnp.tile(jnp.sqrt(jnp.sum(r**2, axis=1))[:, None], (1, self.m))
        f = (1 - r1) * jnp.tile(
            jnp.power(2, jnp.arange(1, self.m + 1)), (r.shape[0], 1)
        )
        return f, state


@evox.jit_class
class MaF5(MaF):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, state, X):
        n, d = jnp.shape(X)
        alpha = 100
        X = X.at[:, : self.m - 1].set((X[:, : self.m - 1] ** alpha))
        g = jnp.sum((X[:, self.m - 1 :] - 0.5) ** 2, axis=1)[:, jnp.newaxis]
        f1 = (
            jnp.tile(1 + g, (1, self.m))
            * jnp.fliplr(
                jnp.cumprod(
                    jnp.hstack(
                        [jnp.ones((n, 1)), jnp.cos(X[:, : self.m - 1] * jnp.pi / 2)]
                    ),
                    axis=1,
                )
            )
            * jnp.hstack(
                [jnp.ones((n, 1)), jnp.sin(X[:, self.m - 2 :: -1] * jnp.pi / 2)]
            )
        )
        f = f1 * jnp.tile(jnp.power(2, jnp.arange(self.m, 0, -1)), (n, 1))
        return f, state

    def pf(self, state):
        n = self.ref_num * self.m
        r = UniformSampling(n, self.m)()[0]
        r1 = r / jnp.tile(
            jnp.sqrt(jnp.sum(r**2, axis=1))[:, jnp.newaxis], (1, self.m)
        )
        f = r1 * jnp.tile(jnp.power(2, jnp.arange(self.m, 0, -1)), (r.shape[0], 1))
        return f, state


@evox.jit_class
class MaF6(MaF):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, state, X):
        n, d = jnp.shape(X)
        i = 2
        g = jnp.sum((X[:, self.m - 1 :] - 0.5) ** 2, axis=1)[:, jnp.newaxis]
        temp = jnp.tile(g, (1, self.m - i))
        X = X.at[:, i - 1 : self.m - 1].set(
            (1 + 2 * temp * X[:, i - 1 : self.m - 1]) / (2 + 2 * temp)
        )
        f = (
            jnp.tile((1 + 100 * g), (1, self.m))
            * jnp.fliplr(
                jnp.cumprod(
                    jnp.hstack(
                        [jnp.ones((n, 1)), jnp.cos(X[:, : self.m - 1] * jnp.pi / 2)]
                    ),
                    axis=1,
                )
            )
            * jnp.hstack(
                [jnp.ones((n, 1)), jnp.sin(X[:, self.m - 2 :: -1] * jnp.pi / 2)]
            )
        )
        return f, state

    def pf(self, state):
        i = 2
        n = self.ref_num * self.m
        r = UniformSampling(n, i)()[0]
        r1 = r / jnp.tile(
            jnp.sqrt(jnp.sum(r**2, axis=1))[:, jnp.newaxis], (1, r.shape[1])
        )
        if r1.shape[1] < self.m:
            r1 = jnp.hstack((r1[:, jnp.zeros((self.m - r1.shape[1])).astype(int)], r1))
        f = r1 / jnp.power(
            jnp.sqrt(2), jnp.tile(jnp.maximum(self.m - i, 0), (r.shape[0], 1))
        )
        return f, state


@evox.jit_class
class MaF7(MaF):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, state, X):
        n, d = jnp.shape(X)
        f = jnp.zeros((n, self.m))
        g = 1 + 9 * jnp.mean(X[:, self.m - 1 :], axis=1)
        f = f.at[:, : self.m - 1].set(X[:, : self.m - 1])
        f = f.at[:, self.m - 1].set(
            (1 + g)
            * (
                self.m
                - jnp.sum(
                    f[:, : self.m - 1]
                    / (1 + jnp.tile(g[:, jnp.newaxis], (1, self.m - 1)))
                    * (1 + jnp.sin(3 * jnp.pi * f[:, : self.m - 1])),
                    axis=1,
                )
            )
        )
        return f, state

    def pf(self, state):
        n = self.ref_num * self.m
        interval = jnp.array([0, 0.251412, 0.631627, 0.859401])
        median = (interval[1] - interval[0]) / (
            interval[3] - interval[2] + interval[1] - interval[0]
        )
        X = self._grid(n, self.m - 1)
        X = jnp.where(
            X <= median, X * (interval[1] - interval[0]) / median + interval[0], X
        )
        X = jnp.where(
            X > median,
            (X - median) * (interval[3] - interval[2]) / (1 - median) + interval[2],
            X,
        )
        f = jnp.hstack(
            [
                X,
                2
                * (self.m - jnp.sum(X / 2 * (1 + jnp.sin(3 * jnp.pi * X)), axis=1))[
                    :, jnp.newaxis
                ],
            ]
        )
        return f, state

    def _grid(self, N, M):
        gap = jnp.linspace(0, 1, int(math.ceil(N ** (1 / M))))
        c = jnp.meshgrid(*([gap] * M))
        W = jnp.vstack([x.ravel() for x in c]).T
        return W


"""
   the dimention only is 2.
"""


class MaF8(MaF):
    def __init__(self, d=None, m=None, ref_num=1000):
        d = 2
        super().__init__(d, m, ref_num)
        self.points = self._getPoints()

    @evox.jit_method
    def evaluate(self, state, X):
        if X.shape[1] != 2:
            X = X[:, :2]
        f = self._eucl_dis(X, self.points)
        return f, state

    def pf(self, state):
        n = self.ref_num * self.m
        temp = jnp.linspace(-1, 1, num=jnp.ceil(jnp.sqrt(n)).astype(int))
        y, x = jnp.meshgrid(temp, temp)
        x = x.ravel(order="F")
        y = y.ravel(order="F")
        _points = jnp.column_stack((x, y))
        ND = jax.vmap(point_in_polygon, in_axes=(None, 0))(self.points, _points)
        f = self._eucl_dis(jnp.column_stack([x[ND], y[ND]]), self.points)
        return f, state

    @evox.jit_method
    def _getPoints(self):
        thera, rho = self._cart2pol(0, 1)
        temp = jnp.arange(1, self.m + 1).reshape((-1, 1))
        x, y = self._pol2cart(thera - temp * 2 * jnp.pi / self.m, rho)
        return jnp.column_stack([x, y])

    @evox.jit_method
    def _cart2pol(self, x, y):
        rho = jnp.sqrt(jnp.power(x, 2) + jnp.power(y, 2))
        theta = jnp.arctan2(y, x)
        return theta, rho

    @evox.jit_method
    def _pol2cart(self, theta, rho):
        x = rho * jnp.cos(theta)
        y = rho * jnp.sin(theta)
        return (x, y)

    @evox.jit_method
    def _eucl_dis(self, x, y):
        dist_matrix = jnp.linalg.norm(x[:, None] - y, axis=-1)
        return dist_matrix


class MaF9(MaF):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)
        # Generate vertexes
        self.points = self._getPoints()

    @evox.jit_method
    def evaluate(self, state, X):
        n, d = jnp.shape(X)
        m, d1 = jnp.shape(self.points)
        f = jnp.zeros((n, m))
        for i in range(m):
            f = f.at[:, i].set(
                self._Point2Line(X, self.points[jnp.mod(jnp.arange(i, i + 2), m), :])
            )
        return f, state

    def pf(self, state):
        n = self.ref_num * self.m
        temp = jnp.linspace(-1, 1, num=jnp.ceil(jnp.sqrt(n)).astype(int))
        y, x = jnp.meshgrid(temp, temp)
        x = x.ravel(order="C")
        y = y.ravel(order="C")
        _points = jnp.column_stack((x, y))
        ND = jax.vmap(point_in_polygon, in_axes=(None, 0))(self.points, _points)
        f, state = self.evaluate(state, jnp.column_stack((x[ND], y[ND])))
        return f, state

    @evox.jit_method
    def _getPoints(self):
        thera, rho = self._cart2pol(0, 1)
        temp = jnp.arange(1, self.m + 1).reshape((-1, 1))
        x, y = self._pol2cart(thera - temp * 2 * jnp.pi / self.m, rho)
        return jnp.column_stack([x, y])

    @evox.jit_method
    def _cart2pol(self, x, y):
        rho = jnp.sqrt(jnp.power(x, 2) + jnp.power(y, 2))
        theta = jnp.arctan2(y, x)
        return theta, rho

    @evox.jit_method
    def _pol2cart(self, theta, rho):
        x = rho * jnp.cos(theta)
        y = rho * jnp.sin(theta)
        return (x, y)

    @evox.jit_method
    def _Point2Line(self, PopDec, Line):
        Distance = jnp.abs(
            (Line[0, 0] - PopDec[:, 0]) * (Line[1, 1] - PopDec[:, 1])
            - (Line[1, 0] - PopDec[:, 0]) * (Line[0, 1] - PopDec[:, 1])
        ) / jnp.sqrt((Line[0, 0] - Line[1, 0]) ** 2 + (Line[0, 1] - Line[1, 1]) ** 2)
        return Distance


@evox.jit_class
class MaF10(MaF):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)
        if m is None:
            self.m = 3
        else:
            self.m = m
        if d is None:
            self.d = self.m + 9
        else:
            self.d = d

    def evaluate(self, state, X):
        n, d = jnp.shape(X)
        M = self.m
        K = M - 1
        L = d - K
        D = 1
        S = jnp.arange(2, 2 * M + 1, 2)
        A = jnp.ones(M - 1)

        z01 = X / jnp.arange(2, d * 2 + 1, 2)
        t1 = jnp.zeros((n, K + L))
        t1 = t1.at[:, :K].set(z01[:, :K])
        t1 = t1.at[:, K:].set(self._s_linear(z01[:, K:], 0.35))

        t2 = jnp.zeros((n, K + L))
        t2 = t2.at[:, :K].set(t1[:, :K])
        t2 = t2.at[:, K:].set(self._b_flat(t1[:, K:], 0.8, 0.75, 0.85))

        t3 = t2**0.02

        t4 = jnp.zeros((n, M))
        t4 = t4.at[:, 0].set(
            self._r_sum(t3[:, int(K / (M - 1) - 1)][:, None], 2 * K / (M - 1))
        )

        def inner_fun(i, t4):
            start1 = ((i - 1) * K / (M - 1)).astype(int)
            length1 = int(K / (M - 1))
            temp1 = lax.dynamic_slice(t3, [0, start1], [t3.shape[0], length1])
            start2 = (2 * ((i - 1) * K / (M - 1) + 1)).astype(int)
            length2 = int(2 * K / (M - 1) + 1)
            temp2 = start2 + jnp.arange(length2) * 2
            return t4.at[:, i - 1].set(self._r_sum(temp1, temp2))

        t4 = lax.fori_loop(2, M, inner_fun, t4)
        t4 = t4.at[:, M - 1].set(
            self._r_sum(
                t3[:, jnp.arange(K, K + L)], jnp.arange(2 * (K + 1), 2 * (K + L) + 1, 2)
            )
        )
        x = jnp.zeros((n, M))

        def inner_fun2(i, x):
            return x.at[:, i - 1].set(
                jnp.maximum(t4[:, M - 1], A[i - 1]) * (t4[:, i - 1] - 0.5) + 0.5
            )

        x = lax.fori_loop(1, M, inner_fun2, x)
        x = x.at[:, M - 1].set(t4[:, M - 1])

        h = self._convex(x)
        h = h.at[:, M - 1].set(self._mixed(x))
        f = jnp.tile((D * x[:, M])[:, jnp.newaxis], (1, M)) + S * h
        return f, state

    """
        If result is not correct for some problems, it is necessary to use float64 globally
    """

    def pf(self, state):
        M = self.m
        N = self.ref_num * self.m
        R = UniformSampling(N, M)()[0]
        c = jnp.ones((R.shape[0], M))

        def inner_fun(i, c):
            def inner_fun2(j, c):
                temp = (
                    R[i - 1, j - 1]
                    / R[i - 1, 0]
                    * jnp.prod(1 - c[i - 1, M - j + 1 : M - 1])
                )
                c = c.at[i - 1, M - j].set(
                    (temp**2 - temp + jnp.sqrt(2 * temp)) / (temp**2 + 1)
                )
                return c

            with jax.disable_jit():
                c = lax.fori_loop(2, M + 1, inner_fun2, c)
            return c

        c = lax.fori_loop(1, R.shape[0] + 1, inner_fun, c)
        x = jnp.arccos(c) * 2 / jnp.pi
        temp = (1 - jnp.sin(jnp.pi / 2 * x[:, 1])) * R[:, M - 1] / R[:, M - 2]
        a = jnp.arange(0, 1.0001, 0.0001)[jnp.newaxis, :]
        E = jnp.abs(
            temp[:, None] * (1 - jnp.cos(jnp.pi / 2 * a))
            - 1
            + (a + jnp.cos(10 * jnp.pi * a + jnp.pi / 2) / 10 / jnp.pi)
        )
        rank = jnp.argsort(E, axis=1)

        def inner_fun3(i, x):
            return x.at[i, 0].set(a[0, jnp.min(rank[i, :10])])

        x = lax.fori_loop(0, x.shape[0], inner_fun3, x)
        f = self._convex(x)
        f = f.at[:, M - 1].set(self._mixed(x))
        f = f * jnp.tile(jnp.arange(2, 2 * M + 1, 2), (f.shape[0], 1))
        return f, state

    def _s_linear(self, y, A):
        output = jnp.abs(y - A) / jnp.abs(jnp.floor(A - y) + A)
        return output

    def _b_flat(self, y, A, B, C):
        output = (
            A
            + jnp.minimum(0, jnp.floor(y - B)) * A * (B - y) / B
            - jnp.minimum(0, jnp.floor(C - y)) * (1 - A) * (y - C) / (1 - C)
        )
        output = jnp.round(output * 1e4) / 1e4
        return output

    def _r_sum(self, y, w):
        return jnp.sum(y * w, axis=1) / jnp.sum(w)

    def _convex(self, x):
        return jnp.fliplr(
            jnp.cumprod(
                jnp.hstack(
                    [jnp.ones((x.shape[0], 1)), 1 - jnp.cos(x[:, :-1] * jnp.pi / 2)]
                ),
                axis=1,
            )
        ) * jnp.hstack(
            [jnp.ones((x.shape[0], 1)), 1 - jnp.sin(x[:, -2::-1] * jnp.pi / 2)]
        )

    def _mixed(self, x):
        return 1 - x[:, 0] - jnp.cos(10 * jnp.pi * x[:, 0] + jnp.pi / 2) / 10 / jnp.pi


class MaF11(MaF):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)
        if m is None:
            self.m = 3
        else:
            self.m = m
        if d is None:
            self.d = self.m + 9
        else:
            self.d = d
        self.d = jnp.ceil((self.d - self.m + 1) / 2) * 2 + self.m - 1

    @evox.jit_method
    def evaluate(self, state, X):
        N, D = X.shape
        M = self.m
        K = M - 1
        L = D - K
        D = 1
        S = jnp.arange(2, 2 * M + 1, 2)
        A = jnp.ones(M - 1)

        z01 = X / jnp.arange(2, X.shape[1] * 2 + 1, 2)

        t1 = jnp.zeros((N, K + L))
        t1 = t1.at[:, :K].set(z01[:, :K])
        t1 = t1.at[:, K:].set(self._s_linear(z01[:, K:], 0.35))

        t2 = jnp.zeros((N, K + L // 2))
        t2 = t2.at[:, :K].set(t1[:, :K])
        t2 = t2.at[:, K : K + L // 2].set(
            (
                t1[:, K::2]
                + t1[:, K + 1 :: 2]
                + 2 * jnp.abs(t1[:, K::2] - t1[:, K + 1 :: 2])
            )
            / 3
        )

        t3 = jnp.zeros((N, M))

        def inner_fun1(i, t3):
            start = (i - 1) * K // (M - 1)
            length = K // (M - 1)
            temp = lax.dynamic_slice(t2, [0, start], [t2.shape[0], length])
            return t3.at[:, i - 1].set(self._r_sum(temp, jnp.ones(K // (M - 1))))

        t3 = lax.fori_loop(1, M, inner_fun1, t3)
        t3 = t3.at[:, M - 1].set(self._r_sum(t2[:, K : K + L // 2], jnp.ones(L // 2)))

        x = jnp.zeros((N, M))

        def inner_fun2(i, x):
            return x.at[:, i - 1].set(
                jnp.maximum(t3[:, M - 1], A[i - 1]) * (t3[:, i - 1] - 0.5) + 0.5
            )

        x = lax.fori_loop(1, M, inner_fun2, x)
        x = x.at[:, M - 1].set(t3[:, M - 1])

        h = self._convex(x)
        h = h.at[:, M - 1].set(self._disc(x))
        f = D * x[:, M - 1].reshape(-1, 1) + S * h
        return f, state

    """
        If result is not correct for some problems, it is necessary to use float64 globally
    """

    def pf(self, state):
        M = self.m
        N = self.ref_num * self.m
        R = UniformSampling(N, M)()[0]
        c = jnp.ones((R.shape[0], M))

        def inner_fun(i, c):
            def inner_fun2(j, c):
                temp = R[i, j] / R[i, 0] * jnp.prod(1 - c[i, M - j : M - 1])
                return c.at[i, M - j - 1].set(
                    (temp**2 - temp + jnp.sqrt(2 * temp)) / (temp**2 + 1)
                )

            with jax.disable_jit():
                c = lax.fori_loop(1, M, inner_fun2, c)
            return c

        c = lax.fori_loop(0, R.shape[0], inner_fun, c)

        x = jnp.arccos(c) * 2 / jnp.pi
        temp = (1 - jnp.sin(jnp.pi / 2 * x[:, 1])) * R[:, M - 1] / R[:, M - 2]
        a = jnp.arange(0, 1.0001, 0.0001)[None, :]
        E = jnp.abs(
            temp[:, None] * (1 - jnp.cos(jnp.pi / 2 * a))
            - 1
            + a * jnp.cos(5 * jnp.pi * a) ** 2
        )
        rank = jnp.argsort(E, axis=1)
        x = x.at[:, 0].set(a[0, jnp.min(rank[:, :10], axis=1)])
        R = self._convex(x)
        R = R.at[:, M - 1].set(self._disc(x))
        non_dominated_rank = non_dominated_sort(R)
        f = R[non_dominated_rank == 0, :]
        f = f * jnp.arange(2, 2 * M + 1, 2)
        return f, state

    @evox.jit_method
    def _s_linear(self, y, A):
        return jnp.abs(y - A) / jnp.abs(jnp.floor(A - y) + A)

    @evox.jit_method
    def _r_nonsep(self, y, A):
        Output = jnp.zeros((y.shape[0], 1))
        for j in range(y.shape[1]):
            Temp = jnp.zeros((y.shape[0], 1))
            for k in range(A - 2):
                Temp += jnp.abs(y[:, j] - y[:, (j + k) % y.shape[1]])
            Output += y[:, j] + Temp
        Output /= (y.shape[1] / A) / jnp.ceil(A / 2) / (1 + 2 * A - 2 * jnp.ceil(A / 2))
        return Output

    @evox.jit_method
    def _r_sum(self, y, w):
        return jnp.sum(y * w, axis=1) / jnp.sum(w)

    @evox.jit_method
    def _convex(self, x):
        return jnp.fliplr(
            jnp.cumprod(
                jnp.hstack(
                    (jnp.ones((x.shape[0], 1)), 1 - jnp.cos(x[:, :-1] * jnp.pi / 2))
                ),
                axis=1,
            )
        ) * jnp.hstack(
            (
                jnp.ones((x.shape[0], 1)),
                1 - jnp.sin(x[:, x.shape[1] - 2 :: -1] * jnp.pi / 2),
            )
        )

    @evox.jit_method
    def _disc(self, x):
        return 1 - x[:, 0] * (jnp.cos(5 * jnp.pi * x[:, 0])) ** 2


@evox.jit_class
class MaF12(MaF):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)
        if m is None:
            self.m = 3
        else:
            self.m = m
        if d is None:
            self.d = self.m + 9
        else:
            self.d = d

    def evaluate(self, state, X):
        N, D = X.shape
        M = self.m
        K = M - 1
        L = D - K
        D = 1
        S = jnp.arange(2, 2 * M + 1, 2)
        A = jnp.ones(M - 1)

        z01 = X / jnp.arange(2, X.shape[1] * 2 + 1, 2)

        t1 = jnp.zeros((N, K + L))
        Y = (jnp.flip(jnp.cumsum(jnp.flip(z01, 1), 1), 1) - z01) / jnp.arange(
            K + L - 1, -1, -1
        )
        t1 = t1.at[:, : K + L - 1].set(
            z01[:, : K + L - 1]
            ** (
                0.02
                + (50 - 0.02)
                * (
                    0.98 / 49.98
                    - (1 - 2 * Y[:, : K + L - 1])
                    * jnp.abs(jnp.floor(0.5 - Y[:, : K + L - 1]) + 0.98 / 49.98)
                )
            )
        )
        t1 = t1.at[:, -1].set(z01[:, -1])

        t2 = jnp.zeros((N, K + L))
        t2 = t2.at[:, :K].set(self._s_decept(t1[:, :K], 0.35, 0.001, 0.05))
        t2 = t2.at[:, K:].set(self._s_multi(t1[:, K:], 30, 95, 0.35))

        t3 = jnp.zeros((N, M))

        def inner_fun(i, t3):
            start = ((i - 1) * K / (M - 1)).astype(int)
            length = int(K / (M - 1))
            temp = lax.dynamic_slice(t2, [0, start], [t2.shape[0], length])
            return t3.at[:, i - 1].set(self._r_nonsep(temp, int(K / (M - 1))))

        t3 = lax.fori_loop(1, M, inner_fun, t3)

        SUM = jnp.zeros(N)

        def inner_fun2(i, SUM):
            def inner_fun3(j, SUM):
                SUM += jnp.abs(t2[:, i] - t2[:, j])
                return SUM

            return lax.fori_loop(i + 1, K + L, inner_fun3, SUM)

        SUM = lax.fori_loop(K, K + L - 1, inner_fun2, SUM)

        t3 = t3.at[:, M - 1].set(
            (jnp.sum(t2[:, K:], axis=1) + SUM * 2)
            / jnp.ceil(L / 2)
            / (1 + 2 * L - 2 * jnp.ceil(L / 2))
        )

        x = jnp.zeros((N, M))

        def inner_fun4(i, x):
            return x.at[:, i].set(
                jnp.maximum(t3[:, M - 1], A[i]) * (t3[:, i] - 0.5) + 0.5
            )

        x = lax.fori_loop(0, M - 1, inner_fun4, x)
        x = x.at[:, M - 1].set(t3[:, M - 1])

        h = self._concave(x)
        f = D * x[:, M - 1].reshape(-1, 1) + S * h
        return f, state

    def pf(self, state):
        M = self.m
        N = self.ref_num * self.m
        R = UniformSampling(N, M)()[0]
        R = R / jnp.sqrt(jnp.sum(R**2, axis=1)).reshape(-1, 1)
        f = jnp.arange(2, 2 * M + 1, 2) * R
        return f, state

    def _s_decept(self, y, A, B, C):
        return 1 + (jnp.abs(y - A) - B) * (
            jnp.floor(y - A + B) * (1 - C + (A - B) / B) / (A - B)
            + jnp.floor(A + B - y) * (1 - C + (1 - A - B) / B) / (1 - A - B)
            + 1 / B
        )

    def _s_multi(self, y, A, B, C):
        return (
            1
            + jnp.cos(
                (4 * A + 2)
                * jnp.pi
                * (0.5 - jnp.abs(y - C) / 2 / (jnp.floor(C - y) + C))
            )
            + 4 * B * (jnp.abs(y - C) / 2 / (jnp.floor(C - y) + C)) ** 2
        ) / (B + 2)

    def _r_nonsep(self, y, A):
        Output = jnp.zeros(y.shape[0])
        for j in range(y.shape[1]):
            Temp = jnp.zeros(y.shape[0])
            for k in range(A - 1):
                Temp += jnp.abs(y[:, j] - y[:, (j + 1 + k) % y.shape[1]])
            Output += y[:, j] + Temp
        return (
            Output
            / (y.shape[1] / A)
            / jnp.ceil(A / 2)
            / (1 + 2 * A - 2 * jnp.ceil(A / 2))
        )

    def _concave(self, x):
        return jnp.fliplr(
            jnp.cumprod(
                jnp.hstack(
                    [jnp.ones((x.shape[0], 1)), jnp.sin(x[:, :-1] * jnp.pi / 2)]
                ),
                axis=1,
            )
        ) * jnp.hstack(
            [
                jnp.ones((x.shape[0], 1)),
                jnp.cos(x[:, x.shape[1] - 2 :: -1] * jnp.pi / 2),
            ]
        )


@evox.jit_class
class MaF13(MaF):
    def __init__(self, d=None, m=None, ref_num=1000):
        if m is None:
            self.m = 3
        else:
            self.m = m
        if d is None:
            self.d = 5
        else:
            self.d = d
        self.m = jnp.maximum(self.m, 3)
        super().__init__(d, m, ref_num)

    def evaluate(self, state, X):
        N, D = X.shape
        Y = X - 2 * X[:, 1].reshape(-1, 1) * jnp.sin(
            2 * jnp.pi * X[:, 0].reshape(-1, 1) + jnp.arange(1, D + 1) * jnp.pi / D
        )
        f = jnp.zeros((N, self.m))
        f = f.at[:, 0].set(
            jnp.sin(X[:, 0] * jnp.pi / 2) + 2 * jnp.mean(Y[:, 3:D:3] ** 2, axis=1)
        )
        f = f.at[:, 1].set(
            jnp.cos(X[:, 0] * jnp.pi / 2) * jnp.sin(X[:, 1] * jnp.pi / 2)
            + 2 * jnp.mean(Y[:, 4:D:3] ** 2, axis=1)
        )
        f = f.at[:, 2].set(
            jnp.cos(X[:, 0] * jnp.pi / 2) * jnp.cos(X[:, 1] * jnp.pi / 2)
            + 2 * jnp.mean(Y[:, 2:D:3] ** 2, axis=1)
        )
        f = f.at[:, 3:].set(
            jnp.tile(
                (
                    f[:, 0] ** 2
                    + f[:, 1] ** 10
                    + f[:, 2] ** 10
                    + 2 * jnp.mean(Y[:, 3:D] ** 2, axis=1)
                )[:, None],
                (1, self.m - 3),
            )
        )
        return f, state

    def pf(self, state):
        M = self.m
        N = self.ref_num * self.m
        R = UniformSampling(N, 3)()[0]
        R = R / (jnp.sqrt(jnp.sum(R**2, axis=1))[:, None])
        f = jnp.hstack(
            [
                R,
                jnp.tile(
                    (R[:, 0] ** 2 + R[:, 1] ** 10 + R[:, 2] ** 10)[:, None], (1, M - 3)
                ),
            ]
        )
        return f, state


@evox.jit_class
class MaF14(MaF):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)
        if m is None:
            self.m = 3
        else:
            self.m = m
        if d is None:
            self.d = 20 * self.m
        else:
            self.d = d
        nk = 2
        c = [3.8 * 0.1 * (1 - 0.1)]
        for i in range(1, self.m):
            c.append(3.8 * c[-1] * (1 - c[-1]))
        c = jnp.array(c)
        self.sublen = jnp.floor(c / jnp.sum(c) * (self.d - self.m + 1) / nk).astype(int)
        self.len = jnp.concatenate([jnp.array([0]), jnp.cumsum(self.sublen * nk)])
        self.sublen = tuple(map(int, self.sublen))
        self.len = tuple(map(int, self.len))

    def evaluate(self, state, X):
        N, D = X.shape
        M = self.m
        nk = 2
        X = X.at[:, M - 1 :].set(
            (1 + jnp.arange(M, D + 1) / D) * X[:, M - 1 :] - (X[:, 0] * 10)[:, None]
        )
        G = jnp.zeros((N, M))

        def inner_loop(i, inner_fun, g):
            for j in range(0, nk):
                start = self.len[i] + self.m - 1 + j * self.sublen[i]
                end = start + self.sublen[i]
                temp = X[:, start:end]
                g = g.at[:, i].set(g[:, i] + inner_fun(temp))
            return g

        for i in range(0, M, 2):
            G = inner_loop(i, self._Rastrigin, G)
        for i in range(1, M, 2):
            G = inner_loop(i, self._Rosenbrock, G)
        G /= jnp.array(self.sublen)[None, :] * nk
        f = (
            (1 + G)
            * jnp.fliplr(
                jnp.cumprod(jnp.hstack([jnp.ones((N, 1)), X[:, : M - 1]]), axis=1)
            )
            * jnp.hstack([jnp.ones((N, 1)), 1 - X[:, M - 2 :: -1]])
        )
        return f, state

    def pf(self, state):
        M = self.m
        N = self.ref_num * self.m
        f = UniformSampling(N, M)()[0]
        return f, state

    def _Rastrigin(self, x):
        return jnp.sum(x**2 - 10 * jnp.cos(2 * jnp.pi * x) + 10, axis=1)

    def _Rosenbrock(self, x):
        return jnp.sum(
            100 * (x[:, :-1] ** 2 - x[:, 1:]) ** 2 + (x[:, :-1] - 1) ** 2, axis=1
        )


@evox.jit_class
class MaF15(MaF):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)
        if m is None:
            self.m = 3
        else:
            self.m = m
        if d is None:
            self.d = 20 * self.m
        else:
            self.d = d
        nk = 2
        c = [3.8 * 0.1 * (1 - 0.1)]
        for i in range(1, self.m):
            c.append(3.8 * c[-1] * (1 - c[-1]))
        c = jnp.array(c)
        self.sublen = jnp.floor(c / jnp.sum(c) * (self.d - self.m + 1) / nk).astype(int)
        self.len = jnp.concatenate([jnp.array([0]), jnp.cumsum(self.sublen * nk)])
        self.sublen = tuple(map(int, self.sublen))
        self.len = tuple(map(int, self.len))
        self.sphere = Sphere()
        self.griewank = Griewank()

    """
        The use of the for loop is due to the two variables: self.sub and self.len, which make dynamic slice and prevent the use of fori_loop.
    """

    def evaluate(self, state, X):
        N, D = X.shape
        M = self.m
        nk = 2
        X = X.at[:, M - 1 :].set(
            (1 + jnp.cos(jnp.arange(M, D + 1) / D * jnp.pi / 2))[None, :]
            * X[:, M - 1 :]
            - (X[:, 0] * 10)[:, None]
        )
        G = jnp.zeros((N, M))

        def inner_loop(i, inner_fun, g):
            for j in range(0, nk):
                start = self.len[i] + self.m - 1 + j * self.sublen[i]
                end = start + self.sublen[i]
                temp = X[:, start:end]
                g = g.at[:, i].set(g[:, i] + inner_fun(state, temp)[0])
            return g

        for i in range(0, M, 2):
            G = inner_loop(i, self.griewank.evaluate, G)
        for i in range(1, M, 2):
            G = inner_loop(i, self.sphere.evaluate, G)
        G /= jnp.array(self.sublen)[None, :] * nk
        f = (1 + G + jnp.hstack([G[:, 1:], jnp.zeros((N, 1))])) * (
            1
            - jnp.fliplr(
                jnp.cumprod(
                    jnp.hstack([jnp.ones((N, 1)), jnp.cos(X[:, : M - 1] * jnp.pi / 2)]),
                    axis=1,
                )
            )
            * jnp.hstack([jnp.ones((N, 1)), jnp.sin(X[:, M - 2 :: -1] * jnp.pi / 2)])
        )
        return f, state

    def pf(self, state):
        M = self.m
        N = self.ref_num * self.m
        R = UniformSampling(N, M)()[0]
        R = 1 - R / jnp.sqrt(jnp.sum(R**2, axis=1)).reshape(-1, 1)
        return R, state
