import evox
import jax
from jax import lax
import jax.numpy as jnp
from src.evox import Problem, State, jit_class
from src.evox.operators.sampling import UniformSampling
from scipy.spatial.distance import pdist
import chex
from functools import partial
from jax import vmap
from matplotlib.path import Path
from jax.config import config
from evox.operators.non_dominated_sort import non_dominated_sort

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

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        chex.assert_type(X, float)
        chex.assert_shape(X, (None, self.ref_num))
        return jax.jit(jax.vmap(self._maf))(X), state

    def pf(self, state: chex.PyTreeDef):
        f = 1 - UniformSampling(self.ref_num * self.m, self.m)()[0]
        # f = LatinHypercubeSampling(self.ref_num * self.m, self.m).random(state.key)[0] / 2
        return f, state


@evox.jit_class
class MaF1(MaF):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        m = self.m
        n, d = jnp.shape(X)
        g = jnp.sum((X[:, m - 1:] - 0.5).__pow__(2), axis=1).reshape((-1, 1))
        ones_col = jnp.ones((n, 1))
        cumprod_term = jnp.fliplr(jnp.cumprod(jnp.hstack([ones_col, X[:, :m - 1]]), axis=1))
        reversed_term = jnp.hstack([ones_col, 1 - X[:, m - 2::-1]])  # Reversed slice for last term
        repeat_g = jnp.tile(1 + g, (1, m))
        f = repeat_g - repeat_g * cumprod_term * reversed_term
        return f, state

    def pf(self, state: chex.PyTreeDef):
        n = self.ref_num * self.m
        # n = 1000
        f = 1 - UniformSampling(n, self.m)()[0]
        return f, state

# @evox.jit_class
class MaF2(MaF):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        m = self.m
        n, d = jnp.shape(X)
        g = jnp.zeros((n, m))
        interval =  int((d - m + 1) / m)
        mask_ones = jnp.ones((X.shape[0], interval),dtype=bool)
        mask_zeros = jnp.zeros_like(X,dtype=bool)
        def body_fun(i, g):
            def true_fun():
                start = (m + i * jnp.floor((d - m + 1) / m) - 1).astype(int)
                mask = lax.dynamic_update_slice(mask_zeros,mask_ones,(0,start))
                temp = jnp.where(mask,X,0.5)
                return temp / 2 + 1 / 4
            def false_fun():
                start = m + (m - 1) * int((d - m + 1) / m) - 1
                mask_ones = jnp.ones((X.shape[0],d - start),dtype=bool)
                mask = lax.dynamic_update_slice(mask_zeros,mask_ones,(0,start))
                temp = jnp.where(mask, X, 0.5)
                return temp / 2 + 1 / 4

            temp1 = lax.cond(i < m - 1, true_fun, false_fun)
            return g.at[:, i].set(jnp.sum((temp1 - 0.5) ** 2, axis=1))
        g = lax.fori_loop(0, m, body_fun, g)

        f1 = jnp.fliplr(
            jnp.cumprod(jnp.hstack([jnp.ones((n, 1)), jnp.cos((X[:, :m - 1] / 2 + 1 / 4) * jnp.pi / 2)]), axis=1))
        f2 = jnp.hstack([jnp.ones((n, 1)), jnp.sin(((X[:, m - 2::-1]) / 2 + 1 / 4) * jnp.pi / 2)])
        f = (1 + g) * f1 * f2
        return f, state

    def pf(self, state: chex.PyTreeDef):
        n = self.ref_num * self.m
        # n = 1000
        r = UniformSampling(n, self.m)()[0]
        c = jnp.zeros((r.shape[0], self.m - 1))
        for i in range(1, n + 1):
            for j in range(2, self.m+1):
                temp = r[i-1, j-1] / r[i-1, 0] * jnp.prod(c[i-1, self.m - j + 1:self.m - 1])
                c = c.at[i-1, self.m - j].set(jnp.sqrt(1 / (1 + temp**2)))
        # lax.fori_loop(1,n+1,)

        if self.m > 5:
            c = c * (jnp.cos(jnp.pi / 8) - jnp.cos(3 * jnp.pi / 8)) + jnp.cos(3 * jnp.pi / 8)
        else:
            # temp = jnp.any(jnp.logical_or(c < jnp.cos(3 * jnp.pi / 8), c > jnp.cos(jnp.pi / 8)), axis=1)
            # c = jnp.delete(c, temp.flatten() == 1, axis=0)
            c = c[jnp.all((c >= jnp.cos(3 * jnp.pi / 8)) & (c <= jnp.cos(jnp.pi / 8)), axis=1)]


        n, _ = jnp.shape(c)
        f = jnp.fliplr(jnp.cumprod(jnp.hstack([jnp.ones((n, 1)), c[:, :self.m - 1]]), axis=1)) * jnp.hstack(
            [jnp.ones((n, 1)), jnp.sqrt(1 - c[:, self.m - 2::-1]**2)])
        return f, state


@evox.jit_class
class MaF3(MaF):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        n, d = jnp.shape(X)
        g = 100 * (d - self.m + 1 + jnp.sum(
            (X[:, self.m - 1:] - 0.5)**2 - jnp.cos(20 * jnp.pi * (X[:, self.m - 1:] - 0.5)), axis=1)).reshape((-1,1))
        f1 = jnp.tile(1 + g, (1, self.m)) * jnp.fliplr(
            jnp.cumprod(jnp.hstack([jnp.ones((n, 1)), jnp.cos(X[:, :self.m - 1] * jnp.pi / 2)]), axis=1)) * jnp.hstack(
            [jnp.ones((n, 1)), jnp.sin(X[:, self.m - 2::-1] * jnp.pi / 2)])
        f = jnp.hstack([f1[:, :self.m - 1]**4, (f1[:, self.m - 1]**2).reshape((-1,1))])
        return f, state

    def pf(self, state: chex.PyTreeDef):
        n = self.ref_num * self.m
        # n = 1000
        r = UniformSampling(n, self.m)()[0]**2
        temp = (jnp.sum(jnp.sqrt(r[:, :-1]), axis=1) + r[:, -1]).reshape((-1,1))
        f = r / jnp.hstack([jnp.tile(temp**2, (1, r.shape[1]-1)), temp])
        return f, state


@evox.jit_class
class MaF4(MaF):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        n, d = jnp.shape(X)
        g = 100 * (d - self.m + 1 + jnp.sum(
            (X[:, self.m - 1:] - 0.5)**2 - jnp.cos(20 * jnp.pi * (X[:, self.m - 1:] - 0.5)), axis=1))[:, jnp.newaxis]
        f1 = jnp.tile(1 + g,(1, self.m)) - jnp.tile(1 + g, (1, self.m)) * jnp.fliplr(
            jnp.cumprod(jnp.hstack([jnp.ones((n, 1)), jnp.cos(X[:, :self.m - 1] * jnp.pi / 2)]), axis=1)) * jnp.hstack(
            [jnp.ones((n, 1)), jnp.sin(X[:, self.m - 2::-1] * jnp.pi / 2)])
        f = f1 * jnp.tile(jnp.power(2, jnp.arange(1, self.m + 1)), (n, 1))
        return f, state

    def pf(self, state: chex.PyTreeDef):
        n = self.ref_num * self.m
        # n = 1000
        r = UniformSampling(n, self.m)()[0]
        r1 = r / jnp.tile(jnp.sqrt(jnp.sum(r**2, axis=1))[:, jnp.newaxis], (1, self.m))
        f = (1 - r1) * jnp.tile(jnp.power(2, jnp.arange(1, self.m + 1)), (r.shape[0], 1))
        return f, state


@evox.jit_class
class MaF5(MaF):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        n, d = jnp.shape(X)
        X = X.at[:, :self.m - 1].set(X[:, :self.m - 1]**100)
        g = jnp.sum((X[:, self.m - 1:] - 0.5)**2, axis=1)[:,jnp.newaxis]
        f1 = jnp.tile(1 + g, (1, self.m)) * jnp.fliplr(
            jnp.cumprod(jnp.hstack([jnp.ones((n, 1)), jnp.cos(X[:, :self.m - 1] * jnp.pi / 2)]), axis=1)) * jnp.hstack(
            [jnp.ones((n, 1)), jnp.sin(X[:, self.m - 2::-1] * jnp.pi / 2)])
        f = f1 * jnp.tile(jnp.power(2, jnp.arange(self.m, 0, -1)), (n, 1))
        return f, state

    def pf(self, state: chex.PyTreeDef):
        n = self.ref_num * self.m
        # n = 1000
        r = UniformSampling(n, self.m)()[0]
        r1 = r / jnp.tile(jnp.sqrt(jnp.sum(r**2, axis=1))[:, jnp.newaxis], (1,self.m))
        f = r1 * jnp.tile(jnp.power(2, jnp.arange(self.m, 0, -1)), (r.shape[0], 1))
        return f, state


@evox.jit_class
class MaF6(MaF):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        n, d = jnp.shape(X)
        i = 2
        g = jnp.sum((X[:, self.m - 1:] - 0.5)**2, axis=1)[:, jnp.newaxis]
        temp = jnp.tile(g, (1, self.m - i))
        X = X.at[:, i - 1:self.m - 1].set((1 + 2 * temp * X[:, i - 1:self.m - 1]) / (2 + 2 * temp))
        f = jnp.tile((1 + 100 * g),(1, self.m)) * jnp.fliplr(
            jnp.cumprod(jnp.hstack([jnp.ones((n, 1)), jnp.cos(X[:, :self.m - 1] * jnp.pi / 2)]), axis=1)) * jnp.hstack(
            [jnp.ones((n, 1)), jnp.sin(X[:, self.m - 2::-1] * jnp.pi / 2)])
        return f, state

    def pf(self, state: chex.PyTreeDef):
        i = 2
        # n = self.ref_num * self.m
        n = 1000
        r = UniformSampling(n, i)()[0]
        r1 = r / jnp.tile(jnp.sqrt(jnp.sum(r**2, axis=1))[:, jnp.newaxis], (1, r.shape[1]))
        if r1.shape[1] < self.m:
            r1 = jnp.hstack((r1[:, jnp.zeros((self.m - r1.shape[1])).astype(int)], r1))
        f = r1 / jnp.power(jnp.sqrt(2),
                           jnp.tile(jnp.maximum(self.m - i, 0), (r.shape[0], 1)))
        return f, state


@evox.jit_class
class MaF7(MaF):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        n, d = jnp.shape(X)
        f = jnp.zeros((n, self.m))
        g = 1 + 9 * jnp.mean(X[:, self.m - 1:], axis=1)
        f = f.at[:, :self.m - 1].set(X[:, :self.m - 1])
        f = f.at[:, self.m - 1].set((1 + g) * (self.m - jnp.sum(
            f[:, :self.m - 1] / (1 + jnp.tile(g[:, jnp.newaxis], (1, self.m - 1))) * (
                    1 + jnp.sin(3 * jnp.pi * f[:, :self.m - 1])), axis=1)))
        return f, state

    def pf(self, state: chex.PyTreeDef):
        n = self.ref_num * self.m
        # n = 1000
        interval = jnp.array([0, 0.251412, 0.631627, 0.859401])
        median = (interval[1] - interval[0]) / (interval[3] - interval[2] + interval[1] - interval[0])
        X = self._grid(n, self.m - 1)
        X = jnp.where(X <= median, X * (interval[1] - interval[0]) / median + interval[0], X)
        X = jnp.where(X > median, (X - median) * (interval[3] - interval[2]) / (1 - median) + interval[2], X)
        f = jnp.hstack([X, 2 * (self.m - jnp.sum(X / 2 * (1 + jnp.sin(3 * jnp.pi * X)), axis=1))[:,jnp.newaxis]])
        return f, state

    def _grid(self,N, M):
        gap = jnp.linspace(0, 1, jnp.ceil(N ** (1 / M)).astype(int))
        c = jnp.meshgrid(*[gap] * M)
        W = jnp.column_stack([c[i].ravel() for i in range(M)])
        return jnp.flip(W,axis=1)


@evox.jit_class
class MaF8(MaF):
    """
    the dimention only is 2.
    """
    def __init__(self, d=None, m=None, ref_num=1000):
        d = 2
        super().__init__(d, m, ref_num)
        self.points = self._getPoints()

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        if(X.shape[1] != 2):
            X = X[:, :2]
        f = self._eucl_dis(X, self.points)
        return f, state

    def pf(self, state: chex.PyTreeDef):
        n = self.ref_num * self.m
        # [X, Y] = ndgrid(linspace(-1, 1, ceil(sqrt(N))));
        temp = jnp.linspace(-1, 1, num=jnp.ceil(jnp.sqrt(n)).astype(int))
        x, y = jnp.meshgrid(temp, temp)
        x = x.ravel(order="F")
        y = y.ravel(order="F")
        # using np based library, this may make some mistakes, but in my test, there is no warning
        poly_path = Path(self.points)
        _points = jnp.column_stack((x, y))
        ND = poly_path.contains_points(_points)

        f = self._eucl_dis(jnp.column_stack([x[ND], y[ND]]), self.points)
        return f, state

    def _getPoints(self):
        thera, rho = self._cart2pol(0, 1)
        temp = jnp.arange(1, self.m + 1).reshape((-1, 1))
        x, y = self._pol2cart(thera - temp * 2 * jnp.pi / self.m, rho)
        return jnp.column_stack([x, y])

    def _cart2pol(self, x, y):
        rho = jnp.sqrt(jnp.power(x, 2) + jnp.power(y, 2))
        theta = jnp.arctan2(y, x)
        return theta, rho

    def _pol2cart(self, theta, rho):
        x = rho * jnp.cos(theta)
        y = rho * jnp.sin(theta)
        return (x, y)

    def _eucl_dis(self, x, y):
        dist_matrix = jnp.linalg.norm(x[:, None] - y, axis=-1)
        return dist_matrix


@evox.jit_class
class MaF9(MaF):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)
        # Generate vertexes
        self.points = self._getPoints()

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        n, d = jnp.shape(X)
        m, d1 = jnp.shape(self.points)
        f = jnp.zeros((n, m))
        for i in range(m):
            f = f.at[:, i].set(self._Point2Line(X, self.points[jnp.mod(jnp.arange(i, i + 2), m), :]))
        return f, state

    def pf(self, state: chex.PyTreeDef):
        n = self.ref_num * self.m
        # n = 1000
        # [X, Y] = ndgrid(linspace(-1, 1, ceil(sqrt(N))));
        temp = jnp.linspace(-1, 1, num=jnp.ceil(jnp.sqrt(n)).astype(int))
        x, y = jnp.meshgrid(temp, temp)
        x = x.ravel(order="C")
        y = y.ravel(order="C")
        # using jnp as np, this may make some mistakes, but in my test, there is no warning
        poly_path = Path(self.points)
        _points = jnp.column_stack((x, y))
        ND = poly_path.contains_points(_points)

        f, state = self.evaluate(state, jnp.column_stack((x[ND], y[ND])))
        return f, state

    def _getPoints(self):
        thera, rho = self._cart2pol(0, 1)
        temp = jnp.arange(1, self.m + 1).reshape((-1, 1))
        x, y = self._pol2cart(thera - temp * 2 * jnp.pi / self.m, rho)
        return jnp.column_stack([x, y])

    def _cart2pol(self, x, y):
        rho = jnp.sqrt(jnp.power(x, 2) + jnp.power(y, 2))
        theta = jnp.arctan2(y, x)
        return theta, rho

    def _pol2cart(self, theta, rho):
        x = rho * jnp.cos(theta)
        y = rho * jnp.sin(theta)
        return (x, y)

    def _Point2Line(self, PopDec, Line):
        Distance = jnp.abs((Line[0, 0] - PopDec[:, 0]) * (Line[1, 1] - PopDec[:, 1]) - (Line[1, 0] - PopDec[:, 0]) * (
                    Line[0, 1] - PopDec[:, 1])) / jnp.sqrt(
            (Line[0, 0] - Line[1, 0]) ** 2 + (Line[0, 1] - Line[1, 1]) ** 2)
        return Distance

        # return distance


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


    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
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

        t3 = t2 ** 0.02

        t4 = jnp.zeros((n, M))
        t4 = t4.at[:, 0].set(self._r_sum(t3[:, int(K/(M-1)-1)][:,None], 2*K/(M-1)))
        for i in range(2, M):
            t4 = t4.at[:, i - 1].set(self._r_sum(t3[:, jnp.arange(int((i - 1) * K / (M - 1)), int(i * K / (M - 1)))],
                                                 jnp.arange(int(2 * ((i - 1) * K / (M - 1) + 1)),
                                                            int(2 * i * K / (M - 1) + 1), 2)))
        t4 = t4.at[:, M - 1].set(
            self._r_sum(t3[:, jnp.arange(K, K + L)], jnp.arange(2 * (K + 1), 2 * (K + L) + 1, 2)))
        x = jnp.zeros((n, M))
        for i in range(1, M):
            x = x.at[:, i - 1].set(jnp.maximum(t4[:, M - 1], A[i - 1]) * (t4[:, i - 1] - 0.5) + 0.5)
        x = x.at[:, M - 1].set(t4[:, M - 1])

        h = self._convex(x)
        h = h.at[:, M-1].set(self._mixed(x))
        f = jnp.tile((D * x[:, M])[:,jnp.newaxis], (1, M)) + S * h
        return f, state

    '''精度必须是float64'''
    def pf(self, state: chex.PyTreeDef):
        config.update("jax_enable_x64", True)
        M = self.m
        N = self.ref_num * self.m
        # N = 1000
        R = UniformSampling(N, M)()[0]
        R = R.astype(jnp.float64)
        c = jnp.ones((R.shape[0], M), dtype=jnp.float64)
        for i in range(1, R.shape[0] + 1):
            for j in range(2, M + 1):
                temp = R[i - 1, j - 1] / R[i - 1, 0] * jnp.prod(1 - c[i - 1, M - j + 1:M - 1])
                c = c.at[i - 1, M - j].set((temp ** 2 - temp + jnp.sqrt(2 * temp)) / (temp ** 2 + 1))
        x = jnp.arccos(c) * 2 / jnp.pi
        temp = (1 - jnp.sin(jnp.pi / 2 * x[:, 1])) * R[:, M - 1] / R[:, M - 2]
        a = jnp.arange(0, 1.0001, 0.0001, dtype=jnp.float64)[jnp.newaxis,:]
        E = jnp.abs(temp[:, None] * (1 - jnp.cos(jnp.pi / 2 * a)) - 1 + (a + jnp.cos(10 * jnp.pi * a + jnp.pi / 2) / 10 / jnp.pi))
        # E = jnp.abs(temp[:, None] @ (1 - jnp.cos(jnp.pi / 2 * a)) - 1 + jnp.tile(a + jnp.cos(10 * jnp.pi * a + jnp.pi / 2) / 10 / jnp.pi, (x.shape[0], 1)))
        rank = jnp.argsort(E, axis=1) # rank is wrong!!!
        for i in range(x.shape[0]):
            x = x.at[i, 0].set(a[0, jnp.min(rank[i, :10])])
        f = self._convex(x)
        f = f.at[:, M - 1].set(self._mixed(x))
        f = f * jnp.tile(jnp.arange(2, 2 * M + 1, 2), (f.shape[0], 1))
        return f, state

    def _s_linear(self, y, A):
        output = jnp.abs(y - A)/jnp.abs(jnp.floor(A - y) + A)
        return output

    def _b_flat(self, y, A, B, C):
        output = A + jnp.minimum(0, jnp.floor(y - B)) * A * (B - y)/B - jnp.minimum(0, jnp.floor(C - y)) * (1 - A) * (
                y - C) / (1 - C)
        output = jnp.round(output * 1e4) / 1e4
        return output

    def _r_sum(self, y, w):
        return jnp.sum(y * w, axis=1) / jnp.sum(w)

    def _convex(self, x):
        return jnp.fliplr(jnp.cumprod(jnp.hstack([jnp.ones((x.shape[0], 1)), 1 - jnp.cos(x[:, :-1] * jnp.pi / 2)]),
                                      axis=1)) * jnp.hstack(
            [jnp.ones((x.shape[0], 1)), 1 - jnp.sin(x[:, -2::-1] * jnp.pi / 2)])

    def _mixed(self, x):
        return 1 - x[:, 0] - jnp.cos(10 * jnp.pi * x[:, 0] + jnp.pi / 2) / 10 / jnp.pi

# @evox.jit_class
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


    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
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
        t2 = t2.at[:, K:K + L // 2].set(
            (t1[:, K::2] + t1[:, K + 1::2] + 2 * jnp.abs(t1[:, K::2] - t1[:, K + 1::2])) / 3)

        t3 = jnp.zeros((N, M))
        for i in range(1, M):
            t3 = t3.at[:, i - 1].set(
                self._r_sum(t2[:, (i - 1) * K // (M - 1):i * K // (M - 1)], jnp.ones(K // (M - 1))))
        t3 = t3.at[:, M - 1].set(self._r_sum(t2[:, K:K + L // 2], jnp.ones(L // 2)))

        x = jnp.zeros((N, M))
        for i in range(1, M):
            x = x.at[:, i - 1].set(jnp.maximum(t3[:, M - 1], A[i - 1]) * (t3[:, i - 1] - 0.5) + 0.5)
        x = x.at[:, M - 1].set(t3[:, M - 1])

        h = self._convex(x)
        h = h.at[:, M - 1].set(self._disc(x))
        f = D * x[:, M - 1].reshape(-1, 1) + S * h
        return f, state

    '''精度必须是float64, 不能使用JIT, 后续计划直接读数据'''
    def pf(self, state: chex.PyTreeDef):
        config.update("jax_enable_x64", True)
        M = self.m
        # N = self.ref_num * self.m
        N = 1000
        R = UniformSampling(N, M)()[0].astype(jnp.float64)
        c = jnp.ones((R.shape[0], M))
        for i in range(R.shape[0]):
            for j in range(1, M):
                temp = R[i, j] / R[i, 0] * jnp.prod(1 - c[i, M - j:M - 1]).astype(jnp.float64)
                c = c.at[i, M - j - 1].set((temp ** 2 - temp + jnp.sqrt(2 * temp)) / (temp ** 2 + 1)).astype(jnp.float64)
        x = jnp.arccos(c) * 2 / jnp.pi
        temp = (1 - jnp.sin(jnp.pi / 2 * x[:, 1])) * R[:, M - 1] / R[:, M - 2]
        a = jnp.arange(0, 1.0001, 0.0001)[None,:].astype(jnp.float64)
        E = jnp.abs(temp[:,None] * (1 - jnp.cos(jnp.pi / 2 * a)) - 1 + a * jnp.cos(5 * jnp.pi * a) ** 2).astype(jnp.float64)
        rank = jnp.argsort(E, axis=1)
        for i in range(x.shape[0]):
            x = x.at[i, 0].set(a[0, jnp.min(rank[i, :10])])
        R = self._convex(x)
        R = R.at[:, M - 1].set(self._disc(x)).astype(jnp.float64)
        non_dominated_rank = non_dominated_sort(R)
        # indices = jnp.nonzero(non_dominated_rank)
        # indices = jnp.argwhere(non_dominated_rank != 0).squeeze()
        mask = (non_dominated_rank != 0)[:,None]
        f = jnp.where(mask, 0, R)
        # f = R[non_dominated_rank == 0].astype(jnp.float64)
        # f = R.at[mask,:].set(0)
        f = f * jnp.arange(2, 2 * M + 1, 2).astype(jnp.float64)
        return f, state

    def _s_linear(self, y, A):
        return jnp.abs(y - A) / jnp.abs(jnp.floor(A - y) + A)

    def _r_nonsep(self, y, A):
        Output = jnp.zeros((y.shape[0], 1))
        for j in range(y.shape[1]):
            Temp = jnp.zeros((y.shape[0], 1))
            for k in range(A - 2):
                Temp += jnp.abs(y[:, j] - y[:, (j + k) % y.shape[1]])
            Output += y[:, j] + Temp
        Output /= (y.shape[1] / A) / jnp.ceil(A / 2) / (1 + 2 * A - 2 * jnp.ceil(A / 2))
        return Output

    def _r_sum(self, y, w):
        return jnp.sum(y * w, axis=1) / jnp.sum(w)

    def _convex(self, x):
        return jnp.fliplr(jnp.cumprod(jnp.hstack((jnp.ones((x.shape[0], 1)), 1 - jnp.cos(x[:, :-1] * jnp.pi / 2))),
                                      axis=1)) * jnp.hstack(
            (jnp.ones((x.shape[0], 1)), 1 - jnp.sin(x[:, x.shape[1] - 2::-1] * jnp.pi / 2)))

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


    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        N, D = X.shape
        M = self.m
        K = M - 1
        L = D - K
        D = 1
        S = jnp.arange(2, 2 * M + 1, 2)
        A = jnp.ones(M - 1)

        z01 = X / jnp.arange(2, X.shape[1] * 2 + 1, 2)

        t1 = jnp.zeros((N, K + L))
        Y = (jnp.flip(jnp.cumsum(jnp.flip(z01, 1), 1), 1) - z01) / jnp.arange(K + L - 1, -1, -1)
        t1 = t1.at[:, :K + L - 1].set(z01[:, :K + L - 1] ** (0.02 + (50 - 0.02) * (
                0.98 / 49.98 - (1 - 2 * Y[:, :K + L - 1]) * jnp.abs(
            jnp.floor(0.5 - Y[:, :K + L - 1]) + 0.98 / 49.98))))
        t1 = t1.at[:, -1].set(z01[:, -1])

        t2 = jnp.zeros((N, K + L))
        t2 = t2.at[:, :K].set(self._s_decept(t1[:, :K], 0.35, 0.001, 0.05))
        t2 = t2.at[:, K:].set(self._s_multi(t1[:, K:], 30, 95, 0.35))

        t3 = jnp.zeros((N, M))
        for i in range(1, M):
            t3 = t3.at[:, i - 1].set(self._r_nonsep(t2[:, int((i - 1) * K / (M - 1)): int(i * K / (M - 1))], int(K / (M - 1))))

        SUM = jnp.zeros(N)
        for i in range(K, K + L - 1):
            for j in range(i+1, K + L):
                SUM += jnp.abs(t2[:, i] - t2[:, j])

        t3 = t3.at[:, M - 1].set(
            (jnp.sum(t2[:, K:], axis=1) + SUM * 2) / jnp.ceil(L / 2) / (1 + 2 * L - 2 * jnp.ceil(L / 2)))

        x = jnp.zeros((N, M))
        for i in range(M - 1):
            x = x.at[:, i].set(jnp.maximum(t3[:, M - 1], A[i]) * (t3[:, i] - 0.5) + 0.5)

        x = x.at[:, M - 1].set(t3[:, M - 1])

        h = self._concave(x)
        f = D * x[:, M - 1].reshape(-1, 1) + S * h
        return f, state

    def pf(self, state: chex.PyTreeDef):
        M = self.m
        N = self.ref_num * self.m
        # N = 1000
        R = UniformSampling(N, M)()[0]
        R = R / jnp.sqrt(jnp.sum(R ** 2, axis=1)).reshape(-1, 1)
        f = jnp.arange(2, 2 * M + 1, 2) * R
        return f, state

    def _s_decept(self, y, A, B, C):
        return 1 + (jnp.abs(y - A) - B) * (
                jnp.floor(y - A + B) * (1 - C + (A - B) / B) / (A - B) + jnp.floor(A + B - y) * (
                1 - C + (1 - A - B) / B) / (1 - A - B) + 1 / B)

    def _s_multi(self, y, A, B, C):
        return (1 + jnp.cos((4 * A + 2) * jnp.pi * (0.5 - jnp.abs(y - C) / 2 / (jnp.floor(C - y) + C))) + 4 * B * (
                jnp.abs(y - C) / 2 / (jnp.floor(C - y) + C)) ** 2) / (B + 2)

    def _r_nonsep(self, y, A):
        Output = jnp.zeros(y.shape[0])
        for j in range(y.shape[1]):
            Temp = jnp.zeros(y.shape[0])
            for k in range(A - 1):
                Temp += jnp.abs(y[:, j] - y[:, (j + 1 + k) % y.shape[1]])
            Output += y[:, j] + Temp
        return Output / (y.shape[1] / A) / jnp.ceil(A / 2) / (1 + 2 * A - 2 * jnp.ceil(A / 2))

    def _concave(self, x):
        return jnp.fliplr(
            jnp.cumprod(jnp.hstack([jnp.ones((x.shape[0], 1)), jnp.sin(x[:, :-1] * jnp.pi / 2)]), axis=1)) * jnp.hstack(
            [jnp.ones((x.shape[0], 1)), jnp.cos(x[:, x.shape[1] - 2::-1] * jnp.pi / 2)])


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

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        N, D = X.shape
        Y = X - 2 * X[:, 1].reshape(-1, 1) * jnp.sin(
            2 * jnp.pi * X[:, 0].reshape(-1, 1) + jnp.arange(1, D + 1) * jnp.pi / D)
        f = jnp.zeros((N, self.m))
        f = f.at[:, 0].set(jnp.sin(X[:, 0] * jnp.pi / 2) + 2 * jnp.mean(Y[:, 3:D:3] ** 2, axis=1))
        f = f.at[:, 1].set(
            jnp.cos(X[:, 0] * jnp.pi / 2) * jnp.sin(X[:, 1] * jnp.pi / 2) + 2 * jnp.mean(Y[:, 4:D:3] ** 2, axis=1))
        f = f.at[:, 2].set(
            jnp.cos(X[:, 0] * jnp.pi / 2) * jnp.cos(X[:, 1] * jnp.pi / 2) + 2 * jnp.mean(Y[:, 2:D:3] ** 2, axis=1))
        f = f.at[:, 3:].set(
            jnp.tile((f[:, 0] ** 2 + f[:, 1] ** 10 + f[:, 2] ** 10 + 2 * jnp.mean(Y[:, 3:D] ** 2, axis=1))[:,None],
                     (1, self.m - 3)))
        return f, state

    def pf(self, state: chex.PyTreeDef):
        M = self.m
        N = 1000
        # N = self.ref_num * self.m
        R = UniformSampling(N, 3)()[0]
        R = R / (jnp.sqrt(jnp.sum(R ** 2, axis=1))[:,None])
        f = jnp.hstack([R, jnp.tile((R[:, 0] ** 2 + R[:, 1] ** 10 + R[:, 2] ** 10)[:,None], (1, M - 3))])
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

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        N, D = X.shape
        M = self.m
        nk = 2
        X = X.at[:, M-1:].set((1 + jnp.arange(M, D + 1) / D) * X[:, M-1:] - (X[:, 0] * 10)[:,None])
        G = jnp.zeros((N, M))
        for i in range(0, M, 2):
            for j in range(nk):
                G = G.at[:, i].set(G[:, i] + self._Rastrigin(
                    X[:, self.len[i] + M - 1 + j * self.sublen[i]:self.len[i] + M - 1 + (j + 1) * self.sublen[i]]))

        for i in range(1, M, 2):
            for j in range(nk):
                G = G.at[:, i].set(G[:, i] + self._Rosenbrock(
                    X[:, self.len[i] + M - 1 + j * self.sublen[i]:self.len[i] + M - 1 + (j + 1) * self.sublen[i]]))

        G /= self.sublen[None, :] * nk
        f = (1 + G) * jnp.fliplr(jnp.cumprod(jnp.hstack([jnp.ones((N, 1)), X[:, :M - 1]]), axis=1)) * jnp.hstack(
            [jnp.ones((N, 1)),  1 - X[:, M - 2::-1]])
        return f, state

    # def pf(self, state: chex.PyTreeDef):
    #     if self.m == 2:
    #         R = self._GetOptimum()
    #         return R, state
    #     elif self.m == 3:
    #         a = jnp.linspace(0, 1, 10).reshape(-1, 1)
    #         R = [a @ a.T, a @ (1 - a).T, (1 - a) @ jnp.ones_like(a).T]
    #         return R, state

    def pf(self, state: chex.PyTreeDef):
        M = self.m
        N = self.ref_num * self.m
        f = UniformSampling(N, M)()[0]
        return f, state

    def _Rastrigin(self, x):
        return jnp.sum(x ** 2 - 10 * jnp.cos(2 * jnp.pi * x) + 10, axis=1)

    def _Rosenbrock(self, x):
        return jnp.sum(100 * (x[:, :-1] ** 2 - x[:, 1:]) ** 2 + (x[:, :-1] - 1) ** 2, axis=1)


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


    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        N, D = X.shape
        M = self.m
        nk = 2
        X = X.at[:, M-1:].set(
            (1 + jnp.cos(jnp.arange(M, D + 1) / D * jnp.pi / 2))[None, :] * X[:, M-1:] - (X[:, 0]* 10)[:,None])
        G = jnp.zeros((N, M))
        for i in range(0, M, 2):
            for j in range(nk):
                G = G.at[:, i].set(G[:, i] + self._Griewank(
                    X[:, self.len[i] + M - 1 + j * self.sublen[i]:self.len[i] + M - 1 + (j + 1) * self.sublen[i]]))

        for i in range(1, M, 2):
            for j in range(nk):
                G = G.at[:, i].set(G[:, i] + self._Sphere(
                    X[:, self.len[i] + M - 1 + j * self.sublen[i]:self.len[i] + M - 1 + (j + 1) * self.sublen[i]]))

        G /= self.sublen.reshape(1, -1) * nk
        f = (1 + G + jnp.hstack([G[:, 1:], jnp.zeros((N, 1))])) * (1 - jnp.fliplr(
            jnp.cumprod(jnp.hstack([jnp.ones((N, 1)), jnp.cos(X[:, :M - 1] * jnp.pi / 2)]), axis=1)) * jnp.hstack(
            [jnp.ones((N, 1)), jnp.sin(X[:, M - 2::-1] * jnp.pi / 2)]))
        return f, state

    # def pf(self, state: chex.PyTreeDef):
    #     if self.m == 2:
    #         R = self._GetOptimum()
    #         return R, state
    #     elif self.m == 3:
    #         a = jnp.linspace(0, jnp.pi / 2, 10).reshape(-1, 1)
    #         R = [1 - jnp.sin(a) * jnp.cos(a.T), 1 - jnp.sin(a) * jnp.sin(a.T), 1 - jnp.cos(a) * jnp.ones_like(a).T]
    #         return R, state

    def pf(self, state: chex.PyTreeDef):
        M = self.m
        N = self.ref_num * self.m
        R = UniformSampling(N, M)()[0]
        R = 1 - R / jnp.sqrt(jnp.sum(R ** 2, axis=1)).reshape(-1, 1)
        return R, state

    def _Griewank(self, x):
        return jnp.sum(x ** 2, axis=1) / 4000 - jnp.prod(jnp.cos(x / jnp.sqrt(jnp.arange(1, x.shape[1] + 1))),
                                                         axis=1) + 1

    def _Sphere(self, x):
        return jnp.sum(x ** 2, axis=1)
