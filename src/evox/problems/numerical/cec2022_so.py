import jax.numpy as jnp
from functools import partial
from evox import Problem, jit_class
from jax import lax, vmap
import pkgutil


# 1. Generate x = ones and evaluate
# X = jnp.zeros((x_num, D))
# F, _ = func.evaluate(state, X)

# 2. Generate x = randnum and evaluate
# X = jax.random.uniform(key, shape=(x_num, D)) * 200 -100
# F, _ = func.evaluate(state, X)

# 3. Generate x = Os and evaluate
# F = func._evaluate(x)


@jit_class
class OperatFunc:
    """Operational functions transform basic problems into test problems"""

    def __init__(self):
        func_num = int(self.__class__.__name__.split("_")[0][1:])
        Os_D = pkgutil.get_data(
            __name__, f"cec2022_input_data/shift_data_{func_num}.txt"
        ).decode()
        M_2D = pkgutil.get_data(
            __name__, f"cec2022_input_data/M_{func_num}_D2.txt"
        ).decode()
        M_10D = pkgutil.get_data(
            __name__, f"cec2022_input_data/M_{func_num}_D10.txt"
        ).decode()
        M_20D = pkgutil.get_data(
            __name__, f"cec2022_input_data/M_{func_num}_D20.txt"
        ).decode()

        if func_num in [9, 10, 11, 12]:
            self.Os_D = jnp.array(
                [list(map(float, line.split())) for line in Os_D.splitlines()]
            )
        else:
            self.Os_D = jnp.array([float(x) for x in Os_D.split()])

        M_2D = jnp.array([list(map(float, line.split())) for line in M_2D.splitlines()])
        M_10D = jnp.array(
            [list(map(float, line.split())) for line in M_10D.splitlines()]
        )
        M_20D = jnp.array(
            [list(map(float, line.split())) for line in M_20D.splitlines()]
        )
        self.M_dict = {2: M_2D, 10: M_10D, 20: M_20D}
        self.func_num = func_num

        if self.func_num in [6, 7, 8]:
            S_10D = pkgutil.get_data(
                __name__,
                f"cec2022_input_data/shuffle_data_{func_num}_D10.txt",
            ).decode()
            S_20D = pkgutil.get_data(
                __name__,
                f"cec2022_input_data/shuffle_data_{func_num}_D20.txt",
            ).decode()
            S_10D = jnp.array([int(x) for x in S_10D.split()])
            S_20D = jnp.array([int(x) for x in S_20D.split()])
            self.S_dict = {10: S_10D, 20: S_20D}

            group_sizes = jnp.round(self.p * 10).astype(int)
            split_points = jnp.cumsum(group_sizes)[:-1]
            ids = jnp.arange(10)
            ids_10 = jnp.split(ids, split_points)

            group_sizes = jnp.round(self.p * 20).astype(int)
            split_points = jnp.cumsum(group_sizes)[:-1]
            ids = jnp.arange(20)
            ids_20 = jnp.split(ids, split_points)

            self.ids = {10: ids_10, 20: ids_20}

    def shift_operat(self, x, Os):
        # Shift the solution to Os
        x_shift = x - Os
        return x_shift

    def rotate_operat(self, x, Mr):
        x_rot = Mr @ x
        return x_rot

    def shuffle_operat(self, x, SS):
        SS_is_None = SS is None
        ones = jnp.ones_like(x, dtype=jnp.int32)
        SS_fill = jnp.cumsum(ones)
        SS_dict = {True: SS_fill, False: SS}
        SS = SS_dict[SS_is_None]

        shuf_ids = SS - 1
        x_shuf = x[shuf_ids]
        return x_shuf

    def split_operat(self, x, p):
        # Group x according to the proportion of p
        # p_is_None = p is None
        group_sizes = jnp.round(p * x.size).astype(int)
        split_points = jnp.cumsum(group_sizes)[:-1]
        x_split = jnp.split(x, split_points)
        return x_split

    def ssr_operat(self, x, Os, Mr, sh_rate, SS=None):
        # Wraps shift_operat, shuffle_operat, rotate_operat and split_operat
        y = self.shift_operat(x, Os)

        y = y * sh_rate
        # The order of scaling and rotate can be changed
        z = self.rotate_operat(y, Mr)

        z = self.shuffle_operat(z, SS)

        return z

    def compose_operat(self, x, fs, Os_mat, bias, sigma, lamb):
        N = fs.shape[0]
        Os_mat = Os_mat[0:N]
        D = x.shape[0]

        diff_square_sum = jnp.sum((x - Os_mat) ** 2, axis=1)
        term1 = 1 / jnp.sqrt(diff_square_sum)
        term2 = jnp.exp(-0.5 * diff_square_sum / (sigma**2 * D))
        W = term1 * term2
        W_norm = W / jnp.sum(W)

        f = jnp.sum(W_norm * (lamb * fs + bias))

        return f


@jit_class
class BasicFunc:
    """These are the basic functions"""

    def zakharov_func(self, z):
        D = z.shape[0]
        i = jnp.arange(1, D + 1)

        term1 = jnp.sum(z**2)
        term2 = jnp.sum(0.5 * i * z)

        f_value = term1 + term2**2 + term2**4
        return f_value

    def rosenbrock_func(self, z):
        z = z + jnp.ones_like(z)
        term1 = 100 * jnp.sum((z[:-1] ** 2 - z[1:]) ** 2)
        term2 = jnp.sum((1 - z[:-1]) ** 2)

        f_value = term1 + term2
        return f_value

    def schafferF7_func(self, z, y):
        # This function is transferred from the python source code
        nx = z.shape[0]

        def body_func(i, f, z, y):
            z = z.at[i].set(lax.pow(y[i] * y[i] + y[i + 1] * y[i + 1], 0.5))
            tmp = jnp.sin(50.0 * lax.pow(z[i], 0.2))
            f += lax.pow(z[i], 0.5) + lax.pow(z[i], 0.5) * tmp * tmp
            return f

        f = lax.fori_loop(0, nx - 1, partial(body_func, z=z, y=y), 0)
        f = f * f / (nx - 1) / (nx - 1)
        return f

    def rastrigin_func(self, z):
        z = z * 0.0512
        f_value = jnp.sum(z**2 - 10 * jnp.cos(2 * jnp.pi * z) + 10)
        return f_value

    def levy_func(self, z):
        w = 1 + (z) / 4  # Levy func should be 1 + (z-1) / 4 here
        w1 = w[:-1]
        w_last = w[-1]

        term1 = jnp.sin(jnp.pi * w[0]) ** 2
        term2 = jnp.sum(((w1 - 1) ** 2) * (1 + 10 * (jnp.sin(jnp.pi * w1 + 1)) ** 2))
        term3 = ((w_last - 1) ** 2) * (1 + (jnp.sin(2 * jnp.pi * w_last)) ** 2)

        f_value = term1 + term2 + term3
        return f_value

    def bentCigar_func(self, z):
        D = z.shape[0]
        term1 = z[0] ** 2
        term2 = jnp.sum(z[1:D] ** 2)
        f_value = term1 + 1e6 * term2
        return f_value

    def hgbat_func(self, z):
        D = z.shape[0]
        z = z * 0.05
        z = z - 1
        term1 = jnp.fabs(jnp.sum(z**2) ** 2 - jnp.sum(z) ** 2) ** 0.5
        term2 = (0.5 * jnp.sum(z**2) + jnp.sum(z)) / D

        f_value = term1 + term2 + 0.5
        return f_value

    def katsuura_func(self, z):
        # This function is transferred from the python source code
        nx = z.shape[0]
        f = 1.0
        tmp3 = lax.pow(1.0 * nx, 1.2)

        z = z * 0.05

        for i in range(nx):
            temp = 0.0
            for j in range(1, 33):
                tmp1 = lax.pow(2.0, jnp.float32(j))
                tmp2 = tmp1 * z[i]
                temp += jnp.fabs(tmp2 - jnp.floor(tmp2 + 0.5)) / tmp1
            f *= lax.pow(1.0 + (i + 1) * temp, 10.0 / tmp3)
        tmp1 = 10.0 / nx / nx
        f = f * tmp1 - tmp1

        return f

    def ackley_func(self, z):
        D = z.shape[0]
        a = 20
        b = 0.2
        c = 2 * jnp.pi

        sum_sq_term = -a * jnp.exp(-b * jnp.sqrt(jnp.sum(z**2) / D))
        cos_term = -jnp.exp(jnp.sum(jnp.cos(c * z)) / D)

        return sum_sq_term + cos_term + a + jnp.exp(1)

    def schwefel_func(self, z):
        # This function is transferred from the python source code
        nx = z.shape[0]
        f = 0.0

        z = z * 10

        for i in range(nx):
            z = z.at[i].set(z[i] + 4.209687462275036e002)

            condition1 = z[i] > 500
            condition2 = z[i] < -500

            result = lax.select(condition1, 1, 0)
            result = lax.select(condition2, 2, result)

            def zi_larger_500(zi, f):
                f -= (500.0 - jnp.fmod(zi, 500)) * jnp.sin(
                    lax.pow(500.0 - jnp.fmod(zi, 500), 0.5)
                )
                tmp = (zi - 500.0) / 100
                f += tmp * tmp / nx
                return f

            def zi_smaller_neg500(zi, f):
                f -= (-500.0 + jnp.fmod(jnp.fabs(zi), 500)) * jnp.sin(
                    lax.pow(500.0 - jnp.fmod(jnp.fabs(zi), 500), 0.5)
                )
                tmp = (zi + 500.0) / 100
                f += tmp * tmp / nx
                return f

            def zi_between_500(zi, f):
                f -= zi * jnp.sin(lax.pow(jnp.fabs(zi), 0.5))
                return f

            branch_func = (zi_between_500, zi_larger_500, zi_smaller_neg500)
            f = lax.switch(result, branch_func, z[i], f)
        f += 4.189828872724338e002 * nx
        return f

    def happycat_func(self, z):
        D = z.shape[0]
        z = z * 0.05 - 1
        zsquare_sum = jnp.sum(z**2)
        zsum = jnp.sum(z)

        term1 = lax.pow(jnp.fabs(zsquare_sum - D), 1 / 4)
        term2 = (0.5 * zsquare_sum + zsum) / D

        f_value = term1 + term2 + 0.5
        return f_value

    def elliptic_func(self, z):
        D = z.shape[0]
        i_arr = jnp.arange(D)
        idx_arr = 6 * i_arr / (D - 1)
        pow_arr = 10**idx_arr

        f_value = jnp.sum(pow_arr * (z**2))

        return f_value

    def discus_func(self, z):
        f_value = 1e6 * (z[0] ** 2) + jnp.sum(z[1:] ** 2)

        return f_value

    def expSchaffer(self, z):
        z_roll = jnp.roll(z, 1)
        square_sum = z**2 + z_roll**2

        term1 = (jnp.sin(jnp.sqrt(square_sum))) ** 2 - 0.5
        term2 = (1 + 0.001 * (square_sum)) ** 2

        f_value = jnp.sum(0.5 + term1 / term2)
        return f_value

    def expGrieRosen_func(self, z):
        # This function is transferred from the python source code
        nx = z.shape[0]

        f = 0.0
        z = z * 0.05
        z = z.at[0].set(z[0] + 1)
        for i in range(nx - 1):
            z = z.at[i + 1].set(z[i + 1] + 1)
            tmp1 = z[i] * z[i] - z[i + 1]
            tmp2 = z[i] - 1.0
            temp = 100.0 * tmp1 * tmp1 + tmp2 * tmp2
            f += (temp * temp) / 4000.0 - jnp.cos(temp) + 1.0

        tmp1 = z[nx - 1] * z[nx - 1] - z[0]
        tmp2 = z[nx - 1] - 1.0
        temp = 100.0 * tmp1 * tmp1 + tmp2 * tmp2
        f += (temp * temp) / 4000.0 - jnp.cos(temp) + 1.0
        return f

    def griewank_func(self, z):
        D = z.shape[0]
        term1 = jnp.sum(z**2) / 4000

        i_arr = jnp.arange(D) + 1
        term2 = jnp.prod(jnp.cos(z / jnp.sqrt(i_arr)))

        f_value = term1 - term2 + 1
        return f_value


@jit_class
class F1_CEC2022(OperatFunc, BasicFunc, Problem):
    """Test problems of CEC2022"""

    def __init__(self):
        OperatFunc.__init__(self)

    def _evaluate(self, x):
        D = x.shape[0]
        Os = self.Os_D[0:D]
        M = self.M_dict[D]
        sh_rate = 1

        z = self.ssr_operat(x, Os, M, sh_rate)
        f = self.zakharov_func(z)
        f = lax.select(f < 1e-8, 0.0, f)
        return f

    def evaluate(self, state, X):
        F = vmap(self._evaluate)(X)
        return F, state


@jit_class
class F2_CEC2022(OperatFunc, BasicFunc, Problem):
    """Test problems of CEC2022"""

    def __init__(self):
        OperatFunc.__init__(self)

    def _evaluate(self, x):
        D = x.shape[0]
        Os = self.Os_D[0:D]
        M = self.M_dict[D]
        sh_rate = 2.048 / 100.0

        z = self.ssr_operat(x, Os, M, sh_rate)
        f = self.rosenbrock_func(z)
        f = lax.select(f < 1e-8, 0.0, f)
        return f

    def evaluate(self, state, X):
        F = vmap(self._evaluate)(X)
        return F, state


@jit_class
class F3_CEC2022(OperatFunc, BasicFunc, Problem):
    """Test problems of CEC2022"""

    def __init__(self):
        OperatFunc.__init__(self)

    def _evaluate(self, x):
        D = x.shape[0]
        Os = self.Os_D[0:D]
        M = self.M_dict[D]
        sh_rate = 1

        y = self.shift_operat(x, Os)
        z = self.ssr_operat(x, Os, M, sh_rate)
        f = self.schafferF7_func(z, y)
        f = lax.select(f < 1e-8, 0.0, f)
        return f

    def evaluate(self, state, X):
        F = vmap(self._evaluate)(X)
        return F, state


@jit_class
class F4_CEC2022(OperatFunc, BasicFunc, Problem):
    """Test problems of CEC2022"""

    def __init__(self):
        OperatFunc.__init__(self)

    def _evaluate(self, x):
        D = x.shape[0]
        Os = self.Os_D[0:D]
        M = self.M_dict[D]
        sh_rate = 1

        z = self.ssr_operat(x, Os, M, sh_rate)
        f = self.rastrigin_func(z)
        f = lax.select(f < 1e-8, 0.0, f)
        return f

    def evaluate(self, state, X):
        F = vmap(self._evaluate)(X)
        return F, state


@jit_class
class F5_CEC2022(OperatFunc, BasicFunc, Problem):
    """Test problems of CEC2022"""

    def __init__(self):
        OperatFunc.__init__(self)

    def _evaluate(self, x):
        D = x.shape[0]
        Os = self.Os_D[0:D]
        M = self.M_dict[D]
        sh_rate = 1

        z = self.ssr_operat(x, Os, M, sh_rate)
        f = self.levy_func(z)
        f = lax.select(f < 1e-8, 0.0, f)
        return f

    def evaluate(self, state, X):
        F = vmap(self._evaluate)(X)
        return F, state


@jit_class
class F6_CEC2022(OperatFunc, BasicFunc, Problem):
    """Test problems of CEC2022"""

    def __init__(self):
        self.p = jnp.array([0.4, 0.4, 0.2])
        OperatFunc.__init__(self)

    def _evaluate(self, x):
        D = x.shape[0]
        Os = self.Os_D[0:D]
        M = self.M_dict[D]
        sh_rate = 1
        S = self.S_dict[D]
        ids = self.ids[D]

        z = self.ssr_operat(x, Os, M, sh_rate, S)

        f1 = self.bentCigar_func(z[ids[0]])
        f2 = self.hgbat_func(z[ids[1]])
        f3 = self.rastrigin_func(z[ids[2]])

        f = f1 + f2 + f3
        # f may be smaller than 0 (round error)
        f = lax.select(f < 1e-8, 0.0, f)
        return f

    def evaluate(self, state, X):
        F = vmap(self._evaluate)(X)
        return F, state


@jit_class
class F7_CEC2022(OperatFunc, BasicFunc, Problem):
    """Test problems of CEC2022"""

    def __init__(self):
        self.p = jnp.array([0.1, 0.2, 0.2, 0.2, 0.1, 0.2])
        OperatFunc.__init__(self)

    def _evaluate(self, x):
        D = x.shape[0]
        Os = self.Os_D[0:D]
        M = self.M_dict[D]
        sh_rate = 1
        S = self.S_dict[D]
        ids = self.ids[D]

        z = self.ssr_operat(x, Os, M, sh_rate, S)

        y_len = len(ids[5])
        y = lax.dynamic_slice_in_dim(z, 0, y_len)

        f1 = self.hgbat_func(z[ids[0]])
        f2 = self.katsuura_func(z[ids[1]])
        f3 = self.ackley_func(z[ids[2]])
        f4 = self.rastrigin_func(z[ids[3]])
        f5 = self.schwefel_func(z[ids[4]])
        f6 = self.schafferF7_func(z[ids[5]], y)

        f = f1 + f2 + f3 + f4 + f5 + f6
        f = lax.select(f < 1e-8, 0.0, f)
        return f

    def evaluate(self, state, X):
        F = vmap(self._evaluate)(X)
        return F, state


@jit_class
class F8_CEC2022(OperatFunc, BasicFunc, Problem):
    """Test problems of CEC2022"""

    def __init__(self):
        self.p = jnp.array([0.3, 0.2, 0.2, 0.1, 0.2])
        OperatFunc.__init__(self)

    def _evaluate(self, x):
        D = x.shape[0]
        Os = self.Os_D[0:D]
        M = self.M_dict[D]
        sh_rate = 1
        S = self.S_dict[D]
        ids = self.ids[D]

        z = self.ssr_operat(x, Os, M, sh_rate, S)

        f1 = self.katsuura_func(z[ids[0]])
        f2 = self.happycat_func(z[ids[1]])
        f3 = self.expGrieRosen_func(z[ids[2]])
        f4 = self.schwefel_func(z[ids[3]])
        f5 = self.ackley_func(z[ids[4]])

        f = f1 + f2 + f3 + f4 + f5
        f = lax.select(f < 1e-8, 0.0, f)
        return f
        # compt_err = {10: 6.1035156e-5, 20: 1.2207031e-4}
        # return f - compt_err[D]

    def evaluate(self, state, X):
        F = vmap(self._evaluate)(X)
        return F, state


@jit_class
class F9_CEC2022(OperatFunc, BasicFunc, Problem):
    """Test problems of CEC2022"""

    def __init__(self):
        OperatFunc.__init__(self)
        self.bias = jnp.array([0, 200, 300, 100, 400])
        self.lamb = jnp.array([1, 1e-6, 1e-26, 1e-6, 1e-6])
        self.sigma = jnp.array([10, 20, 30, 40, 50])

    def _evaluate(self, x):
        D = x.shape[0]
        Os_mat = self.Os_D[:, 0:D]
        Os = jnp.ravel(Os_mat)
        M = self.M_dict[D]
        sh_rate = 1

        z1 = self.ssr_operat(x, Os[0:D], M[0:D], 2.048 / 100.0)
        f1 = self.rosenbrock_func(z1)

        z2 = self.ssr_operat(x, Os[D : 2 * D], M[D : 2 * D], sh_rate)
        f2 = self.elliptic_func(z2)

        z3 = self.ssr_operat(x, Os[2 * D : 3 * D], M[2 * D : 3 * D], sh_rate)
        f3 = self.bentCigar_func(z3)

        z4 = self.ssr_operat(x, Os[3 * D : 4 * D], M[3 * D : 4 * D], sh_rate)
        f4 = self.discus_func(z4)

        z5 = x - Os[4 * D : 5 * D]
        f5 = self.elliptic_func(z5)

        fs = jnp.array([f1, f2, f3, f4, f5])

        f = self.compose_operat(x, fs, Os_mat, self.bias, self.sigma, self.lamb)
        f = lax.select(f < 1e-8, 0.0, f)
        return f

    def evaluate(self, state, X):
        F = vmap(self._evaluate)(X)
        return F, state


@jit_class
class F10_CEC2022(OperatFunc, BasicFunc, Problem):
    """Test problems of CEC2022"""

    def __init__(self):
        OperatFunc.__init__(self)
        self.bias = jnp.array([0, 200, 100])
        self.lamb = jnp.array([1, 1, 1])
        self.sigma = jnp.array([20, 10, 10])

    def _evaluate(self, x):
        D = x.shape[0]
        Os_mat = self.Os_D[:, 0:D]
        Os = jnp.ravel(Os_mat)
        M = self.M_dict[D]
        sh_rate = 1

        z1 = x - Os[0:D]
        f1 = self.schwefel_func(z1)

        z2 = self.ssr_operat(x, Os[D : 2 * D], M[D : 2 * D], sh_rate)
        f2 = self.rastrigin_func(z2)

        z3 = self.ssr_operat(x, Os[2 * D : 3 * D], M[2 * D : 3 * D], sh_rate)
        f3 = self.hgbat_func(z3)

        fs = jnp.array([f1, f2, f3])

        f = self.compose_operat(x, fs, Os_mat, self.bias, self.sigma, self.lamb)
        f = lax.select(f < 1e-8, 0.0, f)
        return f

    def evaluate(self, state, X):
        F = vmap(self._evaluate)(X)
        return F, state


@jit_class
class F11_CEC2022(OperatFunc, BasicFunc, Problem):
    """Test problems of CEC2022"""

    def __init__(self):
        OperatFunc.__init__(self)
        self.bias = jnp.array([0, 200, 300, 400, 200])
        self.lamb = jnp.array([5e-4, 1, 10, 1, 10])
        self.sigma = jnp.array([20, 20, 30, 30, 20])

    def _evaluate(self, x):
        D = x.shape[0]
        Os_mat = self.Os_D[:, 0:D]
        Os = jnp.ravel(Os_mat)
        M = self.M_dict[D]
        sh_rate = 1

        z1 = self.ssr_operat(x, Os[0:D], M[0:D], sh_rate)
        f1 = self.expSchaffer(z1)

        z2 = self.ssr_operat(x, Os[D : 2 * D], M[D : 2 * D], sh_rate)
        f2 = self.schwefel_func(z2)

        z3 = self.ssr_operat(x, Os[2 * D : 3 * D], M[2 * D : 3 * D], 6)
        f3 = self.griewank_func(z3)

        z4 = self.ssr_operat(x, Os[3 * D : 4 * D], M[3 * D : 4 * D], 2.048 / 100.0)
        f4 = self.rosenbrock_func(z4)

        z5 = self.ssr_operat(x, Os[4 * D : 5 * D], M[4 * D : 5 * D], sh_rate)
        f5 = self.rastrigin_func(z5)

        fs = jnp.array([f1, f2, f3, f4, f5])

        f = self.compose_operat(x, fs, Os_mat, self.bias, self.sigma, self.lamb)

        error_10D = 5.07e-6
        error_20D = 1.46e-5
        error = lax.select(D == 10, error_10D, error_20D)
        f = lax.select(f < error, 0.0, f)
        return f

    def evaluate(self, state, X):
        F = vmap(self._evaluate)(X)
        return F, state


@jit_class
class F12_CEC2022(OperatFunc, BasicFunc, Problem):
    """Test problems of CEC2022"""

    def __init__(self):
        OperatFunc.__init__(self)
        self.bias = jnp.array([0, 300, 500, 100, 400, 200])
        self.lamb = jnp.array([10, 10, 2.5, 1e-26, 1e-6, 5e-4])
        self.sigma = jnp.array([10, 20, 30, 40, 50, 60])

    def _evaluate(self, x):
        D = x.shape[0]
        Os_mat = self.Os_D[:, 0:D]
        Os = jnp.ravel(Os_mat)
        M = self.M_dict[D]
        sh_rate = 1

        z1 = self.ssr_operat(x, Os[0:D], M[0:D], sh_rate)
        f1 = self.hgbat_func(z1)

        z2 = self.ssr_operat(x, Os[D : 2 * D], M[D : 2 * D], sh_rate)
        f2 = self.rastrigin_func(z2)

        z3 = self.ssr_operat(x, Os[2 * D : 3 * D], M[2 * D : 3 * D], sh_rate)
        f3 = self.schwefel_func(z3)

        z4 = self.ssr_operat(x, Os[3 * D : 4 * D], M[3 * D : 4 * D], sh_rate)
        f4 = self.bentCigar_func(z4)

        z5 = self.ssr_operat(x, Os[4 * D : 5 * D], M[4 * D : 5 * D], sh_rate)
        f5 = self.elliptic_func(z5)

        z6 = self.ssr_operat(x, Os[4 * D : 5 * D], M[4 * D : 5 * D], sh_rate)
        f6 = self.expSchaffer(z6)

        fs = jnp.array([f1, f2, f3, f4, f5, f6])

        f = self.compose_operat(x, fs, Os_mat, self.bias, self.sigma, self.lamb)
        f = lax.select(f < 1e-8, 0.0, f)
        return f

    def evaluate(self, state, X):
        F = vmap(self._evaluate)(X)
        return F, state


class CEC2022TestSuit:
    """
    Instantiation format: problem_instance = CEC2022.create(1)
    i.e., problem_instance = F1_CEC2022()
    """

    func_num2class = {
        1: F1_CEC2022,
        2: F2_CEC2022,
        3: F3_CEC2022,
        4: F4_CEC2022,
        5: F5_CEC2022,
        6: F6_CEC2022,
        7: F7_CEC2022,
        8: F8_CEC2022,
        9: F9_CEC2022,
        10: F10_CEC2022,
        11: F11_CEC2022,
        12: F12_CEC2022,
    }

    @staticmethod
    def create(func_num: int):
        return CEC2022TestSuit.func_num2class[func_num]()
