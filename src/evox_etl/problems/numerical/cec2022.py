__all__ = [
    "CEC2022",
]


import dataclasses
from math import ceil, pi
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import etl
import etl.numpy as enp

from evox_etl.problems.numerical.basic import (
    ackley_func,
    griewank_func,
    rastrigin_func,
    rosenbrock_func,
    zakharov_func,
)


# Data files shipped with the torch evox package (read-only reference).
# numpy is used ONLY to load this input data; never inside graph code.
DATA_DIR = Path(__file__).resolve().parents[4] / "src" / "evox" / "problems" / "numerical" / "cec2022_input_data"

# Memoized parsed input data (plain numpy arrays, host-side), keyed by (func_num, dim).
_DATA_CACHE: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = {}


def _read_floats(filename: Path) -> List[float]:
    if not filename.is_file():
        raise FileNotFoundError(f"Cannot open {filename} for reading")
    with open(filename, "r") as f:
        return [float(num) for num in f.read().split()]


def _load_data(func_num: int, dim: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load rotation matrix M, shift vector OShift and (for funcs 6-8) shuffle index SS as numpy arrays."""
    key = (func_num, dim)
    if key in _DATA_CACHE:
        return _DATA_CACHE[key]

    # Loading rotation matrix M
    m_filename = DATA_DIR / f"M_{func_num}_D{dim}.txt"
    M_data = _read_floats(m_filename)
    if func_num < 9:
        M = np.array(M_data).reshape(dim, dim).T
    else:
        M = np.array(M_data).reshape(-1, dim).T

    # Loading shift matrix OShift
    shift_filename = DATA_DIR / f"shift_data_{func_num}.txt"
    shift_data = _read_floats(shift_filename)
    if func_num < 9:
        OShift = np.array(shift_data).reshape(1, -1)
    else:
        OShift = np.array(shift_data).reshape(10, -1)[:9, :dim].reshape(1, 9 * dim)

    # Loading shuffle index SS
    SS: Optional[np.ndarray] = None
    if 6 <= func_num <= 8:
        shuffle_filename = DATA_DIR / f"shuffle_data_{func_num}_D{dim}.txt"
        if not shuffle_filename.is_file():
            raise FileNotFoundError(f"Cannot open {shuffle_filename} for reading")
        with open(shuffle_filename, "r") as f:
            shuffle_data = [int(num) for num in f.read().split()]
        # To 0-based index
        SS = np.array(shuffle_data) - 1

    result = (M, OShift, SS)
    _DATA_CACHE[key] = result
    return result


# Transform functions


def _slice_cols(x: etl.SymbolicTensor, start: int, stop: int) -> etl.SymbolicTensor:
    """Gather-based static column slice ``x[:, start:stop]``.

    The IR slice op cannot express a full-axis ``:`` over a dynamic (batch)
    dim, so ``x[:, a:b]`` is done with a constant int32 index array instead.
    """
    return etl.gather(x, etl.constant(etl.tensor(np.arange(start, stop, dtype=np.int32))), axis=1)


def shift(x: etl.SymbolicTensor, offset: etl.SymbolicTensor) -> etl.SymbolicTensor:
    """Shift the input vector."""
    return x - offset[:, : x.shape[1]]


def rotate(x: etl.SymbolicTensor, M: etl.SymbolicTensor) -> etl.SymbolicTensor:
    """Rotate the input vector."""
    return etl.matmul(x, M[: x.shape[1], :])


def cut(
    x: etl.SymbolicTensor,
    Gp: List[float],
    sh_flag: bool,
    rot_flag: bool,
    offset: etl.SymbolicTensor,
    M: etl.SymbolicTensor,
    SS: Optional[etl.SymbolicTensor],
) -> List[etl.SymbolicTensor]:
    nx = x.shape[1]
    G_nx = [ceil(g * nx) for g in Gp]
    G_nx[-1] = nx - sum(G_nx[:-1])
    G = [0] * len(G_nx)
    for i in range(1, len(Gp)):
        G[i] = G[i - 1] + G_nx[i - 1]

    y = sr_func_rate(x, sh_rate=1.0, sh_flag=sh_flag, rot_flag=rot_flag, offset=offset, M=M)
    z = etl.gather(y, SS[:nx], axis=1) if SS is not None else y

    z_piece = []
    for i in range(len(Gp)):
        z_piece.append(_slice_cols(z, G[i], G[i] + G_nx[i]))
    return z_piece


def sr_func_rate(
    x: etl.SymbolicTensor,
    sh_rate: float,
    sh_flag: bool,
    rot_flag: bool,
    offset: etl.SymbolicTensor,
    M: etl.SymbolicTensor,
) -> etl.SymbolicTensor:
    """Shift and rotate function with rate."""
    if sh_flag:
        if rot_flag:
            y = shift(x, offset) * sh_rate
            z = rotate(y, M)
        else:
            z = shift(x, offset) * sh_rate
    else:
        if rot_flag:
            y = x * sh_rate
            z = rotate(y, M)
        else:
            z = x * sh_rate
    return z


def cf_cal(
    x: etl.SymbolicTensor,
    fit: List[etl.SymbolicTensor],
    delta: List[int],
    bias: List[int],
    OShift: etl.SymbolicTensor,
) -> etl.SymbolicTensor:
    nx = x.shape[1]
    w_all = []
    # x * 0.0: zero init of shape (n,); direct leaf reductions hit an etl
    # Dim-vs-None shape mismatch, arithmetic outputs carry the None form.
    w_sum = enp.sum(x * 0.0, axis=1)
    for i, (d, f, b) in enumerate(zip(delta, fit, bias)):
        diff = x - OShift[:, i * nx : (i + 1) * nx]
        w = enp.sum(diff**2, axis=1)
        w = enp.where(etl.not_equal(w, 0), (1 / enp.sqrt(w)) * enp.exp(-w / (2 * nx * d * d)), float("inf"))
        w_sum = w_sum + w
        w_all.append(w * (f + b))
    w_ret = enp.sum(x * 0.0, axis=1)
    w_sum = enp.where(w_sum == 0, 1e-9, w_sum)
    for w in w_all:
        w_ret = w_ret + w / w_sum
    return w_ret


# cSpell:words Zakharov Rosenbrock Schaffer Rastrigin hgbat katsuura ackley schwefel happycat grie_rosen ellips escaffer griewank

# Problem


def cec2022_f1(x: etl.SymbolicTensor, OShift: etl.SymbolicTensor, M: etl.SymbolicTensor) -> etl.SymbolicTensor:
    """Zakharov Function"""
    return zakharov_func(sr_func_rate(x, 1.0, True, True, OShift, M)) + 300.0


def cec2022_f2(x: etl.SymbolicTensor, OShift: etl.SymbolicTensor, M: etl.SymbolicTensor) -> etl.SymbolicTensor:
    """Rosenbrock Function"""
    return rosenbrock_func(1 + sr_func_rate(x, 2.048e-2, True, True, OShift, M)) + 400


def cec2022_f3(x: etl.SymbolicTensor, OShift: etl.SymbolicTensor, M: etl.SymbolicTensor) -> etl.SymbolicTensor:
    """Schaffer F7 Function"""
    return schaffer_F7_func(sr_func_rate(x, 1.0, True, True, OShift, M)) + 600.0


def cec2022_f4(x: etl.SymbolicTensor, OShift: etl.SymbolicTensor, M: etl.SymbolicTensor) -> etl.SymbolicTensor:
    """Step Rastrigin Function (Noncontinuous Rastrigin's)"""
    return rastrigin_func(sr_func_rate(x, 5.12e-2, True, True, OShift, M)) + 800.0


def cec2022_f5(x: etl.SymbolicTensor, OShift: etl.SymbolicTensor, M: etl.SymbolicTensor) -> etl.SymbolicTensor:
    """Levy Function"""
    return levy_func(sr_func_rate(x, 1.0, True, True, OShift, M)) + 900.0


def cec2022_f6(
    x: etl.SymbolicTensor, OShift: etl.SymbolicTensor, M: etl.SymbolicTensor, SS: etl.SymbolicTensor
) -> etl.SymbolicTensor:
    """Hybrid Function 2"""
    # cf_num = 3
    Gp = [0.4, 0.4, 0.2]
    y = cut(x, Gp, True, True, OShift, M, SS)

    fit0 = bent_cigar_func(sr_func_rate(y[0], 1.0, False, False, OShift, M))
    fit1 = hgbat_func(sr_func_rate(y[1], 5.00e-2, False, False, OShift, M))
    fit2 = rastrigin_func(sr_func_rate(y[2], 5.12e-2, False, False, OShift, M))

    return fit0 + fit1 + fit2 + 1800.0


def cec2022_f7(
    x: etl.SymbolicTensor, OShift: etl.SymbolicTensor, M: etl.SymbolicTensor, SS: etl.SymbolicTensor
) -> etl.SymbolicTensor:
    """Hybrid Function 10"""
    # cf_num_ = 6
    Gp = [0.1, 0.2, 0.2, 0.2, 0.1, 0.2]
    y = cut(x, Gp, True, True, OShift, M, SS)

    fit0 = hgbat_func(sr_func_rate(y[0], 5.00e-2, False, False, OShift, M))
    fit1 = katsuura_func(sr_func_rate(y[1], 5.00e-2, False, False, OShift, M))
    fit2 = ackley_func(20.0, 0.2, 2 * pi, sr_func_rate(y[2], 1.0, False, False, OShift, M))
    fit3 = rastrigin_func(sr_func_rate(y[3], 5.12e-2, False, False, OShift, M))
    fit4 = modified_schwefel_func(sr_func_rate(y[4], 10.0, False, False, OShift, M))
    fit5 = schaffer_F7_func(sr_func_rate(y[5], 1.0, False, False, OShift, M))

    return fit0 + fit1 + fit2 + fit3 + fit4 + fit5 + 2000.0


def cec2022_f8(
    x: etl.SymbolicTensor, OShift: etl.SymbolicTensor, M: etl.SymbolicTensor, SS: etl.SymbolicTensor
) -> etl.SymbolicTensor:
    """Hybrid Function 6"""
    # cf_num_ = 5
    Gp = [0.3, 0.2, 0.2, 0.1, 0.2]
    y = cut(x, Gp, True, True, OShift, M, SS)

    fit0 = katsuura_func(sr_func_rate(y[0], 5.00e-2, False, False, OShift, M))
    fit1 = happycat_func(sr_func_rate(y[1], 5.00e-2, False, False, OShift, M))
    fit2 = grie_rosen_func(sr_func_rate(y[2], 5.00e-2, False, False, OShift, M))
    fit3 = modified_schwefel_func(sr_func_rate(y[3], 10.0, False, False, OShift, M))
    fit4 = ackley_func(20.0, 0.2, 2 * pi, sr_func_rate(y[4], 1.0, False, False, OShift, M))

    return fit0 + fit1 + fit2 + fit3 + fit4 + 2200.0


def cec2022_f9(x: etl.SymbolicTensor, OShift: etl.SymbolicTensor, M: etl.SymbolicTensor) -> etl.SymbolicTensor:
    """Composition Function 1"""
    nx = x.shape[1]
    delta = [10, 20, 30, 40, 50]
    bias = [0, 200, 300, 100, 400]
    fit = [
        rosenbrock_func(
            1 + sr_func_rate(x, 2.048e-2, True, True, OShift[:, 0 * nx : 1 * nx], M[:, 0 * nx : 1 * nx])
        )
        * 10000
        / 1e4,
        ellips_func(sr_func_rate(x, 1.0, True, True, OShift[:, 1 * nx : 2 * nx], M[:, 1 * nx : 2 * nx]))
        * 10000
        / 1e10,
        bent_cigar_func(
            sr_func_rate(x, 1.0, True, True, OShift[:, 2 * nx : 3 * nx], M[:, 2 * nx : 3 * nx])
        )
        * 10000
        / 1e10
        / 1e10
        / 1e10,
        # if divide by 1e30 , cause NVRTC compilation error(https://github.com/pytorch/pytorch/issues/62962)
        discus_func(sr_func_rate(x, 1.0, True, True, OShift[:, 3 * nx : 4 * nx], M[:, 3 * nx : 4 * nx]))
        * 10000
        / 1e10,
        ellips_func(
            sr_func_rate(x, 1.0, True, False, OShift[:, 4 * nx : 5 * nx], M[:, 4 * nx : 5 * nx])
        )
        * 10000
        / 1e10,
    ]
    f = cf_cal(x, fit, delta, bias, OShift)
    return f + 2300.0


def cec2022_f10(x: etl.SymbolicTensor, OShift: etl.SymbolicTensor, M: etl.SymbolicTensor) -> etl.SymbolicTensor:
    """Composition Function 2"""
    nx = x.shape[1]
    delta = [20, 10, 10]
    bias = [0, 200, 100]
    fit = [
        modified_schwefel_func(
            sr_func_rate(x, 10.0, True, False, OShift[:, 0 * nx : 1 * nx], M[:, 0 * nx : 1 * nx])
        )
        * 1.0,
        rastrigin_func(
            sr_func_rate(x, 5.12e-2, True, True, OShift[:, 1 * nx : 2 * nx], M[:, 1 * nx : 2 * nx])
        )
        * 1.0,
        hgbat_func(
            sr_func_rate(x, 5.00e-2, True, True, OShift[:, 2 * nx : 3 * nx], M[:, 2 * nx : 3 * nx])
        )
        * 1.0,
    ]
    f = cf_cal(x, fit, delta, bias, OShift)
    return f + 2400.0


def cec2022_f11(x: etl.SymbolicTensor, OShift: etl.SymbolicTensor, M: etl.SymbolicTensor) -> etl.SymbolicTensor:
    """Composition Function 6"""
    nx = x.shape[1]
    delta = [20, 20, 30, 30, 20]
    bias = [0, 200, 300, 400, 200]
    fit = [
        escaffer6_func(
            sr_func_rate(x, 1.0, True, True, OShift[:, 0 * nx : 1 * nx], M[:, 0 * nx : 1 * nx])
        )
        * 10000
        / 2e7,
        modified_schwefel_func(
            sr_func_rate(x, 10.0, True, True, OShift[:, 1 * nx : 2 * nx], M[:, 1 * nx : 2 * nx])
        )
        * 1.0,
        griewank_func(sr_func_rate(x, 6.0, True, True, OShift[:, 2 * nx : 3 * nx], M[:, 2 * nx : 3 * nx]))
        * 1000
        / 100,
        rosenbrock_func(
            1 + sr_func_rate(x, 2.048e-2, True, True, OShift[:, 3 * nx : 4 * nx], M[:, 3 * nx : 4 * nx])
        )
        * 1.0,
        rastrigin_func(
            sr_func_rate(x, 5.12e-2, True, True, OShift[:, 4 * nx : 5 * nx], M[:, 4 * nx : 5 * nx])
        )
        * 10000
        / 1e3,
    ]
    f = cf_cal(x, fit, delta, bias, OShift)
    return f + 2600.0


def cec2022_f12(x: etl.SymbolicTensor, OShift: etl.SymbolicTensor, M: etl.SymbolicTensor) -> etl.SymbolicTensor:
    """Composition Function 7"""
    nx = x.shape[1]
    delta = [10, 20, 30, 40, 50, 60]
    bias = [0, 300, 500, 100, 400, 200]
    fit = [
        hgbat_func(
            sr_func_rate(x, 5.00e-2, True, True, OShift[:, 0 * nx : 1 * nx], M[:, 0 * nx : 1 * nx])
        )
        * 10000
        / 1000,
        rastrigin_func(
            sr_func_rate(x, 5.12e-2, True, True, OShift[:, 1 * nx : 2 * nx], M[:, 1 * nx : 2 * nx])
        )
        * 10000
        / 1e3,
        modified_schwefel_func(
            sr_func_rate(x, 10.0, True, True, OShift[:, 2 * nx : 3 * nx], M[:, 2 * nx : 3 * nx])
        )
        * 10000
        / 4e3,
        bent_cigar_func(
            sr_func_rate(x, 1.0, True, True, OShift[:, 3 * nx : 4 * nx], M[:, 3 * nx : 4 * nx])
        )
        * 10000
        / 1e10
        / 1e10
        / 1e10,
        # if divide by 1e30 , cause NVRTC compilation error(https://github.com/pytorch/pytorch/issues/62962)
        ellips_func(sr_func_rate(x, 1.0, True, True, OShift[:, 4 * nx : 5 * nx], M[:, 4 * nx : 5 * nx]))
        * 10000
        / 1e10,
        escaffer6_func(
            sr_func_rate(x, 1.0, True, True, OShift[:, 5 * nx : 6 * nx], M[:, 5 * nx : 6 * nx])
        )
        * 10000
        / 2e7,
    ]
    f = cf_cal(x, fit, delta, bias, OShift)
    return f + 2700.0


# Basic functions


def levy_func(x: etl.SymbolicTensor) -> etl.SymbolicTensor:
    """Problem number = 5."""
    w = 1.0 + x / 4.0
    tmp1 = enp.sin(pi * w[:, 0]) ** 2
    tmp2 = (w[:, -1] - 1) ** 2 * (1 + enp.sin(2 * pi * w[:, -1]) ** 2)
    wm1 = _slice_cols(w, 0, w.shape[1] - 1)
    sum_ = (wm1 - 1) ** 2 * (1 + 10 * enp.sin(pi * wm1 + 1) ** 2)
    return tmp1 + enp.sum(sum_, axis=1) + tmp2


def bent_cigar_func(x: etl.SymbolicTensor) -> etl.SymbolicTensor:
    return x[:, 0] ** 2 + enp.sum((10.0**6) * _slice_cols(x, 1, x.shape[1]) ** 2, axis=1)


def hgbat_func(x: etl.SymbolicTensor) -> etl.SymbolicTensor:
    alpha = 1.0 / 4.0
    tmp = x - 1
    r2 = enp.sum(tmp**2, axis=1)
    sum_x = enp.sum(tmp, axis=1)
    return enp.abs(r2**2 - sum_x**2) ** (2 * alpha) + (0.5 * r2 + sum_x) / x.shape[1] + 0.5


def katsuura_func(x: etl.SymbolicTensor) -> etl.SymbolicTensor:
    nx = x.shape[1]
    # tmp1 = 2.0 ** arange(1, 33) and the sum over j are unrolled as a static
    # loop: enp.expand_dims cannot carry dynamic batch dims, so the (n, nx, 32)
    # intermediate of the torch version is avoided (math is elementwise identical).
    temp = x * 0.0
    for j in range(1, 33):
        p2 = 2.0**j
        temp = temp + enp.abs(x * p2 - etl.floor(x * p2 + 0.5)) / p2
    tmp3 = enp.arange(1, nx + 1, dtype=x.dtype)
    f = enp.prod((1 + temp * tmp3) ** (10.0 / (nx**1.2)), axis=1)
    return (f - 1) * (10.0 / nx / nx)


def modified_schwefel_func(x: etl.SymbolicTensor) -> etl.SymbolicTensor:
    nx = x.shape[1]
    tmp1 = x + 420.9687462275036
    tmp2 = -tmp1 * enp.sin(enp.sqrt(enp.abs(tmp1)))
    fm = enp.abs(tmp1) - etl.floor(enp.abs(tmp1) / 500.0) * 500.0
    tmp3 = (500.0 - fm) * enp.sin(enp.sqrt(enp.abs(500.0 - fm)))
    tmp5 = enp.where(tmp1 > 500.0, -tmp3 + (tmp1 - 500.0) ** 2 / 10000.0 / nx, tmp2)
    tmp5 = enp.where(tmp1 < -500.0, tmp3 + (tmp1 + 500.0) ** 2 / 10000.0 / nx, tmp5)
    return enp.sum(tmp5, axis=1) + 418.98288727243378 * nx


def schaffer_F7_func(x: etl.SymbolicTensor) -> etl.SymbolicTensor:
    nx = x.shape[1]
    tmp1 = enp.sqrt(_slice_cols(x, 0, nx - 1) ** 2 + _slice_cols(x, 1, nx) ** 2)
    tmp2 = enp.sin(50.0 * (tmp1**0.2))
    f = enp.sqrt(tmp1) * (1 + tmp2 * tmp2)
    f = enp.mean(f, axis=1)
    return f * f


def escaffer6_func(x: etl.SymbolicTensor) -> etl.SymbolicTensor:
    y = etl.concatenate([_slice_cols(x, 1, x.shape[1]), _slice_cols(x, 0, 1)], axis=1)
    tmp1 = enp.sin(enp.sqrt(x**2 + y**2)) ** 2
    tmp2 = 1.0 + 0.001 * (x**2 + y**2)
    return enp.sum(0.5 + (tmp1 - 0.5) / (tmp2**2), axis=1)


def happycat_func(x: etl.SymbolicTensor) -> etl.SymbolicTensor:
    alpha = 1.0 / 8.0
    nx = x.shape[1]
    tmp = x - 1
    r2 = enp.sum(tmp**2, axis=1)
    sum_x = enp.sum(tmp, axis=1)
    return enp.abs(r2 - nx) ** (2 * alpha) + (0.5 * r2 + sum_x) / nx + 0.5


def grie_rosen_func(x: etl.SymbolicTensor) -> etl.SymbolicTensor:
    x = x + 1
    y = etl.concatenate([_slice_cols(x, 1, x.shape[1]), _slice_cols(x, 0, 1)], axis=1)
    tmp = 100.0 * (x**2 - y) ** 2 + (x - 1.0) ** 2
    return enp.sum((tmp**2) / 4000.0 - enp.cos(tmp) + 1.0, axis=1)


def discus_func(x: etl.SymbolicTensor) -> etl.SymbolicTensor:
    return (10.0**6) * x[:, 0] ** 2 + enp.sum(_slice_cols(x, 1, x.shape[1]) ** 2, axis=1)


def ellips_func(x: etl.SymbolicTensor) -> etl.SymbolicTensor:
    nx = x.shape[1]
    idx = enp.arange(nx, dtype=x.dtype)
    powers = 6.0 * idx / (nx - 1)
    return enp.sum((10.0**powers) * x**2, axis=1)


# Config + entry point


@dataclasses.dataclass(frozen=True)
class CEC2022:
    """The CEC 2022 single-objective test suite Problem config."""

    problem_number: int
    dimension: int

    def __post_init__(self) -> None:
        assert self.dimension in [2, 10, 20], f"Test functions are only defined for D=2,10,20, got {self.dimension}."
        assert not (self.problem_number in [6, 7, 8] and self.dimension == 2), (
            f"Function {self.problem_number} is not defined for D=2."
        )


def evaluate(config: CEC2022, problem_state, pop: etl.SymbolicTensor) -> Tuple[etl.SymbolicTensor, object]:
    """Evaluate the configured CEC2022 test function on a population of shape (n, dim)."""
    nx = config.dimension
    assert pop.shape[1] == nx, f"Dimension mismatch! Expect {nx}, got {pop.shape[1]}."
    func_num = config.problem_number

    # Bake the input data ONCE per trace as graph constants.
    M_np, OShift_np, SS_np = _load_data(func_num, nx)
    M = etl.constant(etl.tensor(M_np.astype(np.float32)))
    OShift = etl.constant(etl.tensor(OShift_np.astype(np.float32)))
    SS = etl.constant(etl.tensor(SS_np.astype(np.int32))) if SS_np is not None else None

    if func_num == 1:
        fitness = cec2022_f1(pop, OShift, M)
    elif func_num == 2:
        fitness = cec2022_f2(pop, OShift, M)
    elif func_num == 3:
        fitness = cec2022_f3(pop, OShift, M)
    elif func_num == 4:
        fitness = cec2022_f4(pop, OShift, M)
    elif func_num == 5:
        fitness = cec2022_f5(pop, OShift, M)
    elif func_num == 6:
        fitness = cec2022_f6(pop, OShift, M, SS)
    elif func_num == 7:
        fitness = cec2022_f7(pop, OShift, M, SS)
    elif func_num == 8:
        fitness = cec2022_f8(pop, OShift, M, SS)
    elif func_num == 9:
        fitness = cec2022_f9(pop, OShift, M)
    elif func_num == 10:
        fitness = cec2022_f10(pop, OShift, M)
    elif func_num == 11:
        fitness = cec2022_f11(pop, OShift, M)
    elif func_num == 12:
        fitness = cec2022_f12(pop, OShift, M)
    else:
        raise ValueError(f"Function {func_num} is not defined.")

    return fitness, problem_state
