import os
from math import ceil
from typing import List, Optional

import torch

from ...core import Problem, jit_class


@jit_class
class CEC2022(Problem):
    """The CEC 2022 single-objective test suite Problem"""

    def __init__(self, problem_number: int, dimension: int, device: torch.device | None = None):
        """
        Initialize a single test function instance from the CEC2022 test suite.

        Args:
            problem_number (`int`): The index for the specific test function to be used. Must be ranged from 1 to 12.
            dimension (`int`): The dimensionality of the problem. Must be one of [2, 10, 20].
            device (`torch.device`, optional): The device on which tensors will be allocated. Defaults to None.

        Raises:
            AssertionError: If the dimension is not one of the allowed values or if the function is not defined.
            FileNotFoundError: If the necessary data files for the problem are not found.
        """
        super().__init__()
        self.nx = dimension
        self.func_num = problem_number

        self.OShift: Optional[torch.Tensor] = None
        self.M: Optional[torch.Tensor] = None
        self.SS: Optional[torch.Tensor] = None
        self.sh_flag = True
        self.rot_flag = True

        assert self.nx in [2, 10, 20], f"Test functions are only defined for D=2,10,20, got {self.nx}."
        assert not (self.func_num in [6, 7, 8] and self.nx == 2), f"Function {self.func_num} is not defined for D=2."

        # Loading data preparation
        current_dir = os.path.dirname(__file__)
        data_dir = os.path.join(current_dir, "cec2022_input_data")
        # Loading rotation matrix M
        m_filename = os.path.join(data_dir, f"M_{self.func_num}_D{self.nx}.txt")
        m_filename = m_filename.replace("\\", "/")
        if not os.path.isfile(m_filename):
            raise FileNotFoundError(f"Cannot open {m_filename} for reading")
        with open(m_filename, "r") as f:
            M_data = [float(num) for num in f.read().split()]
        if self.func_num < 9:
            self.M = torch.tensor(M_data, device=device).reshape(self.nx, self.nx)
        else:
            self.M = torch.tensor(M_data, device=device).reshape(-1, self.nx)
        self.M = self.M.t()

        # Loading shift matrix OShift
        shift_filename = os.path.join(data_dir, f"shift_data_{self.func_num}.txt")
        if not os.path.isfile(shift_filename):
            raise FileNotFoundError(f"Cannot open {shift_filename} for reading")
        with open(shift_filename, "r") as f:
            shift_data = [float(num) for num in f.read().split()]
        if self.func_num < 9:
            self.OShift = torch.tensor(shift_data, device=device).unsqueeze(0)
        else:
            self.OShift = torch.tensor(shift_data, device=device).view( 10, -1 )[:9,:self.nx].reshape(9 * self.nx).unsqueeze(0)

        # Loading shuffle index SS
        if 6 <= self.func_num <= 8:
            shuffle_filename = os.path.join(data_dir, f"shuffle_data_{self.func_num}_D{self.nx}.txt")
            if not os.path.isfile(shuffle_filename):
                raise FileNotFoundError(f"Cannot open {shuffle_filename} for reading")
            with open(shuffle_filename, "r") as f:
                shuffle_data = [int(num) for num in f.read().split()]
            # To 0-based index
            self.SS = torch.tensor(shuffle_data, dtype=torch.long, device=device) - 1

    # Transform matrix
    def shift(self, x: torch.Tensor, O: torch.Tensor) -> torch.Tensor:
        """Shift the input vector."""
        return x - O[:, : x.size(1)]

    def rotate(self, x: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """Rotate the input vector."""
        return torch.matmul(x, M[: x.size(1), :] )

    def cut(self, x: torch.Tensor, Gp: List[float], sh_flag: bool, rot_flag: bool, O: torch.Tensor, M: torch.Tensor) -> List[torch.Tensor]:
        nx = x.size(1)
        G_nx = [ceil(g * nx) for g in Gp]
        G_nx[-1] = nx - sum(G_nx[:-1])
        G = [0] * len(G_nx)
        for i in range(1, len(Gp)):
            G[i] = G[i - 1] + G_nx[i - 1]

        y = self.sr_func_rate(x,sh_rate=1.0,sh_flag=sh_flag,rot_flag=rot_flag, O=O, M=M)
        z = y[:, self.SS[:nx]] if self.SS is not None else y

        z_piece = []
        for i in range(len(Gp)):
            z_piece.append(z[:, G[i] : G[i] + G_nx[i]])
        return z_piece

    #def sr_func(self, x: torch.Tensor) -> torch.Tensor:
    #    """Shift and rotate function."""
    #    if self.sh_flag:
    #        if self.rot_flag:
    #            y = self.shift(x)
    #            z = self.rotate(y, self.M)
    #        else:
    #            z = self.shift(x)
    #    else:
    #        if self.rot_flag:
    #            y = x
    #            z = self.rotate(y, self.M)
    #        else:
    #            z = x
    #    return z

    def sr_func_rate(self, x: torch.Tensor, sh_rate: float, sh_flag: bool, rot_flag: bool, O: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """Shift and rotate function with rate."""
        if sh_flag:
            if rot_flag:
                y = self.shift( x, O) * sh_rate
                z = self.rotate(y, M)
            else:
                z = self.shift(x, O) * sh_rate
        else:
            if rot_flag:
                y = x * sh_rate
                z = self.rotate(y, M)
            else:
                z = x * sh_rate
        return z

    def cf_cal(self, x: torch.Tensor, fit: List[torch.Tensor], delta: List[int], bias: List[int]) -> torch.Tensor:
        nx = x.size(1)
        shift = self.OShift
        w_all = []
        w_sum = torch.zeros(x.size(0), device=x.device)
        for i, (d, f, b) in enumerate(zip(delta, fit, bias)):
            diff = x - shift[:, i*nx:(i + 1)*nx]
            w = torch.sum(diff**2, dim=1)
            w = torch.where(w != 0, (1 / torch.sqrt(w)) * torch.exp(-w / (2 * nx * d * d)), torch.inf)
            w_sum += w
            w_all.append(w * (f + b))
        w_ret = torch.zeros(x.size(0), device=x.device)
        w_sum = torch.where( w_sum == 0, 1e-9, w_sum )
        for w in w_all:
            w_ret += w / w_sum
        return w_ret

    # cSpell:words Zakharov Rosenbrock Schaffer Rastrigin hgbat katsuura ackley schwefel happycat grie_rosen ellips escaffer griewank

    # Problem
    def cec2022_f1(self, x: torch.Tensor, sh_flag: bool, rot_flag: bool, O: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """Zakharov Function"""
        return self.zakharov_func(x,sh_flag,rot_flag,O=O,M=M) + 300.0

    def cec2022_f2(self, x: torch.Tensor, sh_flag: bool, rot_flag: bool, O: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """Rosenbrock Function"""
        return self.rosenbrock_func(x,sh_flag,rot_flag,O=O,M=M) + 400

    def cec2022_f3(self, x: torch.Tensor, sh_flag: bool, rot_flag: bool, O: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """Schaffer F7 Function"""
        return self.schaffer_F7_func(x,sh_flag,rot_flag,O=O,M=M) + 600.0

    def cec2022_f4(self, x: torch.Tensor, sh_flag: bool, rot_flag: bool, O: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """Step Rastrigin Function (Noncontinuous Rastrigin's)"""
        return self.step_rastrigin_func(x,sh_flag,rot_flag,O=O,M=M) + 800.0

    def cec2022_f5(self, x: torch.Tensor, sh_flag: bool, rot_flag: bool, O: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """Levy Function"""
        return self.levy_func(x,sh_flag,rot_flag,O=O,M=M) + 900.0

    def cec2022_f6(self, x: torch.Tensor, sh_flag: bool, rot_flag: bool, O: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """Hybrid Function 2"""
        # cf_num = 3
        Gp = [0.4, 0.4, 0.2]
        y = self.cut(x, Gp, sh_flag=sh_flag, rot_flag=rot_flag,O=O,M=M)

        fit0 = self.bent_cigar_func(y[0], False, False,O=O,M=M)
        fit1 = self.hgbat_func(y[1], False, False,O=O,M=M)
        fit2 = self.rastrigin_func(y[2], False, False,O=O,M=M)

        return fit0 + fit1 + fit2 + 1800.0

    def cec2022_f7(self, x: torch.Tensor, sh_flag: bool, rot_flag: bool, O: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """Hybrid Function 10"""
        # cf_num_ = 6
        Gp = [0.1, 0.2, 0.2, 0.2, 0.1, 0.2]
        y = self.cut(x, Gp, sh_flag=sh_flag, rot_flag=rot_flag,O=O,M=M)

        fit0 = self.hgbat_func(y[0], False, False,O=O,M=M)
        fit1 = self.katsuura_func(y[1], False, False,O=O,M=M)
        fit2 = self.ackley_func(y[2], False, False,O=O,M=M)
        fit3 = self.rastrigin_func(y[3], False, False,O=O,M=M)
        fit4 = self.schwefel_func(y[4], False, False,O=O,M=M)
        fit5 = self.schaffer_F7_func(y[5], False, False,O=O,M=M)
        
        return fit0 + fit1 + fit2 + fit3 + fit4 + fit5 + 2000.0

    def cec2022_f8(self, x: torch.Tensor, sh_flag: bool, rot_flag: bool, O: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """Hybrid Function 6"""
        # cf_num_ = 5
        Gp = [0.3, 0.2, 0.2, 0.1, 0.2]
        y = self.cut(x, Gp, sh_flag=sh_flag, rot_flag=rot_flag,O=O,M=M)

        fit0 = self.katsuura_func(y[0], False, False,O=O,M=M)
        fit1 = self.happycat_func(y[1], False, False,O=O,M=M)
        fit2 = self.grie_rosen_func(y[2], False, False,O=O,M=M)
        fit3 = self.schwefel_func(y[3], False, False,O=O,M=M)
        fit4 = self.ackley_func(y[4], False, False,O=O,M=M)
        
        return fit0 + fit1 + fit2 + fit3 + fit4 + 2200.0

    def cec2022_f9(self, x: torch.Tensor, sh_flag: bool, rot_flag: bool, O: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """Composition Function 1"""
        nx = x.size(1)
        delta = [10, 20, 30, 40, 50]
        bias = [0, 200, 300, 100, 400]
        fit = [
            self.rosenbrock_func(x, True, rot_flag,O=O[:,0*nx:1*nx],M=M[:,0*nx:1*nx]) * 10000 / 1e4,
            self.ellips_func(x, True, rot_flag,    O=O[:,1*nx:2*nx],M=M[:,1*nx:2*nx]) * 10000 / 1e10,
            self.bent_cigar_func(x, True, rot_flag,O=O[:,2*nx:3*nx],M=M[:,2*nx:3*nx]) * 10000 / 1e30,
            self.discus_func(x, True, rot_flag,    O=O[:,3*nx:4*nx],M=M[:,3*nx:4*nx]) * 10000 / 1e10,
            self.ellips_func(x, True, False,       O=O[:,4*nx:5*nx],M=M[:,4*nx:5*nx]) * 10000 / 1e10,
        ]
        f = self.cf_cal(x, fit, delta, bias)
        return f + 2300.0

    def cec2022_f10(self, x: torch.Tensor, sh_flag: bool, rot_flag: bool, O: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """Composition Function 2"""
        nx = x.size(1)
        delta = [20, 10, 10]
        bias = [0, 200, 100]
        fit = [
            self.schwefel_func(x,True,False,    O=O[:,0*nx:1*nx],M=M[:,0*nx:1*nx]) * 1.0, 
            self.rastrigin_func(x,True,rot_flag,O=O[:,1*nx:2*nx],M=M[:,1*nx:2*nx]) * 1.0,
            self.hgbat_func(x,True,rot_flag,    O=O[:,2*nx:3*nx],M=M[:,2*nx:3*nx]) * 1.0
        ]
        f = self.cf_cal(x, fit, delta, bias)
        return f + 2400.0

    def cec2022_f11(self, x: torch.Tensor, sh_flag: bool, rot_flag: bool, O: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """Composition Function 6"""
        nx = x.size(1)
        delta = [20, 20, 30, 30, 20]
        bias = [0, 200, 300, 400, 200]
        fit = [
            self.escaffer6_func (x, True, rot_flag ,O=O[:,0*nx:1*nx],M=M[:,0*nx:1*nx]) * 10000 / 2e7,
            self.schwefel_func  (x, True, rot_flag ,O=O[:,1*nx:2*nx],M=M[:,1*nx:2*nx]) * 1.0,
            self.griewank_func  (x, True, rot_flag ,O=O[:,2*nx:3*nx],M=M[:,2*nx:3*nx]) * 1000 / 100,
            self.rosenbrock_func(x, True, rot_flag ,O=O[:,3*nx:4*nx],M=M[:,3*nx:4*nx]) * 1.0,
            self.rastrigin_func (x, True, rot_flag ,O=O[:,4*nx:5*nx],M=M[:,4*nx:5*nx]) * 10000 / 1e3,
        ]
        f = self.cf_cal(x, fit, delta, bias)
        return f + 2600.0

    def cec2022_f12(self, x: torch.Tensor, sh_flag: bool, rot_flag: bool, O: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """Composition Function 7"""
        nx = x.size(1)
        delta = [10, 20, 30, 40, 50, 60]
        bias = [0, 300, 500, 100, 400, 200]
        fit = [
            self.hgbat_func     (x, True, rot_flag ,O=O[:,0*nx:1*nx],M=M[:,0*nx:1*nx]) * 10000 / 1000,
            self.rastrigin_func (x, True, rot_flag ,O=O[:,1*nx:2*nx],M=M[:,1*nx:2*nx]) * 10000 / 1e3,
            self.schwefel_func  (x, True, rot_flag ,O=O[:,2*nx:3*nx],M=M[:,2*nx:3*nx]) * 10000 / 4e3,
            self.bent_cigar_func(x, True, rot_flag ,O=O[:,3*nx:4*nx],M=M[:,3*nx:4*nx]) * 10000 / 1e30,
            self.ellips_func    (x, True, rot_flag ,O=O[:,4*nx:5*nx],M=M[:,4*nx:5*nx]) * 10000 / 1e10,
            self.escaffer6_func (x, True, rot_flag ,O=O[:,5*nx:6*nx],M=M[:,5*nx:6*nx]) * 10000 / 2e7,
        ]
        f = self.cf_cal(x, fit, delta, bias)
        return f + 2700.0

    # Basic functions
    def zakharov_func(self, x: torch.Tensor, sh_flag: bool,rot_flag: bool, O: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """Problem number = 1."""
        x = self.sr_func_rate( x, sh_rate=1.0, sh_flag=sh_flag, rot_flag=rot_flag, O=O, M=M)
        sum1 = x**2
        idx = torch.arange(1, x.size(1) + 1, device=x.device)
        sum2 = torch.sum( (0.5 * idx) * x, dim = 1)
        return torch.sum(sum1,dim=1) + sum2**2 + sum2**4

    def step_rastrigin_func(self, x: torch.Tensor, sh_flag: bool, rot_flag: bool, O: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """Problem number = 4."""
        z = self.sr_func_rate(x, sh_rate=5.12 / 100.0, sh_flag=sh_flag, rot_flag=rot_flag,O=O,M=M)
        return torch.sum(z**2 - 10.0 * torch.cos(2.0 * torch.pi * z) + 10.0, dim=1)

    def levy_func(self, x: torch.Tensor, sh_flag: bool, rot_flag: bool, O: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """Problem number = 5."""
        x = self.sr_func_rate(x, 1.0, sh_flag, rot_flag ,O=O,M=M)
        w = 1.0 + x / 4.0
        tmp1 = torch.sin(torch.pi * w[:, 0]) ** 2
        tmp2 = (w[:, -1] - 1) ** 2 * (1 + torch.sin(2 * torch.pi * w[:, -1]) ** 2)
        sum = (w[:, :-1] - 1) ** 2 * (1 + 10 * torch.sin(torch.pi * w[:, :-1] + 1) ** 2)
        return tmp1 + torch.sum( sum, dim=1 ) + tmp2

    def bent_cigar_func(self, x: torch.Tensor, sh_flag: bool, rot_flag: bool, O: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        x = self.sr_func_rate(x, 1.0, sh_flag, rot_flag ,O=O,M=M)
        return x[:, 0] ** 2 + torch.sum((10.0**6) * x[:, 1:] ** 2, dim=1)

    def hgbat_func(self, x: torch.Tensor, sh_flag: bool, rot_flag: bool, O: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        alpha = 1.0 / 4.0
        x = self.sr_func_rate(x, sh_rate=5.0 / 100.0, sh_flag=sh_flag, rot_flag=rot_flag,O=O,M=M)
        tmp = x - 1
        r2 = torch.sum(tmp**2, dim=1)
        sum_x = torch.sum(tmp, dim=1)
        return torch.abs(r2**2 - sum_x**2) ** (2 * alpha) + (0.5 * r2 + sum_x) / x.size(1) + 0.5

    def rastrigin_func(self, x: torch.Tensor, sh_flag: bool, rot_flag: bool, O: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        x = self.sr_func_rate(x, sh_rate=5.12 / 100.0, sh_flag=sh_flag, rot_flag=rot_flag,O=O,M=M)
        return torch.sum(x**2 - 10.0 * torch.cos(2.0 * torch.pi * x) + 10.0, dim=1)

    def katsuura_func(self, x: torch.Tensor, sh_flag: bool, rot_flag: bool, O: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        x = self.sr_func_rate(x, sh_rate=5.0 / 100.0, sh_flag=sh_flag, rot_flag=rot_flag,O=O,M=M)
        nx = x.size(1)
        tmp1 = 2.0 ** torch.arange(1, 33, device=x.device)
        tmp2 = x.unsqueeze(-1) * tmp1.unsqueeze(0).unsqueeze(0)
        temp = torch.sum(torch.abs(tmp2 - torch.floor(tmp2 + 0.5)) / tmp1, dim=2)
        tmp3 = torch.arange(1, nx + 1, device=x.device)
        f = torch.prod((1 + temp * tmp3.unsqueeze(0)) ** (10.0 / (nx**1.2)), dim=1)
        return (f - 1) * (10.0 / nx / nx)

    def ackley_func(self, x: torch.Tensor, sh_flag: bool, rot_flag: bool, O: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        x = self.sr_func_rate(x,sh_rate=1.0,sh_flag=sh_flag,rot_flag=rot_flag,O=O,M=M)
        mean1 = torch.mean(x**2, dim=1)
        mean2 = torch.mean(torch.cos(2.0 * torch.pi * x), dim=1)
        return torch.e - 20.0 * torch.exp( -0.2 * torch.sqrt(mean1) ) - torch.exp(mean2) + 20.0

    def schwefel_func(self, x: torch.Tensor, sh_flag: bool, rot_flag: bool, O: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        x = self.sr_func_rate(x, sh_rate=1000.0 / 100.0, sh_flag=sh_flag, rot_flag=rot_flag,O=O,M=M)
        nx = x.size(1)
        tmp1 = x + 420.9687462275036
        tmp2 = -tmp1 * tmp1.abs().sqrt().sin()
        tmp3 = (500.0 - torch.fmod( tmp1.abs(), 500 ) ) * ( 500.0 - torch.fmod( tmp1.abs(), 500 ) ).abs().sqrt().sin()
        tmp5 = torch.where( tmp1 >  500.0, -tmp3 + ( tmp1 - 500.0 ) ** 2 / 10000.0 / nx, tmp2 )
        tmp5 = torch.where( tmp1 < -500.0,  tmp3 + ( tmp1 + 500.0 ) ** 2 / 10000.0 / nx, tmp5 )
        return torch.sum(tmp5, dim=1) + 418.98288727243378 * nx

    def schaffer_F7_func(self, x: torch.Tensor, sh_flag: bool, rot_flag: bool, O: torch.Tensor, M: torch.Tensor ) -> torch.Tensor:
        x = self.sr_func_rate(x, sh_rate=1.0, sh_flag=sh_flag, rot_flag=rot_flag ,O=O,M=M)
        tmp1 = torch.hypot(x[:, :-1], x[:, 1:])
        tmp2 = torch.sin(50.0 * (tmp1 ** 0.2) )
        f = torch.sqrt(tmp1) * (1 + tmp2 * tmp2)
        f = torch.mean(f, dim=1)
        return f * f

    def escaffer6_func(self, x: torch.Tensor, sh_flag: bool, rot_flag: bool, O: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        x = self.sr_func_rate(x,sh_rate=1.0,sh_flag=sh_flag,rot_flag=rot_flag,O=O,M=M)
        y = torch.cat( [ x[:,1:], x[:,0:1] ], dim = 1 )
        tmp1 = torch.sin(torch.sqrt(x ** 2 + y ** 2)) ** 2
        tmp2 = 1.0 + 0.001 * (x ** 2 + y ** 2)
        return torch.sum(0.5 + (tmp1 - 0.5) / (tmp2**2), dim=1)

    def happycat_func(self, x: torch.Tensor, sh_flag: bool, rot_flag: bool, O: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        alpha = 1.0 / 8.0
        x = self.sr_func_rate(x, sh_rate=5.0 / 100.0, sh_flag=sh_flag, rot_flag=rot_flag,O=O,M=M)
        nx = x.size(1)
        tmp = x - 1
        r2 = torch.sum(tmp**2, dim=1)
        sum_x = torch.sum(tmp, dim=1)
        return torch.abs(r2 - nx) ** (2 * alpha) + (0.5 * r2 + sum_x) / nx + 0.5

    def grie_rosen_func(self, x: torch.Tensor, sh_flag: bool, rot_flag: bool, O: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        x = self.sr_func_rate(x, sh_rate=5.0 / 100.0, sh_flag=sh_flag, rot_flag=rot_flag,O=O,M=M)
        x = x + 1
        y = torch.cat( [ x[:,1:], x[:,0:1] ], dim = 1 )
        tmp = 100.0 * (x ** 2 - y) ** 2 + (x - 1.0) ** 2
        return torch.sum((tmp**2) / 4000.0 - torch.cos(tmp) + 1.0, dim=1)

    def griewank_func(self, x: torch.Tensor, sh_flag: bool, rot_flag: bool, O: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        x = self.sr_func_rate(x, sh_rate=600.0 / 100.0, sh_flag=sh_flag, rot_flag=rot_flag,O=O,M=M)
        sum_sq = torch.sum(x**2, dim=1)
        idx = torch.arange(1, x.size(1) + 1, device=x.device)
        prod_cos = torch.prod(torch.cos(x / torch.sqrt(idx)), dim=1)
        return 1.0 + sum_sq / 4000.0 - prod_cos

    def rosenbrock_func(self, x: torch.Tensor, sh_flag: bool, rot_flag: bool, O: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        x = self.sr_func_rate(x, sh_rate=2.048 / 100.0, sh_flag=sh_flag, rot_flag=rot_flag,O=O,M=M)
        tmp = x + 1
        return torch.sum(100.0 * (tmp[:, :-1] ** 2 - tmp[:, 1:]) ** 2 + (tmp[:, :-1] - 1.0) ** 2, dim=1)

    def discus_func(self, x: torch.Tensor, sh_flag: bool, rot_flag: bool, O: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        x = self.sr_func_rate(x,sh_rate=1.0,sh_flag=sh_flag, rot_flag=rot_flag,O=O,M=M)
        return (10.0**6) * x[:, 0] ** 2 + torch.sum(x[:, 1:] ** 2, dim=1)

    def ellips_func(self, x: torch.Tensor, sh_flag: bool, rot_flag: bool, O: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        x = self.sr_func_rate(x,sh_rate=1.0,sh_flag=sh_flag, rot_flag=rot_flag,O=O,M=M)
        nx = x.size(1)
        idx = torch.arange(nx, device=x.device)
        powers = 6.0 * idx / (nx - 1)
        return torch.sum((10.0**powers) * x**2, dim=1)

    def evaluate(self, pop: torch.Tensor) -> torch.Tensor:
        assert pop.size(1) == self.nx, f"Dimension mismatch! Expect {self.nx}, got {pop.size(1)}."

        if self.func_num == 1:
            fitness = self.cec2022_f1(pop, True, True,O=self.OShift,M=self.M)
        elif self.func_num == 2:
            fitness = self.cec2022_f2(pop, True, True,O=self.OShift,M=self.M)
        elif self.func_num == 3:
            fitness = self.cec2022_f3(pop, True, True,O=self.OShift,M=self.M)
        elif self.func_num == 4:
            fitness = self.cec2022_f4(pop, True, True,O=self.OShift,M=self.M)
        elif self.func_num == 5:
            fitness = self.cec2022_f5(pop, True, True,O=self.OShift,M=self.M)
        elif self.func_num == 6:
            fitness = self.cec2022_f6(pop, True, True,O=self.OShift,M=self.M)
        elif self.func_num == 7:
            fitness = self.cec2022_f7(pop, True, True,O=self.OShift,M=self.M)
        elif self.func_num == 8:
            fitness = self.cec2022_f8(pop, True, True,O=self.OShift,M=self.M)
        elif self.func_num == 9:
            fitness = self.cec2022_f9(pop, True, True,O=self.OShift,M=self.M)
        elif self.func_num == 10:
            fitness = self.cec2022_f10(pop, True, True,O=self.OShift,M=self.M)
        elif self.func_num == 11:
            fitness = self.cec2022_f11(pop, True, True,O=self.OShift,M=self.M)
        elif self.func_num == 12:
            fitness = self.cec2022_f12(pop, True, True,O=self.OShift,M=self.M)
        else:
            raise ValueError(f"Function {self.func_num} is not defined.")

        return fitness
