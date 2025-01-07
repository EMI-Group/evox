import torch
import os
from ...core.components import Problem
from typing import Optional

class CEC2022(Problem):
    def __init__(
        self,
        problem_number: int,
        dimension: int,
    ):
        super().__init__()
        self.problem_number = problem_number
        self.nx = dimension
        self.func_num = problem_number
        self.initialized = False

        self.n_flag: Optional[int] = None
        self.func_flag: Optional[int] = None
        self.OShift: Optional[torch.Tensor] = None
        self.M: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None
        self.z: Optional[torch.Tensor] = None
        self.x_bound: Optional[torch.Tensor] = None
        self.SS: Optional[torch.Tensor] = None
        self.sh_flag = True
        self.rot_flag = True

        if self.nx not in [2, 10, 20]:
            raise ValueError("Test functions are only defined for D=2,10,20.")
        if self.nx == 2 and self.func_num in [6, 7, 8]:
            raise ValueError("Function not defined for D=2.")

        # Loading data preparation
        current_dir = os.path.dirname(__file__)
        data_dir = os.path.join(current_dir, "cec2022_input_data")
        # Loading rotation matrix M
        m_filename = os.path.join(data_dir, f"M_{self.func_num}_D{self.nx}.txt")
        m_filename = m_filename.replace("\\", "/")
        if not os.path.isfile(m_filename):
            raise FileNotFoundError(f"Cannot open {m_filename} for reading")
        with open(m_filename, 'r') as f:
            M_data = [float(num) for num in f.read().split()]
        if self.func_num < 9:
            self.M = torch.tensor(M_data, dtype=torch.float64).reshape(self.nx, self.nx)
        else:
            # 对于复合函数（func_num >= 9），cf_num 在 C++ 代码里写成 12
            cf_num = 12
            self.M = torch.tensor(M_data, dtype=torch.float64).reshape(cf_num, self.nx, self.nx)

        # Loading shift matrix OShift
        shift_filename = os.path.join(data_dir, f"shift_data_{self.func_num}.txt")
        if not os.path.isfile(shift_filename):
            raise FileNotFoundError(f"Cannot open {shift_filename} for reading")
        with open(shift_filename, 'r') as f:
            shift_data = [float(num) for num in f.read().split()]
        if self.func_num < 9:
            self.OShift = torch.tensor(shift_data, dtype=torch.float64)
        else:
            cf_num = 12
            self.OShift = torch.tensor(shift_data, dtype=torch.float64).reshape(cf_num, self.nx)

        # Loading shuffle index SS
        if 6 <= self.func_num <= 8:
            shuffle_filename = os.path.join(data_dir, f"shuffle_data_{self.func_num}_D{self.nx}.txt")
            if not os.path.isfile(shuffle_filename):
                raise FileNotFoundError(f"Cannot open {shuffle_filename} for reading")
            with open(shuffle_filename, 'r') as f:
                shuffle_data = [int(num) for num in f.read().split()]
            # 转换成 0-based index
            self.SS = torch.tensor(shuffle_data, dtype=torch.long) - 1

        self.y = torch.zeros(self.nx, dtype=torch.float64)
        self.z = torch.zeros(self.nx, dtype=torch.float64)
        self.x_bound = torch.full((self.nx,), 100.0, dtype=torch.float64)

        self.n_flag = self.nx
        self.func_flag = self.func_num
        self.initialized = True

        # Transform matrix
        def shift(x: torch.Tensor) -> torch.Tensor:
            """Shift the input vector."""
            return x - self.OShift

        def rotate(x: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
            """Rotate the input vector."""
            return torch.matmul(x, mat)

        def sr_func(x: torch.Tensor, sh_rate: float = 1.0) -> torch.Tensor:
            """Shift and rotate function."""
            if self.sh_flag:
                if self.rot_flag:
                    y_ = shift(x) * sh_rate
                    z_ = rotate(y_, self.M)
                else:
                    z_ = shift(x) * sh_rate
            else:
                if self.rot_flag:
                    y_ = x * sh_rate
                    z_ = rotate(y_, self.M)
                else:
                    z_ = x * sh_rate
            return z_

        def cec2022_f1(x: torch.Tensor) -> torch.Tensor:
            """Zakharov Function"""
            z_ = sr_func(x)
            sum1 = torch.sum(z_ ** 2)
            idx = torch.arange(1, self.nx + 1, dtype=torch.float64, device=x.device)
            sum2 = torch.sum(0.5 * (idx * z_))
            return sum1 + sum2 ** 2 + sum2 ** 4 + 300.0

        def cec2022_f2(x: torch.Tensor) -> torch.Tensor:
            """Rosenbrock Function"""
            z_ = sr_func(x, sh_rate=2.048 / 100.0)
            z_ += 1.0  # Shift to origin
            tmp = 100.0 * (z_[:-1] ** 2 - z_[1:]) ** 2 + (z_[:-1] - 1.0) ** 2
            return torch.sum(tmp) + 400.0

        def cec2022_f3(x: torch.Tensor) -> torch.Tensor:
            """Schaffer F7 Function"""
            z_ = sr_func(x)
            temp1 = torch.sin(torch.sqrt(z_[:-1] ** 2 + z_[1:] ** 2)) ** 2
            temp2 = 1.0 + 0.001 * (z_[:-1] ** 2 + z_[1:] ** 2)
            f_ = torch.sum(0.5 + (temp1 - 0.5) / (temp2 ** 2))
            return f_ ** 2 / ((self.nx - 1) ** 2) + 600.0

        def cec2022_f4(x: torch.Tensor) -> torch.Tensor:
            """Step Rastrigin Function (Noncontinuous Rastrigin's)"""
            y_ = x.clone()
            if self.OShift is not None:  # 做一次单独 shift
                y_ = y_ - self.OShift
            y_ = torch.floor(2 * y_ + 0.5) / 2
            z_ = sr_func(y_, sh_rate=5.12 / 100.0)
            f_ = torch.sum(z_ ** 2 - 10.0 * torch.cos(2.0 * torch.pi * z_) + 10.0)
            return f_ + 800.0

        def cec2022_f5(x: torch.Tensor) -> torch.Tensor:
            """Levy Function"""
            z_ = sr_func(x)
            w_ = 1.0 + z_ / 4.0
            term1 = torch.sin(torch.pi * w_[0]) ** 2
            term3 = (w_[-1] - 1) ** 2 * (1 + torch.sin(2 * torch.pi * w_[-1]) ** 2)
            sum_terms = torch.sum((w_[:-1] - 1) ** 2 * (1 + 10 * torch.sin(torch.pi * w_[:-1] + 1) ** 2))
            return term1 + sum_terms + term3 + 900.0

        # Basic functions
        def bent_cigar(z_: torch.Tensor) -> torch.Tensor:
            return z_[0] ** 2 + torch.sum((10.0 ** 6) * z_[1:] ** 2)

        def hgbat(z_: torch.Tensor) -> torch.Tensor:
            alpha = 1.0 / 4.0
            tmp = z_.clone()
            tmp -= 1.0
            r2_ = torch.sum(tmp ** 2)
            sum_z_ = torch.sum(tmp)
            return torch.abs(r2_ ** 2 - sum_z_ ** 2) ** (2 * alpha) + (0.5 * r2_ + sum_z_) / self.nx + 0.5

        def rastrigin(z_: torch.Tensor) -> torch.Tensor:
            return torch.sum(z_ ** 2 - 10.0 * torch.cos(2.0 * torch.pi * z_) + 10.0)

        def katsuura(z_: torch.Tensor) -> torch.Tensor:
            f_ = torch.ones(1, dtype=torch.float64, device=z_.device)
            tmp3_ = (10.0 * self.nx) ** 1.2
            for i_ in range(self.nx):
                temp_ = torch.zeros(32, dtype=torch.float64, device=z_.device)
                for j_ in range(1, 33):
                    tmp1_ = 2.0 ** j_
                    tmp2_ = tmp1_ * z_[i_]
                    temp_[j_ - 1] = torch.abs(tmp2_ - torch.floor(tmp2_ + 0.5)) / tmp1_
                temp_sum_ = torch.sum(temp_)
                f_ *= (1.0 + (i_ + 1) * temp_sum_) ** (10.0 / tmp3_)
            return f_ * (10.0 / (self.nx ** 2)) - (10.0 / (self.nx ** 2))

        def ackley(z_: torch.Tensor) -> torch.Tensor:
            sum1_ = torch.sum(z_ ** 2)
            sum2_ = torch.sum(torch.cos(2.0 * torch.pi * z_))
            term1_ = -0.2 * torch.sqrt(sum1_ / self.nx)
            term2_ = -torch.exp(sum2_ / self.nx)
            return torch.e - 20.0 * torch.exp(term1_) - torch.exp(term2_) + 20.0

        def schwefel(z_: torch.Tensor) -> torch.Tensor:
            tmp = z_.clone()
            tmp += 420.9687462275036
            term1_ = -tmp * torch.sin(torch.sqrt(torch.abs(tmp)))
            f_ = torch.sum(term1_)
            f_ += 418.98288727243378 * self.nx
            return f_

        def schaffer_F7(z_: torch.Tensor) -> torch.Tensor:
            # 在原代码中，schaffer_F7 与 escaffer6 有紧密关联，这里照原C++一样直接调用 escaffer6
            return escaffer6(z_)

        def escaffer6(z_: torch.Tensor) -> torch.Tensor:
            temp1_ = torch.sin(torch.sqrt(z_[:-1] ** 2 + z_[1:] ** 2)) ** 2
            temp2_ = 1.0 + 0.001 * (z_[:-1] ** 2 + z_[1:] ** 2)
            f_ = torch.sum(0.5 + (temp1_ - 0.5) / (temp2_ ** 2))
            return f_

        def happycat(z_: torch.Tensor) -> torch.Tensor:
            alpha_ = 1.0 / 8.0
            tmp = z_.clone()
            tmp -= 1.0
            r2_ = torch.sum(tmp ** 2)
            sum_z_ = torch.sum(tmp)
            return torch.abs(r2_ - self.nx) ** (2 * alpha_) + (0.5 * r2_ + sum_z_) / self.nx + 0.5

        def grie_rosen(z_: torch.Tensor) -> torch.Tensor:
            tmp = z_.clone()
            tmp += 1.0
            tmp2_ = 100.0 * (tmp[:-1] ** 2 - tmp[1:]) ** 2 + (tmp[:-1] - 1.0) ** 2
            return torch.sum((tmp2_ ** 2) / 4000.0 - torch.cos(tmp2_) + 1.0)

        def griewank(z_: torch.Tensor) -> torch.Tensor:
            sum_sq_ = torch.sum(z_ ** 2)
            idx_ = torch.arange(1, self.nx + 1, dtype=torch.float64, device=z_.device)
            prod_cos_ = torch.prod(torch.cos(z_ / torch.sqrt(idx_)))
            return 1.0 + sum_sq_ / 4000.0 - prod_cos_

        def rosenbrock(z_: torch.Tensor) -> torch.Tensor:
            tmp = z_.clone()
            tmp += 1.0
            return torch.sum(100.0 * (tmp[:-1] ** 2 - tmp[1:]) ** 2 + (tmp[:-1] - 1.0) ** 2)

        def discus(z_: torch.Tensor) -> torch.Tensor:
            return (10.0 ** 6) * z_[0] ** 2 + torch.sum(z_[1:] ** 2)

        def ellips(z_: torch.Tensor) -> torch.Tensor:
            idx_ = torch.arange(self.nx, dtype=torch.float64, device=z_.device)
            powers_ = 6.0 * idx_ / (self.nx - 1) if self.nx > 1 else 0
            return torch.sum((10.0 ** powers_) * z_ ** 2)

        # 为了组合函数使用，需要同一个作用域下
        def cf_cal(x: torch.Tensor, fit_: torch.Tensor, delta_: torch.Tensor, bias_: torch.Tensor, cf_num_: int) -> torch.Tensor:
            w_ = torch.zeros(cf_num_, dtype=torch.float64, device=x.device)
            for i_ in range(cf_num_):
                # 对应地，取 shift 的块
                block_start = i_ * self.nx
                block_end = (i_ + 1) * self.nx
                diff_ = x - self.OShift[block_start:block_end]
                w_[i_] = torch.sum(diff_ ** 2)
                if w_[i_] != 0:
                    w_[i_] = (1.0 / torch.sqrt(w_[i_])) * torch.exp(-w_[i_] / (2.0 * self.nx * delta_[i_] ** 2))
                else:
                    w_[i_] = torch.tensor(float('inf'), dtype=torch.float64, device=x.device)

            w_max_ = torch.max(w_)
            w_sum_ = torch.sum(w_)
            if w_max_ == 0:
                w_ = torch.ones(cf_num_, dtype=torch.float64, device=x.device)
                w_sum_ = cf_num_
            return torch.sum(w_ / w_sum_ * (fit_ + bias_))

        # 将这些辅助函数、子函数绑定到 self 上
        self.sr_func = sr_func
        self.bent_cigar = bent_cigar
        self.hgbat = hgbat
        self.rastrigin = rastrigin
        self.katsuura = katsuura
        self.ackley = ackley
        self.schwefel = schwefel
        self.schaffer_F7 = schaffer_F7
        self.escaffer6 = escaffer6
        self.happycat = happycat
        self.grie_rosen = grie_rosen
        self.griewank = griewank
        self.rosenbrock = rosenbrock
        self.discus = discus
        self.ellips = ellips
        self.cf_cal = cf_cal

        # cec2022_fX
        self.cec2022_f1 = cec2022_f1
        self.cec2022_f2 = cec2022_f2
        self.cec2022_f3 = cec2022_f3
        self.cec2022_f4 = cec2022_f4
        self.cec2022_f5 = cec2022_f5

        def cec2022_f6(x: torch.Tensor) -> torch.Tensor:
            """Hybrid Function 2"""
            cf_num_ = 3
            Gp_ = torch.tensor([0.4, 0.4, 0.2], dtype=torch.float64, device=x.device)
            G_nx_ = torch.ceil(Gp_ * self.nx).int().tolist()
            G_nx_[-1] = self.nx - sum(G_nx_[:-1])
            G_ = [0] + list(torch.cumsum(torch.tensor(G_nx_[:-1]), dim=0).tolist())

            z_ = self.sr_func(x)
            y_ = z_[self.SS[:self.nx]] if self.SS is not None else z_

            fit_ = torch.zeros(cf_num_, dtype=torch.float64, device=x.device)
            fit_[0] = self.bent_cigar(y_[G_[0] : G_[0] + G_nx_[0]])
            fit_[1] = self.hgbat(y_[G_[1] : G_[1] + G_nx_[1]])
            fit_[2] = self.rastrigin(y_[G_[2] : G_[2] + G_nx_[2]])

            return torch.sum(fit_) + 1800.0

        self.cec2022_f6 = cec2022_f6

        def cec2022_f7(x: torch.Tensor) -> torch.Tensor:
            """Hybrid Function 10"""
            cf_num_ = 6
            Gp_ = torch.tensor([0.1, 0.2, 0.2, 0.2, 0.1, 0.2], dtype=torch.float64, device=x.device)
            G_nx_ = torch.ceil(Gp_ * self.nx).int().tolist()
            G_nx_[-1] = self.nx - sum(G_nx_[:-1])
            G_ = [0] + list(torch.cumsum(torch.tensor(G_nx_[:-1]), dim=0).tolist())

            z_ = self.sr_func(x)
            y_ = z_[self.SS[:self.nx]] if self.SS is not None else z_

            fit_ = torch.zeros(cf_num_, dtype=torch.float64, device=x.device)
            fit_[0] = self.hgbat(y_[G_[0] : G_[0] + G_nx_[0]])
            fit_[1] = self.katsuura(y_[G_[1] : G_[1] + G_nx_[1]])
            fit_[2] = self.ackley(y_[G_[2] : G_[2] + G_nx_[2]])
            fit_[3] = self.rastrigin(y_[G_[3] : G_[3] + G_nx_[3]])
            fit_[4] = self.schwefel(y_[G_[4] : G_[4] + G_nx_[4]])
            fit_[5] = self.schaffer_F7(y_[G_[5] : G_[5] + G_nx_[5]])

            return torch.sum(fit_) + 2000.0

        self.cec2022_f7 = cec2022_f7

        def cec2022_f8(x: torch.Tensor) -> torch.Tensor:
            """Hybrid Function 6"""
            cf_num_ = 5
            Gp_ = torch.tensor([0.3, 0.2, 0.2, 0.1, 0.2], dtype=torch.float64, device=x.device)
            G_nx_ = torch.ceil(Gp_ * self.nx).int().tolist()
            G_nx_[-1] = self.nx - sum(G_nx_[:-1])
            G_ = [0] + list(torch.cumsum(torch.tensor(G_nx_[:-1]), dim=0).tolist())

            z_ = self.sr_func(x)
            y_ = z_[self.SS[:self.nx]] if self.SS is not None else z_

            fit_ = torch.zeros(cf_num_, dtype=torch.float64, device=x.device)
            fit_[0] = self.katsuura(y_[G_[0] : G_[0] + G_nx_[0]])
            fit_[1] = self.happycat(y_[G_[1] : G_[1] + G_nx_[1]])
            fit_[2] = self.grie_rosen(y_[G_[2] : G_[2] + G_nx_[2]])
            fit_[3] = self.schwefel(y_[G_[3] : G_[3] + G_nx_[3]])
            fit_[4] = self.ackley(y_[G_[4] : G_[4] + G_nx_[4]])

            return torch.sum(fit_) + 2200.0

        self.cec2022_f8 = cec2022_f8

        def cec2022_f9(x: torch.Tensor) -> torch.Tensor:
            """Composition Function 1"""
            cf_num_ = 5
            delta_ = torch.tensor([10, 20, 30, 40, 50], dtype=torch.float64, device=x.device)
            bias_ = torch.tensor([0, 200, 300, 100, 400], dtype=torch.float64, device=x.device)

            fit_ = torch.zeros(cf_num_, dtype=torch.float64, device=x.device)
            fit_[0] = self.rosenbrock(x) * 10000 / 1e4
            fit_[1] = self.ellips(x) * 10000 / 1e10
            fit_[2] = self.bent_cigar(x) * 10000 / 1e30
            fit_[3] = self.discus(x) * 10000 / 1e10
            fit_[4] = self.ellips(x) * 10000 / 1e10

            f_ = self.cf_cal(x, fit_, delta_, bias_, cf_num_)
            return f_ + 2300.0

        self.cec2022_f9 = cec2022_f9

        def cec2022_f10(x: torch.Tensor) -> torch.Tensor:
            """Composition Function 2"""
            cf_num_ = 3
            delta_ = torch.tensor([20, 10, 10], dtype=torch.float64, device=x.device)
            bias_ = torch.tensor([0, 200, 100], dtype=torch.float64, device=x.device)

            fit_ = torch.zeros(cf_num_, dtype=torch.float64, device=x.device)
            fit_[0] = self.schwefel(x) * 1.0
            fit_[1] = self.rastrigin(x) * 1.0
            fit_[2] = self.hgbat(x) * 1.0

            f_ = self.cf_cal(x, fit_, delta_, bias_, cf_num_)
            return f_ + 2400.0

        self.cec2022_f10 = cec2022_f10

        def cec2022_f11(x: torch.Tensor) -> torch.Tensor:
            """Composition Function 6"""
            cf_num_ = 5
            delta_ = torch.tensor([20, 20, 30, 30, 20], dtype=torch.float64, device=x.device)
            bias_ = torch.tensor([0, 200, 300, 400, 200], dtype=torch.float64, device=x.device)

            fit_ = torch.zeros(cf_num_, dtype=torch.float64, device=x.device)
            fit_[0] = self.escaffer6(x) * 10000 / 2e7
            fit_[1] = self.schwefel(x) * 1.0
            fit_[2] = self.griewank(x) * 1000 / 100
            fit_[3] = self.rosenbrock(x) * 1.0
            fit_[4] = self.rastrigin(x) * 10000 / 1e3

            f_ = self.cf_cal(x, fit_, delta_, bias_, cf_num_)
            return f_ + 2600.0

        self.cec2022_f11 = cec2022_f11

        def cec2022_f12(x: torch.Tensor) -> torch.Tensor:
            """Composition Function 7"""
            cf_num_ = 6
            delta_ = torch.tensor([10, 20, 30, 40, 50, 60], dtype=torch.float64, device=x.device)
            bias_ = torch.tensor([0, 300, 500, 100, 400, 200], dtype=torch.float64, device=x.device)

            fit_ = torch.zeros(cf_num_, dtype=torch.float64, device=x.device)
            fit_[0] = self.hgbat(x) * 10000 / 1000
            fit_[1] = self.rastrigin(x) * 10000 / 1e3
            fit_[2] = self.schwefel(x) * 10000 / 4e3
            fit_[3] = self.bent_cigar(x) * 10000 / 1e30
            fit_[4] = self.ellips(x) * 10000 / 1e10
            fit_[5] = self.escaffer6(x) * 10000 / 2e7

            f_ = self.cf_cal(x, fit_, delta_, bias_, cf_num_)
            return f_ + 2700.0

        self.cec2022_f12 = cec2022_f12

    def evaluate(self, pop: torch.Tensor) -> torch.Tensor:
        if pop.shape[1] != self.nx:
            raise ValueError(f"Dimension mismatch! Expect {self.nx}, got {pop.shape[1]}.")

        fitness = torch.empty(pop.shape[0], dtype=pop.dtype, device=pop.device)
        for i in range(pop.shape[0]):
            x = pop[i]
            if self.func_num == 1:
                fitness[i] = self.cec2022_f1(x)
            elif self.func_num == 2:
                fitness[i] = self.cec2022_f2(x)
            elif self.func_num == 3:
                fitness[i] = self.cec2022_f3(x)
            elif self.func_num == 4:
                fitness[i] = self.cec2022_f4(x)
            elif self.func_num == 5:
                fitness[i] = self.cec2022_f5(x)
            elif self.func_num == 6:
                fitness[i] = self.cec2022_f6(x)
            elif self.func_num == 7:
                fitness[i] = self.cec2022_f7(x)
            elif self.func_num == 8:
                fitness[i] = self.cec2022_f8(x)
            elif self.func_num == 9:
                fitness[i] = self.cec2022_f9(x)
            elif self.func_num == 10:
                fitness[i] = self.cec2022_f10(x)
            elif self.func_num == 11:
                fitness[i] = self.cec2022_f11(x)
            elif self.func_num == 12:
                fitness[i] = self.cec2022_f12(x)
            else:
                raise ValueError(f"Function {self.func_num} is not defined.")
        return fitness
