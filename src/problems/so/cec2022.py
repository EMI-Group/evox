import torch
import os
from ...core.components import Problem
from typing import Optional, Tuple

class CEC2022(Problem):
    def __init__(
        self,
        problem_number: int,
        dimension: int,
        device: torch.device | None = None
    ):
        super().__init__()
        self.nx = dimension
        self.func_num = problem_number

        self.OShift: Optional[torch.Tensor] = None
        self.M: Optional[torch.Tensor] = None
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
            self.M = torch.tensor(M_data, device=device).reshape(self.nx, self.nx)
        else:
            self.M = torch.tensor(M_data, device=device).reshape(-1, self.nx)

        # Loading shift matrix OShift
        shift_filename = os.path.join(data_dir, f"shift_data_{self.func_num}.txt")
        if not os.path.isfile(shift_filename):
            raise FileNotFoundError(f"Cannot open {shift_filename} for reading")
        with open(shift_filename, 'r') as f:
            shift_data = [float(num) for num in f.read().split()]
        if self.func_num < 9:
            self.OShift = torch.tensor(shift_data, device=device).unsqueeze(0)
        else:
            self.OShift = torch.tensor(shift_data, device=device).unsqueeze(0)

        # Loading shuffle index SS
        if 6 <= self.func_num <= 8:
            shuffle_filename = os.path.join(data_dir, f"shuffle_data_{self.func_num}_D{self.nx}.txt")
            if not os.path.isfile(shuffle_filename):
                raise FileNotFoundError(f"Cannot open {shuffle_filename} for reading")
            with open(shuffle_filename, 'r') as f:
                shuffle_data = [int(num) for num in f.read().split()]
            # To 0-based index
            self.SS = torch.tensor(shuffle_data, dtype=torch.long, device=device) - 1


    # Transform matrix
    def shift(self, x: torch.Tensor) -> torch.Tensor:
        """Shift the input vector."""
        return x - self.OShift[:,:x.size(1)]

    def rotate(self, x: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
        """Rotate the input vector."""
        return torch.matmul(x, mat[:x.size(1)])
    
    def cut(self, x: torch.Tensor, Gp: torch.Tensor) -> Tuple[torch.Tensor]:
        nx = x.size(1)
        G_nx = torch.ceil(Gp * nx).int().tolist()
        G_nx[-1] = nx - sum(G_nx[:-1])
        G = [0] + list(torch.cumsum(torch.tensor(G_nx[:-1]), dim=0).tolist())

        y = self.sr_func(x)
        z = y[:,self.SS[:nx]] if self.SS is not None else y

        z_piece = []
        for i in range (Gp.size(0)):
            z_piece.append(z[:,G[i] : G[i] + G_nx[i]])
        return z_piece

    def sr_func(self, x: torch.Tensor, sh_rate: float = 1.0) -> torch.Tensor:
        """Shift and rotate function."""
        if self.sh_flag:
            if self.rot_flag:
                y = self.shift(x) * sh_rate
                z = self.rotate(y, self.M)
            else:
                z = self.shift(x) * sh_rate
        else:
            if self.rot_flag:
                y = x * sh_rate
                z = self.rotate(y, self.M)
            else:
                z = x * sh_rate
        return z

    # Problem
    def cec2022_f1(self, x: torch.Tensor) -> torch.Tensor:
        """Zakharov Function"""
        return self.zakharov_func(x) + 300.0

    def cec2022_f2(self, x: torch.Tensor) -> torch.Tensor:
        """Rosenbrock Function"""
        return self.rosenbrock_func(x) + 400

    def cec2022_f3(self, x: torch.Tensor) -> torch.Tensor:
        """Schaffer F7 Function"""
        return self.schaffer_F7_func(x) + 600.0

    def cec2022_f4(self, x: torch.Tensor) -> torch.Tensor:
        """Step Rastrigin Function (Noncontinuous Rastrigin's)"""
        return self.step_rastrigin_func(x) + 800.0
        
    def cec2022_f5(self, x: torch.Tensor) -> torch.Tensor:
        """Levy Function"""
        return self.levy_func(x) + 900.0

    def cec2022_f6(self, x: torch.Tensor) -> torch.Tensor:
        """Hybrid Function 2"""
        # cf_num = 3
        Gp = torch.tensor([0.4, 0.4, 0.2], device=x.device)
        y = self.cut(x, Gp)

        fit0 = self.bent_cigar_func(y[0])
        fit1 = self.hgbat_func(y[1])
        fit2 = self.rastrigin_func(y[2])

        return fit0 + fit1 + fit2 + 1800.0

    def cec2022_f7(self, x: torch.Tensor) -> torch.Tensor:
        """Hybrid Function 10"""
        # cf_num_ = 6
        Gp = torch.tensor([0.1, 0.2, 0.2, 0.2, 0.1, 0.2], device=x.device)
        y = self.cut(x, Gp)

        fit0 = self.hgbat_func(y[0])
        fit1 = self.katsuura_func(y[1])
        fit2 = self.ackley_func(y[2])
        fit3 = self.rastrigin_func(y[3])
        fit4 = self.schwefel_func(y[4])
        fit5 = self.schaffer_F7_func(y[5])

        return fit0 + fit1 + fit2 + fit3 + fit4 + fit5 + 2000.0

    def cec2022_f8(self, x: torch.Tensor) -> torch.Tensor:
        """Hybrid Function 6"""
        # cf_num_ = 5
        Gp = torch.tensor([0.3, 0.2, 0.2, 0.1, 0.2], device=x.device)
        y = self.cut(x, Gp)

        fit0 = self.katsuura_func(y[0])
        fit1 = self.happycat_func(y[1])
        fit2 = self.grie_rosen_func(y[2])
        fit3 = self.schwefel_func(y[3])
        fit4 = self.ackley_func(y[4])

        return fit0 + fit1 + fit2 + fit3 + fit4 + 2200.0

    def cec2022_f9(self, x: torch.Tensor) -> torch.Tensor:
        """Composition Function 1"""
        cf_num = 5
        delta = torch.tensor([10, 20, 30, 40, 50], device=x.device)
        bias = torch.tensor([0, 200, 300, 100, 400], device=x.device)

        fit = torch.zeros(x.size(0), cf_num, device=x.device)
        fit[:, 0] = self.rosenbrock_func(x) * 10000 / 1e4
        fit[:, 1] = self.ellips_func(x) * 10000 / 1e10
        fit[:, 2] = self.bent_cigar_func(x) * 10000 / 1e30
        fit[:, 3] = self.discus_func(x) * 10000 / 1e10
        fit[:, 4] = self.ellips_func(x) * 10000 / 1e10

        f = self.cf_cal(x, fit, delta, bias)
        return f + 2300.0

    def cec2022_f10(self, x: torch.Tensor) -> torch.Tensor:
        """Composition Function 2"""
        cf_num = 3
        delta = torch.tensor([20, 10, 10], device=x.device)
        bias = torch.tensor([0, 200, 100], device=x.device)

        fit = torch.zeros(x.size(0), cf_num, device=x.device)
        fit[:, 0] = self.schwefel_func(x) * 1.0
        fit[:, 1] = self.rastrigin_func(x) * 1.0
        fit[:, 2] = self.hgbat_func(x) * 1.0

        f = self.cf_cal(x, fit, delta, bias)
        return f + 2400.0

    def cec2022_f11(self, x: torch.Tensor) -> torch.Tensor:
        """Composition Function 6"""
        cf_num = 5
        delta = torch.tensor([20, 20, 30, 30, 20], device=x.device)
        bias = torch.tensor([0, 200, 300, 400, 200], device=x.device)

        fit = torch.zeros(x.size(0), cf_num, device=x.device)
        fit[:,0] = self.escaffer6_func(x) * 10000 / 2e7
        fit[:,1] = self.schwefel_func(x) * 1.0
        fit[:,2] = self.griewank_func(x) * 1000 / 100
        fit[:,3] = self.rosenbrock_func(x) * 1.0
        fit[:,4] = self.rastrigin_func(x) * 10000 / 1e3

        f = self.cf_cal(x, fit, delta, bias)
        return f + 2600.0        

    def cec2022_f12(self, x: torch.Tensor) -> torch.Tensor:
        """Composition Function 7"""
        cf_num = 6
        delta = torch.tensor([10, 20, 30, 40, 50, 60], device=x.device)
        bias = torch.tensor([0, 300, 500, 100, 400, 200], device=x.device)

        fit = torch.zeros(x.size(0), cf_num, device=x.device)
        fit[:,0] = self.hgbat_func(x) * 10000 / 1000
        fit[:,1] = self.rastrigin_func(x) * 10000 / 1e3
        fit[:,2] = self.schwefel_func(x) * 10000 / 4e3
        fit[:,3] = self.bent_cigar_func(x) * 10000 / 1e30
        fit[:,4] = self.ellips_func(x) * 10000 / 1e10
        fit[:,5] = self.escaffer6_func(x) * 10000 / 2e7

        f = self.cf_cal(x, fit, delta, bias)
        return f + 2700.0

    
    # Basic functions
    def zakharov_func(self, x: torch.Tensor) -> torch.Tensor:
        """Problem number = 1. """
        x = self.sr_func(x)
        sum1 = x ** 2
        idx = torch.arange(1, x.size(1) + 1, device=x.device)
        sum2 = 0.5 * (idx * x)
        return torch.sum(sum1 + sum2 ** 2 + sum2 ** 4, dim=1)   
    
    def step_rastrigin_func(self, x: torch.Tensor) -> torch.Tensor:
        """Problem number = 4. """
        if self.OShift is not None: 
            x = self.shift(x)
        y = torch.floor(2 * x + 0.5) / 2
        z = self.sr_func(y, sh_rate=5.12 / 100.0)
        return torch.sum(z ** 2 - 10.0 * torch.cos(2.0 * torch.pi * z) + 10.0, dim=1)

    def levy_func(self, x: torch.Tensor) -> torch.Tensor:
        """Problem number = 5. """
        x = self.sr_func(x)
        w = 1.0 + x / 4.0
        tmp1 = torch.sin(torch.pi * w[:,0]) ** 2
        tmp2 = (w[:,-1] - 1) ** 2 * (1 + torch.sin(2 * torch.pi * w[:,-1]) ** 2)
        sum = (w[:,:-1] - 1) ** 2 * (1 + 10 * torch.sin(torch.pi * w[:,:-1] + 1) ** 2)
        return torch.sum(tmp1.unsqueeze(1) + sum + tmp2.unsqueeze(1), dim=1)

    def bent_cigar_func(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sr_func(x)
        return x[:,0] ** 2 + torch.sum((10.0 ** 6) * x[:,1:] ** 2, dim=1)

    def hgbat_func(self, x: torch.Tensor) -> torch.Tensor:
        alpha = 1.0 / 4.0
        x = self.sr_func(x)
        tmp = x - 1
        r2 = torch.sum((tmp) ** 2, dim=1)
        sum_x = torch.sum(tmp, dim=1)
        return torch.abs(r2 ** 2 - sum_x ** 2) ** (2 * alpha) + (0.5 * r2 + sum_x) / x.size(1) + 0.5

    def rastrigin_func(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sr_func(x)
        return torch.sum(x ** 2 - 10.0 * torch.cos(2.0 * torch.pi * x) + 10.0, dim=1)

    def katsuura_func(self, x: torch.Tensor) -> torch.Tensor:
        nx = x.size(1)
        f = torch.ones(x.size(0), device=x.device)
        tmp3 = (10.0 * nx) ** 1.2
        x = self.sr_func(x)
        for i in range(nx):
            temp = torch.zeros(x.size(0), device=x.device)
            for j in range(1, 33):
                tmp1 = 2.0 ** j
                tmp2 = tmp1 * x[:,i]
                temp+= torch.abs(tmp2 - torch.floor(tmp2 + 0.5)) / tmp1
            f *= (1.0 + (i + 1) * temp) ** (10.0 / tmp3)
        return (f - 1) * (10.0 / (nx ** 2))

    def ackley_func(self, x: torch.Tensor) -> torch.Tensor:
        nx = x.size(1)
        x = self.sr_func(x)
        sum1 = torch.sum(x ** 2, dim = 1)
        sum2 = torch.sum(torch.cos(2.0 * torch.pi * x), dim=1)
        tmp1 = -0.2 * torch.sqrt(sum1 / nx)
        tmp2 = -torch.exp(sum2 / nx)
        return torch.e - 20.0 * torch.exp(tmp1) - torch.exp(tmp2) + 20.0

    def schwefel_func(self, x: torch.Tensor) -> torch.Tensor:
        nx = x.size(1)
        x = self.sr_func(x)
        tmp1 = x + 420.9687462275036
        tmp2 = -tmp1 * torch.sin(torch.sqrt(torch.abs(tmp1)))
        return torch.sum(tmp2, dim=1) + 418.98288727243378 * nx

    def schaffer_F7_func(self, x: torch.Tensor) -> torch.Tensor:
        nx = x.size(1)
        x = self.sr_func(x)
        f = torch.zeros(x.size(0), device=x.device)
        for i in range(0, nx-1):
            tmp1 = torch.pow(x[:,i]*x[:,i]+x[:,i+1]*x[:,i+1],0.5)
            tmp2 = torch.sin(50.0 * torch.pow(x[:,i],0.2))
            f += torch.pow(tmp1,0.5) * (1 + tmp2 ** 2) 
        f = (f / (nx-1)) ** 2
        return f
    
    def escaffer6_func(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sr_func(x)
        tmp1 = torch.sin(torch.sqrt(x[:,:-1] ** 2 + x[:,1:] ** 2)) ** 2
        tmp2 = 1.0 + 0.001 * (x[:,:-1] ** 2 + x[:,1:] ** 2)
        return torch.sum(0.5 + (tmp1 - 0.5) / (tmp2 ** 2), dim=1)

    def happycat_func(self, x: torch.Tensor) -> torch.Tensor:
        nx = x.size(1)
        alpha = 1.0 / 8.0
        x = self.sr_func(x)
        tmp = x-1
        r2_ = torch.sum(tmp ** 2, dim=1)
        sum_x = torch.sum(tmp, dim=1)
        return torch.abs(r2_ - nx) ** (2 * alpha) + (0.5 * r2_ + sum_x) / nx + 0.5

    def grie_rosen_func(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.sr_func(x)  
        tmp = x + 1
        tmp2_ = 100.0 * (tmp[:,:-1] ** 2 - tmp[:,1:]) ** 2 + (tmp[:,:-1] - 1.0) ** 2
        return torch.sum((tmp2_ ** 2) / 4000.0 - torch.cos(tmp2_) + 1.0, dim =1)

    def griewank_func(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sr_func(x) 
        sum_sq = torch.sum(x ** 2, dim=1)
        idx = torch.arange(1, x.size(1) + 1, device=x.device)
        prod_cos = torch.prod(torch.cos(x / torch.sqrt(idx)))
        return 1.0 + sum_sq / 4000.0 - prod_cos

    def rosenbrock_func(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sr_func(x)
        tmp = x + 1
        return torch.sum(100.0 * (tmp[:,:-1] ** 2 - tmp[:,1:]) ** 2 + (tmp[:,:-1] - 1.0) ** 2, dim=1)

    def discus_func(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sr_func(x)
        return (10.0 ** 6) * x[:,0] ** 2 + torch.sum(x[:,1:] ** 2, dim=1)

    def ellips_func(self, x: torch.Tensor) -> torch.Tensor:
        nx = x.size(1)
        x = self.sr_func(x)
        idx = torch.arange(nx, device=x.device)
        powers = 6.0 * idx / (nx - 1)
        return torch.sum((10.0 ** powers) * x ** 2, dim=1)

    def cf_cal(self, x: torch.Tensor, fit: torch.Tensor, delta: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        nx = x.size(1)
        cf_num = fit.size(1)
        w = torch.zeros(x.size(0), cf_num, device=x.device)
        for i in range(cf_num):
            block_start = i * nx
            block_end = (i + 1) * nx
            diff = x - self.OShift[:,block_start:block_end]
            w[:,i] = torch.sum(diff ** 2)
        w = torch.where(w!=0,(1.0 / torch.sqrt(w)) * torch.exp(-w / (2.0 * nx * delta[i] ** 2)),torch.inf)
        
        # w_max, _ = torch.max(w, dim = 1)
        w_sum = torch.sum(w, dim = 1)
        # if w_max == 0:
        #     w = torch.ones(x.size(0),cf_num, device=x.device)
        #     w_sum = torch.empty(x.size(0), device=x.device).fill(cf_num)

        return torch.sum(w / w_sum.unsqueeze(1) * (fit + bias.unsqueeze(0)), dim = 1)

    def evaluate(self, pop: torch.Tensor) -> torch.Tensor:
        if pop.size(1) != self.nx:
            raise ValueError(f"Dimension mismatch! Expect {self.nx}, got {pop.size(1)}.")

        if self.func_num == 1:
            fitness = self.cec2022_f1(pop)
        elif self.func_num == 2:
            fitness = self.cec2022_f2(pop)
        elif self.func_num == 3:
            fitness = self.cec2022_f3(pop)
        elif self.func_num == 4:
            fitness = self.cec2022_f4(pop)
        elif self.func_num == 5:
            fitness = self.cec2022_f5(pop)
        elif self.func_num == 6:
            fitness = self.cec2022_f6(pop)
        elif self.func_num == 7:
            fitness = self.cec2022_f7(pop)
        elif self.func_num == 8:
            fitness = self.cec2022_f8(pop)
        elif self.func_num == 9:
            fitness = self.cec2022_f9(pop)
        elif self.func_num == 10:
            fitness = self.cec2022_f10(pop)
        elif self.func_num == 11:
            fitness = self.cec2022_f11(pop)
        elif self.func_num == 12:
            fitness = self.cec2022_f12(pop)
        else:
            raise ValueError(f"Function {self.func_num} is not defined.")
        
        return fitness        
