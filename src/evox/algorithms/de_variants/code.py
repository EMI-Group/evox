import torch

from ...core import Algorithm, Mutable, Parameter, jit_class
from ...utils import clamp

from ...operators.selection import select_rand_pbest
from ...operators.crossover import (
    de_diff_sum,
    de_bin_cross,
    de_exp_cross,
    de_arith_recom,
)

"""
Strategy codes(4 bits): [basevect_prim, basevect_sec, diff_num, cross_strategy]
basevect      : 0="rand", 1="best", 2="pbest", 3="current"
cross_strategy: 0=bin   , 1=exp   , 2=arith
"""

rand_1_bin = [0, 0, 1, 0]
rand_2_bin = [0, 0, 2, 0]
current2rand_1 = [0, 0, 1, 2]  # current2rand_1 <==> rand_1_arith
rand2best_2_bin = [0, 1, 2, 0]
current2pbest_1_bin = [3, 2, 1, 0]


@jit_class
class CoDE(Algorithm):
    def __init__(
        self,
        pop_size: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        diff_padding_num: int = 5,
        param_pool: torch.Tensor = torch.tensor([[1, 0.1], [1, 0.9], [0.8, 0.2]]),
        replace: bool = False,
        device: torch.device | None = None,
    ):
        super().__init__()
        device = torch.get_default_device() if device is None else device
        
        dim = lb.shape[0]
        
        # parameters
        self.param_pool = Parameter( param_pool, device=device )
        # set value
        lb = lb[ None, : ].to( device=device )
        ub = ub[ None, : ].to( device=device )
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.replace = replace
        self.pop_size = pop_size
        self.diff_padding_num = diff_padding_num
        self.strategies = torch.tensor( [rand_1_bin, rand_2_bin, current2rand_1], device=device )
        # setup
        self.best_index = Mutable( torch.tensor(0, device=device) )
        self.population = Mutable( torch.randn( pop_size, dim, device=device ) * (ub - lb) + lb )
        self.fitness    = Mutable( torch.full( (self.pop_size,), fill_value=torch.inf, device=device ) )
        
    def step(self):
        device = self.population.device
        indices = torch.arange(self.pop_size, device=device)

        param_ids = torch.randint( 0, 3, ( 3, self.pop_size ), device=device )
        
        basevect_prim_type = self.strategies[:,0]
        basevect_sec_type = self.strategies[:,1]
        num_diff_vects = self.strategies[:,2]
        cross_strategy = self.strategies[:,3]

        params = self.param_pool[param_ids]
        differential_weight = params[:,:,0]
        cross_probability = params[:,:,1]
        
        trial_vectors = torch.zeros( ( 3, self.pop_size, self.dim ), device=device )
        
        for i in range(3) :
            difference_sum, rand_vect_idx = de_diff_sum( self.diff_padding_num, torch.tensor(num_diff_vects[i]), indices, self.population, self.replace )
        
            rand_vect    = self.population[rand_vect_idx]
            best_vect    = torch.tile( self.population[self.best_index].unsqueeze(0), (self.pop_size,1) )
            pbest_vect   = select_rand_pbest(0.05, self.population, self.fitness)
            current_vect = self.population[indices]
        
            vector_merge = torch.stack((rand_vect, best_vect, pbest_vect, current_vect))
            base_vector_prim = vector_merge[basevect_prim_type[i]]
            base_vector_sec  = vector_merge[basevect_sec_type [i]]
            
            base_vector = base_vector_prim + differential_weight[i].unsqueeze(1) * (base_vector_sec - base_vector_prim)
            mutation_vector = base_vector + difference_sum * differential_weight[i].unsqueeze(1)

            trial_vector = torch.zeros( self.pop_size, self.dim, device=device )
            trial_vector = torch.where( cross_strategy[i] == 0, de_bin_cross  ( mutation_vector, current_vect, cross_probability[i] ), trial_vector )
            trial_vector = torch.where( cross_strategy[i] == 1, de_exp_cross  ( mutation_vector, current_vect, cross_probability[i] ), trial_vector )
            trial_vector = torch.where( cross_strategy[i] == 2, de_arith_recom( mutation_vector, current_vect, cross_probability[i] ), trial_vector )
            trial_vectors = torch.where( (torch.arange( 3, device=device) == i).unsqueeze(1).unsqueeze(2), trial_vector.unsqueeze(0), trial_vectors )
        
        trial_vectors = clamp( trial_vectors.reshape( 3 * self.pop_size, self.dim ), self.lb, self.ub )
        trial_fitness = self.evaluate( trial_vectors )

        indices = torch.arange(3 * self.pop_size, device=device).reshape(3, self.pop_size)
        trans_fit = trial_fitness[indices]
        
        min_indices = torch.argmin(trans_fit, dim=0)
        min_indices_global = indices[min_indices, torch.arange(self.pop_size, device=device)]
        
        trial_fitness_select = trial_fitness[min_indices_global]
        trial_vectors_select = trial_vectors[min_indices_global]

        compare = trial_fitness_select <= self.fitness

        self.population = torch.where(compare[:, torch.newaxis], trial_vectors_select, self.population)
        self.fitness    = torch.where(compare, trial_fitness_select, self.fitness)
        self.best_index = torch.argmin(self.fitness)
