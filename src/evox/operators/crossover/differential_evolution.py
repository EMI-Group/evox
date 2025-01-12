import torch

from ...utils import minimum
from ...core import vmap

from typing import Tuple

def de_diff_sum( diff_padding_num: int, num_diff_vects: torch.Tensor, index: torch.Tensor, population: torch.Tensor, replace: bool = False ) -> Tuple[ torch.Tensor, torch.Tensor ]:
    device = population.device
    pop_size, dim = population.size()
    if  len(num_diff_vects.size()) == 0 :
        num_diff_vects = torch.tile( num_diff_vects, (pop_size,) )

    select_len = num_diff_vects.unsqueeze(1) * 2 + 1
    rand_indices = torch.randint(0, pop_size, (pop_size,diff_padding_num), device=device)
    rand_indices = torch.where( rand_indices == index.unsqueeze(1), torch.tensor(pop_size-1, device=device), rand_indices)
    
    pop_permut = population[rand_indices]
    mask = torch.tile( torch.arange( diff_padding_num, device=device ), (pop_size,1) ) < select_len
    pop_permut_padding = torch.where( mask.unsqueeze(2), pop_permut, torch.zeros_like(pop_permut) )

    diff_vects = pop_permut_padding[:,1:]

    even_idx = torch.arange(0, diff_vects.size(1), 2, device=device)
    odd_idx  = torch.arange(1, diff_vects.size(1), 2, device=device)

    difference_sum = diff_vects[ :,even_idx].sum(dim=1) - diff_vects[ :,odd_idx].sum(dim=1)
    return difference_sum, rand_indices[:,0]

def de_bin_cross( mutation_vector: torch.Tensor, current_vect: torch.Tensor, CR: torch.Tensor ):
    device = mutation_vector.device
    pop_size, dim = mutation_vector.size()
    if  len( CR.size()) == 1 :
        CR = CR.unsqueeze( 1 )
    mask = torch.randn( pop_size, dim, device=device) <  CR
    rind = torch.randint( 0, dim, (pop_size,), device=device).unsqueeze(1)
    jind = torch.arange ( dim, device = device).unsqueeze( 0 ) == rind
    trial_vector = torch.where( torch.logical_or( mask, jind ), mutation_vector, current_vect )
    return trial_vector

def de_exp_cross( mutation_vector: torch.Tensor, current_vect: torch.Tensor, CR: torch.Tensor ):
    device = mutation_vector.device
    pop_size, dim = mutation_vector.size()
    n = torch.randint( 0, dim, ( pop_size, ), device=device )
    l = torch.randint( 0, dim, ( pop_size, ), device=device )
    mask = torch.arange(dim).unsqueeze(0) < ( minimum( l,torch.tensor(dim))-1).unsqueeze(1)
    mask = torch.gather( torch.tile( mask, (1,2) ), 1, n.unsqueeze(1) + torch.arange(dim) )
    trial_vector = torch.where( mask, mutation_vector, current_vect )
    return trial_vector

def de_arith_recom( mutation_vector: torch.Tensor, current_vect: torch.Tensor, K: torch.Tensor ):
    if  len(K.size()) == 1 :
        K = K.unsqueeze( 1 )
    trial_vector = current_vect + K * (mutation_vector - current_vect)
    return trial_vector
