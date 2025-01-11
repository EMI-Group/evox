import torch

from ...utils import minimum

def de_diff_sum( diff_padding_num: int, num_diff_vects: int, index: torch.Tensor, population: torch.Tensor, replace: bool = False ) -> torch.Tensor:
    device = population.device
    pop_size, dim = population.size()
    select_len = num_diff_vects * 2 + 1
    rand_indices = torch.randint(0, pop_size, (pop_size,diff_padding_num), device=device)
    rand_indices = torch.where( rand_indices == index.unsqueeze(1), torch.tensor(pop_size-1, device=device), rand_indices)
    
    pop_permut = population[rand_indices]
    mask = torch.arange( diff_padding_num, device=population.device ) < select_len
    pop_permut_padding = torch.where( mask.unsqueeze(1), pop_permut, torch.zeros_like(pop_permut))

    diff_vects = pop_permut_padding[:,1:]

    even_idx = torch.arange(0, diff_vects.size(1), 2, device=device)
    odd_idx  = torch.arange(1, diff_vects.size(1), 2, device=device)

    difference_sum = diff_vects[:,even_idx].sum(dim=1) - diff_vects[:,odd_idx].sum(dim=1)
    return difference_sum

def de_bin_cross( mutation_vector: torch.Tensor, current_vect: torch.Tensor, CR: torch.Tensor ):
    device = mutation_vector.device
    pop_size, dim = mutation_vector.size()
    mask = torch.randn( pop_size, dim, device = device ) < CR.unsqueeze( 1 )
    rind = torch.randint( 0, dim, ( pop_size,), device=device ).unsqueeze(0)
    mask [ torch.cat( [ torch.arange( pop_size, device=device ).unsqueeze(0), rind ], dim = 0 ) ] = True
    trial_vector = torch.where( mask, mutation_vector, current_vect )
    return trial_vector

def de_arith_recom( mutation_vector: torch.Tensor, current_vect: torch.Tensor, K: float ):
    trial_vector = current_vect + K * (mutation_vector - current_vect)
    return trial_vector