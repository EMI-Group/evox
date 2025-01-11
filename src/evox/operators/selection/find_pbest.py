import torch

def select_rand_pbest( percent: float, population: torch.Tensor, fitness: torch.Tensor ) -> torch.Tensor:
    device = population.device
    pop_size = population.size(0)
    top_p_num = max( int(pop_size * percent), 1 )
    pbest_indices_pool = torch.argsort(fitness)[ :top_p_num]
    random_indices = torch.randint(0, top_p_num, (pop_size,), device=device)
    return population[ pbest_indices_pool[random_indices] ]
