try:
    from .brax import Brax
except ImportError as e:
    original_brax_error_msg = str(e)

    def Brax(*args, **kwargs):
        raise ImportError(
            f'Brax requires brax but got "{original_brax_error_msg}" when importing'
        )


try:
    # optional dependency: gym
    from .gym import Gym
except ImportError as e:
    original_gym_error_msg = str(e)

    def Gym(*args, **kwargs):
        raise ImportError(
            f'Gym requires gym, ray but got "{original_gym_error_msg}" when importing'
        )


try:
    # optional dependency: gym
    from .env_pool import EnvPool
except ImportError as e:
    original_envpool_error_msg = str(e)

    def EnvPool(*args, **kwargs):
        raise ImportError(
            f'EnvPool requires env_pool but got "{original_envpool_error_msg}" when importing'
        )


try:
    # optional dependency: gym
    from .neruoevobench import NeuroEvoBench
except ImportError as e:
    original_neruoevobench_error_msg = str(e)

    def NeuroEvoBench(*args, **kwargs):
        raise ImportError(
            f'NeuroEvoBench requires neruoevobench but got "{original_neruoevobench_error_msg}" when importing'
        )
