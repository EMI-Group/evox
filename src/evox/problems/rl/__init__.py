try:
    from .brax import Brax
except ImportError as e:
    original_erorr_msg = str(e)

    def Brax(*args, **kwargs):
        raise ImportError(
            f'Brax requires brax, ray but got "{original_erorr_msg}" when importing'
        )

try:
    # optional dependency: gym
    from .gym import Gym
except ImportError as e:
    original_erorr_msg = str(e)

    def Gym(*args, **kwargs):
        raise ImportError(
            f'Gym requires gym, ray but got "{original_erorr_msg}" when importing'
        )