try:
    from .evoxbench import C10MOP, CitySegMOP, IN1kMOP
except ImportError:

    def C10MOP(*args, **kwargs):
        raise ImportError("evoxbench is not installed")

    def CitySegMOP(*args, **kwargs):
        raise ImportError("evoxbench is not installed")

    def IN1kMOP(*args, **kwargs):
        raise ImportError("evoxbench is not installed")
