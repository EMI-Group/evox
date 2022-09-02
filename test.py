class MetaModule(type):
    """Meta class used by Module

    This meta class will try to wrap methods with use_state,
    which allows easy managing of states.

    It is recommended to use a single underscore as prefix to prevent a method from being wrapped.
    Still, this behavior can be configured by passing ``force_wrap``, ``ignore`` and ``ignore_prefix``.
    """

    def __new__(
        cls,
        name,
        bases,
        class_dict,
        force_wrap=["__call__"],
        ignore=["init", "setup"],
        ignore_prefix="_",
    ):
        print("Metaclass", name)
        wrapped = {}

        for key, value in class_dict.items():
            if key.startswith(ignore_prefix) or key in ignore:
                print("not wrap")
                wrapped[key] = value
            elif callable(value):
                print("wrap")
                wrapped[key] = use_state(value)
        return super().__new__(cls, name, bases, wrapped)

class A(metaclass=MetaModule):
    pass

class B(A):
    pass

class C(B):
    pass

print("end")

C()