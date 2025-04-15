import importlib
import inspect
import pkgutil
import types


def iter_namespace(ns_pkg):
    # Specifying the second argument (prefix) to iter_modules makes the
    # returned name an absolute name instead of a relative one. This allows
    # import_module to work without having to do additional modification to
    # the name.
    return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")


def load_extension(package, exposed_module):
    discovered_plugins = {name: importlib.import_module(name) for finder, name, ispkg in iter_namespace(package)}

    for name, external_module in discovered_plugins.items():
        module_name = external_module.__name__.split(".")[-1]
        # Find all classes inside the module and add them to evox.algorithms
        # if the package already existed in the exposed_module, recursively merge the two packages
        if module_name in exposed_module.__dict__:
            # if the attribute is a module, recursively load its contents
            sub_module = exposed_module.__dict__[module_name]
            if isinstance(sub_module, types.ModuleType):
                load_extension(external_module, sub_module)
        else:
            # directly add it to the exposed_module
            setattr(exposed_module, module_name, external_module)
            # if __all__ is not defined, create it
            if not hasattr(exposed_module, "__all__"):
                exposed_module.__all__ = []
            exposed_module.__all__.append(name)

    for attr_name in dir(package):
        attr = getattr(package, attr_name)
        # if the attribute is a class or function, add it to the exposed_module
        if inspect.isclass(attr) or inspect.isfunction(attr):
            # add to the exposed_module
            setattr(exposed_module, attr_name, attr)
            # if __all__ is not defined, create it
            if not hasattr(exposed_module, "__all__"):
                exposed_module.__all__ = []
            exposed_module.__all__.append(attr_name)


def auto_load_extensions():
    try:
        import evox.utils
        import evox_ext.utils

        load_extension(evox_ext.utils, evox.utils)
    except ImportError:
        pass

    try:
        import evox.algorithms
        import evox_ext.algorithms

        load_extension(evox_ext.algorithms, evox.algorithms)
    except ImportError:
        pass

    try:
        import evox.problems
        import evox_ext.problems

        load_extension(evox_ext.problems, evox.problems)
    except ImportError:
        pass

    try:
        import evox.operators
        import evox_ext.operators

        load_extension(evox_ext.operators, evox.operators)
    except ImportError:
        pass

    try:
        import evox.metrics
        import evox_ext.metrics

        load_extension(evox_ext.metrics, evox.metrics)

    except ImportError:
        pass
