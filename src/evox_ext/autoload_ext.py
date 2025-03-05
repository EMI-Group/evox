import importlib
import pkgutil


def iter_namespace(ns_pkg):
    # Specifying the second argument (prefix) to iter_modules makes the
    # returned name an absolute name instead of a relative one. This allows
    # import_module to work without having to do additional modification to
    # the name.
    return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")


def load_extension(package, exposed_module):
    discovered_plugins = {name: importlib.import_module(name) for finder, name, ispkg in iter_namespace(package)}

    for name, external_module in discovered_plugins.items():
        # Find all classes inside the module and add them to evox.algorithms
        for attr_name in dir(external_module):
            attr = getattr(external_module, attr_name)
            # add to the exposed_module
            setattr(exposed_module, attr_name, attr)
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
