from types import MethodType


def import_module_error_func(module_name):
    """When a function is imported incorrectly due to a missing module, raise
    an import error when the function is called."""

    def decorate(func):

        def new_func(*args, **kwargs):
            raise ImportError(
                f'Please install {module_name} to use {func.__name__}.')

        return new_func

    return decorate


def import_module_error_class(module_name):
    """When a class is imported incorrectly due to a missing module, raise an
    import error when the class is instantiated."""

    def decorate(cls):

        def import_error_init(*args, **kwargs):
            raise ImportError(
                f'Please install {module_name} to use {cls.__name__}.')

        cls.__init__ = MethodType(import_error_init, cls)
        return cls

    return decorate
