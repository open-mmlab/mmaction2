from types import MethodType


def import_module_error_class(module_name):
    """When a class is imported incorrectly due to a missing module, raise an
    import error when the class is instantiated."""

    def decorate(cls):

        def __init__(self, **kwargs):
            raise ImportError(
                f'Please install {module_name} to use {self.__name__}.')

        cls.__init__ = MethodType(__init__, cls)
        return cls

    return decorate


def import_module_error_func(module_name):
    """When a function is imported incorrectly due to a missing module, raise
    an import error when the function is called."""

    def decorate(func):

        def new_func(*args, **kwargs):
            raise ImportError(
                f'Please install {module_name} to use {func.__name__}.')
            return func(*args, **kwargs)

        return new_func

    return decorate
