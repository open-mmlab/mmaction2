# Copyright (c) OpenMMLab. All rights reserved.
import pytest

from mmaction.utils import import_module_error_class, import_module_error_func


def test_import_module_error_class():

    @import_module_error_class('mmdet')
    class ExampleClass:
        pass

    with pytest.raises(ImportError):
        ExampleClass()

    @import_module_error_class('mmdet')
    class ExampleClass:

        def __init__(self, a, b=3):
            self.c = a + b

    with pytest.raises(ImportError):
        ExampleClass(4)


def test_import_module_error_func():

    @import_module_error_func('_add')
    def ExampleFunc(a, b):
        return a + b

    with pytest.raises(ImportError):
        ExampleFunc(3, 4)
