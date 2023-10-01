"""
The `numpy.core` submodule exists solely for backward compatibility
purposes. The original `core` was renamed to `_core` and made private.
`numpy.core` will be removed in the future.
"""
from numpy import _core
from ._utils import _raise_warning


def __getattr__(attr_name):

    attr = getattr(_core, attr_name)
    _raise_warning(attr_name)
    return attr
