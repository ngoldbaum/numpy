from numpy import _core
from ._utils import _raise_warning


def __getattr__(attr):

    _raise_warning(attr)
    return getattr(_core, attr)
