from numpy._core import getlimits
from ._utils import _raise_warning

_raise_warning("*")

_globals = globals()

for item in getlimits.__dir__():
    _globals[item] = getattr(getlimits, item)
