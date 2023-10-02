from numpy._core import overrides
from ._utils import _raise_warning

_raise_warning("*")

_globals = globals()

for item in overrides.__dir__():
    _globals[item] = getattr(overrides, item)
