from numpy._core import fromnumeric
from ._utils import _raise_warning

_raise_warning("*")

_globals = globals()

for item in fromnumeric.__dir__():
    _globals[item] = getattr(fromnumeric, item)
