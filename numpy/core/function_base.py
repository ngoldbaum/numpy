from numpy._core import function_base
from ._utils import _raise_warning

_raise_warning("*")

_globals = globals()

for item in function_base.__dir__():
    _globals[item] = getattr(function_base, item)
