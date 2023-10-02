from numpy._core import shape_base
from ._utils import _raise_warning

_raise_warning("*")

_globals = globals()

for item in shape_base.__dir__():
    _globals[item] = getattr(shape_base, item)
