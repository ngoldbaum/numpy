from numpy._core import arrayprint
from ._utils import _raise_warning

_raise_warning("*")

_globals = globals()

for item in arrayprint.__dir__():
    _globals[item] = getattr(arrayprint, item)
