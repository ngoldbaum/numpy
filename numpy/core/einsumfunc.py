from numpy._core import einsumfunc
from ._utils import _raise_warning

_raise_warning("*")

_globals = globals()

for item in einsumfunc.__dir__():
    _globals[item] = getattr(einsumfunc, item)
