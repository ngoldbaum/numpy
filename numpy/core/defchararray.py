from numpy._core import defchararray
from ._utils import _raise_warning

_raise_warning("*")

_globals = globals()

for item in defchararray.__dir__():
    _globals[item] = getattr(defchararray, item)
