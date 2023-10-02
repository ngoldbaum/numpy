from numpy._core import records
from ._utils import _raise_warning

_raise_warning("*")

_globals = globals()

for item in records.__dir__():
    _globals[item] = getattr(records, item)
