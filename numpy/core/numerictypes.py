from numpy._core import numerictypes
from ._utils import _raise_warning

_raise_warning("*")

_globals = globals()

for item in numerictypes.__dir__():
    _globals[item] = getattr(numerictypes, item)
