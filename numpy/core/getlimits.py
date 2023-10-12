from numpy._core import getlimits

_globals = globals()

for item in getlimits.__dir__():
    _globals[item] = getattr(getlimits, item)
