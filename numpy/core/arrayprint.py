from numpy._core import arrayprint

_globals = globals()

for item in arrayprint.__dir__():
    _globals[item] = getattr(arrayprint, item)
