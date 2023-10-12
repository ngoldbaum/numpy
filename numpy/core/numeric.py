from numpy._core import numeric

_globals = globals()

for item in numeric.__dir__():
    _globals[item] = getattr(numeric, item)
