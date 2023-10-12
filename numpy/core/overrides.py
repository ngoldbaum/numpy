from numpy._core import overrides

_globals = globals()

for item in overrides.__dir__():
    _globals[item] = getattr(overrides, item)
