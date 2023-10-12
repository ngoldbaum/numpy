from numpy._core import shape_base

_globals = globals()

for item in shape_base.__dir__():
    _globals[item] = getattr(shape_base, item)
