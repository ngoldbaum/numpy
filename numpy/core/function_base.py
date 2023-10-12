from numpy._core import function_base

_globals = globals()

for item in function_base.__dir__():
    _globals[item] = getattr(function_base, item)
