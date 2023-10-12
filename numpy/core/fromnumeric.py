from numpy._core import fromnumeric

_globals = globals()

for item in fromnumeric.__dir__():
    _globals[item] = getattr(fromnumeric, item)
