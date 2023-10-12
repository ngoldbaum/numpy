from numpy._core import einsumfunc

_globals = globals()

for item in einsumfunc.__dir__():
    _globals[item] = getattr(einsumfunc, item)
