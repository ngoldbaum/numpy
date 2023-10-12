from numpy._core import records

_globals = globals()

for item in records.__dir__():
    _globals[item] = getattr(records, item)
