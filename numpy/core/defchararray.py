from numpy._core import defchararray

_globals = globals()

for item in defchararray.__dir__():
    _globals[item] = getattr(defchararray, item)
