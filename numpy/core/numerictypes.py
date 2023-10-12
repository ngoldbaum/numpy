from numpy._core import numerictypes

_globals = globals()

for item in numerictypes.__dir__():
    _globals[item] = getattr(numerictypes, item)
