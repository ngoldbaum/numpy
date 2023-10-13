from numpy._core import _internal, ndarray


# Build a new array from the information in a pickle.
# Note that the name numpy.core._internal._reconstruct is embedded in
# pickles of ndarrays made with NumPy before release 1.0
# so don't remove the name here, or you'll
# break backward compatibility.
def _reconstruct(subtype, shape, dtype):
    return ndarray.__new__(subtype, shape, dtype)


_globals = globals()

for item in _internal.__dir__():
    _globals[item] = getattr(_internal, item)
