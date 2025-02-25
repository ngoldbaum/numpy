#ifndef NUMPY_CORE_SRC_COMMON_NPY_PYCOMPAT_H_
#define NUMPY_CORE_SRC_COMMON_NPY_PYCOMPAT_H_

#include "numpy/npy_3kcompat.h"
#include "pythoncapi-compat/pythoncapi_compat.h"

#define Npy_HashDouble _Py_HashDouble

#if Py_GIL_DISABLED
// adapted from CPython's "internal/pycore_critical_section.h"
//
// Specialized version of critical section locking to safely use
// PySequence_Fast APIs without the GIL. For performance, the argument *to*
// PySequence_Fast() is provided to the macro, not the *result* of
// PySequence_Fast(), which would require an extra test to determine if the
// lock must be acquired.
#define NPY_BEGIN_CRITICAL_SECTION_SEQUENCE_FAST(original)         \
    {                                                              \
        PyObject *_orig_seq = (PyObject *)(original);              \
        const int _should_lock_cs = PyList_CheckExact(_orig_seq);  \
        PyCriticalSection _cs;                                     \
        if (_should_lock_cs) {                                     \
            PyCriticalSection_Begin(&_cs, _orig_seq);              \
        }

#define NPY_END_CRITICAL_SECTION_SEQUENCE_FAST()                   \
        if (_should_lock_cs) {                                     \
            PyCriticalSection_End(&_cs);                           \
        }                                                          \
    }

#else
#define NPy_BEGIN_CRITICAL_SECTION_SEQUENCE_FAST(original) {
#define NPy_END_CRITICAL_SECTION_SEQUENCE_FAST() }
#endif

#endif  /* NUMPY_CORE_SRC_COMMON_NPY_PYCOMPAT_H_ */
