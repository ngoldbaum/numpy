#ifndef NUMPY_CORE_SRC_COMMON_NPY_NPY_HASHTABLE_H_
#define NUMPY_CORE_SRC_COMMON_NPY_NPY_HASHTABLE_H_

#include <Python.h>
#ifdef Py_GIL_DISABLED
#ifndef Py_BUILD_CORE
#define Py_BUILD_CORE 1
#endif
#include "internal/pycore_lock.h"
#endif

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"


typedef struct {
    int key_len;  /* number of identities used */
    /* Buckets stores: val1, key1[0], key1[1], ..., val2, key2[0], ... */
    PyObject **buckets;
    npy_intp size;  /* current size */
    npy_intp nelem;  /* number of elements */
#ifdef Py_GIL_DISABLED
    PyMutex mutex;
#endif
} PyArrayIdentityHash;


NPY_NO_EXPORT int
PyArrayIdentityHash_SetItem(PyArrayIdentityHash *tb,
        PyObject *const *key, PyObject *value, int replace);

NPY_NO_EXPORT PyObject *
PyArrayIdentityHash_GetItem(PyArrayIdentityHash const *tb, PyObject *const *key);

NPY_NO_EXPORT PyArrayIdentityHash *
PyArrayIdentityHash_New(int key_len);

NPY_NO_EXPORT void
PyArrayIdentityHash_Dealloc(PyArrayIdentityHash *tb);

#endif  /* NUMPY_CORE_SRC_COMMON_NPY_NPY_HASHTABLE_H_ */
