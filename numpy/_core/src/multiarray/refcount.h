#ifndef NUMPY_CORE_SRC_MULTIARRAY_REFCOUNT_H_
#define NUMPY_CORE_SRC_MULTIARRAY_REFCOUNT_H_

#ifdef __cplusplus
extern "C" {
#endif

NPY_NO_EXPORT int
PyArray_ClearBuffer(
        PyArray_Descr *descr, char *data,
        npy_intp stride, npy_intp size, int aligned);

NPY_NO_EXPORT int
PyArray_ZeroContiguousBuffer(
        PyArray_Descr *descr, char *data,
        npy_intp stride, npy_intp size, int aligned);

NPY_NO_EXPORT int
PyArray_ClearArray(PyArrayObject *arr);

NPY_NO_EXPORT void
PyArray_Item_INCREF(char *data, PyArray_Descr *descr);

NPY_NO_EXPORT void
PyArray_Item_XDECREF(char *data, PyArray_Descr *descr);

NPY_NO_EXPORT int
PyArray_INCREF(PyArrayObject *mp);

NPY_NO_EXPORT int
PyArray_XDECREF(PyArrayObject *mp);

NPY_NO_EXPORT int
PyArray_SetObjectsToNone(PyArrayObject *arr);

#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_REFCOUNT_H_ */
