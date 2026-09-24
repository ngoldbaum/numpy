#ifndef NUMPY_CORE_SRC_MULTIARRAY_ARRAY_COERCION_H_
#define NUMPY_CORE_SRC_MULTIARRAY_ARRAY_COERCION_H_


/*
 * We do not want to coerce arrays many times unless absolutely necessary.
 * The same goes for sequences, so everything we have seen, we will have
 * to store somehow. This is a linked list of these objects.
 */
typedef struct coercion_cache_obj {
    PyObject *converted_obj;
    PyObject *arr_or_sequence;
    struct coercion_cache_obj *next;
    npy_bool sequence;
    int depth;  /* the dimension at which this object was found. */
} coercion_cache_obj;

typedef enum {
    NPY_DISCOVERY_BYTES = 1,
    NPY_DISCOVERY_TEXT = 2,
    NPY_DISCOVERY_OTHER = 4,
} npy_discovery_kind;

/* Opt-in provenance for callers which defer array materialization. */
typedef struct {
    unsigned int kinds;
    PyObject *scalars;  /* owned list of scalar leaves, in discovery order */
    npy_bool has_array;
} npy_discovery_info;

NPY_NO_EXPORT unsigned int
npy_discovery_kind_from_dtype(PyArray_DTypeMeta *dtype);

static inline unsigned int
npy_discovery_kind_from_descr(PyArray_Descr *descr)
{
    return npy_discovery_kind_from_dtype(NPY_DTYPE(descr));
}

NPY_NO_EXPORT int
PyArray_DiscoverDTypeAndShapeWithInfo(
        PyObject *obj, int max_dims, npy_intp out_shape[NPY_MAXDIMS],
        coercion_cache_obj **coercion_cache,
        PyArray_DTypeMeta *fixed_DType, PyArray_Descr *requested_descr,
        PyArray_Descr **out_descr, int copy, int *was_copied_by__array__,
        npy_discovery_info *info);

NPY_NO_EXPORT coercion_cache_obj *
npy_clone_coercion_cache(coercion_cache_obj *cache);

NPY_NO_EXPORT int
_PyArray_MapPyTypeToDType(
        PyArray_DTypeMeta *DType, PyTypeObject *pytype, npy_bool userdef);

NPY_NO_EXPORT PyObject *
PyArray_DiscoverDTypeFromScalarType(PyTypeObject *pytype);

NPY_NO_EXPORT int
npy_cast_raw_scalar_item(
        PyArray_Descr *from_descr, char *from_item,
        PyArray_Descr *to_descr, char *to_item);

NPY_NO_EXPORT int
PyArray_Pack(PyArray_Descr *descr, void *item, PyObject *value);

NPY_NO_EXPORT PyArray_Descr *
PyArray_AdaptDescriptorToArray(
        PyArrayObject *arr, PyArray_DTypeMeta *dtype, PyArray_Descr *descr);

NPY_NO_EXPORT int
PyArray_DiscoverDTypeAndShape(
        PyObject *obj, int max_dims,
        npy_intp out_shape[NPY_MAXDIMS],
        coercion_cache_obj **coercion_cache,
        PyArray_DTypeMeta *fixed_DType, PyArray_Descr *requested_descr,
        PyArray_Descr **out_descr, int copy, int *was_copied_by__array__);

NPY_NO_EXPORT PyObject *
_discover_array_parameters(PyObject *NPY_UNUSED(self),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames);

/* Would make sense to inline the freeing functions everywhere */
/* Frees the coercion cache object recursively. */
NPY_NO_EXPORT void
npy_free_coercion_cache(coercion_cache_obj *first);

/* unlink a single item and return the next */
NPY_NO_EXPORT coercion_cache_obj *
npy_unlink_coercion_cache(coercion_cache_obj *current);

NPY_NO_EXPORT int
PyArray_AssignFromCache(PyArrayObject *self, coercion_cache_obj *cache);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_ARRAY_COERCION_H_ */
