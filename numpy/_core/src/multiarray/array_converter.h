#ifndef NUMPY_CORE_SRC_MULTIARRAY_ARRAY_CONVERTER_H_
#define NUMPY_CORE_SRC_MULTIARRAY_ARRAY_CONVERTER_H_


#include "numpy/ndarraytypes.h"
#include "array_coercion.h"

NPY_NO_EXPORT int
init_array_converter_type(PyObject *module);

typedef enum {
    NPY_CH_ALL_SCALARS = 1 << 0,
    NPY_CH_ALL_PYSCALARS = 1 << 1,
} npy_array_converter_flags;


typedef struct {
    PyObject *object;
    PyArrayObject *array;
    PyArray_DTypeMeta *DType;
    PyArray_Descr *descr;
    int scalar_input;
    PyArray_Descr *inferred_descr;
    coercion_cache_obj *cache;
    npy_discovery_info discovery;
    int ndim;
    npy_intp *shape;
} creation_item;


typedef struct {
    PyObject_VAR_HEAD
    int narrs;
    /* store if all objects are scalars (unless zero objects) */
    npy_array_converter_flags flags;
    /* __array_wrap__ cache */
    PyObject *wrap;
    PyObject *wrap_type;
    creation_item items[];
}  PyArrayArrayConverterObject;


/* operands is a tuple; return a new tuple of arrays with literal flags. */
NPY_NO_EXPORT PyObject *
npy_convert_operands(PyObject *operands, int with_context, int strict_strings);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_ARRAY_CONVERTER_H_ */
