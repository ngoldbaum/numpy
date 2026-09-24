/*
 * This file defines an _array_converter object used internally in NumPy to
 * deal with `__array_wrap__` and `result_type()` for multiple arguments
 * where converting inputs to arrays would lose the necessary information.
 *
 * The helper thus replaces many asanyarray/asarray calls.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <limits.h>
#include <structmember.h>

#include "numpy/arrayobject.h"
#include "arrayobject.h"
#include "array_converter.h"
#include "array_coercion.h"
#include "arraywrap.h"
#include "numpy/arrayscalars.h"
#include "npy_argparse.h"
#include "abstractdtypes.h"
#include "alloc.h"
#include "convert_datatype.h"
#include "descriptor.h"
#include "dtypemeta.h"
#include "npy_static_data.h"
#include "module_state.h"
#include "ctors.h"

#include "npy_config.h"


#include "array_assign.h"

#include "common.h"
#include "get_attr_string.h"



static PyObject *
array_converter_new(
        PyTypeObject *cls, PyObject *args, PyObject *kwds)
{
    if (kwds != NULL && PyDict_GET_SIZE(kwds) != 0) {
        PyErr_SetString(PyExc_TypeError,
                "Array creation helper doesn't support keywords.");
        return NULL;
    }

    Py_ssize_t narrs_ssize_t = (args == NULL) ? 0 : PyTuple_GET_SIZE(args);
    int narrs = (int)narrs_ssize_t;
    /* result_type() may append one extra dtype. */
    if (narrs_ssize_t >= INT_MAX) {
        PyErr_SetString(PyExc_RuntimeError,
            "too many arrays.");
        return NULL;
    }

    PyArrayArrayConverterObject *self = (PyArrayArrayConverterObject *)
            PyType_GenericAlloc(cls, narrs);
    if (self == NULL) {
        return NULL;
    }

    self->narrs = 0;
    self->flags = 0;
    self->wrap = NULL;
    self->wrap_type = NULL;

    if (narrs == 0) {
        return (PyObject *)self;
    }
    self->flags = (NPY_CH_ALL_PYSCALARS | NPY_CH_ALL_SCALARS);

    creation_item *item = self->items;
    for (int i = 0; i < narrs; i++, item++) {
        item->object = Py_NewRef(PyTuple_GET_ITEM(args, i));
        self->narrs++;  /* all fields are zero-initialized for cleanup */

        if (PyArray_Check(item->object)) {
            item->array = (PyArrayObject *)Py_NewRef(item->object);
            item->inferred_descr = (PyArray_Descr *)Py_NewRef(
                    PyArray_DESCR(item->array));
            item->discovery.has_array = NPY_TRUE;
            item->discovery.kinds = npy_discovery_kind_from_descr(
                    item->inferred_descr);
        }
        else {
            int was_copied = 0;
            npy_intp shape[NPY_MAXDIMS];
            item->ndim = PyArray_DiscoverDTypeAndShapeWithInfo(
                    item->object, NPY_MAXDIMS, shape, &item->cache,
                    NULL, NULL, &item->inferred_descr, -1, &was_copied,
                    &item->discovery);
            if (item->ndim < 0) {
                goto fail;
            }
            if (item->ndim > 0) {
                item->shape = PyMem_Malloc(item->ndim * sizeof(npy_intp));
                if (item->shape == NULL) {
                    PyErr_NoMemory();
                    goto fail;
                }
                memcpy(item->shape, shape, item->ndim * sizeof(npy_intp));
            }
            item->scalar_input = (item->cache == NULL);
            if (item->inferred_descr == NULL) {
                item->inferred_descr = PyArray_DescrFromType(NPY_DEFAULT_TYPE);
                if (item->inferred_descr == NULL) {
                    goto fail;
                }
            }
        }
        item->DType = NPY_DTYPE(item->inferred_descr);
        if (item->scalar_input && npy_mark_tmp_array_if_pyscalar(
                item->object, NULL, &item->DType)) {
            item->descr = NULL;
        }
        else {
            item->descr = (PyArray_Descr *)Py_NewRef(item->inferred_descr);
            self->flags &= ~NPY_CH_ALL_PYSCALARS;
            if (!item->scalar_input) {
                self->flags &= ~NPY_CH_ALL_SCALARS;
            }
        }
        Py_INCREF(item->DType);
    }

    return (PyObject *)self;

  fail:
    Py_DECREF(self);
    return NULL;
}


static PyArrayObject *
materialize_item(creation_item *item, PyArray_Descr *hint)
{
    if (hint == NULL && item->array != NULL) {
        return (PyArrayObject *)Py_NewRef(item->array);
    }
    coercion_cache_obj *cache = npy_clone_coercion_cache(item->cache);
    if (item->cache != NULL && cache == NULL) {
        return NULL;
    }
    PyArray_Descr *descr = (PyArray_Descr *)Py_NewRef(
            hint == NULL ? item->inferred_descr : hint);
    PyArrayObject *array = (PyArrayObject *)PyArray_FromDiscovery(
            item->object, hint, NULL, 0, item->ndim, item->shape, descr, cache, 0);
    if (array != NULL && hint == NULL) {
        item->array = (PyArrayObject *)Py_NewRef(array);
    }
    return array;
}


static PyObject *
array_converter_get_scalar_input(PyArrayArrayConverterObject *self)
{
    PyObject *ret = PyTuple_New(self->narrs);
    if (ret == NULL) {
        return NULL;
    }

    creation_item *item = self->items;
    for (int i = 0; i < self->narrs; i++, item++) {
        if (item->scalar_input) {
            Py_INCREF(Py_True);
            PyTuple_SET_ITEM(ret, i, Py_True);
        }
        else {
            Py_INCREF(Py_False);
            PyTuple_SET_ITEM(ret, i, Py_False);
        }
    }
    return ret;
}


static int
find_wrap(PyArrayArrayConverterObject *self)
{
    if (self->wrap != NULL) {
        return 0;  /* nothing to do */
    }

    /* Allocate scratch space (could be optimized away) */
    PyObject **objects = PyMem_Malloc(self->narrs * sizeof(PyObject *));
    if (objects == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    for (int i = 0; i < self->narrs; i++) {
        objects[i] = self->items[i].object;
    }
    int ret = npy_find_array_wrap(
            self->narrs, objects, &self->wrap, &self->wrap_type);
    PyMem_FREE(objects);
    return ret;
}


static int
check_string_promotion(unsigned int kinds)
{
    if ((kinds & (NPY_DISCOVERY_BYTES | NPY_DISCOVERY_TEXT)) &&
            (kinds & (kinds - 1))) {
        PyErr_SetString(_npy_module_state->static_pydata.DTypePromotionError,
                "Strict string promotion does not allow mixing bytes, text, "
                "or non-string DTypes. Convert the inputs explicitly.");
        return -1;
    }
    return 0;
}


typedef enum {
    CONVERT = 0,
    PRESERVE = 1,
    CONVERT_IF_NO_ARRAY = 2,
    PRESERVE_ALL = 3,
} scalar_policy;


static int
pyscalar_mode_conv(PyObject *obj, scalar_policy *policy)
{
    if (!PyUnicode_Check(obj)) {
        PyErr_Format(PyExc_TypeError,
                "invalid pyscalar mode %.100R, must be a string", obj);
        return 0;
    }
    npy_interned_str_struct *interned_str = &_npy_module_state->interned_str;
    PyObject *strings[3] = {
            interned_str->convert, interned_str->preserve,
            interned_str->convert_if_no_array};

    /* First quick pass using the identity (should practically always match) */
    for (int i = 0; i < 3; i++) {
        if (obj == strings[i]) {
            *policy = i;
            return 1;
        }
    }
    for (int i = 0; i < 3; i++) {
        int cmp = PyObject_RichCompareBool(obj, strings[i], Py_EQ);
        if (cmp < 0) {
            return 0;
        }
        if (cmp) {
            *policy = i;
            return 1;
        }
    }
    if (PyUnicode_CompareWithASCIIString(obj, "preserve_all") == 0) {
        *policy = PRESERVE_ALL;
        return 1;
    }
    PyErr_SetString(PyExc_ValueError,
            "invalid pyscalar mode, must be 'convert', 'preserve', "
            "'preserve_all', or 'convert_if_no_array' (default).");
    return 0;
}


static PyObject *
converter_as_arrays(PyArrayArrayConverterObject *self, npy_bool subok,
        scalar_policy policy, npy_bool strict_strings)
{
    if (strict_strings) {
        unsigned int kinds = 0;
        for (int i = 0; i < self->narrs; i++) {
            kinds |= self->items[i].discovery.kinds;
        }
        if (check_string_promotion(kinds) < 0) {
            return NULL;
        }
    }
    if (policy == CONVERT_IF_NO_ARRAY) {
        policy = (self->flags & NPY_CH_ALL_PYSCALARS) ? CONVERT : PRESERVE;
    }
    PyObject *result = PyTuple_New(self->narrs);
    if (result == NULL) {
        return NULL;
    }
    for (int i = 0; i < self->narrs; i++) {
        creation_item *item = &self->items[i];
        PyObject *value;
        if ((item->descr == NULL && policy == PRESERVE) ||
                (item->scalar_input && policy == PRESERVE_ALL)) {
            value = Py_NewRef(item->object);
        }
        else {
            value = (PyObject *)materialize_item(item, NULL);
            if (value != NULL && !subok) {
                value = PyArray_EnsureArray(value);  /* steals reference */
            }
            if (value == NULL) {
                Py_DECREF(result);
                return NULL;
            }
        }
        PyTuple_SET_ITEM(result, i, value);
    }
    return result;
}


/* array__wrapit calls this by interned name with the subok keyword. */
static PyObject *
array_converter_as_arrays(PyArrayArrayConverterObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    npy_bool subok = NPY_TRUE;
    scalar_policy policy = CONVERT_IF_NO_ARRAY;
    npy_bool strict_strings = NPY_FALSE;
    NPY_PREPARE_ARGPARSER;
    if (npy_parse_arguments("as_arrays", args, len_args, kwnames,
            {"$subok", &PyArray_BoolConverter, &subok},
            {"$strict_strings", &PyArray_BoolConverter, &strict_strings},
            {"$pyscalars", &pyscalar_mode_conv, &policy}) < 0) {
        return NULL;
    }
    return converter_as_arrays(self, subok, policy, strict_strings);
}



static PyObject *
array_converter_wrap(PyArrayArrayConverterObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *obj;
    PyObject *to_scalar = Py_None;
    npy_bool ensure_scalar;

    if (find_wrap(self) < 0) {
        return NULL;
    }

    NPY_PREPARE_ARGPARSER;
    /* to_scalar is three-way "bool", if `None` inspect input to decide. */
    if (npy_parse_arguments("wrap", args, len_args, kwnames,
            {"", NULL, &obj},
            {"$to_scalar", NULL, &to_scalar}) < 0) {
        return NULL;
    }
    if (to_scalar == Py_None) {
        ensure_scalar = self->flags & NPY_CH_ALL_SCALARS;
    }
    else {
        if (!PyArray_BoolConverter(to_scalar, &ensure_scalar)) {
            return NULL;
        }
    }

    return npy_apply_wrap(
        obj, NULL, self->wrap, self->wrap_type, NULL, ensure_scalar, NPY_FALSE);
}


static PyObject *
array_converter_result_type(PyArrayArrayConverterObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyArray_Descr *result = NULL;
    npy_dtype_info dt_info = {NULL, NULL};
    npy_bool ensure_inexact = NPY_FALSE;
    npy_bool strict_strings = NPY_FALSE;

    /* Allocate scratch space (could be optimized away) */
    void *DTypes_and_descrs = PyMem_Malloc(
            (((size_t)self->narrs + 1) * 2) * sizeof(PyObject *));
    if (DTypes_and_descrs == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    PyArray_DTypeMeta **DTypes = DTypes_and_descrs;
    PyArray_Descr **descrs = (PyArray_Descr **)(DTypes + self->narrs + 1);

    NPY_PREPARE_ARGPARSER;
    if (npy_parse_arguments("result_type", args, len_args, kwnames,
            {"|extra_dtype", &PyArray_DTypeOrDescrConverterOptional, &dt_info},
            {"|ensure_inexact", &PyArray_BoolConverter, &ensure_inexact},
            {"$strict_strings", &PyArray_BoolConverter, &strict_strings}) < 0) {
        goto finish;
    }

    int ndescrs = 0;
    int nDTypes = 0;
    creation_item *item = self->items;
    for (int i = 0; i < self->narrs; i++, item++) {
        DTypes[nDTypes] = item->DType;
        nDTypes++;
        if (item->descr != NULL) {
            descrs[ndescrs] = item->descr;
            ndescrs++;
        }
    }

    if (ensure_inexact) {
        if (dt_info.dtype != NULL) {
            PyErr_SetString(PyExc_TypeError,
                    "extra_dtype and ensure_inexact are mutually exclusive.");
            goto finish;
        }
        Py_INCREF(&PyArray_PyFloatDType);
        dt_info.dtype = &PyArray_PyFloatDType;
    }

    if (dt_info.dtype != NULL) {
        DTypes[nDTypes] = dt_info.dtype;
        nDTypes++;
    }
    if (dt_info.descr != NULL) {
        descrs[ndescrs] = dt_info.descr;
        ndescrs++;
    }

    if (strict_strings) {
        unsigned int kinds = 0;
        for (int i = 0; i < self->narrs; i++) {
            kinds |= self->items[i].discovery.kinds;
        }
        if (dt_info.dtype != NULL) {
            /* An explicit extra dtype participates in the same check. */
            kinds |= npy_discovery_kind_from_dtype(dt_info.dtype);
        }
        if (check_string_promotion(kinds) < 0) {
            goto finish;
        }
    }
    PyArray_DTypeMeta *common_dtype = PyArray_PromoteDTypeSequence(
            nDTypes, DTypes);
    if (common_dtype == NULL) {
        goto finish;
    }
    if (ndescrs == 0) {
        result = NPY_DT_CALL_default_descr(common_dtype);
    }
    else {
        result = PyArray_CastToDTypeAndPromoteDescriptors(
                ndescrs, descrs, common_dtype);
    }
    Py_DECREF(common_dtype);

  finish:
    Py_XDECREF(dt_info.descr);
    Py_XDECREF(dt_info.dtype);
    PyMem_Free(DTypes_and_descrs);
    return (PyObject *)result;
}


static PyGetSetDef array_converter_getsets[] = {
    {"scalar_input",
        (getter)array_converter_get_scalar_input,
        NULL,
        NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL},
};


static PyMethodDef array_converter_methods[] = {
    {"as_arrays", 
        (PyCFunction)array_converter_as_arrays,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    {"result_type",
        (PyCFunction)array_converter_result_type,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    {"wrap",
        (PyCFunction)array_converter_wrap,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    {NULL, NULL, 0, NULL}
};


/*
 * Only the first `narrs` items own their references; `array_converter_new`
 * increments the count as it fills them in.
 */
static int
array_converter_traverse(
        PyArrayArrayConverterObject *self, visitproc visit, void *arg)
{
    Py_VISIT(Py_TYPE(self));

    creation_item *item = self->items;
    for (int i = 0; i < self->narrs; i++, item++) {
        Py_VISIT(item->array);
        Py_VISIT(item->object);
        Py_VISIT(item->DType);
        Py_VISIT(item->descr);
        Py_VISIT(item->inferred_descr);
        Py_VISIT(item->discovery.scalars);
        for (coercion_cache_obj *cache = item->cache;
                cache != NULL; cache = cache->next) {
            Py_VISIT(cache->arr_or_sequence);
        }
    }

    Py_VISIT(self->wrap);
    Py_VISIT(self->wrap_type);
    return 0;
}

static int
array_converter_clear(PyArrayArrayConverterObject *self)
{
    creation_item *item = self->items;
    int narrs = self->narrs;
    self->narrs = 0;
    for (int i = 0; i < narrs; i++, item++) {
        Py_CLEAR(item->array);
        Py_CLEAR(item->object);
        Py_CLEAR(item->DType);
        Py_CLEAR(item->descr);
        Py_CLEAR(item->inferred_descr);
        Py_CLEAR(item->discovery.scalars);
        npy_free_coercion_cache(item->cache);
        item->cache = NULL;
        PyMem_Free(item->shape);
        item->shape = NULL;
    }

    Py_CLEAR(self->wrap);
    Py_CLEAR(self->wrap_type);
    return 0;
}

static void
array_converter_dealloc(PyArrayArrayConverterObject *self)
{
    PyObject_GC_UnTrack(self);
    array_converter_clear(self);

    PyTypeObject *type = Py_TYPE(self);
    type->tp_free((PyObject *)self);
    Py_DECREF(type);
}


static Py_ssize_t
array_converter_length(PyArrayArrayConverterObject *self)
{
    return self->narrs;
}


static PyObject *
array_converter_item(PyArrayArrayConverterObject *self, Py_ssize_t item)
{
    /* Python ensures no negative indices (and probably the below also) */
    if (item < 0 || item >= self->narrs) {
        PyErr_SetString(PyExc_IndexError, "invalid index");
        return NULL;
    }

    /* Follow the `as_arrays` default of `CONVERT_IF_NO_ARRAY`: */
    PyObject *res;
    if (self->items[item].descr == NULL
            && !(self->flags & NPY_CH_ALL_PYSCALARS)) {
        res = self->items[item].object;
    }
    else {
        return (PyObject *)materialize_item(&self->items[item], NULL);
    }

    Py_INCREF(res);
    return res;
}


static PyType_Slot array_converter_slots[] = {
    {Py_tp_new, array_converter_new},
    {Py_tp_dealloc, array_converter_dealloc},
    {Py_tp_traverse, array_converter_traverse},
    {Py_tp_clear, array_converter_clear},
    {Py_tp_getset, array_converter_getsets},
    {Py_tp_methods, array_converter_methods},
    {Py_sq_length, array_converter_length},
    {Py_sq_item, array_converter_item},
    {0, NULL},
};

static PyType_Spec array_converter_spec = {
    .name = "numpy._core._multiarray_umath._array_converter",
    .basicsize = sizeof(PyArrayArrayConverterObject),
    .itemsize = sizeof(creation_item),
    .flags = (Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC
              | Py_TPFLAGS_IMMUTABLETYPE),
    .slots = array_converter_slots,
};

NPY_NO_EXPORT int
init_array_converter_type(PyObject *module)
{
    PyObject *type = PyType_FromModuleAndSpec(
            module, &array_converter_spec, NULL);
    if (type == NULL) {
        return -1;
    }
    get_module_state(module)->PyArrayArrayConverter_Type =
            (PyTypeObject *)type;
    return 0;
}
