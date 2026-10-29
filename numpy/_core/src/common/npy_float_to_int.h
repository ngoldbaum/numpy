#ifndef NUMPY_CORE_SRC_COMMON_NPY_FLOAT_TO_INT_H_
#define NUMPY_CORE_SRC_COMMON_NPY_FLOAT_TO_INT_H_

#include "numpy/npy_math.h"

/*
 * Truncate toward zero and convert to intp.  Return -1 for non-finite or
 * out-of-range values without changing *out, or 0 on success.
 */
static inline int
npy_double_to_intp(double value, npy_intp *out)
{
    if (!npy_isfinite(value)) {
        return -1;
    }
    double truncated = npy_trunc(value);
    /* Unlike NPY_MAX_INTP, both bounds are exactly representable as doubles. */
    if (truncated < (double)NPY_MIN_INTP
            || truncated >= -(double)NPY_MIN_INTP) {
        return -1;
    }
    *out = (npy_intp)truncated;
    return 0;
}

#endif  /* NUMPY_CORE_SRC_COMMON_NPY_FLOAT_TO_INT_H_ */
