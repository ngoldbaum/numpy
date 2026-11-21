#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#include <Python.h>

#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ankerl/unordered_dense.h>

#include <numpy/npy_common.h>
#include "numpy/arrayobject.h"
#include "numpy/dtype_api.h"
#include "gil_utils.h"
#include "minmax.h"
#include "numpy_tag.hpp"
#include "raii_utils.hpp"
#include "utils.hpp"
extern "C" {
    #include "npy_argparse.h"
    #include "numpy/npy_math.h"
    #include "numpy/halffloat.h"
}

// HASH_TABLE_INITIAL_BUCKETS is the reserve hashset capacity used in the
// hash set instances in the various unique_* functions.
// We use min(input_size, HASH_TABLE_INITIAL_BUCKETS) as the initial bucket
// count:
// - Reserving for all elements (isize) may over-allocate when there are few
//   unique values.
// - Using a moderate upper bound HASH_TABLE_INITIAL_BUCKETS(1024) keeps
//   memory usage reasonable (4 KiB for pointers).
// See https://github.com/numpy/numpy/pull/28767#discussion_r2064267631
const npy_intp HASH_TABLE_INITIAL_BUCKETS = 1024;

//
// Create a 1-d array with the given length that has the same
// dtype as the input `arr`.
//
static inline PyArrayObject *
empty_array_like(PyArrayObject *arr, npy_intp length)
{
    PyArray_Descr *descr = PyArray_DESCR(arr);
    Py_INCREF(descr);

    // Create the output array.
    PyArrayObject *res_obj =
        reinterpret_cast<PyArrayObject *>(
            PyArray_NewFromDescr(
                &PyArray_Type,
                descr,
                1,                      // ndim
                &length,                // shape
                NULL,                   // strides
                NULL,                   // data
                NPY_ARRAY_WRITEABLE,    // flags
                NULL                    // obj
            )
        );
    return res_obj;
}

// The smaller index holds at most 2**32 values.
using small_index = ankerl::unordered_dense::bucket_type::group;
using large_index = ankerl::unordered_dense::bucket_type::group_big;

static inline bool
fits_small_index(PyArrayObject *arr)
{
    return static_cast<npy_uint64>(PyArray_SIZE(arr)) <= NPY_MAX_UINT32;
}

template <
    typename Index,
    typename Key,
    typename Hash = ankerl::unordered_dense::hash<Key>,
    typename Equal = std::equal_to<Key>
>
using unique_set = ankerl::unordered_dense::set<Key, Hash, Equal, std::allocator<Key>, Index>;

// Wraps a hash whose 64-bit result is already well mixed, so that the set
// does not mix it a second time.
template <typename F>
struct mixed_hash {
    using is_avalanching = void;
    F func;
    template <typename K>
    npy_uint64 operator()(const K &key) const { return func(key); }
};

// The byte hash that comes with unordered_dense, a descendant of wyhash
// (https://github.com/wangyi-fudan/wyhash).
static inline npy_uint64
hash_bytes(const void *data, size_t len)
{
    std::string_view view{static_cast<const char *>(data), len};
    return ankerl::unordered_dense::hash<std::string_view>{}(view);
}

// Each part goes through std::hash, which gives -0.0 the hash of 0.0.
template <typename S>
static inline npy_uint64
hash_complex(S real, S imag)
{
    return ankerl::unordered_dense::hash<std::pair<S, S>>{}({real, imag});
}

#if NPY_SIZEOF_LONGDOUBLE != NPY_SIZEOF_DOUBLE

constexpr size_t LONGDOUBLE_KEY_MAX_BYTES = 32;

// Writes the bytes that identify a long double that is not NaN and returns
// how many were written. Equal values give equal bytes, so -0.0 is written
// as 0.0.
static inline size_t
longdouble_key_bytes(npy_longdouble value, unsigned char *out)
{
    if (value == 0) {
        value = 0;
    }

    // Some long double formats (e.g., the one behind npy_complex256) include undefined or
    // unused bits in their binary representation
    // (see: https://github.com/numpy/numpy/blob/main/numpy/_core/src/npymath/npy_math_private.h#L254-L261).
    // Because hashing the raw bit pattern would make the hash depend on those
    // undefined bits, we extract the mantissa, exponent, and sign components
    // explicitly and pack them into a buffer to ensure the hash is well-defined.
    #if defined(HAVE_LDOUBLE_INTEL_EXTENDED_12_BYTES_LE) || \
        defined(HAVE_LDOUBLE_INTEL_EXTENDED_16_BYTES_LE) || \
        defined(HAVE_LDOUBLE_MOTOROLA_EXTENDED_12_BYTES_BE)

    constexpr size_t SIZEOF_LDOUBLE_MAN = sizeof(ldouble_man_t);
    constexpr size_t SIZEOF_LDOUBLE_EXP = sizeof(ldouble_exp_t);
    constexpr size_t SIZEOF_LDOUBLE_SIGN = sizeof(ldouble_sign_t);

    union IEEEl2bitsrep bits{value};
    size_t offset = 0;

    ldouble_man_t manh = GET_LDOUBLE_MANH(bits);
    ldouble_man_t manl = GET_LDOUBLE_MANL(bits);
    ldouble_exp_t exp = GET_LDOUBLE_EXP(bits);
    ldouble_sign_t sign = GET_LDOUBLE_SIGN(bits);

    std::memcpy(out + offset, &manh, SIZEOF_LDOUBLE_MAN);
    offset += SIZEOF_LDOUBLE_MAN;
    std::memcpy(out + offset, &manl, SIZEOF_LDOUBLE_MAN);
    offset += SIZEOF_LDOUBLE_MAN;
    std::memcpy(out + offset, &exp, SIZEOF_LDOUBLE_EXP);
    offset += SIZEOF_LDOUBLE_EXP;
    std::memcpy(out + offset, &sign, SIZEOF_LDOUBLE_SIGN);
    offset += SIZEOF_LDOUBLE_SIGN;

    return offset;
    #else

    std::memcpy(out, &value, sizeof(value));
    return sizeof(value);

    #endif
}

// std::hash is slow for long double in libstdc++, so its bytes are hashed.
static inline npy_uint64
hash_complex(npy_longdouble real, npy_longdouble imag)
{
    unsigned char key[2 * LONGDOUBLE_KEY_MAX_BYTES];
    size_t len = longdouble_key_bytes(real, key);
    len += longdouble_key_bytes(imag, key + len);
    return hash_bytes(key, len);
}

#endif

template <typename Tag, typename Index>
static PyObject*
unique_complex_with_index(PyArrayObject *self, npy_bool equal_nan)
{
    /*
     * Returns a new NumPy array containing the unique values of the input
     * array of complex numbers.
     * This function uses hashing to identify uniqueness efficiently.
     * NaNs are kept apart from the hash set: the one that sorts first if
     * equal_nan is true, all of them otherwise.
     * Returns NotImplemented, leaving the work to sorting, for an array in
     * non-native byte order.
     */
    using T = typename Tag::type;

    // The values are interpreted, so they have to be in native byte order.
    if (!PyArray_ISNOTSWAPPED(self)) {
        return Py_NewRef(Py_NotImplemented);
    }

    auto hash = [](const T &value) -> npy_uint64 {
        return hash_complex(Tag::creal(value), Tag::cimag(value));
    };
    auto equal = [](const T &lhs, const T &rhs) -> bool {
        return Tag::creal(lhs) == Tag::creal(rhs) && Tag::cimag(lhs) == Tag::cimag(rhs);
    };

    using set_type = unique_set<Index, T, mixed_hash<decltype(hash)>, decltype(equal)>;

    // number of elements in the input array
    npy_intp isize = PyArray_SIZE(self);
    set_type hashset(std::min(isize, HASH_TABLE_INITIAL_BUCKETS),
                     mixed_hash<decltype(hash)>{hash}, equal);
    std::vector<T> nans;
    {
        np::raii::SaveThreadState save_thread_state{};

        char *idata = PyArray_BYTES(self);
        npy_intp istride = PyArray_STRIDES(self)[0];
        for (npy_intp i = 0; i < isize; i++, idata += istride) {
            T value;
            std::memcpy(&value, idata, sizeof(T));
            if (!npy_isnan(Tag::creal(value)) && !npy_isnan(Tag::cimag(value))) {
                hashset.insert(value);
            }
            else if (!equal_nan || nans.empty()) {
                nans.push_back(value);
            }
            else if (Tag::less(value, nans[0])) {
                nans[0] = value;
            }
        }
    }

    PyArrayObject *res_obj = empty_array_like(self, hashset.size() + nans.size());
    if (res_obj == NULL) {
        return NULL;
    }

    {
        np::raii::SaveThreadState save_thread_state{};

        char *odata = PyArray_BYTES(res_obj);
        if (hashset.size() > 0) {
            std::memcpy(odata, hashset.values().data(), hashset.size() * sizeof(T));
        }
        if (nans.size() > 0) {
            std::memcpy(odata + hashset.size() * sizeof(T), nans.data(), nans.size() * sizeof(T));
        }
    }

    return reinterpret_cast<PyObject *>(res_obj);
}

template <typename Tag>
static PyObject*
unique_complex(PyArrayObject *self, npy_bool equal_nan, bool *NPY_UNUSED(needs_sort))
{
    if (fits_small_index(self)) {
        return unique_complex_with_index<Tag, small_index>(self, equal_nan);
    }
    return unique_complex_with_index<Tag, large_index>(self, equal_nan);
}

template <typename T> struct minmax_typenum;
template <> struct minmax_typenum<npy_int> { static constexpr int value = NPY_INT; };
template <> struct minmax_typenum<npy_uint> { static constexpr int value = NPY_UINT; };
template <> struct minmax_typenum<npy_long> { static constexpr int value = NPY_LONG; };
template <> struct minmax_typenum<npy_ulong> { static constexpr int value = NPY_ULONG; };
template <> struct minmax_typenum<npy_longlong> { static constexpr int value = NPY_LONGLONG; };
template <> struct minmax_typenum<npy_ulonglong> { static constexpr int value = NPY_ULONGLONG; };

// The value of entry `i` of a table whose first entry is for `min`.
template <typename T>
static inline T
table_value(T min, size_t i)
{
    if constexpr (std::is_signed_v<T>) {
        return static_cast<T>(static_cast<npy_int64>(min) + static_cast<npy_int64>(i));
    }
    else {
        return static_cast<T>(static_cast<npy_uint64>(min) + i);
    }
}

template <typename T>
static bool
unique_with_table(PyArrayObject *self, T min, npy_uint64 num_values, PyObject **result)
{
    /*
     * Stores a new NumPy array with the unique values of the input array of
     * integers in *result (NULL on error) and returns true, given that the
     * values are `min` or one of the `num_values - 1` integers above it.
     * A table with an entry for each of these replaces the hash set, and the
     * unique values come out in ascending order.
     * Returns false if there is no memory for the table or a value is out of
     * range, hashing still works then.
     */
    npy_intp isize = PyArray_SIZE(self);
    std::vector<npy_uint8> seen;
    npy_intp osize = 0;

    {
        np::raii::SaveThreadState save_thread_state{};

        try {
            seen.resize(num_values, 0);
        }
        catch (const std::bad_alloc &) {
            return false;
        }
        char *idata = PyArray_BYTES(self);
        npy_intp istride = PyArray_STRIDES(self)[0];
        for (npy_intp i = 0; i < isize; i++, idata += istride) {
            T value;
            std::memcpy(&value, idata, sizeof(T));
            npy_uint64 offset = static_cast<npy_uint64>(value) - static_cast<npy_uint64>(min);
            if (offset >= num_values) {
                // the array was written to since its range was found
                return false;
            }
            seen[offset] = 1;
        }
        for (npy_uint8 flag : seen) {
            osize += flag;
        }
    }

    PyArrayObject *res_obj = empty_array_like(self, osize);
    *result = reinterpret_cast<PyObject *>(res_obj);
    if (res_obj == NULL) {
        return true;
    }

    {
        np::raii::SaveThreadState save_thread_state{};

        char *odata = PyArray_BYTES(res_obj);
        for (size_t i = 0; i < seen.size(); i++) {
            if (seen[i]) {
                T value = table_value(min, i);
                std::memcpy(odata, &value, sizeof(T));
                odata += sizeof(T);
            }
        }
    }

    return true;
}

// The number of elements to look at before the first check of the range,
// and the most to look at between two checks. A range that is too large
// usually shows early, so the first checks come quickly.
constexpr npy_intp RANGE_FIRST_CHUNK_SIZE = 64;
constexpr npy_intp RANGE_MAX_CHUNK_SIZE = 8192;

template <typename T>
static bool
integer_table_range(PyArrayObject *self, T *table_min, npy_uint64 *num_values,
                    npy_uint64 small_type_ratio = 1)
{
    /*
     * Finds a range for a table of the values of the input array of
     * integers, and returns whether it is no larger than the array has
     * elements. For 8 and 16 bit integers that is the range of the type,
     * which may be `small_type_ratio` times the size of the array.
     * For wider ones the range is looked for, and false returned as soon as
     * it is seen to be larger.
     */
    npy_intp isize = PyArray_SIZE(self);
    if constexpr (sizeof(T) <= 2) {
        *table_min = std::numeric_limits<T>::min();
        *num_values = npy_uint64{1} << (8 * sizeof(T));
        return static_cast<npy_uint64>(isize) * small_type_ratio >= *num_values;
    }
    else {
        char *idata = PyArray_BYTES(self);
        npy_intp istride = PyArray_STRIDES(self)[0];
        // the loops that find the minimum and maximum need aligned data
        if (isize < RANGE_FIRST_CHUNK_SIZE
                || (reinterpret_cast<npy_uintp>(idata) % alignof(T)) != 0
                || (istride % static_cast<npy_intp>(alignof(T))) != 0) {
            return false;
        }

        PyArrayMethod_StridedLoop *minmax_loop =
                npy_get_integer_minimummaximum_loop(minmax_typenum<T>::value);
        if (minmax_loop == NULL) {
            return false;
        }

        T min, max;
        std::memcpy(&min, idata, sizeof(T));
        max = min;
        char *min_data = reinterpret_cast<char *>(&min);
        char *max_data = reinterpret_cast<char *>(&max);
        const npy_intp minmax_strides[5] = {0, 0, istride, 0, 0};

        {
            np::raii::SaveThreadState save_thread_state{};

            npy_intp chunk = RANGE_FIRST_CHUNK_SIZE;
            for (npy_intp i = 0; i < isize; i += chunk, chunk *= 2) {
                chunk = std::min({chunk, RANGE_MAX_CHUNK_SIZE, isize - i});
                char *const args[5] = {min_data, max_data, idata + i * istride, min_data, max_data};
                minmax_loop(NULL, args, &chunk, minmax_strides, NULL);
                if (static_cast<npy_uint64>(max) - static_cast<npy_uint64>(min)
                        >= static_cast<npy_uint64>(isize)) {
                    return false;
                }
            }
        }

        *table_min = min;
        *num_values = static_cast<npy_uint64>(max) - static_cast<npy_uint64>(min) + 1;
        return true;
    }
}

// The number of elements to hash between two checks of the size of the set.
constexpr npy_intp GIVE_UP_CHUNK_SIZE = 1024;

// The most distinct values that are worth hashing when a sort, or an argsort,
// is the alternative. A larger set does not fit in the cache, and each
// element then costs a cache miss that sorting avoids. x86 has the fastest
// sorts and small caches for each core, so hashing stops paying sooner there.
#if defined(NPY_CPU_AMD64) || defined(NPY_CPU_X86)
constexpr size_t SORT_SET_LIMIT = size_t{1} << 15;
constexpr size_t ARGSORT_SET_LIMIT = size_t{1} << 17;
#else
constexpr size_t SORT_SET_LIMIT = size_t{1} << 20;
constexpr size_t ARGSORT_SET_LIMIT = std::numeric_limits<size_t>::max();
#endif

// The number of distinct values at which to give up on hashing an array:
// a fraction of its elements, or fewer for a large array.
static inline size_t
give_up_limit(npy_intp isize, size_t set_limit, npy_intp fraction = 4)
{
    return std::min(static_cast<size_t>(isize / fraction), set_limit);
}

// How the elements of an array become the keys of a hash set. An integer is
// its own key.
template <typename T>
struct integer_key {
    using key_type = T;
    static bool is_nan(T) { return false; }
    T encode(T value) { return value; }
    T decode(T key) const { return key; }
    bool less(T a, T b) const { return a < b; }
};

// The key of a floating point number is its bit pattern, with one key for
// both zeros and one for all NaNs. The first zero and the first NaN seen
// stand for theirs.
template <typename F, typename K>
struct float_key {
    using key_type = K;
    static constexpr K NAN_KEY = std::numeric_limits<K>::max();
    bool zero_seen = false, nan_seen = false;
    F zero = 0, nan = 0;

    static bool is_nan(F value) { return value != value; }

    K
    encode(F value)
    {
        if (value != value) {
            if (!nan_seen) {
                nan_seen = true;
                nan = value;
            }
            return NAN_KEY;
        }
        if (value == 0) {
            if (!zero_seen) {
                zero_seen = true;
                zero = value;
            }
            return 0;
        }
        K key;
        std::memcpy(&key, &value, sizeof(F));
        return key;
    }

    F
    decode(K key) const
    {
        if (key == 0) {
            return zero;
        }
        if (key == NAN_KEY) {
            return nan;
        }
        F value;
        std::memcpy(&value, &key, sizeof(F));
        return value;
    }

    // NaN sorts last
    bool
    less(K a, K b) const
    {
        return b == NAN_KEY ? a != NAN_KEY : decode(a) < decode(b);
    }
};

// A hash set is cheap for as long as it fits in the cache. Once it has more
// entries than this, the rest of the first half of the array is only looked
// at to estimate how many distinct values it holds, which decides whether to
// carry on hashing.
constexpr size_t EXACT_SET_SIZE = 1 << 14;

// The number of elements to add to an estimate between two looks at it.
constexpr npy_intp ESTIMATE_BLOCK_SIZE = 1 << 16;

// HyperLogLog (Flajolet et al., 2007): estimates the number of distinct
// hashes it was given to within a few percent, from the longest run of
// leading zeros among the hashes that share their first bits.
struct hyperloglog {
    static constexpr int INDEX_BITS = 12;
    static constexpr size_t NUM_REGISTERS = size_t{1} << INDEX_BITS;
    npy_uint8 registers[NUM_REGISTERS] = {};

    void
    add(npy_uint64 hash)
    {
        size_t index = hash >> (64 - INDEX_BITS);
        npy_uint64 rest = (hash << INDEX_BITS) | (npy_uint64{1} << (INDEX_BITS - 1));
        npy_uint8 rank = static_cast<npy_uint8>(64 - np::BitScanReverse(rest));
        // a write for every hash would make each wait for the one before
        if (rank > registers[index]) {
            registers[index] = rank;
        }
    }

    // Only for more distinct values than a few times NUM_REGISTERS, below
    // that the estimate needs a correction.
    double
    estimate() const
    {
        double sum = 0;
        for (npy_uint8 rank : registers) {
            sum += 1.0 / static_cast<double>(npy_uint64{1} << rank);
        }
        constexpr double m = NUM_REGISTERS;
        return 0.7213 / (1 + 1.079 / m) * m * m / sum;
    }
};

template <typename T, typename Keys>
static bool
estimate_exceeds(Keys &keys, const char *data, npy_intp stride, npy_intp count, size_t limit)
{
    /*
     * Returns whether `count` elements hold more than about `limit` distinct
     * values, as soon as that shows.
     */
    hyperloglog sketch;
    ankerl::unordered_dense::hash<typename Keys::key_type> hash;
    while (count > 0) {
        npy_intp block = std::min(count, ESTIMATE_BLOCK_SIZE);
        for (npy_intp i = 0; i < block; i++, data += stride) {
            T value;
            std::memcpy(&value, data, sizeof(T));
            sketch.add(hash(keys.encode(value)));
        }
        count -= block;
        if (sketch.estimate() > static_cast<double>(limit)) {
            return true;
        }
    }
    return false;
}


template <typename T, typename Index, size_t N>
static PyObject*
unique_integer_with_index(PyArrayObject *self, bool may_give_up)
{
    /*
     * Returns a new NumPy array containing the unique values of the input
     * array of integers, or of anything else of N bytes that fits into T.
     * This function uses hashing to identify uniqueness efficiently.
     * If may_give_up is true, it returns NotImplemented once give_up_limit
     * distinct values have turned up. To sort a quarter of the values
     * afterwards costs about as much as sorting the array to begin with.
     * Past the middle of the array it carries on, most of the work is done.
     * A set that has outgrown EXACT_SET_SIZE is not filled to find that out,
     * the number of distinct values in the first half is estimated instead,
     * and that settles it.
     */
    static_assert(N > 0 && N <= sizeof(T));

    // number of elements in the input array
    npy_intp isize = PyArray_SIZE(self);
    unique_set<Index, T> hashset;
    size_t limit = may_give_up ? give_up_limit(isize, SORT_SET_LIMIT)
                               : std::numeric_limits<size_t>::max();
    bool give_up = false, estimated = !may_give_up;

    {
        np::raii::SaveThreadState save_thread_state{};

        char *idata = PyArray_BYTES(self);
        npy_intp istride = PyArray_STRIDES(self)[0];
        for (npy_intp start = 0; start < isize && !give_up; start += GIVE_UP_CHUNK_SIZE) {
            npy_intp stop = std::min(start + GIVE_UP_CHUNK_SIZE, isize);
            for (npy_intp i = start; i < stop; i++, idata += istride) {
                T value = 0;
                std::memcpy(&value, idata, N);
                hashset.insert(value);
            }
            give_up = stop <= isize / 2 && hashset.size() > limit;
            if (!estimated && !give_up && hashset.size() > EXACT_SET_SIZE) {
                estimated = true;
                integer_key<T> keys;
                give_up = estimate_exceeds<T>(keys, idata, istride, isize / 2 - stop, limit);
                limit = std::numeric_limits<size_t>::max();
            }
        }
    }
    if (give_up) {
        return Py_NewRef(Py_NotImplemented);
    }

    PyArrayObject *res_obj = empty_array_like(self, hashset.size());
    if (res_obj == NULL) {
        return NULL;
    }

    {
        np::raii::SaveThreadState save_thread_state{};

        char *odata = PyArray_BYTES(res_obj);
        if constexpr (N == sizeof(T)) {
            if (hashset.size() > 0) {
                std::memcpy(odata, hashset.values().data(), hashset.size() * sizeof(T));
            }
        }
        else {
            for (T value : hashset) {
                std::memcpy(odata, &value, N);
                odata += N;
            }
        }
    }

    return reinterpret_cast<PyObject *>(res_obj);
}

template <typename T, size_t N = sizeof(T)>
static PyObject*
unique_integer(PyArrayObject *self, npy_bool NPY_UNUSED(equal_nan), bool *needs_sort)
{
    if constexpr (N == sizeof(T)) {
        T min;
        npy_uint64 num_values;
        PyObject *result;
        if (integer_table_range<T>(self, &min, &num_values)
                && unique_with_table<T>(self, min, num_values, &result)) {
            *needs_sort = false;
            return result;
        }
    }
    if (fits_small_index(self)) {
        return unique_integer_with_index<T, small_index, N>(self, *needs_sort);
    }
    return unique_integer_with_index<T, large_index, N>(self, *needs_sort);
}

template <typename F, typename K, typename Index>
static PyObject*
unique_float_with_index(PyArrayObject *self, npy_bool equal_nan, bool may_give_up)
{
    /*
     * Returns a new NumPy array containing the unique values of the input
     * array of floating point numbers, or NotImplemented under the rule of
     * unique_integer_with_index. If equal_nan is false every NaN is kept.
     */
    npy_intp isize = PyArray_SIZE(self);
    unique_set<Index, K> hashset;
    float_key<F, K> keys;
    std::vector<F> nans;
    size_t limit = may_give_up ? give_up_limit(isize, SORT_SET_LIMIT)
                               : std::numeric_limits<size_t>::max();
    bool give_up = false, estimated = !may_give_up;

    {
        np::raii::SaveThreadState save_thread_state{};

        char *idata = PyArray_BYTES(self);
        npy_intp istride = PyArray_STRIDES(self)[0];
        for (npy_intp start = 0; start < isize && !give_up; start += GIVE_UP_CHUNK_SIZE) {
            npy_intp stop = std::min(start + GIVE_UP_CHUNK_SIZE, isize);
            for (npy_intp i = start; i < stop; i++, idata += istride) {
                F value;
                std::memcpy(&value, idata, sizeof(F));
                if (!equal_nan && keys.is_nan(value)) {
                    nans.push_back(value);
                }
                else {
                    hashset.insert(keys.encode(value));
                }
            }
            give_up = stop <= isize / 2 && hashset.size() > limit;
            if (!estimated && !give_up && hashset.size() > EXACT_SET_SIZE) {
                estimated = true;
                give_up = estimate_exceeds<F>(keys, idata, istride, isize / 2 - stop, limit);
                limit = std::numeric_limits<size_t>::max();
            }
        }
    }
    if (give_up) {
        return Py_NewRef(Py_NotImplemented);
    }

    PyArrayObject *res_obj = empty_array_like(self, hashset.size() + nans.size());
    if (res_obj == NULL) {
        return NULL;
    }

    {
        np::raii::SaveThreadState save_thread_state{};

        char *odata = PyArray_BYTES(res_obj);
        for (K key : hashset) {
            F value = keys.decode(key);
            std::memcpy(odata, &value, sizeof(F));
            odata += sizeof(F);
        }
        if (nans.size() > 0) {
            std::memcpy(odata, nans.data(), nans.size() * sizeof(F));
        }
    }

    return reinterpret_cast<PyObject *>(res_obj);
}

template <typename F, typename K>
static PyObject*
unique_float(PyArrayObject *self, npy_bool equal_nan, bool *needs_sort)
{
    static_assert(sizeof(F) == sizeof(K));
    // The values are interpreted, so they have to be in native byte order.
    if (!PyArray_ISNOTSWAPPED(self)) {
        return Py_NewRef(Py_NotImplemented);
    }
    if (fits_small_index(self)) {
        return unique_float_with_index<F, K, small_index>(self, equal_nan, *needs_sort);
    }
    return unique_float_with_index<F, K, large_index>(self, equal_nan, *needs_sort);
}

// With the optional results, 16 bit integers get a table of the range of
// their type for an array of at least one in this many elements of it, and
// 8 bit integers always.
template <typename T>
constexpr npy_uint64 SMALL_TYPE_TABLE_RATIO = sizeof(T) == 1 ? 256 : 4;

// Which of the optional results of np.unique are asked for.
struct unique_wanted {
    bool index, inverse, counts;
};

// The results of np.unique: the unique values in ascending order, for each
// of them the index of its first occurrence and how often it occurs, and
// for each element the index of its unique value.
struct unique_results {
    PyArrayObject *values = NULL, *index = NULL, *inverse = NULL, *counts = NULL;

    ~unique_results()
    {
        Py_XDECREF(values);
        Py_XDECREF(index);
        Py_XDECREF(inverse);
        Py_XDECREF(counts);
    }

    static PyArrayObject *
    empty_intp(npy_intp length)
    {
        return reinterpret_cast<PyArrayObject *>(PyArray_EMPTY(1, &length, NPY_INTP, 0));
    }

    // Allocates what is wanted and not allocated yet, false on error.
    bool
    allocate(PyArrayObject *self, npy_intp osize, unique_wanted want)
    {
        values = empty_array_like(self, osize);
        if (values == NULL) {
            return false;
        }
        if (want.index && (index = empty_intp(osize)) == NULL) {
            return false;
        }
        if (want.counts && (counts = empty_intp(osize)) == NULL) {
            return false;
        }
        if (want.inverse && inverse == NULL
                && (inverse = empty_intp(PyArray_SIZE(self))) == NULL) {
            return false;
        }
        return true;
    }

    static npy_intp *
    data(PyArrayObject *arr)
    {
        return arr == NULL ? NULL : reinterpret_cast<npy_intp *>(PyArray_BYTES(arr));
    }

    // A tuple of the four results, None for one that was not wanted.
    PyObject *
    as_tuple()
    {
        auto get = [](PyArrayObject *arr) {
            return arr == NULL ? Py_None : reinterpret_cast<PyObject *>(arr);
        };
        return PyTuple_Pack(4, get(values), get(index), get(inverse), get(counts));
    }
};

template <typename T>
static bool
unique_all_with_table(PyArrayObject *self, T min, npy_uint64 num_values, unique_wanted want,
                      PyObject **result)
{
    /*
     * As unique_with_table, with the optional results of np.unique. The
     * table counts the occurrences of each value and afterwards holds the
     * index of each value among the unique ones.
     */
    npy_intp isize = PyArray_SIZE(self);
    char *idata = PyArray_BYTES(self);
    npy_intp istride = PyArray_STRIDES(self)[0];
    std::vector<npy_intp> table, first;
    npy_intp osize = 0;

    {
        np::raii::SaveThreadState save_thread_state{};

        try {
            table.resize(num_values, 0);
            if (want.index) {
                first.resize(num_values);
            }
        }
        catch (const std::bad_alloc &) {
            return false;
        }
        char *data = idata;
        for (npy_intp i = 0; i < isize; i++, data += istride) {
            T value;
            std::memcpy(&value, data, sizeof(T));
            npy_uint64 offset = static_cast<npy_uint64>(value) - static_cast<npy_uint64>(min);
            if (offset >= num_values) {
                // the array was written to since its range was found
                return false;
            }
            if (table[offset]++ == 0 && want.index) {
                first[offset] = i;
            }
        }
        for (npy_intp count : table) {
            osize += count != 0;
        }
    }

    unique_results res;
    *result = NULL;
    if (!res.allocate(self, osize, want)) {
        return true;
    }

    {
        np::raii::SaveThreadState save_thread_state{};

        char *odata = PyArray_BYTES(res.values);
        npy_intp *index = res.data(res.index), *counts = res.data(res.counts);
        npy_intp k = 0;
        for (size_t i = 0; i < table.size(); i++) {
            if (table[i] == 0) {
                continue;
            }
            T value = table_value(min, i);
            std::memcpy(odata + k * sizeof(T), &value, sizeof(T));
            if (index != NULL) {
                index[k] = first[i];
            }
            if (counts != NULL) {
                counts[k] = table[i];
            }
            table[i] = k++;
        }
        if (want.inverse) {
            npy_intp *inverse = res.data(res.inverse);
            char *data = idata;
            for (npy_intp i = 0; i < isize; i++, data += istride) {
                T value;
                std::memcpy(&value, data, sizeof(T));
                npy_uint64 offset = static_cast<npy_uint64>(value) - static_cast<npy_uint64>(min);
                inverse[i] = offset < num_values ? table[offset] : 0;
            }
        }
    }

    *result = res.as_tuple();
    return true;
}

// An array counts as nearly sorted if at most one in this many of its
// elements is smaller than the one before it.
constexpr npy_intp NEARLY_SORTED_RATIO = 256;

template <typename T>
static bool
unique_all_sorted(PyArrayObject *self, unique_wanted want, PyObject **result, bool *nearly_sorted)
{
    /*
     * Stores the results of np.unique in *result (NULL on error) and returns
     * true if the input array is in ascending order and has no NaN, in which
     * case equal values are next to each other. Otherwise it reports whether
     * the array is nearly sorted.
     */
    npy_intp isize = PyArray_SIZE(self);
    char *idata = PyArray_BYTES(self);
    npy_intp istride = PyArray_STRIDES(self)[0];
    if (isize == 0) {
        return false;
    }
    auto read = [idata, istride](npy_intp i) {
        T value;
        std::memcpy(&value, idata + i * istride, sizeof(T));
        return value;
    };
    npy_intp osize = 1, descents = 0;
    *nearly_sorted = false;

    {
        np::raii::SaveThreadState save_thread_state{};

        T previous = read(0);
        if (previous != previous) {
            return false;
        }
        for (npy_intp i = 1; i < isize; i++) {
            T value = read(i);
            if (!(value >= previous) && ++descents > isize / NEARLY_SORTED_RATIO) {
                return false;
            }
            osize += value != previous;
            previous = value;
        }
    }
    if (descents > 0) {
        *nearly_sorted = true;
        return false;
    }

    unique_results res;
    *result = NULL;
    if (!res.allocate(self, osize, want)) {
        return true;
    }

    {
        np::raii::SaveThreadState save_thread_state{};

        char *odata = PyArray_BYTES(res.values);
        npy_intp *index = res.data(res.index), *counts = res.data(res.counts);
        npy_intp *inverse = res.data(res.inverse);
        npy_intp k = -1, start = 0;
        T previous = 0;
        for (npy_intp i = 0; i < isize; i++) {
            T value = read(i);
            // the array may have been written to since it was found sorted
            if ((k < 0 || value != previous) && k + 1 < osize) {
                if (k >= 0 && counts != NULL) {
                    counts[k] = i - start;
                }
                k++;
                start = i;
                std::memcpy(odata + k * sizeof(T), &value, sizeof(T));
                if (index != NULL) {
                    index[k] = i;
                }
                previous = value;
            }
            if (inverse != NULL) {
                inverse[i] = k;
            }
        }
        if (counts != NULL) {
            counts[k] = isize - start;
            for (npy_intp j = k + 1; j < osize; j++) {
                counts[j] = 0;
            }
        }
        for (npy_intp j = k + 1; j < osize; j++) {
            std::memcpy(odata + j * sizeof(T), &previous, sizeof(T));
            if (index != NULL) {
                index[j] = isize - 1;
            }
        }
    }

    *result = res.as_tuple();
    return true;
}

template <typename T, typename Index, typename Keys>
static PyObject*
unique_all_with_index(PyArrayObject *self, unique_wanted want, npy_bool equal_nan)
{
    /*
     * As unique_integer_with_index, with the optional results of np.unique.
     * The hash map counts each value and keeps them in the order they were
     * first seen, so the position of a value in it serves as its index until
     * the unique values are sorted. A NaN is left to sorting if equal_nan is
     * false.
     */
    using Key = typename Keys::key_type;
    // a count as small as the index keeps the entries of 32 bit keys small
    using Count = std::conditional_t<std::is_same_v<Index, small_index>, npy_uint32, npy_intp>;
    npy_intp isize = PyArray_SIZE(self);
    Keys keys;
    ankerl::unordered_dense::map<Key, Count, ankerl::unordered_dense::hash<Key>,
                                 std::equal_to<Key>,
                                 std::allocator<std::pair<Key, Count>>, Index> hashset;
    std::vector<npy_intp> first, order, rank;
    // To count takes more than to tell values apart, and is up against a sort.
    size_t limit = want.index || want.inverse ? give_up_limit(isize, ARGSORT_SET_LIMIT)
                                              : give_up_limit(isize, SORT_SET_LIMIT, 8);
    bool give_up = false, estimated = false;

    unique_results res;
    if (want.inverse && (res.inverse = res.empty_intp(isize)) == NULL) {
        return NULL;
    }
    npy_intp *inverse = res.data(res.inverse);

    {
        np::raii::SaveThreadState save_thread_state{};

        char *idata = PyArray_BYTES(self);
        npy_intp istride = PyArray_STRIDES(self)[0];
        for (npy_intp start = 0; start < isize && !give_up; start += GIVE_UP_CHUNK_SIZE) {
            npy_intp stop = std::min(start + GIVE_UP_CHUNK_SIZE, isize);
            bool separate_nan = false;
            for (npy_intp i = start; i < stop; i++, idata += istride) {
                T value;
                std::memcpy(&value, idata, sizeof(T));
                separate_nan |= !equal_nan && keys.is_nan(value);
                auto [it, inserted] = hashset.try_emplace(keys.encode(value), 0);
                it->second++;
                if (inserted && want.index) {
                    first.push_back(i);
                }
                if (inverse != NULL) {
                    inverse[i] = it - hashset.begin();
                }
            }
            give_up = separate_nan || (stop <= isize / 2 && hashset.size() > limit);
            if (!estimated && !give_up && hashset.size() > EXACT_SET_SIZE) {
                estimated = true;
                give_up = estimate_exceeds<T>(keys, idata, istride, isize / 2 - stop, limit);
                limit = std::numeric_limits<size_t>::max();
            }
        }
    }
    if (give_up) {
        return Py_NewRef(Py_NotImplemented);
    }

    npy_intp osize = hashset.size();
    if (!res.allocate(self, osize, want)) {
        return NULL;
    }

    {
        np::raii::SaveThreadState save_thread_state{};

        const auto &values = hashset.values();
        order.resize(osize);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&values, &keys](npy_intp a, npy_intp b) {
            return keys.less(values[a].first, values[b].first);
        });

        char *odata = PyArray_BYTES(res.values);
        npy_intp *index = res.data(res.index), *ocounts = res.data(res.counts);
        for (npy_intp k = 0; k < osize; k++) {
            T value = keys.decode(values[order[k]].first);
            std::memcpy(odata + k * sizeof(T), &value, sizeof(T));
            if (index != NULL) {
                index[k] = first[order[k]];
            }
            if (ocounts != NULL) {
                ocounts[k] = static_cast<npy_intp>(values[order[k]].second);
            }
        }
        if (inverse != NULL) {
            rank.resize(osize);
            for (npy_intp k = 0; k < osize; k++) {
                rank[order[k]] = k;
            }
            for (npy_intp i = 0; i < isize; i++) {
                inverse[i] = rank[inverse[i]];
            }
        }
    }

    return res.as_tuple();
}

template <typename F, typename K>
static PyObject*
unique_float_all(PyArrayObject *self, unique_wanted want, npy_bool equal_nan)
{
    PyObject *result;
    bool nearly_sorted;
    if (unique_all_sorted<F>(self, want, &result, &nearly_sorted)) {
        return result;
    }
    if (nearly_sorted && want.index) {
        return Py_NewRef(Py_NotImplemented);
    }
    if (fits_small_index(self)) {
        return unique_all_with_index<F, small_index, float_key<F, K>>(self, want, equal_nan);
    }
    return unique_all_with_index<F, large_index, float_key<F, K>>(self, want, equal_nan);
}

template <typename T>
static PyObject*
unique_integer_all(PyArrayObject *self, unique_wanted want)
{
    T min;
    npy_uint64 num_values;
    PyObject *result;
    bool nearly_sorted;
    if (unique_all_sorted<T>(self, want, &result, &nearly_sorted)) {
        return result;
    }
    if (integer_table_range<T>(self, &min, &num_values, SMALL_TYPE_TABLE_RATIO<T>)
            && unique_all_with_table<T>(self, min, num_values, want, &result)) {
        return result;
    }
    // The stable sort behind the first indices is quick on a nearly sorted
    // array, and 16 bit integers are sorted by radix, which hashing does not beat.
    if (sizeof(T) <= 2 || (nearly_sorted && want.index)) {
        return Py_NewRef(Py_NotImplemented);
    }
    if (fits_small_index(self)) {
        return unique_all_with_index<T, small_index, integer_key<T>>(self, want, NPY_TRUE);
    }
    return unique_all_with_index<T, large_index, integer_key<T>>(self, want, NPY_TRUE);
}

static PyObject*
unique_datetime(PyArrayObject *self, npy_bool equal_nan, bool *NPY_UNUSED(needs_sort))
{
    // Datetimes are hashed as integers. The integers are in another order
    // if there is a NaT, and to sort datetimes is slow.
    // With a SIMD sort for datetimes, giving up would pay off as it does for integers.
    bool unused = false;
    return unique_integer<npy_uint64>(self, equal_nan, &unused);
}

template <typename Index>
static PyObject*
unique_string_with_index(PyArrayObject *self)
{
    /*
     * Returns a new NumPy array containing the unique values of the input
     * array of fixed size strings.
     * This function uses hashing to identify uniqueness efficiently.
     */

    npy_intp itemsize = PyArray_ITEMSIZE(self);

    auto hash = [itemsize](const char *value) -> npy_uint64 {
        return hash_bytes(value, itemsize);
    };
    auto equal = [itemsize](const char *lhs, const char *rhs) -> bool {
        return std::memcmp(lhs, rhs, itemsize) == 0;
    };

    using set_type = unique_set<Index, const char *, mixed_hash<decltype(hash)>, decltype(equal)>;

    // number of elements in the input array
    npy_intp isize = PyArray_SIZE(self);
    set_type hashset(std::min(isize, HASH_TABLE_INITIAL_BUCKETS),
                     mixed_hash<decltype(hash)>{hash}, equal);

    {
        np::raii::SaveThreadState save_thread_state{};

        const char *idata = PyArray_BYTES(self);
        npy_intp istride = PyArray_STRIDES(self)[0];
        for (npy_intp i = 0; i < isize; i++, idata += istride) {
            hashset.insert(idata);
        }
    }

    PyArrayObject *res_obj = empty_array_like(self, hashset.size());
    if (res_obj == NULL) {
        return NULL;
    }

    {
        np::raii::SaveThreadState save_thread_state{};

        char *odata = PyArray_BYTES(res_obj);
        npy_intp ostride = PyArray_STRIDES(res_obj)[0];
        for (auto it = hashset.begin(); it != hashset.end(); it++, odata += ostride) {
            std::memcpy(odata, *it, itemsize);
        }
    }

    return reinterpret_cast<PyObject *>(res_obj);
}

static PyObject*
unique_string(PyArrayObject *self, npy_bool equal_nan, bool *NPY_UNUSED(needs_sort))
{
    // A string of at most 8 bytes is hashed as an integer. The integers are
    // in another order than the strings, and to sort strings is slow.
    bool unused = false;
    switch (PyArray_ITEMSIZE(self)) {
        case 1:
            return unique_integer<npy_uint8>(self, equal_nan, &unused);
        case 2:
            return unique_integer<npy_uint16>(self, equal_nan, &unused);
        case 3:
            return unique_integer<npy_uint32, 3>(self, equal_nan, &unused);
        case 4:
            return unique_integer<npy_uint32>(self, equal_nan, &unused);
        case 5:
            return unique_integer<npy_uint64, 5>(self, equal_nan, &unused);
        case 6:
            return unique_integer<npy_uint64, 6>(self, equal_nan, &unused);
        case 7:
            return unique_integer<npy_uint64, 7>(self, equal_nan, &unused);
        case 8:
            return unique_integer<npy_uint64>(self, equal_nan, &unused);
    }
    if (fits_small_index(self)) {
        return unique_string_with_index<small_index>(self);
    }
    return unique_string_with_index<large_index>(self);
}

template <typename Index>
static PyObject*
unique_vstring_with_index(PyArrayObject *self, npy_bool equal_nan)
{
    /*
     * Returns a new NumPy array containing the unique values of the input array.
     * This function uses hashing to identify uniqueness efficiently.
     */

    auto *descr = reinterpret_cast<PyArray_StringDTypeObject *>(
            PyArray_DESCR(self));
    bool equal_nulls = equal_nan || !descr->has_nan_na;
    const npy_static_string *na_string =
            descr->has_string_na ? &descr->default_string : nullptr;

    auto hash = [equal_nulls, na_string](const npy_static_string *value) -> size_t {
        if (value->buf == NULL) {
            if (na_string != nullptr) {
                value = na_string;
            }
            else if (equal_nulls) {
                return 0;
            } else {
                return std::hash<const npy_static_string *>{}(value);
            }
        }
        return hash_bytes(value->buf, value->size * sizeof(char));
    };
    auto equal = [equal_nulls, na_string](const npy_static_string *lhs, const npy_static_string *rhs) -> bool {
        // String-valued nulls compare equal to their ordinary string value.
        if (na_string != nullptr) {
            if (lhs->buf == NULL) {
                lhs = na_string;
            }
            if (rhs->buf == NULL) {
                rhs = na_string;
            }
        }
        if (lhs->buf == NULL && rhs->buf == NULL) {
            if (equal_nulls) {
                return true;
            } else {
                return lhs == rhs;
            }
        }
        if (lhs->buf == NULL || rhs->buf == NULL) {
            return false;
        }
        if (lhs->size != rhs->size) {
            return false;
        }
        return std::memcmp(lhs->buf, rhs->buf, lhs->size) == 0;
    };

    npy_intp isize = PyArray_SIZE(self);
    // unpacked_strings must live as long as hashset because hashset points
    // to values in this vector.
    std::vector<npy_static_string> unpacked_strings(isize, {0, NULL});

    using set_type = unique_set<Index, npy_static_string *,
                                        decltype(hash), decltype(equal)>;
    set_type hashset(std::min(isize, HASH_TABLE_INITIAL_BUCKETS), hash, equal);

    {
        np::raii::SaveThreadState save_thread_state{};
        np::raii::NpyStringAcquireAllocator alloc(descr);

        char *idata = PyArray_BYTES(self);
        npy_intp istride = PyArray_STRIDES(self)[0];

        for (npy_intp i = 0; i < isize; i++, idata += istride) {
            npy_packed_static_string *packed_string =
                reinterpret_cast<npy_packed_static_string *>(idata);
            int is_null = NpyString_load(alloc.allocator(), packed_string,
                                         &unpacked_strings[i]);
            if (is_null == -1) {
                // Unexpected error. Throw a C++ exception that will be caught
                // by the caller of unique_vstring() and converted into a Python
                // RuntimeError.
                throw std::runtime_error("Failed to load string from packed "
                                         "static string.");
            }
            hashset.insert(&unpacked_strings[i]);
        }
    }

    PyArrayObject *res_obj = empty_array_like(self, hashset.size());
    if (res_obj == NULL) {
        return NULL;
    }

    {
        PyArray_StringDTypeObject *res_descr =
            reinterpret_cast<PyArray_StringDTypeObject *>(PyArray_DESCR(res_obj));
        np::raii::SaveThreadState save_thread_state{};
        np::raii::NpyStringAcquireAllocator alloc(res_descr);

        char *odata = PyArray_BYTES(res_obj);
        npy_intp ostride = PyArray_STRIDES(res_obj)[0];
        for (auto it = hashset.begin(); it != hashset.end(); it++, odata += ostride) {
            npy_packed_static_string *packed_string =
                reinterpret_cast<npy_packed_static_string *>(odata);
            int pack_status = 0;
            if ((*it)->buf == NULL) {
                pack_status = NpyString_pack_null(alloc.allocator(), packed_string);
            } else {
                pack_status = NpyString_pack(alloc.allocator(), packed_string,
                                             (*it)->buf, (*it)->size);
            }
            if (pack_status == -1) {
                // string packing failed
                return NULL;
            }
        }
    }
    return reinterpret_cast<PyObject *>(res_obj);
}

static PyObject*
unique_vstring(PyArrayObject *self, npy_bool equal_nan, bool *NPY_UNUSED(needs_sort))
{
    if (fits_small_index(self)) {
        return unique_vstring_with_index<small_index>(self, equal_nan);
    }
    return unique_vstring_with_index<large_index>(self, equal_nan);
}

// this map contains the functions used for each item size.
// The last argument tells the functions whether sorted values are wanted.
// They clear it if the values, read as integers of the template type, come
// out in ascending order.
typedef std::function<PyObject *(PyArrayObject *, npy_bool, bool *)> function_type;
std::unordered_map<int, function_type> unique_funcs = {
    {NPY_BYTE, unique_integer<npy_byte>},
    {NPY_UBYTE, unique_integer<npy_ubyte>},
    {NPY_SHORT, unique_integer<npy_short>},
    {NPY_USHORT, unique_integer<npy_ushort>},
    {NPY_INT, unique_integer<npy_int>},
    {NPY_UINT, unique_integer<npy_uint>},
    {NPY_LONG, unique_integer<npy_long>},
    {NPY_ULONG, unique_integer<npy_ulong>},
    {NPY_LONGLONG, unique_integer<npy_longlong>},
    {NPY_ULONGLONG, unique_integer<npy_ulonglong>},
    {NPY_FLOAT, unique_float<npy_float, npy_uint32>},
    {NPY_DOUBLE, unique_float<npy_double, npy_uint64>},
    {NPY_CFLOAT, unique_complex<npy::cfloat_tag>},
    {NPY_CDOUBLE, unique_complex<npy::cdouble_tag>},
    {NPY_CLONGDOUBLE, unique_complex<npy::clongdouble_tag>},
    {NPY_INT8, unique_integer<npy_int8>},
    {NPY_INT16, unique_integer<npy_int16>},
    {NPY_INT32, unique_integer<npy_int32>},
    {NPY_INT64, unique_integer<npy_int64>},
    {NPY_UINT8, unique_integer<npy_uint8>},
    {NPY_UINT16, unique_integer<npy_uint16>},
    {NPY_UINT32, unique_integer<npy_uint32>},
    {NPY_UINT64, unique_integer<npy_uint64>},
    {NPY_DATETIME, unique_datetime},
    {NPY_COMPLEX64, unique_complex<npy::cfloat_tag>},
    {NPY_COMPLEX128, unique_complex<npy::cdouble_tag>},
    {NPY_STRING, unique_string},
    {NPY_UNICODE, unique_string},
    {NPY_VSTRING, unique_vstring},
};


/**
 * Python exposed implementation of `_unique_hash`.
 *
 * This is a C only function wrapping code that may cause C++ exceptions into
 * try/catch.
 *
 * @param arr NumPy array to find the unique values of.
 * @param sorted Whether to sort the unique values.
 * @return Base-class NumPy array with unique values, `NotImplemented` if the
 * type is unsupported or `NULL` with an error set.
 */
extern "C" NPY_NO_EXPORT PyObject *
array__unique_hash(PyObject *NPY_UNUSED(module),
                   PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyArrayObject *arr = NULL;
    npy_bool equal_nan = NPY_TRUE;  // default to True
    npy_bool sort = NPY_FALSE;

    NPY_PREPARE_ARGPARSER;
    if (npy_parse_arguments("_unique_hash", args, len_args, kwnames,
            {"arr", (void *)&PyArray_Converter, &arr},
            {"|equal_nan", (void *)&PyArray_BoolConverter, &equal_nan},
            {"|sorted", (void *)&PyArray_BoolConverter, &sort}) < 0) {
        Py_XDECREF(arr);
        return NULL;
    }

    PyObject *result = NULL;
    try {
        auto type = PyArray_TYPE(arr);
        // we only support data types present in our unique_funcs map
        if (unique_funcs.find(type) == unique_funcs.end()) {
            result = Py_NewRef(Py_NotImplemented);
        }
        else {
            bool needs_sort = sort;
            result = unique_funcs[type](arr, equal_nan, &needs_sort);
            // byte-swapped integers are not in the order of the integers read
            if (PyArray_ISBYTESWAPPED(arr)) {
                needs_sort = sort;
            }
            if (needs_sort && result != NULL && result != Py_NotImplemented
                    && PyArray_Sort(reinterpret_cast<PyArrayObject *>(result),
                                    0, NPY_QUICKSORT) < 0) {
                Py_CLEAR(result);
            }
        }
    }
    catch (const std::bad_alloc &e) {
        PyErr_NoMemory();
        result = NULL;
    }
    catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        result = NULL;
    }
    Py_DECREF(arr);
    return result;
}

/**
 * Python exposed implementation of `_unique_hash_all`.
 *
 * @param arr NumPy array to find the unique values of.
 * @return A tuple of the unique values in ascending order, the indices of
 * their first occurrences, the inverse indices and the counts, with `None`
 * for those of the last three that were not asked for. `NotImplemented` if
 * the array is not supported or better sorted, or `NULL` with an error set.
 */
extern "C" NPY_NO_EXPORT PyObject *
array__unique_hash_all(PyObject *NPY_UNUSED(module),
                       PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyArrayObject *arr = NULL;
    npy_bool index = NPY_FALSE, inverse = NPY_FALSE, counts = NPY_FALSE;
    npy_bool equal_nan = NPY_TRUE;

    NPY_PREPARE_ARGPARSER;
    if (npy_parse_arguments("_unique_hash_all", args, len_args, kwnames,
            {"arr", (void *)&PyArray_Converter, &arr},
            {"|return_index", (void *)&PyArray_BoolConverter, &index},
            {"|return_inverse", (void *)&PyArray_BoolConverter, &inverse},
            {"|return_counts", (void *)&PyArray_BoolConverter, &counts},
            {"|equal_nan", (void *)&PyArray_BoolConverter, &equal_nan}) < 0) {
        Py_XDECREF(arr);
        return NULL;
    }

    unique_wanted want = {index != 0, inverse != 0, counts != 0};
    PyObject *result = NULL;
    try {
        if (PyArray_NDIM(arr) != 1 || PyArray_ISBYTESWAPPED(arr)) {
            result = Py_NewRef(Py_NotImplemented);
        }
        else {
            switch (PyArray_TYPE(arr)) {
#define UNIQUE_ALL_CASE(TYPENUM, TYPE) \
                case TYPENUM: result = unique_integer_all<TYPE>(arr, want); break
                UNIQUE_ALL_CASE(NPY_BYTE, npy_byte);
                UNIQUE_ALL_CASE(NPY_UBYTE, npy_ubyte);
                UNIQUE_ALL_CASE(NPY_SHORT, npy_short);
                UNIQUE_ALL_CASE(NPY_USHORT, npy_ushort);
                UNIQUE_ALL_CASE(NPY_INT, npy_int);
                UNIQUE_ALL_CASE(NPY_UINT, npy_uint);
                UNIQUE_ALL_CASE(NPY_LONG, npy_long);
                UNIQUE_ALL_CASE(NPY_ULONG, npy_ulong);
                UNIQUE_ALL_CASE(NPY_LONGLONG, npy_longlong);
                UNIQUE_ALL_CASE(NPY_ULONGLONG, npy_ulonglong);
#undef UNIQUE_ALL_CASE
                case NPY_FLOAT:
                    result = unique_float_all<npy_float, npy_uint32>(arr, want, equal_nan);
                    break;
                case NPY_DOUBLE:
                    result = unique_float_all<npy_double, npy_uint64>(arr, want, equal_nan);
                    break;
                default: result = Py_NewRef(Py_NotImplemented);
            }
        }
    }
    catch (const std::bad_alloc &e) {
        PyErr_NoMemory();
        result = NULL;
    }
    catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        result = NULL;
    }
    Py_DECREF(arr);
    return result;
}
