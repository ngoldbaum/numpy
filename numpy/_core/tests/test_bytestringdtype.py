"""Tests for the bytes-specific behavior of ByteStringDType.

Coverage shared with StringDType lives in test_stringdtype.py
(TestVariableWidthShared).
"""

import copy
import pickle
import subprocess
import sys
import textwrap
import warnings

import pytest

import numpy as np
from numpy._core.tests.test_stringdtype import INVALID_UTF8
from numpy.dtypes import ByteStringDType, StringDType
from numpy.testing import assert_array_equal


def R(*values):
    return np.array(list(values), dtype=ByteStringDType())


NUL_AND_HIGH_BYTE_VALUES = [
    b"hello world",
    b"",
    b"x\x00",            # trailing NUL
    b"a\x00b",           # embedded NUL
    b"\x00\x00",         # all NULs
    b"\xff\xfe\x80",     # invalid UTF-8 lead/continuation bytes
    b"caf\xc3\xa9",      # valid UTF-8 (must still be treated as raw bytes)
    b"  spaced  ",
    b"\x00\xff" * 12,           # arena-length (>15 bytes), NULs + high bytes
    b"long " * 5 + b"tail\x00",  # arena-length, trailing NUL
]


@pytest.fixture
def dtype():
    return ByteStringDType()


class TestConstruction:
    def test_default(self, dtype):
        assert not hasattr(dtype, "coerce")
        with pytest.raises(AttributeError):
            dtype.na_object

    def test_no_coerce_kwarg(self):
        with pytest.raises(TypeError):
            ByteStringDType(coerce=False)
        with pytest.raises(TypeError):
            ByteStringDType(coerce=True)

    @pytest.mark.parametrize("na", [b"", b"\x00", b"NA", np.nan, None])
    def test_na_object(self, na):
        dt = ByteStringDType(na_object=na)
        if na is np.nan:
            assert dt.na_object is np.nan
        else:
            assert dt.na_object == na
        assert dt != ByteStringDType()
        assert dt == ByteStringDType(na_object=na)
        assert repr(dt) == f"ByteStringDType(na_object={na!r})"

    def test_from_char_and_letter(self, dtype):
        assert np.dtype("R") == dtype
        assert dtype.char == "R"
        assert dtype.kind == "R"
        assert np.dtype("R").name == "ByteStringDType128"

    def test_pickle_roundtrip(self, dtype):
        for dt in [dtype, ByteStringDType(na_object=b"\x00"),
                   ByteStringDType(na_object=None)]:
            assert pickle.loads(pickle.dumps(dt)) == dt

    def test_bytes_inference_unchanged(self):
        # registering ByteStringDType must not change default inference
        assert np.array([b"x"]).dtype == np.dtype("S1")
        assert np.array(b"x").dtype == np.dtype("S1")


class TestStrictBytesInput:
    @pytest.mark.parametrize("value", [
        "text",
        1,
        1.5,
        None,
        object(),
    ])
    def test_rejected_with_typeerror(self, dtype, value):
        with pytest.raises(TypeError,
                           match="only allows bytes data"):
            np.array([value], dtype=dtype)
        arr = np.empty(1, dtype=dtype)
        with pytest.raises(TypeError, match="only allows bytes data"):
            arr[0] = value

    @pytest.mark.parametrize("value", [
        bytearray(b"buffer"),
        memoryview(b"buffer"),
    ])
    def test_buffer_protocol_rejected(self, dtype, value):
        # setitem rejects buffer-protocol objects with the clear message;
        # array coercion reads them as uint8 buffers first, so np.array
        # fails on the (unregistered) uint8 cast instead
        arr = np.empty(1, dtype=dtype)
        with pytest.raises(TypeError, match="only allows bytes data"):
            arr[0] = value
        with pytest.raises(TypeError):
            np.array([value], dtype=dtype)

    def test_str_rejection_mentions_encode(self, dtype):
        with pytest.raises(TypeError, match="str.encode"):
            np.array(["x"], dtype=dtype)
        with pytest.raises(TypeError) as exc_info:
            np.array([1], dtype=dtype)
        assert "str.encode" not in str(exc_info.value)

    def test_bytes_and_np_bytes_accepted(self, dtype):
        arr = np.array([b"x", np.bytes_(b"y")], dtype=dtype)
        assert arr.tolist() == [b"x", b"y"]

    def test_bytes_subclasses_accepted(self, dtype):
        # matches the fixed-width S dtype's input domain
        class MyBytes(bytes):
            pass
        arr = np.empty(2, dtype=dtype)
        arr[0] = MyBytes(b"q\x00")
        arr[1] = np.bytes_(b"a\x00b")
        assert arr.tolist() == [b"q\x00", b"a\x00b"]
        assert type(arr[0]) is np.vbytes
        arr2 = np.array([MyBytes(b"x\x00")], dtype=dtype)
        assert arr2.tolist() == [b"x\x00"]


class TestSetGet:
    def test_returns_vbytes(self, dtype):
        arr = np.array([b"abc"], dtype=dtype)
        assert type(arr[0]) is np.vbytes
        assert isinstance(arr[0], bytes)
        assert type(arr.item()) is bytes
        assert type(arr.tolist()[0]) is bytes

    @pytest.mark.parametrize("value", NUL_AND_HIGH_BYTE_VALUES)
    def test_roundtrip(self, dtype, value):
        arr = np.array([value], dtype=dtype)
        assert arr[0] == value
        arr2 = np.empty(1, dtype=dtype)
        arr2[0] = value
        assert arr2[0] == value

    def test_np_bytes_trailing_nul_preserved(self, dtype):
        # np.bytes_ must be a known scalar type: discovered through its
        # fixed-width 'S' descriptor instead, the S->R cast strips the NULs
        value = np.bytes_(b"q\x00")
        arr = np.empty(1, dtype=dtype)
        arr[0] = value
        assert arr[0] == b"q\x00"
        arr[0:1] = [value]
        assert arr[0] == b"q\x00"
        assert np.array([value], dtype=dtype)[0] == b"q\x00"

    def test_zeros_is_empty_bytes(self, dtype):
        assert np.zeros(3, dtype=dtype).tolist() == [b""] * 3


class TestSortingAndSelection:
    def test_sort_high_bytes(self, dtype):
        # bytewise, not codepoint, ordering
        arr = np.array([b"\xff", b"a", b"\x80"], dtype=dtype)
        assert np.sort(arr).tolist() == [b"a", b"\x80", b"\xff"]

    @pytest.mark.parametrize("na", [b"", b"x", b"\x00"])
    def test_null_truthiness_matches_bytes_sentinel(self, na):
        # a bytes na_object takes the string-NA path: a null is truthy
        # exactly when the sentinel is a nonempty bytes
        arr = np.array([na, b"y"], dtype=ByteStringDType(na_object=na))
        assert arr.astype(bool).tolist() == [bool(na), True]
        assert arr.nonzero()[0].tolist() == ([0, 1] if na else [1])


class TestCasts:
    def test_self_cast(self, dtype):
        arr = np.array([b"x" * 30, b"a\x00b"], dtype=dtype)
        assert arr.astype(dtype).tolist() == arr.tolist()
        assert arr.astype(ByteStringDType(na_object=b"")).tolist() == \
            arr.tolist()

    def test_self_cast_null_to_no_na(self, dtype):
        # the null becomes the NA's bytes, not its repr (b"b'NAA'")
        arr = np.array([b"yo", b"NAA"], dtype=ByteStringDType(na_object=b"NAA"))
        assert arr.astype(dtype, casting="unsafe").tolist() == [b"yo", b"NAA"]
        sarr = np.array(["yo", "NAA"], dtype=StringDType(na_object="NAA"))
        assert sarr.astype(StringDType(), casting="unsafe").tolist() == \
            ["yo", "NAA"]

    def test_fixed_width_roundtrip_high_bytes(self, dtype):
        # no ASCII gate in either direction, unlike StringDType
        arr = np.array([b"ab", b"\xff\xfe", b""], dtype=dtype)
        fixed = arr.astype("S4")
        assert fixed.tolist() == [b"ab", b"\xff\xfe", b""]
        assert fixed.astype(dtype).tolist() == [b"ab", b"\xff\xfe", b""]

    def test_fixed_width_truncates(self, dtype):
        arr = np.array([b"abcdef"], dtype=dtype)
        assert arr.astype("S3").tolist() == [b"abc"]

    @pytest.mark.parametrize("bad", INVALID_UTF8)
    def test_fixed_width_source_skips_utf8_validation(self, dtype, bad):
        # the same input is rejected by the S -> StringDType cast
        arr = np.array([bad], dtype=f"S{len(bad)}")
        assert arr.astype(dtype).tolist() == [bad]

    def test_fixed_width_source_strips_trailing_nuls(self, dtype):
        # 'S' cannot represent trailing NULs, so they are already gone in
        # the source of the S -> ByteStringDType cast
        fixed = np.array([b"x\x00"], dtype="S4")
        assert fixed.astype(dtype).tolist() == [b"x"]

    def test_fixed_width_source_is_safe_cast(self, dtype):
        assert np.can_cast("S4", dtype, casting="safe")
        assert not np.can_cast(dtype, "S4", casting="safe")
        assert not np.can_cast("V4", dtype, casting="safe")
        # searchsorted needs a safe cast of the needle
        arr = np.array([b"a", b"c"], dtype=dtype)
        assert arr.searchsorted(np.array([b"b"], dtype="S1")).tolist() == [1]

    def test_void_roundtrip_preserves_nuls(self, dtype):
        # void -> ByteStringDType is length-explicit, with no UTF-8 validation
        arr = np.array([b"ab", b"\xff", b""], dtype=dtype)
        v = arr.astype("V4")
        assert v.astype(dtype).tolist() == [
            b"ab\x00\x00", b"\xff\x00\x00\x00", b"\x00\x00\x00\x00"]

    def test_bool_casts(self, dtype):
        arr = np.array([b"x", b"", b"\x00"], dtype=dtype)
        assert arr.astype(bool).tolist() == [True, False, True]
        assert np.array([True, False]).astype(dtype).tolist() == \
            [b"True", b"False"]

    def test_object_roundtrip(self, dtype):
        values = [b"ab", b"\xff", b"a\x00b"]
        obj = np.array(values, dtype=object)
        arr = obj.astype(dtype)
        assert arr.tolist() == values
        assert arr.astype(object).tolist() == values

    @pytest.mark.parametrize("other", [
        "int64", "uint64", "float64", "complex128", "datetime64[s]",
        "timedelta64[s]", "U4", StringDType()])
    def test_no_cast_to_or_from(self, dtype, other):
        # text and numeric conversions are deliberately unregistered;
        # text goes through the explicit encode/decode ufuncs
        arr = np.array([b"1"], dtype=dtype)
        with pytest.raises(TypeError):
            arr.astype(other)
        if isinstance(other, str):
            other_arr = np.array(["1"], dtype=other) \
                if other == "U4" else np.zeros(1, dtype=other)
        else:
            other_arr = np.array(["1"], dtype=other)
        with pytest.raises(TypeError):
            other_arr.astype(dtype)

    def test_structured_field_rejected(self, dtype):
        with pytest.raises(TypeError, match="not currently supported"):
            np.dtype([("field", dtype)])
        with pytest.raises(TypeError, match="not currently supported"):
            np.dtype([("field", dtype, 2)])
        with pytest.raises(TypeError, match="not currently supported"):
            np.dtype({"names": ["a"], "formats": [dtype]})
        with pytest.raises(TypeError, match="not currently supported"):
            np.dtype("R,i4")

    def test_subarray_dtype_rejected(self, dtype):
        with pytest.raises(TypeError,
                           match="not currently supported within subarray"):
            np.dtype((dtype, 2))
        # (dtype, ()) is equivalent to the dtype itself and remains allowed
        assert np.dtype((dtype, ())) == dtype


class TestUnsizedFixedWidthCasts:
    """Converting to an unsized "S" or "V" dtype infers the width, counted
    in bytes, from the values in the array being converted."""

    @pytest.mark.parametrize(
        "convert",
        [pytest.param(lambda arr, req: arr.astype(req), id="astype"),
         pytest.param(lambda arr, req: np.array(arr, dtype=req),
                      id="np.array")])
    @pytest.mark.parametrize("kind", ["S", "V"])
    @pytest.mark.parametrize(
        "values,width",
        [
            ([b"this", b"is", b"an", b"array"], 5),
            ([b"a" * 100, b"", b"b"], 100),
            # embedded and trailing NULs count as data
            ([b"x\0", b"y\0\0z", b""], 4),
            ([b"\xff\xfe", b"\x00" * 3], 3),
            # empty arrays and all-empty entries produce width 1
            ([], 1),
            ([b"", b"", b""], 1),
        ],
    )
    def test_infer_width(self, convert, kind, values, width, dtype):
        arr = np.array(values, dtype=dtype)
        res = convert(arr, kind)
        assert res.dtype == np.dtype(f"{kind}{width}")
        assert_array_equal(res, np.array(values, dtype=f"{kind}{width}"))

    @pytest.mark.parametrize(
        "na,width",
        [
            (None, 4),           # missing entries count as the repr "None"
            (np.nan, 3),
            (b"", 1),
            (b"miss\0", 5),
        ],
    )
    def test_missing_values(self, na, width):
        dt = ByteStringDType(na_object=na)
        arr = np.array([b"ab", na], dtype=dt)
        assert arr.astype("S").dtype == np.dtype(f"S{max(width, 2)}")
        all_null = np.array([na, na], dtype=dt)
        assert all_null.astype("S").dtype == np.dtype(f"S{width}")

    def test_explicit_width_still_truncates(self, dtype):
        arr = np.array([b"abcdef"], dtype=dtype)
        assert_array_equal(arr.astype("S3"), np.array([b"abc"], dtype="S3"))
        assert_array_equal(arr.astype("V3"), np.array([b"abc"], dtype="V3"))

    def test_descriptor_only_resolution_still_fails(self, dtype):
        # functions that adapt descriptors without inspecting array values
        # cannot infer a width
        arr = np.array([b"abc"], dtype=dtype)
        with pytest.raises(TypeError, match="cast"):
            np.concatenate([arr, arr], dtype="S")

    @pytest.mark.parametrize("unicode_dtype", ["U", "U3"])
    def test_unicode_target_fails(self, unicode_dtype, dtype):
        # no R to U cast, sized or not; the width discovery must not run
        arr = np.array([b"\xff\xfe"], dtype=dtype)
        with pytest.raises(TypeError, match="cast"):
            arr.astype(unicode_dtype)


class TestNAObject:
    def test_bytes_na_is_stringlike(self):
        dt = ByteStringDType(na_object=b"\x00")
        arr = np.empty(2, dtype=dt)
        arr[0] = b"x"
        assert np.sort(arr).tolist() == [b"", b"x"]

    def test_nan_na(self):
        import math
        dt = ByteStringDType(na_object=np.nan)
        arr = np.array([b"x", np.nan], dtype=dt)
        assert math.isnan(arr[1])
        assert np.isnan(arr).tolist() == [False, True]

    def test_scalar_unpickle_guard(self, dtype):
        arr = np.array([b"x"], dtype=dtype)
        # the list-pickle path stores full arrays
        import numpy._core.multiarray as mu
        with pytest.raises(TypeError, match="Cannot unpickle"):
            mu.scalar(dtype, b"x")


class TestScalar:
    def test_type_registration(self, dtype):
        assert ByteStringDType.type is np.vbytes
        assert issubclass(np.vbytes, bytes)
        assert issubclass(np.vbytes, np.generic)
        assert np.dtype(np.vbytes) == dtype
        assert np.dtype("vbytes") == dtype
        assert np.vbytes(b"x").dtype == dtype
        assert np.sctypeDict["vbytes"] is np.vbytes
        assert np.vbytes in np.ScalarType
        assert np.issubdtype(dtype, np.generic)
        assert np.issubdtype(dtype, np.vbytes)
        assert not np.issubdtype(dtype, np.bytes_)

    @pytest.mark.parametrize("value", NUL_AND_HIGH_BYTE_VALUES)
    def test_bytes_api(self, value):
        x = np.vbytes(value)
        assert type(x) is np.vbytes
        assert x == value
        assert type(x == value) is bool
        assert bytes(x) == value
        assert len(x) == len(value)
        assert hash(x) == hash(value)
        assert x[:] == value
        assert x.item() == value
        assert type(x.item()) is bytes

    def test_constructor_arguments(self):
        assert np.vbytes() == b""
        assert np.vbytes("caf\xe9", "utf-8") == b"caf\xc3\xa9"
        with pytest.raises(TypeError):
            np.vbytes("text")

    def test_repr_str(self):
        x = np.vbytes(b"x\x00")
        assert repr(x) == "np.vbytes(b'x\\x00')"
        assert str(x) == "b'x\\x00'"
        with np.printoptions(legacy="1.25"):
            assert repr(x) == "b'x\\x00'"

    @pytest.mark.parametrize("value", NUL_AND_HIGH_BYTE_VALUES)
    def test_element_access(self, dtype, value):
        arr = np.array([value], dtype=dtype)
        for x in (arr[0], arr.flat[0], next(iter(arr))):
            assert type(x) is np.vbytes
            assert x == value
        assert arr.item() == value
        assert type(arr.item()) is bytes
        assert np.array(value, dtype=dtype)[()] == value
        assert type(np.array(value, dtype=dtype)[()]) is np.vbytes

    def test_inference(self, dtype):
        arr = np.array([np.vbytes(b"x\x00")])
        assert arr.dtype == dtype
        assert arr[0] == b"x\x00"
        assert np.array(np.vbytes(b"a\x00"))[()] == b"a\x00"
        mixed = np.array([np.vbytes(b"x"), b"y"])
        assert mixed.dtype == dtype
        assert mixed.tolist() == [b"x", b"y"]
        assert np.array([b"x"]).dtype == np.dtype("S1")
        assert np.full(2, np.vbytes(b"x\x00")).tolist() == [b"x\x00"] * 2

    def test_setitem(self, dtype):
        arr = np.empty(2, dtype=dtype)
        arr[0] = np.vbytes(b"q\x00")
        arr[1:] = [np.vbytes(b"a\x00b")]
        assert arr.tolist() == [b"q\x00", b"a\x00b"]
        arr.fill(np.vbytes(b"z\x00"))
        assert arr.tolist() == [b"z\x00"] * 2

    def test_ufunc_operand(self, dtype):
        arr = np.array([b"x\x00", b"a\x00b"], dtype=dtype)
        assert np.strings.find(arr, np.vbytes(b"\x00")).tolist() == [1, 1]
        assert (arr + np.vbytes(b"y\x00")).tolist() == [b"x\x00y\x00", b"a\x00by\x00"]
        assert (arr == np.vbytes(b"x\x00")).tolist() == [True, False]
        assert type(arr.max()) is np.vbytes
        assert arr.max() == b"x\x00"

    def test_scalar_expressions_keep_bytes_semantics(self):
        x = np.vbytes(b"a")
        assert x + b"b" == b"ab"
        assert type(x + b"b") is bytes
        assert x.decode() == "a"

    def test_pickle_and_copy(self):
        x = np.vbytes(b"x\x00\xff")
        for y in (pickle.loads(pickle.dumps(x)), copy.copy(x), copy.deepcopy(x)):
            assert type(y) is np.vbytes
            assert y == x

    def test_null_returns_na_object(self):
        dt = ByteStringDType(na_object=None)
        arr = np.array([b"x", None], dtype=dt)
        assert type(arr[0]) is np.vbytes
        assert arr[1] is None
        assert list(arr) == [b"x", None]

    def test_no_implicit_text_conversion(self):
        with pytest.raises(TypeError):
            np.array([np.vbytes(b"x")], dtype=StringDType())

    def test_subclass(self, dtype):
        class MyV(np.vbytes):
            pass

        x = MyV(b"x\x00")
        assert isinstance(x, np.vbytes)
        arr = np.array([x])
        assert arr.dtype == dtype
        assert arr[0] == b"x\x00"
        assert type(arr[0]) is np.vbytes


class TestComparisonUfuncs:
    def test_eq_ne_ordering(self):
        a = R(b"a\x00b", b"", b"\xff")
        b = R(b"a\x00b", b"x", b"\xff")
        assert (a == b).tolist() == [True, False, True]
        assert (a != b).tolist() == [False, True, False]
        assert (a < b).tolist() == [False, True, False]
        assert (a >= b).tolist() == [True, False, True]

    def test_trailing_nul_distinct(self):
        # the 'S' defect motivating the dtype
        x = R(b"x", b"x\x00")
        y = R(b"x\x00", b"x\x00")
        assert (x == y).tolist() == [False, True]
        assert (x < y).tolist() == [True, False]

    def test_vs_fixed_width(self):
        a = R(b"abc", b"\xff")
        s = np.array([b"abc", b"z"], dtype="S3")
        assert (a == s).tolist() == [True, False]
        assert (s == a).tolist() == [True, False]
        assert (a > s).tolist() == [False, True]
        # the fixed elements are NUL-padded to the itemsize; the padding is
        # not part of the value on the ByteStringDType side either
        padded = np.array([b"abc", b"\xff"], dtype="S6")
        assert (a == padded).tolist() == [True, True]

    def test_vs_text_follows_python(self):
        # "a" != b"a": equality is elementwise False, ordering raises
        a = R(b"a", b"b")
        for text in [np.array(["a", "b"]),
                     np.array(["a", "b"], dtype=StringDType())]:
            assert (a == text).tolist() == [False, False]
            assert (text == a).tolist() == [False, False]
            assert (a != text).tolist() == [True, True]
            with pytest.raises(TypeError):
                a < text
            with pytest.raises(TypeError):
                text < a

    def test_vs_numeric(self):
        a = R(b"1", b"2")
        assert (a == np.array([1, 2])).tolist() == [False, False]
        with pytest.raises(TypeError):
            a < np.array([1, 2])

    def test_vs_object(self):
        # object operands promote to the object loop, so each element
        # compares with Python semantics, like StringDType and 'S'
        a = R(b"a\x00", b"b")
        obj = np.array([b"a\x00", b"z"], dtype=object)
        assert (a == obj).tolist() == [True, False]
        assert (obj == a).tolist() == [True, False]
        assert (a != obj).tolist() == [False, True]
        assert (a < obj).tolist() == [False, True]
        assert (obj > a).tolist() == [False, True]
        mixed = np.array(["a\x00", None], dtype=object)
        assert (a == mixed).tolist() == [False, False]
        assert (mixed != a).tolist() == [True, True]
        with pytest.raises(TypeError):
            a < np.array(["a", "b"], dtype=object)

    def test_no_promotion_with_text(self):
        with pytest.raises(TypeError):
            np.result_type(ByteStringDType(), StringDType())
        with pytest.raises(TypeError):
            np.result_type(ByteStringDType(), np.dtype("U4"))
        assert np.result_type(ByteStringDType(), np.dtype("S4")) == \
            ByteStringDType()


class TestStringUfuncs:
    """Oracle: the matching Python bytes method, elementwise."""

    def test_str_len_is_byte_length(self):
        a = np.array(NUL_AND_HIGH_BYTE_VALUES, dtype=ByteStringDType())
        assert np.strings.str_len(a).tolist() == \
            [len(v) for v in NUL_AND_HIGH_BYTE_VALUES]

    def test_isalpha(self):
        a = np.array(NUL_AND_HIGH_BYTE_VALUES, dtype=ByteStringDType())
        assert np.strings.isalpha(a).tolist() == \
            [v.isalpha() for v in NUL_AND_HIGH_BYTE_VALUES]

    @pytest.mark.parametrize("needle", [b"l", b"\x00", b"\xff", b"a\x00"])
    def test_find_count(self, needle):
        vals = NUL_AND_HIGH_BYTE_VALUES
        a = np.array(vals, dtype=ByteStringDType())
        n = np.array([needle], dtype=ByteStringDType())
        assert np.strings.find(a, n).tolist() == \
            [v.find(needle) for v in vals]
        assert np.strings.count(a, n).tolist() == \
            [v.count(needle) for v in vals]

    def test_find_with_offsets(self):
        a = R(b"ababab")
        n = R(b"ab")
        assert np.strings.find(a, n, 1)[0] == b"ababab".find(b"ab", 1)
        assert np.strings.find(a, n, 1, 3)[0] == b"ababab".find(b"ab", 1, 3)

    @pytest.mark.parametrize("old,new", [
        (b"\x00", b"-"),
        (b"\xff", b"ZZ"),
        (b"l", b""),
        (b"", b"@"),
    ])
    def test_replace(self, old, new):
        vals = NUL_AND_HIGH_BYTE_VALUES
        a = np.array(vals, dtype=ByteStringDType())
        r = np.strings.replace(a, np.array([old], dtype=ByteStringDType()),
                               np.array([new], dtype=ByteStringDType()))
        assert r.tolist() == [v.replace(old, new) for v in vals]

    def test_strip_whitespace_preserves_nuls(self):
        vals = [b"  hi  ", b"x\x00 ", b" \x00x", b"\x00", b""]
        a = np.array(vals, dtype=ByteStringDType())
        assert np.strings.strip(a).tolist() == [v.strip() for v in vals]
        assert np.strings.lstrip(a).tolist() == [v.lstrip() for v in vals]
        assert np.strings.rstrip(a).tolist() == [v.rstrip() for v in vals]

    def test_add_multiply(self):
        vals = NUL_AND_HIGH_BYTE_VALUES
        a = np.array(vals, dtype=ByteStringDType())
        assert np.strings.add(a, a).tolist() == [v + v for v in vals]
        assert np.strings.multiply(a, 3).tolist() == [v * 3 for v in vals]
        assert (a * np.array([2] * len(vals))).tolist() == \
            [v * 2 for v in vals]

    def test_minimum_maximum(self):
        x = R(b"a", b"z", b"\xff", b"x\x00", b"a\x00b")
        y = R(b"b", b"b", b"a", b"x", b"a\x00a")
        assert np.minimum(x, y).tolist() == \
            [min(a, b) for a, b in zip(x.tolist(), y.tolist())]
        assert np.maximum(x, y).tolist() == \
            [max(a, b) for a, b in zip(x.tolist(), y.tolist())]

    def test_minimum_maximum_mixed_operands(self):
        x = R(b"b", b"y")
        assert np.minimum(x, b"c\x00").tolist() == [b"b", b"c\x00"]
        assert np.maximum(x, b"c\x00").tolist() == [b"c\x00", b"y"]
        fixed = np.array([b"c", b"c"], dtype="S1")
        assert np.minimum(x, fixed).tolist() == [b"b", b"c"]
        assert np.maximum(fixed, x).dtype == ByteStringDType()

    @pytest.mark.parametrize("val, sep", [
        (b"a\x00b\x00c", b"\x00"),
        (b"\xff\x00hi\xff tail\x00", b"\xff "),
    ])
    def test_partition_rpartition(self, val, sep):
        a = R(val)
        for fn, oracle in [(np.strings.partition, val.partition),
                           (np.strings.rpartition, val.rpartition)]:
            parts = fn(a, R(sep))
            assert tuple(p[0] for p in parts) == oracle(sep)

    def test_partition_bytes_scalar_sep(self):
        # a plain bytes separator converts directly to ByteStringDType;
        # through a fixed-width 'S' intermediate a b"\x00" separator
        # would collapse to the empty string
        vals = [b"a b\x00c", b"d e"]
        a = R(*vals)
        for sep in (b" ", b"\x00"):
            for fn, oracle in [(np.strings.partition, bytes.partition),
                               (np.strings.rpartition, bytes.rpartition)]:
                parts = fn(a, sep)
                assert parts[0].dtype == a.dtype
                for i, val in enumerate(vals):
                    assert tuple(p[i] for p in parts) == oracle(val, sep)
        with pytest.raises(ValueError, match="empty separator"):
            np.strings.partition(a, b"")

    def test_pybytes_scalar_ufunc_operand_preserves_nulls(self):
        # an exact bytes operand converts directly to ByteStringDType, so
        # trailing nulls survive; a fixed-width 'S' intermediate would
        # strip them as padding
        arr = R(b"abc\x00", b"abc")

        assert (arr == b"abc\x00").tolist() == [True, False]
        assert (arr != b"abc\x00").tolist() == [False, True]
        assert (arr + b"x\x00").tolist() == [b"abc\x00x\x00", b"abcx\x00"]
        assert (b"x\x00" + arr).tolist() == [b"x\x00abc\x00", b"x\x00abc"]
        assert np.strings.str_len(arr + b"x\x00").tolist() == [6, 5]

        arr2 = arr.copy()
        arr2 += b"\x00"
        assert arr2.tolist() == [b"abc\x00\x00", b"abc\x00"]

        assert np.strings.count(arr, b"\x00").tolist() == [1, 0]
        assert np.strings.find(arr, b"c\x00").tolist() == [2, -1]
        assert np.strings.replace(arr, b"\x00", b"!").tolist() == \
            [b"abc!", b"abc"]

        # subclass operands, np.bytes_ included, keep fixed-width semantics
        assert (arr + np.bytes_(b"x\x00")).tolist() == [b"abc\x00x", b"abcx"]
        assert np.strings.find(arr, np.bytes_(b"c\x00")).tolist() == [2, 2]
        assert (arr == np.bytes_(b"abc\x00")).tolist() == [False, True]

        # ops that resolve to fixed-width dtypes keep fixed-width semantics
        s = np.array([b"abc"], dtype="S4")
        assert np.strings.find(s, np.bytes_(b"abc\x00")).tolist() == [0]

    def test_pybytes_scalar_ufunc_outer_preserves_nulls(self):
        arr = R(b"x")
        assert np.add.outer(arr, b"y\x00").item() == b"xy\x00"
        assert np.add.outer(b"y\x00", arr).item() == b"y\x00x"

    def test_pybytes_scalar_ufunc_at_preserves_nulls(self):
        arr = R(b"x")
        np.add.at(arr, 0, b"y\x00")
        assert arr[0] == b"xy\x00"


class TestSlice:
    def test_high_bytes_no_hang(self):
        # the utf8 slice loop would spin forever on 0x80-0xBF/0xF8-0xFF
        # lead bytes (their utf8 character length is 0), so a regression
        # is a hang; run in a subprocess to fail fast instead
        code = textwrap.dedent("""
            import numpy as np
            from numpy.dtypes import ByteStringDType
            a = np.array([b"\\xff\\xfe", b"\\x80\\x81\\x82", b"ab"],
                         dtype=ByteStringDType())
            sliced = np.strings.slice(a, 0, 1)
            assert sliced.tolist() == [b"\\xff", b"\\x80", b"a"], sliced
        """)
        result = subprocess.run([sys.executable, "-c", code], timeout=120,
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr

    @pytest.mark.parametrize("start,stop,step", [
        (0, None, 1), (1, 3, 1), (-2, None, 1), (None, None, 2),
        (None, None, -1), (3, 0, -2), (0, 100, 1),
    ])
    def test_matches_python_slicing(self, start, stop, step):
        vals = NUL_AND_HIGH_BYTE_VALUES
        a = np.array(vals, dtype=ByteStringDType())
        result = np.strings.slice(a, start, stop, step)
        assert result.tolist() == [v[start:stop:step] for v in vals]


class TestMixedFixedWidth:
    def test_add(self):
        a = R(b"1", b"2")
        s = np.array([b"abc", b"x"], dtype="S3")
        for res in [np.strings.add(a, s), a + s]:
            assert isinstance(res.dtype, ByteStringDType)
            assert res.tolist() == [b"1abc", b"2x"]
        res = s + a
        assert res.tolist() == [b"abc1", b"x2"]


class TestMaskedArray:
    def test_default_fill_value(self, dtype):
        # without a default_filler entry the fill value falls back to the
        # str '?' and filled() degrades to an object array mixing str/bytes
        masked = np.ma.array(np.array([b"x", b"y"], dtype=dtype),
                             mask=[True, False])
        assert masked.fill_value == b"N/A"
        filled = masked.filled()
        assert filled.dtype == dtype
        assert filled.tolist() == [b"N/A", b"y"]


class TestTextIO:
    def test_loadtxt_requires_converters(self, dtype):
        import io
        with pytest.raises(TypeError, match="never assumes an encoding"):
            np.loadtxt(io.StringIO("ab\ncd\n"), dtype=dtype)
        arr = np.loadtxt(io.StringIO("ab\ncd\n"), dtype=dtype,
                         converters=str.encode)
        assert arr.dtype == dtype
        assert arr.tolist() == [b"ab", b"cd"]

    def test_genfromtxt_requires_converters(self, dtype):
        import io
        # previously the legacy StringConverter bytes entry silently
        # latin-1-encoded the text
        with pytest.raises(TypeError, match="never assumes an encoding"):
            np.genfromtxt(io.StringIO("héllo\nx\n"), dtype=dtype)
        arr = np.genfromtxt(io.StringIO("héllo\nx\n"), dtype=dtype,
                            converters={0: str.encode})
        assert arr.tolist() == ["héllo".encode(), b"x"]


class TestWrapperDispatch:
    def test_type_character_collision(self):
        # rational2 also uses the 'R' character; wrappers dispatch on the class
        from numpy._core._rational_tests import rational2
        arr = np.array([rational2(1)], dtype=rational2)
        assert arr.dtype.char == "R"
        for fn, args in [
            (np.strings.decode, ()),
            (np.strings.multiply, (2,)),
            (np.strings.center, (3,)),
            (np.strings.replace, (b"a", b"b")),
            (np.strings.partition, (b"a",)),
        ]:
            with pytest.raises(TypeError) as exc:
                fn(arr, *args)
            assert "ByteStringDType" not in str(exc.value)


class TestUnsupportedOps:
    def test_unregistered_ops_raise_cleanly(self):
        a = R(b"abc")
        b = R(b"b")
        for fn, args in [
            (np.strings.rfind, (a, b)),
            (np.strings.index, (a, b)),
            (np.strings.startswith, (a, b)),
            (np.strings.endswith, (a, b)),
            (np.strings.isdigit, (a,)),
            (np.strings.isdecimal, (a,)),
            (np.strings.isnumeric, (a,)),
            (np.strings.strip, (a, b"a")),
            (np.strings.expandtabs, (a,)),
        ]:
            with pytest.raises(TypeError):
                fn(*args)
        for fn, args in [
            (np.strings.capitalize, (a,)),
            (np.strings.upper, (a,)),
            (np.strings.mod, (a, b"x")),
            (np.strings.translate, (a, None)),
        ]:
            with pytest.raises((TypeError, ValueError)):
                fn(*args)
        # the pad family rejects up front: the fixed-width path it would
        # fall into half-executes before failing on a width-parametrized
        # 'R<width>' dtype string
        for fn in [np.strings.center, np.strings.ljust, np.strings.rjust]:
            with pytest.raises(NotImplementedError, match="ByteStringDType"):
                fn(a, 10, b" ")
            with pytest.raises(NotImplementedError, match="ByteStringDType"):
                fn(a, 10)
        with pytest.raises(NotImplementedError, match="ByteStringDType"):
            np.strings.zfill(a, 10)


class TestEncodeDecodeBridge:
    """Oracle: str.encode / bytes.decode."""

    def test_roundtrip(self):
        # the long values force the arena path when packing the outputs
        vals = ["héllo", "", "a\x00b", "x\x00", "\N{SNOWMAN}",
                "ünïcödé\x00" * 5, "y" * 30 + "\x00",
                "long variable width " * 4]
        s = np.array(vals, dtype=StringDType())
        b = np.strings.encode(s, "utf-8", dtype=ByteStringDType())
        assert isinstance(b.dtype, ByteStringDType)
        assert b.tolist() == [v.encode("utf-8") for v in vals]
        back = np.strings.decode(b, "utf-8")
        assert isinstance(back.dtype, StringDType)
        assert back.tolist() == vals

    def test_default_encoding_is_utf8(self):
        s = np.array(["héllo"], dtype=StringDType())
        assert np.strings.encode(
            s, dtype=ByteStringDType())[0] == "héllo".encode()

    @pytest.mark.parametrize("bad", [
        b"\xff\xfe",     # invalid lead bytes
        b"caf\xc3",      # truncated multibyte sequence
        b"\x80abc",      # bare continuation byte
        b"x" * 20 + b"\xff",  # arena-length value
    ])
    def test_strict_decode_raises(self, bad):
        arr = np.array([bad], dtype=ByteStringDType())
        with pytest.raises(UnicodeDecodeError) as exc:
            np.strings.decode(arr)
        with pytest.raises(UnicodeDecodeError) as expected:
            bad.decode()
        assert (exc.value.start, exc.value.reason) == \
            (expected.value.start, expected.value.reason)

    def test_unsupported_encodings_and_errors(self):
        r = np.array([b"x"], dtype=ByteStringDType())
        s = np.array(["x"], dtype=StringDType())
        with pytest.raises(NotImplementedError):
            np.strings.decode(r, "latin-1")
        with pytest.raises(NotImplementedError):
            np.strings.encode(s, "cp037", dtype=ByteStringDType())
        with pytest.raises(NotImplementedError):
            np.strings.decode(r, "utf-8", "replace")
        with pytest.raises(NotImplementedError):
            np.strings.encode(s, "utf-8", "ignore", dtype=ByteStringDType())
        # utf-8 aliases resolve through codecs
        assert np.strings.decode(r, "UTF8")[0] == "x"
        assert np.strings.encode(s, "utf_8", dtype=ByteStringDType())[0] == b"x"

    def test_directional_type_errors(self):
        r = np.array([b"x"], dtype=ByteStringDType())
        s = np.array(["x"], dtype=StringDType())
        with pytest.raises(TypeError, match="np.strings.decode"):
            np.strings.encode(r)
        with pytest.raises(TypeError, match="np.strings.encode"):
            np.strings.decode(s)

    def test_null_propagation(self):
        for na in [None, np.nan]:
            s = np.array(["a"], dtype=StringDType(na_object=na))
            sn = np.insert(s, 0, na)
            b = np.strings.encode(sn, dtype=ByteStringDType())
            back = np.strings.decode(b)
            if na is None:
                assert b[0] is None and back[0] is None
            else:
                import math
                assert math.isnan(b[0]) and math.isnan(back[0])
            assert b[1] == b"a" and back[1] == "a"

    def test_string_na_translates_across_bridge(self):
        # a string-like sentinel is itself encoded/decoded
        s = np.array(["a"], dtype=StringDType(na_object="MISSING"))
        sn = np.insert(s, 0, "MISSING")
        b = np.strings.encode(sn, dtype=ByteStringDType())
        assert b.dtype.na_object == b"MISSING"
        assert b[0] == b"MISSING"
        assert np.sort(b).tolist() == [b"MISSING", b"a"]
        assert np.strings.str_len(b).tolist() == [7, 1]
        back = np.strings.decode(b)
        assert back.dtype.na_object == "MISSING"
        assert back[0] == "MISSING"

    def test_non_utf8_bytes_na_fails_decode(self):
        # the sentinel crosses the bridge through the same strict codec as
        # the data, so a non-UTF-8 bytes sentinel fails decode up front
        r = np.array([b"ok"], dtype=ByteStringDType(na_object=b"\xff"))
        with pytest.raises(UnicodeDecodeError):
            np.strings.decode(r)

    def test_fixed_width_paths_unchanged(self):
        c = np.array([b"\x81\xc1"], dtype="S2")
        assert np.strings.decode(c, "cp037").tolist() == ["aA"]
        u = np.array(["aA"])
        assert np.strings.encode(u, "cp037").tolist() == [b"\x81\xc1"]

    def test_default_encode_warns_and_keeps_fixed_width(self):
        # without dtype=, StringDType input keeps its pre-ByteStringDType
        # behavior behind a DeprecationWarning
        s = np.array(["x\x00", "héllo"], dtype=StringDType())
        with pytest.warns(DeprecationWarning, match="ByteStringDType"):
            res = np.strings.encode(s, "utf-8")
        assert res.dtype.kind == "S"
        assert res[0] == b"x"
        with pytest.warns(DeprecationWarning):
            full = np.strings.encode(s, "utf-16", "replace")
        # the fixed-width result strips trailing NULs of the encoded
        # bytes (UTF-16-LE of ASCII ends in one), the pre-existing
        # behavior this path preserves
        assert full.tolist() == \
            [v.encode("utf-16", "replace").rstrip(b"\x00")
             for v in s.tolist()]
        with pytest.warns(DeprecationWarning):
            zd = np.strings.encode(np.array("x", dtype=StringDType()))
        assert isinstance(zd, np.ndarray)
        assert zd.shape == () and zd.dtype.kind == "S"

    def test_encode_dtype_selects_and_silences(self):
        s = np.array(["x\x00"], dtype=StringDType())
        u = np.array(["x"])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fixed = np.strings.encode(s, "utf-8", dtype=np.bytes_)
            assert fixed.dtype.kind == "S" and fixed[0] == b"x"
            var = np.strings.encode(s, "utf-8", dtype=ByteStringDType())
            assert isinstance(var.dtype, ByteStringDType)
            assert var[0] == b"x\x00"
            assert np.strings.encode(
                s, dtype=np.dtypes.ByteStringDType)[0] == b"x\x00"
            assert np.strings.encode(
                s, dtype=np.dtypes.BytesDType)[0] == b"x"
            assert np.strings.encode(s, dtype="R")[0] == b"x\x00"
            # str input never warns, with or without the fixed-width dtype
            assert np.strings.encode(u, "utf-8").dtype.kind == "S"
            assert np.strings.encode(
                u, "utf-8", dtype=np.bytes_).dtype.kind == "S"

    def test_encode_dtype_validation(self):
        s = np.array(["x"], dtype=StringDType())
        with pytest.raises(TypeError, match="StringDType input"):
            np.strings.encode(np.array(["x"]), dtype=ByteStringDType())
        with pytest.raises(ValueError, match="ByteStringDType or"):
            np.strings.encode(s, dtype=np.int64)
        with pytest.raises(ValueError, match="np.bytes_"):
            np.strings.encode(s, dtype="S5")
        with pytest.raises(ValueError, match="parametrized"):
            np.strings.encode(s, dtype=ByteStringDType(na_object=b""))
