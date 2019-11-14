from itertools import product

import numpy as np
import pytest

import rmm


def array_tester(dtype, nelem):
    # data
    h_in = np.full(nelem, 3.2, dtype)
    h_result = np.empty(nelem, dtype)

    d_in = rmm.to_device(h_in)
    d_result = rmm.device_array_like(d_in)

    d_result.copy_to_device(d_in)
    h_result = d_result.copy_to_host()

    np.testing.assert_array_equal(h_result, h_in)


_dtypes = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.float32,
    np.float64,
    np.bool_,
]
_nelems = [1, 2, 7, 8, 9, 32, 128]


@pytest.mark.parametrize("dtype,nelem", list(product(_dtypes, _nelems)))
def test_rmm_alloc(dtype, nelem):
    array_tester(dtype, nelem)


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize("dtype", _dtypes)
@pytest.mark.parametrize("nelem", _nelems)
@pytest.mark.parametrize(
    "managed, pool", list(product([False, True], [False, True]))
)
def test_rmm_modes(dtype, nelem, managed, pool):
    rmm.reinitialize(pool_allocator=pool, managed_memory=managed)

    assert rmm.is_initialized()

    array_tester(dtype, nelem)


def test_uninitialized():
    rmm._finalize()
    assert not rmm.is_initialized()
    rmm.reinitialize()  # so further tests will pass


@pytest.mark.parametrize("dtype", _dtypes)
@pytest.mark.parametrize("nelem", _nelems)
def test_rmm_csv_log(dtype, nelem):
    # data
    h_in = np.full(nelem, 3.2, dtype)

    d_in = rmm.to_device(h_in)
    d_result = rmm.device_array_like(d_in)

    d_result.copy_to_device(d_in)

    csv = rmm.csv_log()

    assert (
        csv.find(
            "Event Type,Device ID,Address,Stream,Size (bytes),"
            "Free Memory,Total Memory,Current Allocs,Start,End,"
            "Elapsed,Location"
        )
        >= 0
    )
