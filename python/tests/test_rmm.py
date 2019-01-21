import pytest
from itertools import product

import numpy as np

from librmm_cffi import librmm as rmm

_dtypes = [np.int32]
_nelems = [1, 2, 7, 8, 9, 32, 128]


@pytest.mark.parametrize('dtype,nelem', list(product(_dtypes, _nelems)))
def test_rmm_alloc(dtype, nelem):
    # data
    h_in = np.full(nelem, 3.2, dtype)
    h_result = np.empty(nelem, dtype)

    d_in = rmm.to_device(h_in)
    d_result = rmm.device_array_like(d_in)

    d_result.copy_to_device(d_in)
    h_result = d_result.copy_to_host()

    print('expect')
    print(h_in)
    print('got')
    print(h_result)

    np.testing.assert_array_equal(h_result, h_in)


def test_rmm_csv_log():
    dtype = np.int32
    nelem = 1024

    # data
    h_in = np.full(nelem, 3.2, dtype)
    h_result = np.empty(nelem, dtype)

    d_in = rmm.to_device(h_in)
    d_result = rmm.device_array_like(d_in)

    d_result.copy_to_device(d_in)
    h_result = d_result.copy_to_host()

    csv = rmm.csv_log()

    print(csv[:1000])

    assert(csv.find("Event Type,Device ID,Address,Stream,Size (bytes),"
                    "Free Memory,Total Memory,Current Allocs,Start,End,"
                    "Elapsed,Location") >= 0)
