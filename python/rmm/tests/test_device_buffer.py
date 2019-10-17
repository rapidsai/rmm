from numba import cuda
import numpy as np
import pytest

import rmm

_dtypes = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.float32,
    np.float64,
    np.bool_,
    'datetime64[ms]'
]
_nelems = [1, 2, 7, 8, 9, 32, 128]


@pytest.mark.parametrize("dtype", _dtypes)
@pytest.mark.parametrize("nelems", _nelems)
def test_device_buffer_from_array(dtype, nelems):
    expect = np.random.rand(nelems).astype(dtype)
    buf = rmm.DeviceBuffer(expect)
    got = rmm.device_array_from_ptr(buf.data, len(expect), expect.dtype).copy_to_host()
    np.testing.assert_array_equal(expect, got)
