import numpy as np

from libcpp.memory cimport make_unique

from rmm._lib.device_buffer cimport DeviceBuffer, device_buffer, move


def test_c_release():
    expect = DeviceBuffer.to_device(b'abc')
    cdef DeviceBuffer buf = DeviceBuffer.to_device(b'abc')
    got = DeviceBuffer.c_from_unique_ptr(
        make_unique[device_buffer](buf.c_release())
    )
    np.testing.assert_equal(expect.copy_to_host(), got.copy_to_host())


test_c_release()
