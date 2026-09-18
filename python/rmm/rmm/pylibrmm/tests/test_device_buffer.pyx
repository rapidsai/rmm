# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from cuda.bindings.cyruntime cimport cudaStream_t, cudaStreamDefault
from libcpp.memory cimport make_unique

from rmm.librmm.device_buffer cimport device_buffer
from rmm.pylibrmm.device_buffer cimport DeviceBuffer


def test_release():
    expect = DeviceBuffer.to_device(b'abc')
    cdef DeviceBuffer buf = DeviceBuffer.to_device(b'abc')

    got = DeviceBuffer.c_from_unique_ptr(
        make_unique[device_buffer](buf.c_release(),
                                   <cudaStream_t>cudaStreamDefault)
    )
    np.testing.assert_equal(expect.copy_to_host(), got.copy_to_host())


def test_size_after_release():
    cdef DeviceBuffer buf = DeviceBuffer.to_device(b'abc')
    buf.c_release()
    assert buf.size == 0
