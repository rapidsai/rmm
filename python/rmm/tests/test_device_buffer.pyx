# Copyright (c) 2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cython
import numpy as np

from libcpp.memory cimport make_unique
from libcpp.utility cimport move

from rmm._lib.device_buffer cimport DeviceBuffer, device_buffer


def test_release():
    expect = DeviceBuffer.to_device(b'abc')
    cdef DeviceBuffer buf = DeviceBuffer.to_device(b'abc')
    got = DeviceBuffer.c_from_unique_ptr(
        make_unique[device_buffer](buf.c_release())
    )
    np.testing.assert_equal(expect.copy_to_host(), got.copy_to_host())


def test_size_after_release():
    cdef DeviceBuffer buf = DeviceBuffer.to_device(b'abc')
    buf.c_release()
    assert buf.size == 0
