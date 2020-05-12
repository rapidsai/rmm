# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

# cython: profile = False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from collections import namedtuple

from libc.stdint cimport uintptr_t
from libcpp.vector cimport vector

from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AS_STRING


# Utility Functions
def _get_error_msg(errcode):
    """
    Get error message for the given error code.
    """
    msg = rmmGetErrorString(<rmmError_t>errcode)
    cdef bytes py_msg = msg
    return py_msg.decode("utf-8")


def check_error(errcode):
    """
    Checks the error of a function that returns rmmError_t and raises a Python
    exception based on the error code
    """
    from rmm import RMMError

    if errcode != RMM_SUCCESS:
        msg = _get_error_msg(errcode)
        raise RMMError(errcode, msg)


cdef ptrdiff_t c_getallocationoffset(
    void *ptr, cudaStream_t stream
):
    """
    Gets the offset of ptr from its base allocation by calling the librmm
    functions via Cython
    """
    cdef ptrdiff_t offset

    with nogil:
        rmm_error = rmmGetAllocationOffset(&offset, ptr, stream)

    check_error(rmm_error)

    return offset


def rmm_getallocationoffset(ptr, stream):
    """
    Gets the offset of ptr from its base allocation by calling the librmm
    functions via Cython
    """
    cdef void * c_ptr = <void *><uintptr_t>ptr
    cdef cudaStream_t c_stream = <cudaStream_t><uintptr_t>stream

    cdef ptrdiff_t c_offset = c_getallocationoffset(c_ptr, c_stream)

    result = int(c_offset)
    return result
