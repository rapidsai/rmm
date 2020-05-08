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


cdef caller_pair _get_caller() except *:
    """
    Finds the file and line number of the caller (first caller outside this
    file)
    """
    import inspect

    # Go up stack to find first caller outside this file (more useful)
    cdef rmmOptions_t opts = Manager.getOptions()

    if opts.enable_logging:
        frame = inspect.currentframe().f_back
        while frame:
            filename = inspect.getfile(frame)
            # May need to tweak this to handle Cython
            if not filename.endswith("rmm.py"):
                break
            else:
                frame = frame.f_back
        line_number = frame.f_lineno
        del frame
    else:
        filename = None
        line_number = 0

    cdef const char* file = <const char*>NULL
    if filename is not None:
        filename = filename.encode()
        file = filename
    cdef unsigned int line = line_number

    return caller_pair(file, line)


def rmm_csv_log():
    """
    Returns a CSV log of all events logged by RMM, if logging is enabled by
    calling the librmm functions via Cython
    """
    cdef size_t logsize = rmmLogSize()
    cdef bytes py_buf = PyBytes_FromStringAndSize(NULL, logsize)
    cdef char* buf = PyBytes_AS_STRING(py_buf)

    with nogil:
        rmm_error = rmmGetLog(buf, logsize)

    check_error(rmm_error)

    return py_buf.decode("utf-8")


cdef uintptr_t c_alloc(
    size_t size, cudaStream_t stream
) except? <uintptr_t>NULL:
    """
    Allocates size bytes using the RMM memory manager by calling the librmm
    functions via Cython
    """
    cdef caller_pair tmp_caller_pair = _get_caller()
    cdef const char* file = tmp_caller_pair.first
    cdef unsigned int line = tmp_caller_pair.second

    cdef void* ptr
    with nogil:
        rmm_error = rmmAlloc(
            <void **>&ptr,
            <size_t>size,
            <cudaStream_t>stream,
            <const char*>file,
            <unsigned int>line
        )

    check_error(rmm_error)

    return <uintptr_t>ptr


def rmm_alloc(size, stream):
    """
    Allocates size bytes using the RMM memory manager by calling the librmm
    functions via Cython
    """
    cdef size_t c_size = size
    cdef cudaStream_t c_stream = <cudaStream_t><uintptr_t>stream

    cdef uintptr_t c_addr = c_alloc(
        <size_t>c_size,
        <cudaStream_t>c_stream
    )

    return int(c_addr)


cdef rmmError_t c_free(void *ptr, cudaStream_t stream,
                       const char* file=NULL, unsigned int line=0) except *:
    """
    Deallocates ptr, which was allocated using rmmAlloc by calling the librmm
    functions via Cython
    """
    cdef rmmError_t rmm_error

    # Call RMM to free
    with nogil:
        rmm_error = rmmFree(
            ptr,
            stream,
            file,
            line
        )

    return rmm_error


def rmm_free(ptr, stream):
    """
    Deallocates ptr, which was allocated using rmmAlloc by calling the librmm
    functions via Cython
    """
    cdef void * c_ptr = <void *><uintptr_t>ptr
    cdef cudaStream_t c_stream = <cudaStream_t><uintptr_t>stream

    cdef caller_pair tmp_caller_pair = _get_caller()
    cdef const char* file = tmp_caller_pair.first
    cdef unsigned int line = tmp_caller_pair.second

    rmm_error = c_free(
        c_ptr,
        c_stream,
        file,
        line
    )


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
