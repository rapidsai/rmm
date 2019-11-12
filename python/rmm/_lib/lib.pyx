# Copyright (c) 2019, NVIDIA CORPORATION.
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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libc.stdint cimport uintptr_t
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector


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


# API Functions
def rmm_initialize(
        allocation_mode, initial_pool_size, devices, enable_logging
):
    """
    Initializes the RMM library by calling the librmm functions via Cython
    """
    cdef rmmOptions_t opts = rmmOptions_t()
    opts.allocation_mode = <rmmAllocationMode_t>allocation_mode
    opts.initial_pool_size = <size_t>initial_pool_size
    opts.enable_logging = <bool>enable_logging
    opts.devices = <vector[int]>devices

    with nogil:
        rmm_error = rmmInitialize(
            <rmmOptions_t *>&opts
        )

    check_error(rmm_error)

    return 0


def rmm_finalize():
    """
    Finalizes the RMM library by calling the librmm functions via Cython
    """
    with nogil:
        rmm_error = rmmFinalize()

    check_error(rmm_error)

    return 0


cdef void _rmmFinalizeWrapper ():
    rmmFinalize()


def register_atexit_finalize():
    atexit(&_rmmFinalizeWrapper)


def rmm_is_initialized():
    """
    Returns True if RMM has been initialized, false otherwise by calling the
    librmm functions via Cython
    """
    with nogil:
        result = rmmIsInitialized(
            <rmmOptions_t *>NULL
        )

    return result


def rmm_csv_log():
    """
    Returns a CSV log of all events logged by RMM, if logging is enabled by
    calling the librmm functions via Cython
    """
    cdef size_t logsize = rmmLogSize()
    cdef char* buf = <char*>malloc(logsize)

    with nogil:
        rmm_error = rmmGetLog(
            <char*>buf,
            <size_t>logsize
        )

    check_error(rmm_error)

    cdef bytes py_buf = buf
    free(buf)
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
    cdef cudaStream_t c_stream = <cudaStream_t><size_t>stream

    cdef uintptr_t c_addr = c_alloc(
        <size_t>c_size,
        <cudaStream_t>c_stream
    )

    return int(c_addr)


cdef void c_free(void *ptr, cudaStream_t stream) except *:
    """
    Deallocates ptr, which was allocated using rmmAlloc by calling the librmm
    functions via Cython
    """
    cdef caller_pair tmp_caller_pair = _get_caller()
    cdef const char* file = tmp_caller_pair.first
    cdef unsigned int line = tmp_caller_pair.second

    # Call RMM to free
    with nogil:
        rmm_error = rmmFree(
            <void *>ptr,
            <cudaStream_t>stream,
            <const char*>file,
            <unsigned int>line
        )

    check_error(rmm_error)


def rmm_free(ptr, stream):
    """
    Deallocates ptr, which was allocated using rmmAlloc by calling the librmm
    functions via Cython
    """
    cdef void * c_ptr = <void *><uintptr_t>ptr
    cdef cudaStream_t c_stream = <cudaStream_t><size_t>stream

    c_free(
        <void *>c_ptr,
        <cudaStream_t>c_stream
    )


cdef ptrdiff_t* c_getallocationoffset(
    void *ptr, cudaStream_t stream
) except? <ptrdiff_t*>NULL:
    """
    Gets the offset of ptr from its base allocation by calling the librmm
    functions via Cython
    """
    cdef ptrdiff_t * offset = <ptrdiff_t *>malloc(sizeof(ptrdiff_t))

    with nogil:
        rmm_error = rmmGetAllocationOffset(
            <ptrdiff_t *>offset,
            <void *>ptr,
            <cudaStream_t>stream
        )

    check_error(rmm_error)

    return offset


def rmm_getallocationoffset(ptr, stream):
    """
    Gets the offset of ptr from its base allocation by calling the librmm
    functions via Cython
    """
    cdef void * c_ptr = <void *><uintptr_t>ptr
    cdef cudaStream_t c_stream = <cudaStream_t><size_t>stream

    cdef ptrdiff_t * c_offset = c_getallocationoffset(
        <void *>c_ptr,
        <cudaStream_t>c_stream
    )

    result = int(c_offset[0])
    free(c_offset)
    return result
