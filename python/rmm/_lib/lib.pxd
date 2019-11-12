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
from libcpp cimport bool
from libcpp.utility cimport pair
from libcpp.vector cimport vector

ctypedef pair[const char*, unsigned int] caller_pair


cdef extern from * nogil:

    ctypedef void* cudaStream_t "cudaStream_t"


cdef uintptr_t c_alloc(
    size_t size,
    cudaStream_t stream
) except? <uintptr_t>NULL

cdef void c_free(
    void *ptr,
    cudaStream_t stream
) except *

cdef ptrdiff_t* c_getallocationoffset(
    void *ptr,
    cudaStream_t stream
) except? <ptrdiff_t*>NULL

cdef caller_pair _get_caller() except *


cdef extern from "rmm/rmm.h" nogil:

    ctypedef enum rmmError_t:
        RMM_SUCCESS = 0,
        RMM_ERROR_CUDA_ERROR,
        RMM_ERROR_INVALID_ARGUMENT,
        RMM_ERROR_NOT_INITIALIZED,
        RMM_ERROR_OUT_OF_MEMORY,
        RMM_ERROR_UNKNOWN,
        RMM_ERROR_IO,
        N_RMM_ERROR,

    ctypedef enum rmmAllocationMode_t:
        CudaDefaultAllocation = 0,
        PoolAllocation = 1,
        CudaManagedMemory = 2,

    cdef cppclass rmmOptions_t:
        rmmOptions_t() except +
        rmmAllocationMode_t allocation_mode
        size_t initial_pool_size
        bool enable_logging
        vector[int] devices

    cdef rmmError_t rmmInitialize(
        rmmOptions_t *options
    ) except +

    cdef rmmError_t rmmFinalize() except +

    cdef bool rmmIsInitialized(
        rmmOptions_t *options
    ) except +

    cdef const char * rmmGetErrorString(
        rmmError_t errcode
    ) except +

    cdef rmmError_t rmmAlloc(
        void **ptr,
        size_t size,
        cudaStream_t stream,
        const char* file,
        unsigned int line
    ) except +

    cdef rmmError_t rmmFree(
        void *ptr,
        cudaStream_t stream,
        const char* file,
        unsigned int line
    ) except +

    cdef rmmError_t rmmGetAllocationOffset(
        ptrdiff_t *offset,
        void *ptr,
        cudaStream_t stream
    ) except +

    cdef rmmError_t rmmGetInfo(
        size_t *freeSize,
        size_t *totalSize,
        cudaStream_t stream
    ) except +

    cdef rmmError_t rmmWriteLog(
        const char* filename
    ) except +

    cdef size_t rmmLogSize() except +

    cdef rmmError_t rmmGetLog(
        char* buffer,
        size_t buffer_size
    ) except +


cdef extern from "rmm/rmm.hpp" namespace "rmm" nogil:

    cdef rmmError_t alloc[T](
        T** ptr,
        size_t size,
        cudaStream_t stream,
        const char* file,
        unsigned int line
    ) except +

    cdef rmmError_t free(
        void* ptr,
        cudaStream_t stream,
        const char* file,
        unsigned int line
    ) except +

    cdef cppclass Manager:
        @staticmethod
        rmmOptions_t getOptions() except +


cdef extern from "cstdlib":
    int atexit(void (*func)())
