# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# This import is needed for Cython typing in translate_python_except_to_cpp
# See https://github.com/cython/cython/issues/5589
from builtins import BaseException

from cuda.bindings.cyruntime cimport cudaMemPool_t
from libc.stddef cimport size_t
from libc.stdint cimport int8_t, int32_t, int64_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.optional cimport optional
from libcpp.pair cimport pair
from libcpp.string cimport string

from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.memory_resource cimport device_memory_resource
from rmm.librmm.per_device_resource cimport device_async_resource_ref


# Type alias for any_resource to avoid Cython template complexity
# Also includes C++ helpers to work around Cython varargs limitations and
# resource_ref operations and to handle resource_ref operations without Cython
# needing default constructors
cdef extern from *:
    """
    #include <cuda/memory_resource>
    #include <rmm/mr/arena_memory_resource.hpp>
    #include <rmm/mr/logging_resource_adaptor.hpp>
    #include <rmm/resource_ref.hpp>
    #include <rmm/cuda_stream_view.hpp>
    #include <memory>
    #include <string>
    #include <optional>

    using any_device_resource =
        cuda::mr::any_resource<cuda::mr::device_accessible>;

    // Helper to create arena_memory_resource with optional
    // Avoids Cython varargs issues
    inline std::unique_ptr<
        rmm::mr::arena_memory_resource<rmm::device_async_resource_ref>>
    make_arena_memory_resource(
        rmm::device_async_resource_ref upstream_ref,
        std::optional<std::size_t> arena_size,
        bool dump_log_on_failure)
    {
        return std::make_unique<
            rmm::mr::arena_memory_resource<rmm::device_async_resource_ref>>(
            upstream_ref, arena_size, dump_log_on_failure);
    }

    // Helper to create logging_resource_adaptor with string
    // Avoids Cython varargs issues
    inline std::unique_ptr<
        rmm::mr::logging_resource_adaptor<rmm::device_async_resource_ref>>
    make_logging_resource_adaptor(
        rmm::device_async_resource_ref upstream_ref,
        const std::string& filename)
    {
        return std::make_unique<
            rmm::mr::logging_resource_adaptor<rmm::device_async_resource_ref>>(
            upstream_ref, filename);
    }
    """
    cdef cppclass any_device_resource:
        any_device_resource() except +
        any_device_resource(device_async_resource_ref) except +
        void* allocate_sync(size_t bytes) except +
        void deallocate_sync(void* ptr, size_t bytes) noexcept
        void* allocate(
            cuda_stream_view stream,
            size_t bytes
        ) except + nogil
        void deallocate(
            cuda_stream_view stream,
            void* ptr,
            size_t bytes
        ) noexcept nogil


cdef extern from "rmm/detail/cccl_shared_resource.hpp" \
        namespace "cuda::mr" nogil:
    cdef cppclass shared_resource[T]:
        shared_resource() except +
        shared_resource(const shared_resource&) noexcept
        shared_resource(shared_resource&&) noexcept
        shared_resource& operator=(const shared_resource&) noexcept
        shared_resource& operator=(shared_resource&&) noexcept
        T& get() noexcept nogil
        const T& get() noexcept nogil
        T* operator_arrow "operator->"() noexcept nogil
        T& operator_star "operator*"() noexcept nogil
        void* allocate_sync(size_t bytes, size_t alignment) except + nogil
        void deallocate_sync(void* ptr, size_t bytes, size_t alignment) noexcept nogil
        void* allocate(
            cuda_stream_view stream,
            size_t bytes,
            size_t alignment
        ) except + nogil
        void deallocate(
            cuda_stream_view stream,
            void* ptr,
            size_t bytes,
            size_t alignment
        ) noexcept nogil
        void swap(shared_resource&) noexcept nogil
        bool operator==(const shared_resource&) noexcept nogil
        bool operator!=(const shared_resource&) noexcept nogil


cdef extern from "rmm/mr/device_memory_resource.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass device_memory_resource:
        void* allocate_sync(size_t bytes) except +
        void deallocate_sync(void* ptr, size_t bytes) noexcept
        void* allocate(
            cuda_stream_view stream,
            size_t bytes
        ) except +
        void deallocate(
            cuda_stream_view stream,
            void* ptr,
            size_t bytes
        ) noexcept

cdef extern from "rmm/cuda_device.hpp" namespace "rmm" nogil:
    size_t percent_of_free_device_memory(int percent) except +
    pair[size_t, size_t] available_device_memory() except +

# Transparent handle of a C++ exception
ctypedef pair[int, string] CppExcept

cdef inline CppExcept translate_python_except_to_cpp(err: BaseException) noexcept:
    """Translate a Python exception into a C++ exception handle

    The returned exception handle can then be thrown by `throw_cpp_except()`,
    which MUST be done without holding the GIL.

    This is useful when C++ calls a Python function and needs to catch or
    propagate exceptions.
    """
    if isinstance(err, MemoryError):
        return CppExcept(0, str.encode(str(err)))
    return CppExcept(-1, str.encode(str(err)))

# Implementation of `throw_cpp_except()`, which throws a given `CppExcept`.
# This function MUST be called without the GIL otherwise the thrown C++
# exception are translated back into a Python exception.
cdef extern from *:
    """
    #include <rmm/error.hpp>

    #include <stdexcept>
    #include <utility>

    void throw_cpp_except(std::pair<int, std::string> res) {
        switch(res.first) {
            case 0:
                throw rmm::out_of_memory(res.second);
            default:
                throw std::runtime_error(res.second);
        }
    }
    """
    void throw_cpp_except(CppExcept) nogil


cdef extern from "rmm/mr/cuda_memory_resource.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass cuda_memory_resource(device_memory_resource):
        cuda_memory_resource() except +

cdef extern from "rmm/mr/managed_memory_resource.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass managed_memory_resource(device_memory_resource):
        managed_memory_resource() except +

cdef extern from "rmm/mr/system_memory_resource.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass system_memory_resource(device_memory_resource):
        system_memory_resource() except +

cdef extern from "rmm/mr/pinned_host_memory_resource.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass pinned_host_memory_resource(device_memory_resource):
        pinned_host_memory_resource() except +

cdef extern from "rmm/mr/sam_headroom_memory_resource.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass sam_headroom_memory_resource(device_memory_resource):
        sam_headroom_memory_resource(size_t headroom) except +

cdef extern from "rmm/mr/cuda_async_memory_resource.hpp" \
        namespace "rmm::mr" nogil:

    cdef cppclass cuda_async_memory_resource(device_memory_resource):
        cuda_async_memory_resource(
            optional[size_t] initial_pool_size,
            optional[size_t] release_threshold,
            optional[allocation_handle_type] export_handle_type) except +

cdef extern from "rmm/mr/cuda_async_view_memory_resource.hpp" \
        namespace "rmm::mr" nogil:

    cdef cppclass cuda_async_view_memory_resource(device_memory_resource):
        cuda_async_view_memory_resource(
            cudaMemPool_t pool_handle) except +
        cudaMemPool_t pool_handle() const

cdef extern from "rmm/mr/cuda_async_managed_memory_resource.hpp" \
        namespace "rmm::mr" nogil:

    cdef cppclass cuda_async_managed_memory_resource(device_memory_resource):
        cuda_async_managed_memory_resource() except +
        cudaMemPool_t pool_handle() const

cdef extern from "rmm/mr/cuda_async_memory_resource.hpp" \
        namespace \
        "rmm::mr::cuda_async_memory_resource" \
        nogil:
    cpdef enum class allocation_handle_type(int32_t):
        none
        posix_file_descriptor
        win32
        win32_kmt
        fabric


cdef extern from "rmm/mr/pool_memory_resource.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass pool_memory_resource[Upstream](device_memory_resource):
        pool_memory_resource(
            device_async_resource_ref upstream_mr,
            size_t initial_pool_size,
            optional[size_t] maximum_pool_size) except +
        size_t pool_size()

cdef extern from "rmm/mr/arena_memory_resource.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass arena_memory_resource[Upstream](device_memory_resource):
        arena_memory_resource(
            device_async_resource_ref upstream_mr,
            optional[size_t] arena_size,
            bool dump_log_on_failure
        ) except +

cdef extern from "rmm/mr/fixed_size_memory_resource.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass fixed_size_memory_resource[Upstream](device_memory_resource):
        fixed_size_memory_resource(
            device_async_resource_ref upstream_mr,
            size_t block_size,
            size_t block_to_preallocate) except +

cdef extern from "rmm/mr/callback_memory_resource.hpp" \
        namespace "rmm::mr" nogil:
    ctypedef void* (*allocate_callback_t)(size_t, cuda_stream_view, void*)
    ctypedef void (*deallocate_callback_t)(void*, size_t, cuda_stream_view, void*)

    cdef cppclass callback_memory_resource(device_memory_resource):
        callback_memory_resource(
            allocate_callback_t allocate_callback,
            deallocate_callback_t deallocate_callback,
            void* allocate_callback_arg,
            void* deallocate_callback_arg
        ) except +

cdef extern from "rmm/mr/binning_memory_resource.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass binning_memory_resource[Upstream](device_memory_resource):
        binning_memory_resource(device_async_resource_ref upstream_mr) except +
        binning_memory_resource(
            device_async_resource_ref upstream_mr,
            int8_t min_size_exponent,
            int8_t max_size_exponent) except +

        void add_bin(size_t allocation_size) except +
        void add_bin(
            size_t allocation_size,
            device_async_resource_ref bin_resource) except +

cdef extern from "rmm/mr/limiting_resource_adaptor.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass limiting_resource_adaptor[Upstream](device_memory_resource):
        limiting_resource_adaptor(
            device_async_resource_ref upstream_mr,
            size_t allocation_limit) except +

        size_t get_allocated_bytes() except +
        size_t get_allocation_limit() except +

cdef extern from "rmm/mr/logging_resource_adaptor.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass logging_resource_adaptor[Upstream](device_memory_resource):
        logging_resource_adaptor(
            device_async_resource_ref upstream_mr,
            string filename) except +

        void flush() except +

# C++ helper functions to avoid varargs issues with optional and string parameters
# Declared here after the type declarations to avoid template resolution issues
cdef extern from *:
    cdef unique_ptr[
        arena_memory_resource[device_async_resource_ref]
    ] make_arena_memory_resource(
        device_async_resource_ref upstream_ref,
        optional[size_t] arena_size,
        bool dump_log_on_failure
    ) except +

    cdef unique_ptr[
        logging_resource_adaptor[device_async_resource_ref]
    ] make_logging_resource_adaptor(
        device_async_resource_ref upstream_ref,
        const string& filename
    ) except +

cdef extern from "rmm/mr/statistics_resource_adaptor.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass statistics_resource_adaptor[Upstream](device_memory_resource):
        struct counter:
            counter()

            int64_t value
            int64_t peak
            int64_t total

        statistics_resource_adaptor(device_async_resource_ref upstream_mr) except +

        counter get_bytes_counter() except +
        counter get_allocations_counter() except +
        pair[counter, counter] pop_counters() except +
        pair[counter, counter] push_counters() except +

cdef extern from "rmm/mr/tracking_resource_adaptor.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass tracking_resource_adaptor[Upstream](device_memory_resource):
        tracking_resource_adaptor(
            device_async_resource_ref upstream_mr,
            bool capture_stacks) except +

        size_t get_allocated_bytes() except +
        string get_outstanding_allocations_str() except +
        void log_outstanding_allocations() except +

cdef extern from "rmm/mr/failure_callback_resource_adaptor.hpp" \
        namespace "rmm::mr" nogil:
    ctypedef bool (*failure_callback_t)(size_t, void*)
    cdef cppclass failure_callback_resource_adaptor[Upstream](
        device_memory_resource
    ):
        failure_callback_resource_adaptor(
            device_async_resource_ref upstream_mr,
            failure_callback_t callback,
            void* callback_arg
        ) except +

cdef extern from "rmm/mr/prefetch_resource_adaptor.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass prefetch_resource_adaptor[Upstream](device_memory_resource):
        prefetch_resource_adaptor(device_async_resource_ref upstream_mr) except +
