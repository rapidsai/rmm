# Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

from libc.stdint cimport int8_t, int64_t
from libcpp cimport bool
from libcpp.optional cimport optional

cdef extern from "rmm/mr/device/callback_memory_resource.hpp" \
        namespace "rmm::mr" nogil:
    ctypedef void* (*allocate_callback_t)(size_t, cuda_stream_view, void*)
    ctypedef void (*deallocate_callback_t)(void*, size_t, cuda_stream_view, void*)

cdef extern from "rmm/mr/device/failure_callback_resource_adaptor.hpp" \
        namespace "rmm::mr" nogil:
    ctypedef bool (*failure_callback_t)(size_t, void*)

# TODO: when we adopt Cython 3.0 use enum class
cdef extern from "rmm/mr/device/cuda_async_memory_resource.hpp" \
        namespace \
        "rmm::mr::cuda_async_memory_resource::allocation_handle_type" \
        nogil:
    enum allocation_handle_type \
            "rmm::mr::cuda_async_memory_resource::allocation_handle_type":
        none
        posix_file_descriptor
        win32
        win32_kmt

cdef extern from "rmm/mr/device/device_memory_resource.hpp" namespace "rmm::mr" nogil:
    cdef cppclass device_memory_resource:
        
        device_memory_resource() except +

cdef extern from "rmm/mr/device/cuda_memory_resource.hpp" namespace "rmm::mr" nogil:
    cdef cppclass cuda_memory_resource(device_memory_resource):
        
        cuda_memory_resource() except +

cdef extern from "rmm/mr/device/managed_memory_resource.hpp" namespace "rmm::mr" nogil:
    cdef cppclass managed_memory_resource(device_memory_resource):
        
        managed_memory_resource() except +

cdef extern from "rmm/mr/device/system_memory_resource.hpp" namespace "rmm::mr" nogil:
    cdef cppclass system_memory_resource(device_memory_resource):
        
        system_memory_resource() except +

cdef extern from "rmm/mr/device/sam_headroom_memory_resource.hpp" namespace "rmm::mr" nogil:
    cdef cppclass sam_headroom_memory_resource(device_memory_resource):
        
        sam_headroom_memory_resource(size_t headroom) except +

cdef extern from "rmm/mr/device/cuda_async_memory_resource.hpp" namespace "rmm::mr" nogil:
    cdef cppclass cuda_async_memory_resource(device_memory_resource):
        
        cuda_async_memory_resource(
            optional[size_t] initial_pool_size,
            optional[size_t] release_threshold,
            optional[allocation_handle_type] export_handle_type
        ) except +

cdef extern from "rmm/mr/device/pool_memory_resource.hpp" namespace "rmm::mr" nogil:
    cdef cppclass pool_memory_resource[Upstream](device_memory_resource):
        
        size_t pool_size()
    
        pool_memory_resource(
            device_async_resource_ref upstream_mr,
            size_t initial_pool_size,
            optional[size_t] maximum_pool_size
        ) except +

cdef extern from "rmm/mr/device/fixed_size_memory_resource.hpp" namespace "rmm::mr" nogil:
    cdef cppclass fixed_size_memory_resource[Upstream](device_memory_resource):
        
        fixed_size_memory_resource(
            Upstream* upstream_mr,
            size_t block_size,
            size_t block_to_preallocate
        ) except +

cdef extern from "rmm/mr/device/binning_memory_resource.hpp" namespace "rmm::mr" nogil:
    cdef cppclass binning_memory_resource[Upstream](device_memory_resource):
        
        binning_memory_resource(
            Upstream* upstream_mr,
            int8_t min_size_exponent,
            int8_t max_size_exponent) except +

        void add_bin(size_t allocation_size) except +
        void add_bin(
            size_t allocation_size,
            device_memory_resource* bin_resource) except +
    
        binning_memory_resource(Upstream* upstream_mr) except +

cdef extern from "rmm/mr/device/callback_memory_resource.hpp" namespace "rmm::mr" nogil:
    cdef cppclass callback_memory_resource(device_memory_resource):
        
        callback_memory_resource(
            allocate_callback_t allocate_callback,
            deallocate_callback_t deallocate_callback,
            void* allocate_callback_arg,
            void* deallocate_callback_arg    
        ) except +

cdef extern from "rmm/mr/device/limiting_resource_adaptor.hpp" namespace "rmm::mr" nogil:
    cdef cppclass limiting_resource_adaptor[Upstream](device_memory_resource):
        
        size_t get_allocated_bytes() except +
        size_t get_allocation_limit() except +
    
        limiting_resource_adaptor(Upstream* upstream_mr, size_t allocation_limit) except +

cdef extern from "rmm/mr/device/logging_resource_adaptor.hpp" namespace "rmm::mr" nogil:
    cdef cppclass logging_resource_adaptor[Upstream](device_memory_resource):
        
        void flush() except +
    
        logging_resource_adaptor(
            Upstream* upstream_mr,
            string filename
        ) except +

cdef extern from "rmm/mr/device/statistics_resource_adaptor.hpp" namespace "rmm::mr" nogil:
    cdef cppclass statistics_resource_adaptor[Upstream](device_memory_resource):
        
        struct counter:
            counter()

            int64_t value
            int64_t peak
            int64_t total

        counter get_bytes_counter() except +
        counter get_allocations_counter() except +
        pair[counter, counter] pop_counters() except +
        pair[counter, counter] push_counters() except +
    
        statistics_resource_adaptor(Upstream* upstream_mr) except +

cdef extern from "rmm/mr/device/tracking_resource_adaptor.hpp" namespace "rmm::mr" nogil:
    cdef cppclass tracking_resource_adaptor[Upstream](device_memory_resource):
        
        size_t get_allocated_bytes() except +
        string get_outstanding_allocations_str() except +
        void log_outstanding_allocations() except +
    
        tracking_resource_adaptor(
            Upstream* upstream_mr,
            bool capture_stacks
        ) except +

cdef extern from "rmm/mr/device/failure_callback_resource_adaptor.hpp" namespace "rmm::mr" nogil:
    cdef cppclass failure_callback_resource_adaptor[Upstream](device_memory_resource):
        
        failure_callback_resource_adaptor(
            Upstream* upstream_mr,
            failure_callback_t callback,
            void* callback_arg
        ) except +

cdef extern from "rmm/mr/device/prefetch_resource_adaptor.hpp" namespace "rmm::mr" nogil:
    cdef cppclass prefetch_resource_adaptor[Upstream](device_memory_resource):
        
        prefetch_resource_adaptor(Upstream* upstream_mr) except +
