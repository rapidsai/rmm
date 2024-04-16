# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

from libc.stdint cimport int8_t
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

from rmm._lib.cuda_stream_view cimport cuda_stream_view


cdef extern from "rmm/mr/device/device_memory_resource.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass device_memory_resource:
        void* allocate(size_t bytes) except +
        void* allocate(size_t bytes, cuda_stream_view stream) except +
        void deallocate(void* ptr, size_t bytes) except +
        void deallocate(
            void* ptr,
            size_t bytes,
            cuda_stream_view stream
        ) except +

cdef class DeviceMemoryResource:
    cdef shared_ptr[device_memory_resource] c_obj
    cdef device_memory_resource* get_mr(self) noexcept nogil

cdef class UpstreamResourceAdaptor(DeviceMemoryResource):
    cdef readonly DeviceMemoryResource upstream_mr

    cpdef DeviceMemoryResource get_upstream(self)

cdef class CudaMemoryResource(DeviceMemoryResource):
    pass

cdef class ManagedMemoryResource(DeviceMemoryResource):
    pass

cdef class CudaAsyncMemoryResource(DeviceMemoryResource):
    pass

cdef class PoolMemoryResource(UpstreamResourceAdaptor):
    pass

cdef class FixedSizeMemoryResource(UpstreamResourceAdaptor):
    pass

cdef class BinningMemoryResource(UpstreamResourceAdaptor):

    cdef readonly list _bin_mrs

    cpdef add_bin(
        self,
        size_t allocation_size,
        DeviceMemoryResource bin_resource=*)

cdef class CallbackMemoryResource(DeviceMemoryResource):
    cdef object _allocate_func
    cdef object _deallocate_func

cdef class LimitingResourceAdaptor(UpstreamResourceAdaptor):
    pass

cdef class LoggingResourceAdaptor(UpstreamResourceAdaptor):
    cdef object _log_file_name
    cpdef get_file_name(self)
    cpdef flush(self)

cdef class StatisticsResourceAdaptor(UpstreamResourceAdaptor):
    pass

cdef class TrackingResourceAdaptor(UpstreamResourceAdaptor):
    pass

cdef class FailureCallbackResourceAdaptor(UpstreamResourceAdaptor):
    cdef object _callback

cpdef DeviceMemoryResource get_current_device_resource()
