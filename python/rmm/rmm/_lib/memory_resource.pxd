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

from libc.stdint cimport int8_t
from libcpp.memory cimport make_shared, shared_ptr
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.vector cimport vector

from rmm._lib.cuda_stream_view cimport cuda_stream_view
from rmm._lib.resource_ref cimport (
    CUDA_ALLOCATION_ALIGNMENT,
    device_async_resource_ref,
    stream_ref,
)

include "extern_memory_resources.pxd"

cdef extern from *:
    """
    template <typename T>
    rmm::device_async_resource_ref as_ref(T *p) { return p; }
    """

    device_async_resource_ref as_ref[T](T *p) noexcept nogil


cdef extern from "rmm/cuda_device.hpp" namespace "rmm" nogil:
    size_t percent_of_free_device_memory(int percent) except +
    pair[size_t, size_t] available_device_memory() except +


cdef class DeviceMemoryResource:
    cdef device_memory_resource* get_mr(self) noexcept nogil
    cdef shared_ptr[device_async_resource_ref] get_ref(self) noexcept nogil


cdef class UpstreamResourceAdaptor(DeviceMemoryResource):
    cdef readonly DeviceMemoryResource upstream_mr
    cpdef DeviceMemoryResource get_upstream(self)


cdef class CudaMemoryResource(DeviceMemoryResource):
    cdef shared_ptr[cuda_memory_resource] c_obj
    cdef inline device_memory_resource* get_mr(self) noexcept nogil:
        return self.c_obj.get()
    cdef inline shared_ptr[device_async_resource_ref] get_ref(self) noexcept nogil:
        return make_shared[device_async_resource_ref](as_ref(self.c_obj.get()))


cdef class ManagedMemoryResource(DeviceMemoryResource):
    cdef shared_ptr[managed_memory_resource] c_obj
    cdef inline device_memory_resource* get_mr(self) noexcept nogil:
        return self.c_obj.get()
    cdef inline shared_ptr[device_async_resource_ref] get_ref(self) noexcept nogil:
        return make_shared[device_async_resource_ref](as_ref(self.c_obj.get()))


cdef class SystemMemoryResource(DeviceMemoryResource):
    cdef shared_ptr[system_memory_resource] c_obj
    cdef inline device_memory_resource* get_mr(self) noexcept nogil:
        return self.c_obj.get()
    cdef inline shared_ptr[device_async_resource_ref] get_ref(self) noexcept nogil:
        return make_shared[device_async_resource_ref](as_ref(self.c_obj.get()))


cdef class SamHeadroomMemoryResource(DeviceMemoryResource):
    cdef shared_ptr[sam_headroom_memory_resource] c_obj
    cdef inline device_memory_resource* get_mr(self) noexcept nogil:
        return self.c_obj.get()
    cdef inline shared_ptr[device_async_resource_ref] get_ref(self) noexcept nogil:
        return make_shared[device_async_resource_ref](as_ref(self.c_obj.get()))


cdef class CudaAsyncMemoryResource(DeviceMemoryResource):
    cdef shared_ptr[cuda_async_memory_resource] c_obj
    cdef inline device_memory_resource* get_mr(self) noexcept nogil:
        return self.c_obj.get()
    cdef inline shared_ptr[device_async_resource_ref] get_ref(self) noexcept nogil:
        return make_shared[device_async_resource_ref](as_ref(self.c_obj.get()))


cdef class PoolMemoryResource(UpstreamResourceAdaptor):
    cdef shared_ptr[pool_memory_resource[device_memory_resource]] c_obj
    cdef inline device_memory_resource* get_mr(self) noexcept nogil:
        return self.c_obj.get()
    cdef inline shared_ptr[device_async_resource_ref] get_ref(self) noexcept nogil:
        return make_shared[device_async_resource_ref](as_ref(self.c_obj.get()))


cdef class FixedSizeMemoryResource(UpstreamResourceAdaptor):
    cdef shared_ptr[fixed_size_memory_resource[device_memory_resource]] c_obj
    cdef inline device_memory_resource* get_mr(self) noexcept nogil:
        return self.c_obj.get()
    cdef inline shared_ptr[device_async_resource_ref] get_ref(self) noexcept nogil:
        return make_shared[device_async_resource_ref](as_ref(self.c_obj.get()))


cdef class BinningMemoryResource(UpstreamResourceAdaptor):
    cdef shared_ptr[binning_memory_resource[device_memory_resource]] c_obj
    cdef inline device_memory_resource* get_mr(self) noexcept nogil:
        return self.c_obj.get()
    cdef inline shared_ptr[device_async_resource_ref] get_ref(self) noexcept nogil:
        return make_shared[device_async_resource_ref](as_ref(self.c_obj.get()))

    cdef readonly list _bin_mrs

    cpdef add_bin(
        self,
        size_t allocation_size,
        DeviceMemoryResource bin_resource=*)

cdef class CallbackMemoryResource(DeviceMemoryResource):
    cdef shared_ptr[callback_memory_resource] c_obj
    cdef inline device_memory_resource* get_mr(self) noexcept nogil:
        return self.c_obj.get()
    cdef inline shared_ptr[device_async_resource_ref] get_ref(self) noexcept nogil:
        return make_shared[device_async_resource_ref](as_ref(self.c_obj.get()))

    cdef object _allocate_func
    cdef object _deallocate_func

cdef class LimitingResourceAdaptor(UpstreamResourceAdaptor):
    cdef shared_ptr[limiting_resource_adaptor[device_memory_resource]] c_obj
    cdef inline device_memory_resource* get_mr(self) noexcept nogil:
        return self.c_obj.get()
    cdef inline shared_ptr[device_async_resource_ref] get_ref(self) noexcept nogil:
        return make_shared[device_async_resource_ref](as_ref(self.c_obj.get()))


cdef class LoggingResourceAdaptor(UpstreamResourceAdaptor):
    cdef shared_ptr[logging_resource_adaptor[device_memory_resource]] c_obj
    cdef inline device_memory_resource* get_mr(self) noexcept nogil:
        return self.c_obj.get()
    cdef inline shared_ptr[device_async_resource_ref] get_ref(self) noexcept nogil:
        return make_shared[device_async_resource_ref](as_ref(self.c_obj.get()))

    cdef object _log_file_name
    cpdef get_file_name(self)
    cpdef flush(self)

cdef class StatisticsResourceAdaptor(UpstreamResourceAdaptor):
    cdef shared_ptr[statistics_resource_adaptor[device_memory_resource]] c_obj
    cdef inline device_memory_resource* get_mr(self) noexcept nogil:
        return self.c_obj.get()
    cdef inline shared_ptr[device_async_resource_ref] get_ref(self) noexcept nogil:
        return make_shared[device_async_resource_ref](as_ref(self.c_obj.get()))


cdef class TrackingResourceAdaptor(UpstreamResourceAdaptor):
    cdef shared_ptr[tracking_resource_adaptor[device_memory_resource]] c_obj
    cdef inline device_memory_resource* get_mr(self) noexcept nogil:
        return self.c_obj.get()
    cdef inline shared_ptr[device_async_resource_ref] get_ref(self) noexcept nogil:
        return make_shared[device_async_resource_ref](as_ref(self.c_obj.get()))


cdef class FailureCallbackResourceAdaptor(UpstreamResourceAdaptor):
    cdef shared_ptr[failure_callback_resource_adaptor[device_memory_resource]] c_obj
    cdef inline device_memory_resource* get_mr(self) noexcept nogil:
        return self.c_obj.get()
    cdef inline shared_ptr[device_async_resource_ref] get_ref(self) noexcept nogil:
        return make_shared[device_async_resource_ref](as_ref(self.c_obj.get()))

    cdef object _callback


cdef class PrefetchResourceAdaptor(UpstreamResourceAdaptor):
    cdef shared_ptr[prefetch_resource_adaptor[device_memory_resource]] c_obj
    cdef inline device_memory_resource* get_mr(self) noexcept nogil:
        return self.c_obj.get()
    cdef inline shared_ptr[device_async_resource_ref] get_ref(self) noexcept nogil:
        return make_shared[device_async_resource_ref](as_ref(self.c_obj.get()))


cpdef DeviceMemoryResource get_current_device_resource()
