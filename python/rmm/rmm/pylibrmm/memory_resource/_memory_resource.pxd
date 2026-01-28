# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport shared_ptr, unique_ptr

from rmm.librmm.memory_resource cimport (
    any_device_resource,
    arena_memory_resource,
    binning_memory_resource,
    callback_memory_resource,
    cuda_async_memory_resource,
    cuda_async_view_memory_resource,
    cuda_memory_resource,
    device_memory_resource,
    failure_callback_resource_adaptor,
    fixed_size_memory_resource,
    limiting_resource_adaptor,
    logging_resource_adaptor,
    managed_memory_resource,
    pinned_host_memory_resource,
    pool_memory_resource,
    prefetch_resource_adaptor,
    sam_headroom_memory_resource,
    shared_resource,
    shared_resource_wrapper,
    statistics_resource_adaptor,
    statistics_resource_adaptor_t,
    system_memory_resource,
    tracking_resource_adaptor,
)
from rmm.librmm.per_device_resource cimport device_async_resource_ref


# Import C++ helper function to create resource_ref from pointer
cdef extern from "rmm/resource_ref.hpp" namespace "rmm" nogil:
    cdef device_async_resource_ref to_device_async_resource_ref_checked[
        Resource
    ](Resource* res)


cdef class DeviceMemoryResource:
    cdef any_device_resource c_any_mr
    cdef device_memory_resource* get_mr(self) noexcept nogil

cdef class UpstreamResourceAdaptor(DeviceMemoryResource):
    cdef readonly DeviceMemoryResource upstream_mr

    cpdef DeviceMemoryResource get_upstream(self)

cdef class ArenaMemoryResource(UpstreamResourceAdaptor):
    cdef unique_ptr[arena_memory_resource[device_async_resource_ref]] _typed_mr

cdef class CudaMemoryResource(DeviceMemoryResource):
    cdef unique_ptr[cuda_memory_resource] _typed_mr

cdef class ManagedMemoryResource(DeviceMemoryResource):
    cdef unique_ptr[managed_memory_resource] _typed_mr

cdef class SystemMemoryResource(DeviceMemoryResource):
    cdef unique_ptr[system_memory_resource] _typed_mr

cdef class PinnedHostMemoryResource(DeviceMemoryResource):
    cdef unique_ptr[pinned_host_memory_resource] _typed_mr

cdef class SamHeadroomMemoryResource(DeviceMemoryResource):
    cdef unique_ptr[sam_headroom_memory_resource] _typed_mr

cdef class CudaAsyncMemoryResource(DeviceMemoryResource):
    cdef unique_ptr[cuda_async_memory_resource] _typed_mr

cdef class CudaAsyncViewMemoryResource(DeviceMemoryResource):
    cdef unique_ptr[cuda_async_view_memory_resource] _typed_mr

cdef class PoolMemoryResource(UpstreamResourceAdaptor):
    cdef unique_ptr[pool_memory_resource[device_async_resource_ref]] _typed_mr

cdef class FixedSizeMemoryResource(UpstreamResourceAdaptor):
    cdef unique_ptr[fixed_size_memory_resource[device_async_resource_ref]] _typed_mr

cdef class BinningMemoryResource(UpstreamResourceAdaptor):
    cdef unique_ptr[binning_memory_resource[device_async_resource_ref]] _typed_mr
    cdef readonly list _bin_mrs

    cpdef add_bin(
        self,
        size_t allocation_size,
        DeviceMemoryResource bin_resource=*)

cdef class CallbackMemoryResource(DeviceMemoryResource):
    cdef unique_ptr[callback_memory_resource] _typed_mr
    cdef object _allocate_func
    cdef object _deallocate_func

cdef class LimitingResourceAdaptor(UpstreamResourceAdaptor):
    cdef unique_ptr[limiting_resource_adaptor[device_async_resource_ref]] _typed_mr

cdef class LoggingResourceAdaptor(UpstreamResourceAdaptor):
    cdef unique_ptr[logging_resource_adaptor[device_async_resource_ref]] _typed_mr
    cdef object _log_file_name
    cpdef get_file_name(self)
    cpdef flush(self)

cdef class StatisticsResourceAdaptor(UpstreamResourceAdaptor):
    cdef unique_ptr[shared_resource_wrapper[statistics_resource_adaptor_t]] c_shared_mr

cdef class TrackingResourceAdaptor(UpstreamResourceAdaptor):
    cdef unique_ptr[tracking_resource_adaptor[device_async_resource_ref]] _typed_mr

cdef class FailureCallbackResourceAdaptor(UpstreamResourceAdaptor):
    cdef unique_ptr[
        failure_callback_resource_adaptor[device_async_resource_ref]
    ] _typed_mr
    cdef object _callback

cdef class PrefetchResourceAdaptor(UpstreamResourceAdaptor):
    cdef unique_ptr[prefetch_resource_adaptor[device_async_resource_ref]] _typed_mr

cpdef DeviceMemoryResource get_current_device_resource()
