# Copyright (c) 2020, NVIDIA CORPORATION.

from libc.stdint cimport int8_t
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector


cdef extern from "rmm/mr/device/device_memory_resource.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass device_memory_resource:
        pass

cdef class DeviceMemoryResource:
    cdef shared_ptr[device_memory_resource] c_obj

    cdef device_memory_resource* get_mr(self)

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

    cdef readonly list bin_mrs

    cpdef add_bin(
        self,
        size_t allocation_size,
        DeviceMemoryResource bin_resource=*)

cdef class LoggingResourceAdaptor(UpstreamResourceAdaptor):
    cdef object _log_file_name
    cpdef get_file_name(self)
    cpdef flush(self)

cdef class StatisticsResourceAdaptor(UpstreamResourceAdaptor):
    pass

cdef class TrackingResourceAdaptor(UpstreamResourceAdaptor):
    pass

cpdef DeviceMemoryResource get_current_device_resource()
