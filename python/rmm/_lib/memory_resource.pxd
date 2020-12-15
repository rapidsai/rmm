# Copyright (c) 2020, NVIDIA CORPORATION.

from libc.stdint cimport int8_t
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.memory cimport shared_ptr

cdef extern from "thrust/optional.h" namespace "thrust" nogil:

    struct nullopt_t:
        pass

    cdef nullopt_t nullopt

    cdef cppclass optional[T]:
        optional()
        optional(T v)

    cdef optional[T] make_optional[T](T v)

cdef extern from "rmm/mr/device/device_memory_resource.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass device_memory_resource:
        pass

cdef extern from "rmm/mr/device/cuda_memory_resource.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass cuda_memory_resource(device_memory_resource):
        cuda_memory_resource() except +

cdef extern from "rmm/mr/device/managed_memory_resource.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass managed_memory_resource(device_memory_resource):
        managed_memory_resource() except +

cdef extern from "rmm/mr/device/pool_memory_resource.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass pool_memory_resource[Upstream](device_memory_resource):
        pool_memory_resource(
            Upstream* upstream_mr,
            optional[size_t] initial_pool_size,
            optional[size_t] maximum_pool_size) except +

cdef extern from "rmm/mr/device/fixed_size_memory_resource.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass fixed_size_memory_resource[Upstream](device_memory_resource):
        fixed_size_memory_resource(
            Upstream* upstream_mr,
            size_t block_size,
            size_t block_to_preallocate) except +

cdef extern from "rmm/mr/device/binning_memory_resource.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass binning_memory_resource[Upstream](device_memory_resource):
        binning_memory_resource(Upstream* upstream_mr) except +
        binning_memory_resource(
            Upstream* upstream_mr,
            int8_t min_size_exponent,
            int8_t max_size_exponent) except +

        void add_bin(size_t allocation_size) except +
        void add_bin(
            size_t allocation_size,
            device_memory_resource* bin_resource) except +

cdef extern from "rmm/mr/device/logging_resource_adaptor.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass logging_resource_adaptor[Upstream](device_memory_resource):
        logging_resource_adaptor(
            Upstream* upstream_mr,
            string filename) except +

        void flush() except +

cdef extern from "rmm/mr/device/per_device_resource.hpp" namespace "rmm" nogil:

    cdef cppclass cuda_device_id:
        ctypedef int value_type

        cuda_device_id(value_type id)

        value_type value()

    cdef device_memory_resource* _set_current_device_resource \
        "rmm::mr::set_current_device_resource" (device_memory_resource* new_mr)
    cdef device_memory_resource* _get_current_device_resource \
        "rmm::mr::get_current_device_resource" ()

    cdef device_memory_resource* _set_per_device_resource \
        "rmm::mr::set_per_device_resource" (
            cuda_device_id id,
            device_memory_resource* new_mr
        )
    cdef device_memory_resource* _get_per_device_resource \
        "rmm::mr::get_per_device_resource"(cuda_device_id id)


cdef class DeviceMemoryResource:
    cdef shared_ptr[device_memory_resource] c_obj

    cdef device_memory_resource* get_mr(self)

cdef class UpstreamResourceAdaptor(DeviceMemoryResource):
    cdef readonly DeviceMemoryResource upstream_mr

cdef class CudaMemoryResource(DeviceMemoryResource):
    pass

cdef class ManagedMemoryResource(DeviceMemoryResource):
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

cpdef DeviceMemoryResource get_current_device_resource()
