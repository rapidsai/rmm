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

cdef extern from "rmm/mr/device/device_memory_resource.hpp" namespace "rmm::mr" nogil:
    cdef cppclass device_memory_resource:
        pass

cdef extern from "rmm/mr/device/cuda_memory_resource.hpp" namespace "rmm::mr" nogil:
    cdef cppclass cuda_memory_resource(device_memory_resource):
        cuda_memory_resource() except +

cdef extern from "rmm/mr/device/managed_memory_resource.hpp" namespace "rmm::mr" nogil:
    cdef cppclass managed_memory_resource(device_memory_resource):
        managed_memory_resource() except +

cdef extern from "rmm/mr/device/pool_memory_resource.hpp" namespace "rmm::mr" nogil:
    cdef cppclass pool_memory_resource[Upstream](device_memory_resource):
        pool_memory_resource(Upstream* upstream_mr, optional[size_t] initial_pool_size, optional[size_t] maximum_pool_size) except +

cdef extern from "rmm/mr/device/fixed_size_memory_resource.hpp" namespace "rmm::mr" nogil:
    cdef cppclass fixed_size_memory_resource[Upstream](device_memory_resource):
        fixed_size_memory_resource(Upstream* upstream_mr, size_t block_size, size_t block_to_preallocate) except +

cdef extern from "rmm/mr/device/binning_memory_resource.hpp" namespace "rmm::mr" nogil:
    cdef cppclass binning_memory_resource[Upstream](device_memory_resource):
        binning_memory_resource(Upstream* upstream_mr) except +
        binning_memory_resource(Upstream* upstream_mr, int8_t min_size_exponent, int8_t max_size_exponent) except +

        void add_bin(size_t allocation_size) except +
        void add_bin(size_t allocation_size, device_memory_resource* bin_resource) except +

cdef extern from "rmm/mr/device/logging_resource_adaptor.hpp" namespace "rmm::mr" nogil:
    cdef cppclass logging_resource_adaptor[Upstream](device_memory_resource):
        logging_resource_adaptor(Upstream* upstream_mr, string filename) except +

        void flush() except +


cdef extern from "memory_resource_wrappers.hpp" nogil:
    cdef cppclass device_memory_resource_wrapper:
        shared_ptr[device_memory_resource_wrapper] get_mr() except +

    cdef cppclass default_memory_resource_wrapper(
        device_memory_resource_wrapper
    ):
        default_memory_resource_wrapper(int device) except +

    cdef cppclass cuda_memory_resource_wrapper(device_memory_resource_wrapper):
        cuda_memory_resource_wrapper() except +

    cdef cppclass managed_memory_resource_wrapper(
        device_memory_resource_wrapper
    ):
        managed_memory_resource_wrapper() except +

    cdef cppclass pool_memory_resource_wrapper(device_memory_resource_wrapper):
        pool_memory_resource_wrapper(
            shared_ptr[device_memory_resource_wrapper] upstream_mr,
            size_t initial_pool_size,
            size_t maximum_pool_size
        ) except +

    cdef cppclass fixed_size_memory_resource_wrapper(
        device_memory_resource_wrapper
    ):
        fixed_size_memory_resource_wrapper(
            shared_ptr[device_memory_resource_wrapper] upstream_mr,
            size_t block_size,
            size_t blocks_to_preallocate
        ) except +

    cdef cppclass binning_memory_resource_wrapper(
        device_memory_resource_wrapper
    ):
        binning_memory_resource_wrapper(
            shared_ptr[device_memory_resource_wrapper] upstream_mr
        ) except +
        binning_memory_resource_wrapper(
            shared_ptr[device_memory_resource_wrapper] upstream_mr,
            int8_t min_size_exponent,
            int8_t max_size_exponent
        ) except +
        void add_bin(
            size_t allocation_size,
            shared_ptr[device_memory_resource_wrapper] bin_mr
        ) except +
        void add_bin(
            size_t allocation_size
        ) except +

    cdef cppclass logging_resource_adaptor_wrapper(
        device_memory_resource_wrapper
    ):
        logging_resource_adaptor_wrapper(
            shared_ptr[device_memory_resource_wrapper] upstream_mr,
            string filename
        ) except +
        void flush() except +

    cdef cppclass thread_safe_resource_adaptor_wrapper(
        device_memory_resource_wrapper
    ):
        thread_safe_resource_adaptor_wrapper(
            shared_ptr[device_memory_resource_wrapper] upstream_mr,
        ) except +

    void set_per_device_resource(
        int device,
        shared_ptr[device_memory_resource_wrapper] new_resource
    ) except +


cdef class DeviceMemoryResource:
    cdef shared_ptr[device_memory_resource] c_obj

    cdef device_memory_resource* get_mr(self)

cdef class UpstreamResourceAdaptor(DeviceMemoryResource):
    cdef readonly DeviceMemoryResource upstream_mr

cdef class CudaMemoryResource2(DeviceMemoryResource):
    pass

cdef class ManagedMemoryResource2(DeviceMemoryResource):
    pass

cdef class PoolMemoryResource2(UpstreamResourceAdaptor):
    pass

cdef class FixedSizeMemoryResource2(UpstreamResourceAdaptor):
    pass

cdef class BinningMemoryResource2(UpstreamResourceAdaptor):

    cdef readonly list bin_mrs

    cpdef add_bin(self, size_t allocation_size, DeviceMemoryResource bin_resource=*)

cdef class LoggingResourceAdaptor2(UpstreamResourceAdaptor):
    cdef object _log_file_name
    cpdef get_file_name(self)
    cpdef flush(self)


cdef class MemoryResource:
    cdef shared_ptr[device_memory_resource_wrapper] c_obj

cdef class CudaMemoryResource(MemoryResource):
    pass

cdef class ManagedMemoryResource(MemoryResource):
    pass

cdef class PoolMemoryResource(MemoryResource):
    pass

cdef class FixedSizeMemoryResource(MemoryResource):
    pass

cdef class BinningMemoryResource(MemoryResource):
    cpdef add_bin(self, size_t allocation_size, object bin_resource=*)

cdef class LoggingResourceAdaptor(MemoryResource):
    cdef object _log_file_name
    cpdef get_file_name(self)
    cpdef flush(self)

cpdef MemoryResource get_current_device_resource()
