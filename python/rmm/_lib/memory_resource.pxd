# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.memory cimport shared_ptr


cdef extern from "memory_resource_wrappers.hpp" nogil:
    cdef cppclass device_memory_resource_wrapper:
        shared_ptr[device_memory_resource_wrapper] get_mr() except +

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

    void set_default_resource(
        shared_ptr[device_memory_resource_wrapper] new_resource
    ) except +


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
    cpdef flush(self)

cpdef get_default_resource_type()

cpdef is_initialized()
