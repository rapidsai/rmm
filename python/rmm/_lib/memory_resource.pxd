# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.memory cimport shared_ptr


cdef extern from "memory_resource_wrappers.hpp":
    cdef cppclass device_memory_resource_wrapper:
        shared_ptr[device_memory_resource_wrapper] get_mr() except +

    cdef cppclass cuda_memory_resource_wrapper(device_memory_resource_wrapper):
        cuda_memory_resource_wrapper() except +

    cdef cppclass managed_memory_resource_wrapper(
        device_memory_resource_wrapper
    ):
        managed_memory_resource_wrapper() except +

    cdef cppclass cnmem_memory_resource_wrapper(
        device_memory_resource_wrapper
    ):
        cnmem_memory_resource_wrapper(
            size_t initial_pool_size,
            vector[int] devices
        ) except +

    cdef cppclass cnmem_managed_memory_resource_wrapper(
        device_memory_resource_wrapper
    ):
        cnmem_managed_memory_resource_wrapper(
            size_t initial_pool_size,
            vector[int] devices
        ) except +

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

    cdef cppclass fixed_multisize_memory_resource_wrapper(
        device_memory_resource_wrapper
    ):
        fixed_multisize_memory_resource_wrapper(
            shared_ptr[device_memory_resource_wrapper] upstream_mr,
            size_t size_base,
            size_t min_size_exponent,
            size_t max_size_exponent,
            size_t initial_blocks_per_size
        ) except +

    cdef cppclass hybrid_memory_resource_wrapper(
        device_memory_resource_wrapper
    ):
        hybrid_memory_resource_wrapper(
            shared_ptr[device_memory_resource_wrapper] small_alloc_mr,
            shared_ptr[device_memory_resource_wrapper] large_alloc_mr,
            size_t threshold_size
        ) except +

    cdef cppclass logging_resource_adaptor_wrapper(
        device_memory_resource_wrapper
    ):
        logging_resource_adaptor_wrapper(
            shared_ptr[device_memory_resource_wrapper] upstream_mr,
            string filename
        ) except +
        void flush() except +

    void set_default_resource(
        shared_ptr[device_memory_resource_wrapper] new_resource
    ) except +


cdef class MemoryResource:
    cdef shared_ptr[device_memory_resource_wrapper] c_obj

cdef class CudaMemoryResource(MemoryResource):
    pass

cdef class ManagedMemoryResource(MemoryResource):
    pass

cdef class CNMemMemoryResource(MemoryResource):
    pass

cdef class CNMemManagedMemoryResource(MemoryResource):
    pass

cdef class PoolMemoryResource(MemoryResource):
    pass

cdef class FixedSizeMemoryResource(MemoryResource):
    pass

cdef class FixedMultiSizeMemoryResource(MemoryResource):
    pass

cdef class HybridMemoryResource(MemoryResource):
    pass

cdef class LoggingResourceAdaptor(MemoryResource):
    cpdef flush(self)
