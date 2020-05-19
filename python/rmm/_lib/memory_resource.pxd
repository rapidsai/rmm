from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.string cimport string

from rmm._lib.lib cimport cudaStream_t
from libcpp.memory cimport unique_ptr, make_unique, shared_ptr, make_shared


cdef extern from "memory_resource_util.hpp":
    cdef cppclass device_memory_resource:
        shared_ptr[device_memory_resource] get_mr() except +

    cdef cppclass cuda_memory_resource(device_memory_resource):
        cuda_memory_resource() except +

    cdef cppclass managed_memory_resource(device_memory_resource):
        managed_memory_resource() except +

    cdef cppclass cnmem_memory_resource(device_memory_resource):
        cnmem_memory_resource(
            size_t initial_pool_size,
            vector[int] devices
        ) except +

    cdef cppclass cnmem_managed_memory_resource(device_memory_resource):
        cnmem_managed_memory_resource(
            size_t initial_pool_size,
            vector[int] devices
        ) except +

    cdef cppclass pool_memory_resource(device_memory_resource):
        pool_memory_resource(
            shared_ptr[device_memory_resource] upstream_mr,
            size_t initial_pool_size,
            size_t maximum_pool_size
        ) except +

    cdef cppclass fixed_size_memory_resource(device_memory_resource):
        fixed_size_memory_resource(
            shared_ptr[device_memory_resource] upstream_mr,
            size_t block_size,
            size_t blocks_to_preallocate
        ) except +

    cdef cppclass fixed_multisize_memory_resource(device_memory_resource):
        fixed_multisize_memory_resource(
            shared_ptr[device_memory_resource] upstream_mr,
            size_t size_base,
            size_t min_size_exponent,
            size_t max_size_exponent,
            size_t initial_blocks_per_size
        ) except +

    cdef cppclass hybrid_memory_resource(device_memory_resource):
        hybrid_memory_resource(
            shared_ptr[device_memory_resource] small_alloc_mr,
            shared_ptr[device_memory_resource] large_alloc_mr,
            size_t threshold_size
        ) except +

    cdef cppclass logging_resource_adaptor(device_memory_resource):
        logging_resource_adaptor(
            shared_ptr[device_memory_resource] upstream_mr,
            string filename
        ) except +
        void flush()

    void set_default_resource(shared_ptr[device_memory_resource] new_resource)
