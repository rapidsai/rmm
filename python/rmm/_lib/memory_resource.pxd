from libcpp cimport bool
from libcpp.vector cimport vector

from rmm._lib.lib cimport cudaStream_t


cdef extern from "rmm/mr/device/device_memory_resource.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass device_memory_resource:
        device_memory_resource()

cdef extern from "rmm/mr/device/cuda_memory_resource.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass cuda_memory_resource(device_memory_resource):
        bool supports_streams()

cdef extern from "rmm/mr/device/managed_memory_resource.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass managed_memory_resource(device_memory_resource):
        bool supports_streams()

cdef extern from "rmm/mr/device/cnmem_memory_resource.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass cnmem_memory_resource(device_memory_resource):
        cnmem_memory_resource()
        cnmem_memory_resource(size_t initial_pool_size,)
        cnmem_memory_resource(
            size_t initial_pool_size,
            const vector[int]& devices
        )
        bool supports_streams()

cdef extern from "rmm/mr/device/cnmem_managed_memory_resource.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass cnmem_managed_memory_resource(cnmem_memory_resource):
        cnmem_managed_memory_resource()
        cnmem_managed_memory_resource(size_t initial_pool_size,)
        cnmem_managed_memory_resource(
            size_t initial_pool_size,
            const vector[int]& devices
        )

cdef extern from "rmm/mr/device/default_memory_resource.hpp" \
        namespace "rmm::mr" nogil:
    cdef device_memory_resource* get_default_resource()
    cdef device_memory_resource* set_default_resource(
        device_memory_resource* new_resource
    ) except +
