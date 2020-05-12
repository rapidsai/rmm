from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.string cimport string

from rmm._lib.lib cimport cudaStream_t


cdef extern from "rmm/mr/device/device_memory_resource.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass device_memory_resource:
        device_memory_resource() except +

cdef extern from "rmm/mr/device/cuda_memory_resource.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass cuda_memory_resource(device_memory_resource):
        pass

cdef extern from "rmm/mr/device/managed_memory_resource.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass managed_memory_resource(device_memory_resource):
        pass

cdef extern from "rmm/mr/device/cnmem_memory_resource.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass cnmem_memory_resource(device_memory_resource):
        cnmem_memory_resource() except +
        cnmem_memory_resource(size_t initial_pool_size) except +
        cnmem_memory_resource(
            size_t initial_pool_size,
            const vector[int]& devices
        ) except +

cdef extern from "rmm/mr/device/cnmem_managed_memory_resource.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass cnmem_managed_memory_resource(cnmem_memory_resource):
        cnmem_managed_memory_resource() except +
        cnmem_managed_memory_resource(size_t initial_pool_size) except +
        cnmem_managed_memory_resource(
            size_t initial_pool_size,
            const vector[int]& devices
        ) except +

cdef extern from "rmm/mr/device/default_memory_resource.hpp" \
        namespace "rmm::mr" nogil:
    cdef device_memory_resource* get_default_resource() except +
    cdef device_memory_resource* set_default_resource(
        device_memory_resource* new_resource
    ) except +


cdef extern from "rmm/mr/device/logging_resource_adaptor.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass logging_resource_adaptor[Upstream](device_memory_resource):
        logging_resource_adaptor(Upstream* upstream) except +
        logging_resource_adaptor(
            Upstream* upstream,
            const string& filename
        ) except +
        Upstream* get_upstream() except +
        void flush() except +
