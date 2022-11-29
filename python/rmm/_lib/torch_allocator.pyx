from cuda.ccudart cimport cudaStream_t
from libc.stdint cimport uintptr_t
from libc.stdio cimport printf

from rmm._lib.memory_resource cimport device_memory_resource


cdef extern from "rmm/mr/device/per_device_resource.hpp" namespace "rmm" nogil:
    cdef device_memory_resource* get_current_device_resource \
        "rmm::mr::get_current_device_resource" ()

cdef public void* allocate(ssize_t size, int device, void* stream) except *:
    cdef device_memory_resource* mr = get_current_device_resource()
    return mr[0].allocate(size, <cudaStream_t> stream)

cdef public void deallocate(void* ptr, ssize_t size, void* stream) except *:
    cdef device_memory_resource* mr = get_current_device_resource()
    mr[0].deallocate(ptr, size, <cudaStream_t> stream)
