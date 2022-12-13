from cuda.ccudart cimport cudaStream_t
from libc.stdint cimport uintptr_t
from libc.stdio cimport printf

from rmm._lib.memory_resource cimport device_memory_resource
from rmm._lib.per_device_resource cimport get_current_device_resource


cdef public void* allocate(
    ssize_t size, int device, void* stream
) except * with gil:
    cdef device_memory_resource* mr = get_current_device_resource()
    return mr[0].allocate(size, <cudaStream_t> stream)

cdef public void deallocate(
    void* ptr, ssize_t size, void* stream
) except * with gil:
    cdef device_memory_resource* mr = get_current_device_resource()
    mr[0].deallocate(ptr, size, <cudaStream_t> stream)
