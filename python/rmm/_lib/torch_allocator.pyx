from cuda.ccudart cimport cudaStream_t

from rmm._lib.cuda_stream_view cimport cuda_stream_view
from rmm._lib.memory_resource cimport device_memory_resource
from rmm._lib.per_device_resource cimport get_current_device_resource


cdef public void* allocate(
    ssize_t size, int device, void* stream
) except * with gil:
    cdef device_memory_resource* mr = get_current_device_resource()
    cdef cuda_stream_view stream_view = cuda_stream_view(
        <cudaStream_t>(stream)
    )
    return mr[0].allocate(size, stream_view)

cdef public void deallocate(
    void* ptr, ssize_t size, void* stream
) except * with gil:
    cdef device_memory_resource* mr = get_current_device_resource()
    cdef cuda_stream_view stream_view = cuda_stream_view(
        <cudaStream_t>(stream)
    )
    mr[0].deallocate(ptr, size, stream_view)
