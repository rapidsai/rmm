from rmm._cuda.gpu cimport cudaError_t
from rmm._lib.lib cimport cudaStream_t

cdef extern from * nogil:

    ctypedef void* cudaEvent_t "cudaEvent_t"

    cudaError_t cudaEventCreate(cudaEvent_t* event)

    cudaError_t cudaEventRecord(cudaEvent_t event)
    cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)

    cudaError_t cudaStreamWaitEvent(cudaStream_t stream,
                                    cudaEvent_t event,
                                    unsigned int flags)

cdef class Event:
    cdef cudaEvent_t _cuda_event
