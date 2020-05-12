from libcpp.memory cimport unique_ptr, make_unique
from libcpp.string cimport string

cdef class MemoryResource:
    cdef device_memory_resource* c_obj

    def __dealloc__(self):
        del self.c_obj

cdef class CudaMemoryResource(MemoryResource):
    def __cinit__(self):
        self.c_obj = new cuda_memory_resource()

cdef class ManagedMemoryResource(MemoryResource):
    def __cinit__(self):
        self.c_obj = new managed_memory_resource()

cdef class CNMemMemoryResource(MemoryResource):
    def __cinit__(self, size_t initial_pool_size=0, vector[int] devices=[]):
        self.c_obj = new cnmem_memory_resource(
            initial_pool_size,
            devices
        )

cdef class CNMemManagedMemoryResource(MemoryResource):
    def __cinit__(self, size_t initial_pool_size=0, vector[int] devices=[]):
        self.c_obj = new cnmem_managed_memory_resource(
            initial_pool_size,
            devices
        )

cdef class LoggingResourceAdaptor(MemoryResource):
    cdef MemoryResource upstream

    def __cinit__(self, MemoryResource upstream, string log_file_name):
        self.c_obj = new logging_resource_adaptor[device_memory_resource](
            upstream.c_obj,
            log_file_name
        )
        self.upstream = upstream

    def flush(self):
        (
            <logging_resource_adaptor[device_memory_resource]*>
            self.c_obj
        )[0].flush()


cdef MemoryResource mr


def _set_default_resource(
    kind,
    initial_pool_size=0,
    devices=[],
    logging=False,
    log_file_name=None
):
    global mr

    mr = MemoryResource()

    if kind == "cuda":
        typ = CudaMemoryResource
        args = ()
    elif kind == "managed":
        typ = ManagedMemoryResource
        args = ()
    elif kind == "cnmem":
        typ = CNMemMemoryResource
        args = (initial_pool_size, devices)
    elif kind == "cnmem_managed":
        typ = CNMemManagedMemoryResource
        args = (initial_pool_size, devices)
    else:
        raise TypeError(f"Unsupported resource kind: {kind}")

    if logging:
        mr = LoggingResourceAdaptor(typ(*args), log_file_name.encode())
    else:
        mr = typ(*args)

    cdef device_memory_resource* c_mr = mr.c_obj
    set_default_resource(c_mr)


def is_initialized():
    global mr
    return mr.c_obj is not NULL


def flush_logs():
    global mr
    mr.flush()
