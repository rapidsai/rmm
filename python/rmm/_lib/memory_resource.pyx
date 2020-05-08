from libcpp.memory cimport unique_ptr, make_unique

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


def _set_default_resource(kind, initial_pool_size=0, devices=[]):
    cdef MemoryResource mr
    if kind == "cuda":
        mr = CudaMemoryResource()
    elif kind == "managed":
        mr = ManagedMemoryResource()
    elif kind == "cnmem":
        mr = CNMemMemoryResource(initial_pool_size, devices)
    elif kind == "cnmem_managed":
        mr = CNMemManagedMemoryResource(initial_pool_size, devices)
    else:
        raise TypeError(f"Unsupported resource kinnd: {kind}")
    cdef device_memory_resource* c_mr = mr.c_obj
    set_default_resource(c_mr)
    return mr
