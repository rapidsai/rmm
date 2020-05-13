from libcpp.memory cimport unique_ptr, make_unique
from libcpp.string cimport string


cdef class MemoryResource:
    cdef device_memory_resource* c_obj
    cdef object __weakref__

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

cdef class PoolMemoryResource(MemoryResource):
    cdef MemoryResource _upstream

    def __cinit__(
            self,
            MemoryResource upstream,
            size_t initial_pool_size=~0,
            size_t maximum_pool_size=~0
    ):
        self.c_obj = new pool_memory_resource[device_memory_resource](
            upstream.c_obj,
            initial_pool_size,
            maximum_pool_size
        )
        self._upstream = upstream


cdef class FixedSizeMemoryResource(MemoryResource):
    cdef MemoryResource _upstream

    def __cinit__(
            self,
            MemoryResource upstream,
            size_t block_size=1<<20,
            size_t blocks_to_preallocate=128
    ):
        self.c_obj = new pool_memory_resource[device_memory_resource](
            upstream.c_obj,
            block_size,
            blocks_to_preallocate
        )
        self._upstream = upstream


cdef class FixedMultiSizeMemoryResource(MemoryResource):
    cdef MemoryResource _upstream

    def __cinit__(
        self,
        MemoryResource upstream,
        size_t size_base=2,
        size_t min_size_exponent=18,
        size_t max_size_exponent=22,
        size_t initial_blocks_per_size=128
    ):
        self.c_obj = new fixed_multisize_memory_resource[
            device_memory_resource
        ](
            upstream.c_obj,
            size_base,
            min_size_exponent,
            max_size_exponent,
            initial_blocks_per_size
        )
        self._upstream = upstream


cdef class HybridMemoryResource(MemoryResource):
    cdef MemoryResource _small_alloc_mr
    cdef MemoryResource _large_alloc_mr

    def __cinit__(
        self,
        MemoryResource small_alloc_mr,
        MemoryResource large_alloc_mr,
        size_t threshold_size=1<<22
    ):
        self.c_obj = new hybrid_memory_resource[
            device_memory_resource,
            device_memory_resource
        ](
            small_alloc_mr.c_obj,
            large_alloc_mr.c_obj,
            threshold_size
        )
        self._small_alloc_mr = small_alloc_mr
        self._large_alloc_mr = large_alloc_mr


cdef class LoggingResourceAdaptor(MemoryResource):
    cdef MemoryResource _upstream

    def __cinit__(self, MemoryResource upstream, string log_file_name):
        self.c_obj = new logging_resource_adaptor[device_memory_resource](
            upstream.c_obj,
            log_file_name
        )
        self._upstream = upstream

    def flush(self):
        (
            <logging_resource_adaptor[device_memory_resource]*>
            self.c_obj
        )[0].flush()


# Global memory resource:
cdef MemoryResource _mr


def _initialize(
    pool_allocator=False,
    managed_memory=False,
    initial_pool_size=None,
    devices=0,
    logging=False,
    log_file_name=None,
):
    """
    Initializes RMM library using the options passed
    """
    global _mr
    _mr = MemoryResource()

    if not pool_allocator:
        if not managed_memory:
            typ = CudaMemoryResource
        else:
            typ = ManagedMemoryResource
        args = ()
    else:
        if not managed_memory:
            typ = CNMemMemoryResource
        else:
            typ = CNMemManagedMemoryResource

        if initial_pool_size is None:
            initial_pool_size = 0

        if devices is None:
            devices = [0]
        elif isinstance(devices, int):
            devices = [devices]

        args = (initial_pool_size, devices)

    cdef MemoryResource mr

    if logging:
        mr = LoggingResourceAdaptor(typ(*args), log_file_name.encode())
    else:
        mr = typ(*args)

    _set_default_resource(
        mr
    )


def _set_default_resource(MemoryResource mr):
    global _mr

    _mr = mr

    cdef device_memory_resource* c_mr = _mr.c_obj
    set_default_resource(c_mr)


def get_default_resource():
    global _mr

    import weakref
    return weakref.ref(_mr)


def is_initialized():
    global _mr
    return _mr.c_obj is not NULL


def _flush_logs():
    global _mr
    _mr.flush()
