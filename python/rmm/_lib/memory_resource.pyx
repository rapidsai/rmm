import cython
import os

from libcpp.memory cimport unique_ptr, make_unique
from libcpp.string cimport string


cdef class MemoryResource:
    cdef device_memory_resource* c_obj
    cdef object __weakref__


cdef class CudaMemoryResource(MemoryResource):
    """
    Memory resource that uses cudaMalloc/Free for allocation/deallocation
    """
    def __cinit__(self):
        self.c_obj = new cuda_memory_resource()

    def __dealloc__(self):
        del self.c_obj


cdef class ManagedMemoryResource(MemoryResource):
    """
    Memory resource that uses cudaMallocManaged/Free for
    allocation/deallocation.
    """
    def __cinit__(self):
        self.c_obj = new managed_memory_resource()

    def __dealloc__(self):
        del self.c_obj


cdef class CNMemMemoryResource(MemoryResource):
    """
    Memory resource that uses the cnmem pool sub-allocator.

    Parameters
    ----------
    initial_pool_size : int, optional
        Initial pool size in bytes. By default, an implementation defined
        pool size is used.
    devices : list of int
        List of GPU device IDs to register with CNMEM.
    """
    def __cinit__(self, size_t initial_pool_size=0, vector[int] devices=[]):
        self.c_obj = new cnmem_memory_resource(
            initial_pool_size,
            devices
        )

    def __dealloc__(self):
        del self.c_obj


cdef class CNMemManagedMemoryResource(MemoryResource):
    """
    Memory resource that uses the cnmem pool sub-allocator for
    allocating/deallocating managed device memory.

    Parameters
    ----------
    initial_pool_size : int, optional
        Initial pool size in bytes. By default, an implementation defined
        pool size is used.
    devices : list of int
        List of GPU device IDs to register with CNMEM.
    """
    def __cinit__(self, size_t initial_pool_size=0, vector[int] devices=[]):
        self.c_obj = new cnmem_managed_memory_resource(
            initial_pool_size,
            devices
        )

    def __dealloc__(self):
        del self.c_obj

cdef class PoolMemoryResource(MemoryResource):
    """
    Coalescing best-fit suballocator which uses a pool of memory allocated
    from an upstream memory resource.

    Parameters
    ----------
    upstream : MemoryResource
        The MemoryResource from which to allocate blocks for the pool.
    initial_pool_size : int,optional
        Initial pool size in bytes. By default, an implementation defined
        pool size is used.
    maximum_pool_size : int, optional
        Maximum size in bytes, that the pool can grow to.
    """
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

    def __dealloc__(self):
        del self.c_obj

cdef class FixedSizeMemoryResource(MemoryResource):
    """
    Memory resource which allocates memory blocks of a single fixed size.

    Parameters
    ----------
    upstream : MemoryResource
        The MemoryResource from which to allocate blocks for the pool.
    block_size : int, optional
        The size of blocks to allocate (default is 1MiB).
    blocks_to_preallocate : int, optional
        The number of blocks to allocate to initialize the pool.

    Notes
    -----
    Supports only allocations of size smaller than the configured block_size.
    """
    cdef MemoryResource _upstream

    def __cinit__(
            self,
            MemoryResource upstream,
            size_t block_size=1<<20,
            size_t blocks_to_preallocate=128
    ):
        self.c_obj = new fixed_size_memory_resource[device_memory_resource](
            upstream.c_obj,
            block_size,
            blocks_to_preallocate
        )
        self._upstream = upstream

    def __dealloc__(self):
        del self.c_obj

cdef class FixedMultiSizeMemoryResource(MemoryResource):
    """
    Allocates blocks in the range `[min_size], max_size]` in power oftwo
    steps, where `min_size` and `max_size` are both powers of two.

    Parameters
    ----------
    upstream : MemoryResource
        The upstream memory resource used to allocate pools of blocks.
    size_base : int, optional
        The base of allocation block sizes (defaults is 2).
    min_size_exponent : int, optional
        The exponent of the minimum fixed block size to allocate
        (default is 18).
    max_size_exponent : int, optional
        The exponent of the maximum fixed block size to allocate
        (default is 22).
    initial_blocks_per_size : int, optional
        The number of blocks to preallocate from the upstream memory resource,
        and to allocate when all current blocks are in use.
    """
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

    def __dealloc__(self):
        del self.c_obj

cdef class HybridMemoryResource(MemoryResource):
    """"
    Allocates memory from one of two allocators based on the requested size.

    Parameters
    ----------
    small_alloc_mr : MemoryResource
        The memory resource to use for small allocations.
    large_alloc_mr : MemoryResource
        The memory resource to use for large allocations.
    threshold_size : int, optional
        Size in bytes representing the threshold beyond which `large_alloc_mr`
        is used.
    """
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

    def __dealloc__(self):
        del self.c_obj

cdef class LoggingResourceAdaptor(MemoryResource):
    """
    Memory resource that logs information about allocations/deallocations
    performed by an upstream memory resource.

    Parameters
    ----------
    upstream : MemoryResource
        The upstream memory resource.
    log_file_name : str
        Path to the file to which logs are written.
    """
    cdef MemoryResource _upstream

    def __cinit__(self, MemoryResource upstream, object log_file_name=None):
        if log_file_name is None:
            log_file_name = os.getenv("RMM_LOG_FILE")
            if not log_file_name:
                raise TypeError(
                    "RMM log file must be specified either using "
                    "log_file_name= argument or RMM_LOG_FILE "
                    "environment variable"
                )
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

    def __dealloc__(self):
        del self.c_obj


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
