# Copyright (c) 2020, NVIDIA CORPORATION.
import os

from libcpp cimport bool
from libcpp.cast cimport dynamic_cast
from libcpp.memory cimport make_shared, make_unique, shared_ptr, unique_ptr
from libcpp.string cimport string


cdef class CudaMemoryResource(MemoryResource):
    def __cinit__(self):
        self.c_obj.reset(
            new cuda_memory_resource_wrapper()
        )

    def __init__(self):
        """
        Memory resource that uses cudaMalloc/Free for allocation/deallocation
        """
        pass


cdef class ManagedMemoryResource(MemoryResource):
    def __cinit__(self):
        self.c_obj.reset(
            new managed_memory_resource_wrapper()
        )

    def __init__(self):
        """
        Memory resource that uses cudaMallocManaged/Free for
        allocation/deallocation.
        """
        pass


cdef class CNMemMemoryResource(MemoryResource):
    def __cinit__(self, size_t initial_pool_size=0, vector[int] devices=()):
        self.c_obj.reset(
            new cnmem_memory_resource_wrapper(
                initial_pool_size,
                devices
            )
        )

    def __init__(self, size_t initial_pool_size=0, vector[int] devices=()):
        """
        Memory resource that uses the cnmem pool sub-allocator.

        Parameters
        ----------
        initial_pool_size : int, optional
            Initial pool size in bytes. By default, an implementation defined
            pool size is used.
        devices : tuple of int, optional
            List of GPU device IDs to register with CNMEM.
        """
        pass


cdef class CNMemManagedMemoryResource(MemoryResource):
    def __cinit__(self, size_t initial_pool_size=0, vector[int] devices=()):
        self.c_obj.reset(
            new cnmem_managed_memory_resource_wrapper(
                initial_pool_size,
                devices
            )
        )

    def __init__(self, size_t initial_pool_size=0, vector[int] devices=()):
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
        pass


cdef class PoolMemoryResource(MemoryResource):

    def __cinit__(
            self,
            MemoryResource upstream,
            size_t initial_pool_size=~0,
            size_t maximum_pool_size=~0
    ):
        self.c_obj.reset(
            new pool_memory_resource_wrapper(
                upstream.c_obj,
                initial_pool_size,
                maximum_pool_size
            )
        )

    def __init__(
            self,
            MemoryResource upstream,
            size_t initial_pool_size=~0,
            size_t maximum_pool_size=~0
    ):
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
        pass


cdef class FixedSizeMemoryResource(MemoryResource):
    def __cinit__(
            self,
            MemoryResource upstream,
            size_t block_size=1<<20,
            size_t blocks_to_preallocate=128
    ):
        self.c_obj.reset(
            new thread_safe_resource_adaptor_wrapper(
                shared_ptr[device_memory_resource_wrapper](
                    new fixed_size_memory_resource_wrapper(
                        upstream.c_obj,
                        block_size,
                        blocks_to_preallocate
                    )
                )
            )
        )

    def __init__(
            self,
            MemoryResource upstream,
            size_t block_size=1<<20,
            size_t blocks_to_preallocate=128
    ):
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
        Supports only allocations of size smaller than the configured
        block_size.
        """
        pass


cdef class FixedMultiSizeMemoryResource(MemoryResource):
    def __cinit__(
        self,
        MemoryResource upstream,
        size_t size_base=2,
        size_t min_size_exponent=18,
        size_t max_size_exponent=22,
        size_t initial_blocks_per_size=128
    ):
        self.c_obj.reset(
            new thread_safe_resource_adaptor_wrapper(
                shared_ptr[device_memory_resource_wrapper](
                    new fixed_multisize_memory_resource_wrapper(
                        upstream.c_obj,
                        size_base,
                        min_size_exponent,
                        max_size_exponent,
                        initial_blocks_per_size
                    )
                )
            )
        )

    def __init__(
        self,
        MemoryResource upstream,
        size_t size_base=2,
        size_t min_size_exponent=18,
        size_t max_size_exponent=22,
        size_t initial_blocks_per_size=128
    ):
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
            The number of blocks to preallocate from the upstream memory
            resource, and to allocate when all current blocks are in use.
        """
        pass


cdef class HybridMemoryResource(MemoryResource):
    def __cinit__(
        self,
        MemoryResource small_alloc_mr,
        MemoryResource large_alloc_mr,
        size_t threshold_size=1<<22
    ):
        self.c_obj.reset(
            new thread_safe_resource_adaptor_wrapper(
                shared_ptr[device_memory_resource_wrapper](
                    new hybrid_memory_resource_wrapper(
                        small_alloc_mr.c_obj,
                        large_alloc_mr.c_obj,
                        threshold_size
                    )
                )
            )
        )

    def __init__(
        self,
        MemoryResource small_alloc_mr,
        MemoryResource large_alloc_mr,
        size_t threshold_size=1<<22
    ):
        """"
        Allocates memory from one of two allocators based on the requested
        size.

        Parameters
        ----------
        small_alloc_mr : MemoryResource
            The memory resource to use for small allocations.
        large_alloc_mr : MemoryResource
            The memory resource to use for large allocations.
        threshold_size : int, optional
            Size in bytes representing the threshold beyond which
            `large_alloc_mr` is used.
        """
        pass


cdef class LoggingResourceAdaptor(MemoryResource):
    def __cinit__(self, MemoryResource upstream, object log_file_name=None):
        if log_file_name is None:
            log_file_name = os.getenv("RMM_LOG_FILE")
            if not log_file_name:
                raise TypeError(
                    "RMM log file must be specified either using "
                    "log_file_name= argument or RMM_LOG_FILE "
                    "environment variable"
                )
        self.c_obj.reset(
            new logging_resource_adaptor_wrapper(
                upstream.c_obj,
                log_file_name
            )
        )

    def __init__(self, MemoryResource upstream, object log_file_name=None):
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
        pass

    cpdef flush(self):
        (<logging_resource_adaptor_wrapper*>(self.c_obj.get()))[0].flush()


# Global memory resource:
cdef MemoryResource _mr


cpdef _initialize(
    bool pool_allocator=False,
    bool managed_memory=False,
    object initial_pool_size=None,
    object devices=0,
    bool logging=False,
    object log_file_name=None,
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


cpdef _set_default_resource(MemoryResource mr):
    """
    Set the memory resource to use for RMM device allocations.

    Parameters
    ----------
    mr : MemoryResource
        A MemoryResource object. See `rmm.mr` for the different
        MemoryResource types available.
    """
    global _mr
    _mr = mr
    set_default_resource(_mr.c_obj)


cpdef get_default_resource_type():
    """
    Get the default memory resource type used for RMM device allocations.
    """
    return type(_mr)


cpdef is_initialized():
    global _mr
    return _mr.c_obj.get() is not NULL


cpdef _flush_logs():
    global _mr
    _mr.flush()
