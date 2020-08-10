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
            new fixed_size_memory_resource_wrapper(
                upstream.c_obj,
                block_size,
                blocks_to_preallocate
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


cdef class BinningMemoryResource(MemoryResource):
    def __cinit__(
        self,
        MemoryResource upstream_mr,
        int min_size_exponent=-1,
        int max_size_exponent=-1,
    ):
        if (min_size_exponent == -1 or max_size_exponent == -1):
            self.c_obj.reset(
                new binning_memory_resource_wrapper(
                    upstream_mr.c_obj
                )
            )
        else:
            self.c_obj.reset(
                new binning_memory_resource_wrapper(
                    upstream_mr.c_obj,
                    min_size_exponent,
                    max_size_exponent
                )
            )

    def __init__(
        self,
        MemoryResource upstream_mr,
        int min_size_exponent=-1,
        int max_size_exponent=-1,
    ):
        """
        Allocates memory from a set of specified "bin" sizes based on a
        specified allocation size.

        If min_size_exponent and max_size_exponent are specified, initializes
        with one or more FixedSizeMemoryResource bins in the range
        [2^min_size_exponent, 2^max_size_exponent].

        Call add_bin to add additional bin allocators.

        Parameters
        ----------
        upstream_mr : MemoryResource
            The memory resource to use for allocations larger than any of the
            bins
        min_size_exponent : size_t
            The base-2 exponent of the minimum size FixedSizeMemoryResource
            bin to create.
        max_size_exponent : size_t
            The base-2 exponent of the maximum size FixedSizeMemoryResource
            bin to create.
        """
        pass

    cpdef add_bin(
        self,
        size_t allocation_size,
        object bin_resource=None
    ):
        """
        Adds a bin of the specified maximum allocation size to this memory
        resource. If specified, uses bin_resource for allocation for this bin.
        If not specified, creates and uses a FixedSizeMemoryResource for
        allocation for this bin.

        Allocations smaller than allocation_size and larger than the next
        smaller bin size will use this fixed-size memory resource.

        Parameters
        ----------
        allocation_size : size_t
            The maximum allocation size in bytes for the created bin
        bin_resource : MemoryResource
            The resource to use for this bin (optional)
        """
        cdef MemoryResource _bin_resource

        if bin_resource is None:
            (<binning_memory_resource_wrapper*>(self.c_obj.get()))[0].add_bin(
                allocation_size
            )
        else:
            # Coerce Python object `bin_resource` to C object `_bin_resource`
            _bin_resource = bin_resource

            (<binning_memory_resource_wrapper*>(self.c_obj.get()))[0].add_bin(
                allocation_size,
                _bin_resource.c_obj
            )

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
    bool logging=False,
    object log_file_name=None,
):
    """
    Initializes RMM library using the options passed
    """
    global _mr
    _mr = MemoryResource()

    if managed_memory:
        upstream = ManagedMemoryResource
    else:
        upstream = CudaMemoryResource

    if pool_allocator:
        if initial_pool_size is None:
            initial_pool_size = 0

        typ = PoolMemoryResource
        args = (upstream(), initial_pool_size)
    else:
        typ = upstream
        args = ()

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
