# Copyright (c) 2020, NVIDIA CORPORATION.
import os
import warnings

from libc.stdint cimport int8_t
from libcpp cimport bool
from libcpp.cast cimport dynamic_cast
from libcpp.memory cimport make_shared, make_unique, shared_ptr, unique_ptr
from libcpp.string cimport string

from rmm._lib.lib cimport cudaGetDevice, cudaSetDevice, cudaSuccess


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
        int8_t min_size_exponent=-1,
        int8_t max_size_exponent=-1,
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
        int8_t min_size_exponent=-1,
        int8_t max_size_exponent=-1,
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


def _append_id(filename, id):
    """
    Append ".dev<ID>" onto a filename before the extension

    Example: _append_id("hello.txt", 1) returns "hello.dev1.txt"

    Parameters
    ----------
    filename : string
        The filename, possibly with extension
    id : int
        The ID to append
    """
    name, ext = os.path.splitext(filename)
    return f"{name}.dev{id}{ext}"


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
        # Append the device ID before the file extension
        log_file_name = _append_id(
            log_file_name.decode(), get_current_device()
        )

        _log_file_name = log_file_name

        self.c_obj.reset(
            new logging_resource_adaptor_wrapper(
                upstream.c_obj,
                log_file_name.encode()
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

    cpdef get_file_name(self):
        return self._log_file_name

# Global per-device memory resources; dict of int:MemoryResource
cdef dict _per_device_mrs = {}


cpdef int get_current_device() except -1:
    """
    Get the current CUDA device
    """
    cdef int current_device
    err = cudaGetDevice(&current_device)
    if err != cudaSuccess:
        raise RuntimeError(f"Failed to get CUDA device with error: {err}")
    return current_device


cpdef void set_current_device(int device) except *:
    """
    Set the current CUDA device

    Parameters
    ----------
    device : int
        The ID of the device to set as current
    """
    err = cudaSetDevice(device)
    if err != cudaSuccess:
        raise RuntimeError(f"Failed to set CUDA device with error: {err}")


cpdef void _initialize(
    bool pool_allocator=False,
    bool managed_memory=False,
    object initial_pool_size=None,
    object devices=0,
    bool logging=False,
    object log_file_name=None,
    bool cuda_initialization=True,
) except *:
    """
    Initializes RMM library using the options passed
    """
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
    cdef int original_device

    # Save the current device so we can reset it
    try:
        if cuda_initialization:
            original_device = get_current_device()
    except RuntimeError:
        warnings.warn("No CUDA Device Found", ResourceWarning)
    else:
        # reset any previously specified per device resources
        global _per_device_mrs
        _per_device_mrs.clear()

        if devices is None:
            devices = [0]
        elif isinstance(devices, int):
            devices = [devices]

        if not cuda_initialization and devices != [0]:
            raise RuntimeError(
                "Avoiding CUDA initialization is not allowed with a device "
                "other than 0"
            )

        # create a memory resource per specified device
        for device in devices:
            if cuda_initialization:
                set_current_device(device)

            if logging:
                mr = LoggingResourceAdaptor(typ(*args), log_file_name.encode())
            else:
                mr = typ(*args)

            _set_per_device_resource(device, mr)

        if cuda_initialization:
            # reset CUDA device to original
            set_current_device(original_device)


cpdef void _import_initialize() except *:
    """
    Function used to import RMM at import. Checks if either a
    ``RMM_NO_INITIALIZE`` or ``RAPIDS_NO_INITIALIZE`` environment variable
    exists, and avoids initializing the CUDA Driver / Runtime.
    """
    cuda_initialization = True
    if (
            "RMM_NO_INITIALIZE" in os.environ or
            "RAPIDS_NO_INITIALIZE" in os.environ
    ):
        cuda_initialization = False

    _initialize(cuda_initialization=cuda_initialization)


cpdef get_per_device_resource(int device):
    """
    Get the default memory resource for the specified device.

    Parameters
    ----------
    device : int
        The ID of the device for which to get the memory resource.
    """
    global _per_device_mrs
    return _per_device_mrs[device]


cpdef _set_per_device_resource(int device, MemoryResource mr):
    """
    Set the default memory resource for the specified device.

    Parameters
    ----------
    device : int
        The ID of the device for which to get the memory resource.
    mr : MemoryResource
        The memory resource to set.
    """
    global _per_device_mrs
    _per_device_mrs[device] = mr
    _mr = mr  # coerce Python object to C object
    set_per_device_resource(device, _mr.c_obj)


cpdef set_current_device_resource(MemoryResource mr):
    """
    Set the default memory resource for the current device.

    Parameters
    ----------
    mr : MemoryResource
        The memory resource to set.
    """
    _set_per_device_resource(get_current_device(), mr)


cpdef get_per_device_resource_type(int device):
    """
    Get the memory resource type used for RMM device allocations on the
    specified device.

    Parameters
    ----------
    device : int
        The device ID
    """
    return type(get_per_device_resource(device))


cpdef get_current_device_resource():
    """
    Get the memory resource used for RMM device allocations on the current
    device.
    """
    return get_per_device_resource(get_current_device())


cpdef get_current_device_resource_type():
    """
    Get the memory resource type used for RMM device allocations on the
    current device.
    """
    return type(get_current_device_resource())


cpdef is_initialized():
    """
    Check whether RMM is initialized
    """
    global _per_device_mrs
    cdef MemoryResource each_mr
    return all(
        [each_mr.c_obj.get() is not NULL
            for each_mr in _per_device_mrs.values()]
    )


cpdef _flush_logs():
    """
    Flush the logs of all currently initialized LoggingResourceAdaptor
    memory resources
    """
    global _per_device_mrs
    cdef MemoryResource each_mr
    for each_mr in _per_device_mrs.values():
        if isinstance(each_mr, LoggingResourceAdaptor):
            each_mr.flush()
