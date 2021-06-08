# Copyright (c) 2020, NVIDIA CORPORATION.
import os
import warnings
from collections import defaultdict

from cython.operator cimport dereference as deref
from libc.stdint cimport int8_t, int64_t
from libcpp cimport bool
from libcpp.cast cimport dynamic_cast
from libcpp.memory cimport make_shared, make_unique, shared_ptr, unique_ptr
from libcpp.string cimport string

from rmm._cuda.gpu import CUDARuntimeError, cudaError, getDevice, setDevice


# NOTE: Keep extern declarations in .pyx file as much as possible to avoid
# leaking dependencies when importing RMM Cython .pxd files
cdef extern from "thrust/optional.h" namespace "thrust" nogil:

    struct nullopt_t:
        pass

    cdef nullopt_t nullopt

    cdef cppclass optional[T]:
        optional()
        optional(T v)

    cdef optional[T] make_optional[T](T v)

cdef extern from "rmm/mr/device/cuda_memory_resource.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass cuda_memory_resource(device_memory_resource):
        cuda_memory_resource() except +

cdef extern from "rmm/mr/device/managed_memory_resource.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass managed_memory_resource(device_memory_resource):
        managed_memory_resource() except +

cdef extern from "rmm/mr/device/cuda_async_memory_resource.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass cuda_async_memory_resource(device_memory_resource):
        cuda_async_memory_resource(optional[size_t] initial_pool_size,
                                   optional[size_t] release_threshold) except +

cdef extern from "rmm/mr/device/pool_memory_resource.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass pool_memory_resource[Upstream](device_memory_resource):
        pool_memory_resource(
            Upstream* upstream_mr,
            optional[size_t] initial_pool_size,
            optional[size_t] maximum_pool_size) except +

cdef extern from "rmm/mr/device/fixed_size_memory_resource.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass fixed_size_memory_resource[Upstream](device_memory_resource):
        fixed_size_memory_resource(
            Upstream* upstream_mr,
            size_t block_size,
            size_t block_to_preallocate) except +

cdef extern from "rmm/mr/device/binning_memory_resource.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass binning_memory_resource[Upstream](device_memory_resource):
        binning_memory_resource(Upstream* upstream_mr) except +
        binning_memory_resource(
            Upstream* upstream_mr,
            int8_t min_size_exponent,
            int8_t max_size_exponent) except +

        void add_bin(size_t allocation_size) except +
        void add_bin(
            size_t allocation_size,
            device_memory_resource* bin_resource) except +

cdef extern from "rmm/mr/device/logging_resource_adaptor.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass logging_resource_adaptor[Upstream](device_memory_resource):
        logging_resource_adaptor(
            Upstream* upstream_mr,
            string filename) except +

        void flush() except +

cdef extern from "rmm/mr/device/statistics_resource_adaptor.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass statistics_resource_adaptor[Upstream](
            device_memory_resource):
        struct counter:
            counter()

            int64_t value
            int64_t peak
            int64_t total

        statistics_resource_adaptor(
            Upstream* upstream_mr) except +

        counter get_bytes_counter() except +
        counter get_allocations_counter() except +

cdef extern from "rmm/mr/device/tracking_resource_adaptor.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass tracking_resource_adaptor[Upstream](device_memory_resource):
        tracking_resource_adaptor(
            Upstream* upstream_mr,
            bool capture_stacks) except +

        size_t get_allocated_bytes() except +
        string get_outstanding_allocations_str() except +
        void log_outstanding_allocations() except +

cdef extern from "rmm/mr/device/per_device_resource.hpp" namespace "rmm" nogil:

    cdef cppclass cuda_device_id:
        ctypedef int value_type

        cuda_device_id(value_type id)

        value_type value()

    cdef device_memory_resource* _set_current_device_resource \
        "rmm::mr::set_current_device_resource" (device_memory_resource* new_mr)
    cdef device_memory_resource* _get_current_device_resource \
        "rmm::mr::get_current_device_resource" ()

    cdef device_memory_resource* _set_per_device_resource \
        "rmm::mr::set_per_device_resource" (
            cuda_device_id id,
            device_memory_resource* new_mr
        )
    cdef device_memory_resource* _get_per_device_resource \
        "rmm::mr::get_per_device_resource"(cuda_device_id id)


cdef class DeviceMemoryResource:

    cdef device_memory_resource* get_mr(self):
        return self.c_obj.get()


cdef class UpstreamResourceAdaptor(DeviceMemoryResource):

    def __cinit__(self, DeviceMemoryResource upstream_mr, *args, **kwargs):

        if (upstream_mr is None):
            raise Exception("Argument `upstream_mr` must not be None")

        self.upstream_mr = upstream_mr

    def __dealloc__(self):
        # Must cleanup the base MR before any upstream MR
        self.c_obj.reset()

    cpdef DeviceMemoryResource get_upstream(self):
        return self.upstream_mr


cdef class CudaMemoryResource(DeviceMemoryResource):
    def __cinit__(self):
        self.c_obj.reset(
            new cuda_memory_resource()
        )

    def __init__(self):
        """
        Memory resource that uses cudaMalloc/Free for allocation/deallocation
        """
        pass


cdef class CudaAsyncMemoryResource(DeviceMemoryResource):
    """
    Memory resource that uses cudaMallocAsync/Free for
    allocation/deallocation.

    Parameters
    ----------
    initial_pool_size : int,optional
        Initial pool size in bytes. By default, half the available memory
        on the device is used.
    release_threshold: int, optional
        Release threshold in bytes. If the pool size grows beyond this
        value, unused memory held by the pool will be released at the
        next synchronization point.
    """
    def __cinit__(self, initial_pool_size=None, release_threshold=None):
        cdef optional[size_t] c_initial_pool_size = (
            optional[size_t]()
            if initial_pool_size is None
            else optional[size_t](initial_pool_size)
        )

        cdef optional[size_t] c_release_threshold = (
            optional[size_t]()
            if release_threshold is None
            else optional[size_t](release_threshold)
        )

        self.c_obj.reset(
            new cuda_async_memory_resource(
                c_initial_pool_size,
                c_release_threshold
            )
        )


cdef class ManagedMemoryResource(DeviceMemoryResource):
    def __cinit__(self):
        self.c_obj.reset(
            new managed_memory_resource()
        )

    def __init__(self):
        """
        Memory resource that uses cudaMallocManaged/Free for
        allocation/deallocation.
        """
        pass


cdef class PoolMemoryResource(UpstreamResourceAdaptor):

    def __cinit__(
            self,
            DeviceMemoryResource upstream_mr,
            initial_pool_size=None,
            maximum_pool_size=None
    ):
        cdef optional[size_t] c_initial_pool_size
        cdef optional[size_t] c_maximum_pool_size
        c_initial_pool_size = (
            optional[size_t]() if
            initial_pool_size is None
            else make_optional[size_t](initial_pool_size)
        )
        c_maximum_pool_size = (
            optional[size_t]() if
            maximum_pool_size is None
            else make_optional[size_t](maximum_pool_size)
        )
        self.c_obj.reset(
            new pool_memory_resource[device_memory_resource](
                upstream_mr.get_mr(),
                c_initial_pool_size,
                c_maximum_pool_size
            )
        )

    def __init__(
            self,
            DeviceMemoryResource upstream_mr,
            object initial_pool_size=None,
            object maximum_pool_size=None
    ):
        """
        Coalescing best-fit suballocator which uses a pool of memory allocated
        from an upstream memory resource.

        Parameters
        ----------
        upstream_mr : DeviceMemoryResource
            The DeviceMemoryResource from which to allocate blocks for the
            pool.
        initial_pool_size : int,optional
            Initial pool size in bytes. By default, half the available memory
            on the device is used.
        maximum_pool_size : int, optional
            Maximum size in bytes, that the pool can grow to.
        """
        pass


cdef class FixedSizeMemoryResource(UpstreamResourceAdaptor):
    def __cinit__(
            self,
            DeviceMemoryResource upstream_mr,
            size_t block_size=1<<20,
            size_t blocks_to_preallocate=128
    ):
        self.c_obj.reset(
            new fixed_size_memory_resource[device_memory_resource](
                upstream_mr.get_mr(),
                block_size,
                blocks_to_preallocate
            )
        )

    def __init__(
            self,
            DeviceMemoryResource upstream_mr,
            size_t block_size=1<<20,
            size_t blocks_to_preallocate=128
    ):
        """
        Memory resource which allocates memory blocks of a single fixed size.

        Parameters
        ----------
        upstream_mr : DeviceMemoryResource
            The DeviceMemoryResource from which to allocate blocks for the
            pool.
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


cdef class BinningMemoryResource(UpstreamResourceAdaptor):
    def __cinit__(
        self,
        DeviceMemoryResource upstream_mr,
        int8_t min_size_exponent=-1,
        int8_t max_size_exponent=-1,
    ):

        self.bin_mrs = []

        if (min_size_exponent == -1 or max_size_exponent == -1):
            self.c_obj.reset(
                new binning_memory_resource[device_memory_resource](
                    upstream_mr.get_mr()
                )
            )
        else:
            self.c_obj.reset(
                new binning_memory_resource[device_memory_resource](
                    upstream_mr.get_mr(),
                    min_size_exponent,
                    max_size_exponent
                )
            )

    def __dealloc__(self):

        # Must cleanup the base MR before any upstream or referenced Bins
        self.c_obj.reset()

    def __init__(
        self,
        DeviceMemoryResource upstream_mr,
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
        upstream_mr : DeviceMemoryResource
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
        DeviceMemoryResource bin_resource=None
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
        bin_resource : DeviceMemoryResource
            The resource to use for this bin (optional)
        """
        cdef DeviceMemoryResource _bin_resource

        if bin_resource is None:
            (<binning_memory_resource[device_memory_resource]*>(
                self.c_obj.get()))[0].add_bin(allocation_size)
        else:
            # Save the ref to the new bin resource to ensure its lifetime
            self.bin_mrs.append(bin_resource)

            (<binning_memory_resource[device_memory_resource]*>(
                self.c_obj.get()))[0].add_bin(
                    allocation_size,
                    bin_resource.get_mr())


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


cdef class LoggingResourceAdaptor(UpstreamResourceAdaptor):
    def __cinit__(
        self,
        DeviceMemoryResource upstream_mr,
        object log_file_name=None
    ):
        if log_file_name is None:
            log_file_name = os.getenv("RMM_LOG_FILE")
            if not log_file_name:
                raise ValueError(
                    "RMM log file must be specified either using "
                    "log_file_name= argument or RMM_LOG_FILE "
                    "environment variable"
                )

        # Append the device ID before the file extension
        log_file_name = _append_id(
            log_file_name, getDevice()
        )
        log_file_name = os.path.abspath(log_file_name)
        self._log_file_name = log_file_name

        self.c_obj.reset(
            new logging_resource_adaptor[device_memory_resource](
                upstream_mr.get_mr(),
                log_file_name.encode()
            )
        )

    def __init__(
        self,
        DeviceMemoryResource upstream_mr,
        object log_file_name=None
    ):
        """
        Memory resource that logs information about allocations/deallocations
        performed by an upstream memory resource.

        Parameters
        ----------
        upstream : DeviceMemoryResource
            The upstream memory resource.
        log_file_name : str
            Path to the file to which logs are written.
        """
        pass

    cpdef flush(self):
        (<logging_resource_adaptor[device_memory_resource]*>(
            self.get_mr()))[0].flush()

    cpdef get_file_name(self):
        return self._log_file_name

    def __dealloc__(self):
        self.c_obj.reset()

cdef class StatisticsResourceAdaptor(UpstreamResourceAdaptor):

    def __cinit__(
        self,
        DeviceMemoryResource upstream_mr
    ):
        self.c_obj.reset(
            new statistics_resource_adaptor[device_memory_resource](
                upstream_mr.get_mr()
            )
        )

    def __init__(
        self,
        DeviceMemoryResource upstream_mr
    ):
        """
        Memory resource that tracks the current, peak and total
        allocations/deallocations performed by an upstream memory resource.
        Includes the ability to query these statistics at any time.

        Parameters
        ----------
        upstream : DeviceMemoryResource
            The upstream memory resource.
        """
        pass

    @property
    def allocation_counts(self) -> dict:
        """
        Gets the current, peak, and total allocated bytes and number of
        allocations.

        Returns:
            dict: Dictionary containing allocation counts and bytes.
        """

        counts = (<statistics_resource_adaptor[device_memory_resource]*>(
            self.c_obj.get()))[0].get_allocations_counter()
        byte_counts = (<statistics_resource_adaptor[device_memory_resource]*>(
            self.c_obj.get()))[0].get_bytes_counter()

        return {
            "current_bytes": byte_counts.value,
            "current_count": counts.value,
            "peak_bytes": byte_counts.peak,
            "peak_count": counts.peak,
            "total_bytes": byte_counts.total,
            "total_count": counts.total,
        }

cdef class TrackingResourceAdaptor(UpstreamResourceAdaptor):

    def __cinit__(
        self,
        DeviceMemoryResource upstream_mr,
        bool capture_stacks=False
    ):
        self.c_obj.reset(
            new tracking_resource_adaptor[device_memory_resource](
                upstream_mr.get_mr(),
                capture_stacks
            )
        )

    def __init__(
        self,
        DeviceMemoryResource upstream_mr,
        bool capture_stacks=False
    ):
        """
        Memory resource that logs tracks allocations/deallocations
        performed by an upstream memory resource. Includes the ability to
        query all outstanding allocations with the stack trace, if desired.

        Parameters
        ----------
        upstream : DeviceMemoryResource
            The upstream memory resource.
        capture_stacks : bool
            Whether or not to capture the stack trace with each allocation.
        """
        pass

    def get_allocated_bytes(self) -> size_t:
        """
        Query the number of bytes that have been allocated. Note that this can
        not be used to know how large of an allocation is possible due to both
        possible fragmentation and also internal page sizes and alignment that
        is not tracked by this allocator.
        """
        return (<tracking_resource_adaptor[device_memory_resource]*>(
            self.c_obj.get())
        )[0].get_allocated_bytes()

    def get_outstanding_allocations_str(self) -> str:
        """
        Returns a string containing information about the current outstanding
        allocations. For each allocation, the address, size and optional
        stack trace are shown.
        """

        return (<tracking_resource_adaptor[device_memory_resource]*>(
            self.c_obj.get())
        )[0].get_outstanding_allocations_str().decode('UTF-8')

    def log_outstanding_allocations(self):
        """
        Logs the output of `get_outstanding_allocations_str` to the current
        RMM log file if enabled.
        """

        (<tracking_resource_adaptor[device_memory_resource]*>(
            self.c_obj.get()))[0].log_outstanding_allocations()


# Global per-device memory resources; dict of int:DeviceMemoryResource
cdef _per_device_mrs = defaultdict(CudaMemoryResource)


cpdef void _initialize(
    bool pool_allocator=False,
    bool managed_memory=False,
    object initial_pool_size=None,
    object maximum_pool_size=None,
    object devices=0,
    bool logging=False,
    object log_file_name=None,
) except *:
    """
    Initializes RMM library using the options passed
    """
    if managed_memory:
        upstream = ManagedMemoryResource
    else:
        upstream = CudaMemoryResource

    if pool_allocator:
        typ = PoolMemoryResource
        args = (upstream(),)
        kwargs = dict(
            initial_pool_size=initial_pool_size,
            maximum_pool_size=maximum_pool_size
        )
    else:
        typ = upstream
        args = ()
        kwargs = {}

    cdef DeviceMemoryResource mr
    cdef int original_device

    # Save the current device so we can reset it
    try:
        original_device = getDevice()
    except CUDARuntimeError as e:
        if e.status == cudaError.cudaErrorNoDevice:
            warnings.warn(e.msg)
        else:
            raise e
    else:
        # reset any previously specified per device resources
        global _per_device_mrs
        _per_device_mrs.clear()

        if devices is None:
            devices = [0]
        elif isinstance(devices, int):
            devices = [devices]

        # create a memory resource per specified device
        for device in devices:
            setDevice(device)

            if logging:
                mr = LoggingResourceAdaptor(
                    typ(*args, **kwargs),
                    log_file_name
                )
            else:
                mr = typ(*args, **kwargs)

            set_per_device_resource(device, mr)

        # reset CUDA device to original
        setDevice(original_device)


cpdef get_per_device_resource(int device):
    """
    Get the default memory resource for the specified device.

    If the returned memory resource is used when a different device is the
    active CUDA device, behavior is undefined.

    Parameters
    ----------
    device : int
        The ID of the device for which to get the memory resource.
    """
    global _per_device_mrs
    return _per_device_mrs[device]


cpdef set_per_device_resource(int device, DeviceMemoryResource mr):
    """
    Set the default memory resource for the specified device.

    Parameters
    ----------
    device : int
        The ID of the device for which to get the memory resource.
    mr : DeviceMemoryResource
        The memory resource to set.  Must have been created while device was
        the active CUDA device.
    """
    global _per_device_mrs
    _per_device_mrs[device] = mr

    # Since cuda_device_id does not have a default constructor, it must be heap
    # allocated
    cdef unique_ptr[cuda_device_id] device_id = \
        make_unique[cuda_device_id](device)

    _set_per_device_resource(deref(device_id), mr.get_mr())


cpdef set_current_device_resource(DeviceMemoryResource mr):
    """
    Set the default memory resource for the current device.

    Parameters
    ----------
    mr : DeviceMemoryResource
        The memory resource to set. Must have been created while the current
        device is the active CUDA device.
    """
    set_per_device_resource(getDevice(), mr)


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


cpdef DeviceMemoryResource get_current_device_resource():
    """
    Get the memory resource used for RMM device allocations on the current
    device.

    If the returned memory resource is used when a different device is the
    active CUDA device, behavior is undefined.
    """
    return get_per_device_resource(getDevice())


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
    cdef DeviceMemoryResource each_mr
    return all(
        [each_mr.get_mr() is not NULL
            for each_mr in _per_device_mrs.values()]
    )


cpdef _flush_logs():
    """
    Flush the logs of all currently initialized LoggingResourceAdaptor
    memory resources
    """
    global _per_device_mrs
    cdef DeviceMemoryResource each_mr
    for each_mr in _per_device_mrs.values():
        if isinstance(each_mr, LoggingResourceAdaptor):
            each_mr.flush()


def enable_logging(log_file_name=None):
    """
    Enable logging of run-time events for all devices.

    Parameters
    ----------
    log_file_name:  str, optional
        Name of the log file. If not specified, the environment variable
        RMM_LOG_FILE is used. A ValueError is thrown if neither is available.
        A separate log file is produced for each device,
        and the suffix `".dev{id}"` is automatically added to the log file
        name.

    Notes
    -----
    Note that if you use the environment variable CUDA_VISIBLE_DEVICES
    with logging enabled, the suffix may not be what you expect. For
    example, if you set CUDA_VISIBLE_DEVICES=1, the log file produced
    will still have suffix `0`. Similarly, if you set
    CUDA_VISIBLE_DEVICES=1,0 and use devices 0 and 1, the log file
    with suffix `0` will correspond to the GPU with device ID `1`.
    Use `rmm.get_log_filenames()` to get the log file names
    corresponding to each device.
    """
    global _per_device_mrs

    devices = [0] if not _per_device_mrs.keys() else _per_device_mrs.keys()

    for device in devices:
        each_mr = <DeviceMemoryResource>_per_device_mrs[device]
        if not isinstance(each_mr, LoggingResourceAdaptor):
            set_per_device_resource(
                device,
                LoggingResourceAdaptor(each_mr, log_file_name)
            )


def disable_logging():
    """
    Disable logging if it was enabled previously using
    `rmm.initialize()` or `rmm.enable_logging()`.
    """
    global _per_device_mrs
    for i, each_mr in _per_device_mrs.items():
        if isinstance(each_mr, LoggingResourceAdaptor):
            set_per_device_resource(i, each_mr.get_upstream())


def get_log_filenames():
    """
    Returns the log filename (or `None` if not writing logs)
    for each device in use.

    Examples
    --------
    >>> import rmm
    >>> rmm.reinitialize(devices=[0, 1], logging=True, log_file_name="rmm.log")
    >>> rmm.get_log_filenames()
    {0: '/home/user/workspace/rapids/rmm/python/rmm.dev0.log',
     1: '/home/user/workspace/rapids/rmm/python/rmm.dev1.log'}
    """
    global _per_device_mrs

    return {
        i: each_mr.get_file_name()
        if isinstance(each_mr, LoggingResourceAdaptor)
        else None
        for i, each_mr in _per_device_mrs.items()
    }
