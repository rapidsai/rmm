# Copyright (c) 2020-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import warnings
# This import is needed for Cython typing in translate_python_except_to_cpp
# See https://github.com/cython/cython/issues/5589
from builtins import BaseException
from collections import defaultdict

cimport cython
from cython.operator cimport dereference as deref
from libc.stddef cimport size_t
from libc.stdint cimport int8_t, int32_t, uintptr_t
from libcpp cimport bool
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.optional cimport optional
from libcpp.pair cimport pair

from cuda.bindings cimport cyruntime
from cuda.bindings import driver, runtime

from rmm._cuda.gpu import CUDARuntimeError, getDevice, setDevice

from rmm.pylibrmm.stream cimport Stream

from rmm.pylibrmm.stream import DEFAULT_STREAM

from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.per_device_resource cimport (
    cuda_device_id,
    set_per_device_resource as cpp_set_per_device_resource,
)
from rmm.pylibrmm.helper cimport parse_bytes

from rmm.statistics import Statistics

from rmm.librmm.memory_resource cimport (
    CppExcept,
    allocate_callback_t,
    allocation_handle_type,
    arena_memory_resource,
    available_device_memory as c_available_device_memory,
    binning_memory_resource,
    callback_memory_resource,
    cuda_async_memory_resource,
    cuda_async_view_memory_resource,
    cuda_memory_resource,
    deallocate_callback_t,
    device_memory_resource,
    failure_callback_resource_adaptor,
    failure_callback_t,
    fixed_size_memory_resource,
    limiting_resource_adaptor,
    logging_resource_adaptor,
    managed_memory_resource,
    percent_of_free_device_memory as c_percent_of_free_device_memory,
    pool_memory_resource,
    prefetch_resource_adaptor,
    sam_headroom_memory_resource,
    statistics_resource_adaptor,
    system_memory_resource,
    throw_cpp_except,
    tracking_resource_adaptor,
    translate_python_except_to_cpp,
)


cdef class DeviceMemoryResource:

    cdef device_memory_resource* get_mr(self) noexcept nogil:
        """Get the underlying C++ memory resource object."""
        return self.c_obj.get()

    def allocate(self, size_t nbytes, Stream stream=DEFAULT_STREAM):
        """Allocate ``nbytes`` bytes of memory.

        Note
        ----
        On integrated memory systems, attempting to allocate more memory than
        available can cause the process to be killed by the operating system
        instead of raising a catchable ``MemoryError``.

        Raises
        ------
        MemoryError
            If allocation fails.

        Parameters
        ----------
        nbytes : size_t
            The size of the allocation in bytes
        stream : Stream
            Optional stream for the allocation
        """
        cdef uintptr_t ptr
        with nogil:
            ptr = <uintptr_t>self.c_obj.get().allocate(nbytes, stream.view())
        return ptr

    def deallocate(self, uintptr_t ptr, size_t nbytes, Stream stream=DEFAULT_STREAM):
        """Deallocate memory pointed to by ``ptr`` of size ``nbytes``.

        Parameters
        ----------
        ptr : uintptr_t
            Pointer to be deallocated
        nbytes : size_t
            Size of the allocation in bytes
        stream : Stream
            Optional stream for the deallocation
        """
        with nogil:
            self.c_obj.get().deallocate(<void*>(ptr), nbytes, stream.view())

    def __dealloc__(self):
        # See the __dealloc__ method on DeviceBuffer for discussion of why we must
        # explicitly call reset here instead of relying on the unique_ptr's
        # destructor.
        with nogil:
            self.c_obj.reset()


# See the note about `no_gc_clear` in `device_buffer.pyx`.
@cython.no_gc_clear
cdef class UpstreamResourceAdaptor(DeviceMemoryResource):
    """Parent class for all memory resources that track an upstream.

    Upstream resource tracking requires maintaining a reference to the upstream
    mr so that it is kept alive and may be accessed by any downstream resource
    adaptors.
    """

    def __cinit__(self, DeviceMemoryResource upstream_mr, *args, **kwargs):

        if (upstream_mr is None):
            raise Exception("Argument `upstream_mr` must not be None")

        self.upstream_mr = upstream_mr

    cpdef DeviceMemoryResource get_upstream(self):
        return self.upstream_mr

    def __dealloc__(self):
        # Need to override the parent method with an identical implementation
        # to ensure that self.upstream_mr is still alive when the C++ mr's
        # destructor is invoked since it will reference self.upstream_mr.c_obj.
        with nogil:
            self.c_obj.reset()


cdef class CudaMemoryResource(DeviceMemoryResource):
    def __cinit__(self):
        self.c_obj.reset(
            new cuda_memory_resource()
        )

    def __init__(self):
        """
        Memory resource that uses ``cudaMalloc``/``cudaFree`` for
        allocation/deallocation.
        """
        pass


cdef class CudaAsyncMemoryResource(DeviceMemoryResource):
    """
    Memory resource that uses ``cudaMallocAsync``/``cudaFreeAsync`` for
    allocation/deallocation.

    Parameters
    ----------
    initial_pool_size : int | str, optional
        Initial pool size in bytes. By default, half the available memory
        on the device is used. A string argument is parsed using `parse_bytes`.
    release_threshold: int, optional
        Release threshold in bytes. If the pool size grows beyond this
        value, unused memory held by the pool will be released at the
        next synchronization point.
    enable_ipc: bool, optional
        If True, enables export of POSIX file descriptor handles for the memory
        allocated by this resource so that it can be used with CUDA IPC.
    enable_fabric: bool, optional
        If True, enables export of fabric handles for the memory allocated by
        this resource.
    """
    def __cinit__(
        self,
        initial_pool_size=None,
        release_threshold=None,
        enable_ipc=False,
        enable_fabric=False
    ):
        cdef optional[size_t] c_initial_pool_size = (
            optional[size_t]()
            if initial_pool_size is None
            else optional[size_t](<size_t> parse_bytes(initial_pool_size))
        )

        cdef optional[size_t] c_release_threshold = (
            optional[size_t]()
            if release_threshold is None
            else optional[size_t](<size_t> release_threshold)
        )

        # If IPC or fabric memory handles are enabled but not supported, the
        # constructor below will raise an error from C++.
        cdef allocation_handle_type descriptor = allocation_handle_type.none
        if enable_ipc:
            descriptor = <allocation_handle_type>(
                <int32_t?>descriptor |
                <int32_t?>allocation_handle_type.posix_file_descriptor
            )
        if enable_fabric:
            descriptor = <allocation_handle_type>(
                <int32_t?>descriptor |
                <int32_t?>allocation_handle_type.fabric
            )

        cdef optional[allocation_handle_type] c_export_handle_type = (
            optional[allocation_handle_type](descriptor)
            if (enable_ipc or enable_fabric)
            else optional[allocation_handle_type]()
        )

        self.c_obj.reset(
            new cuda_async_memory_resource(
                c_initial_pool_size,
                c_release_threshold,
                c_export_handle_type
            )
        )


cdef class CudaAsyncViewMemoryResource(DeviceMemoryResource):
    """
    Memory resource that uses ``cudaMallocAsync``/``cudaFreeAsync`` for
    allocation/deallocation with an existing CUDA memory pool.

    This resource uses an existing CUDA memory pool handle (such as the default pool)
    instead of creating a new one. This is useful for integrating with existing GPU
    applications that already use a CUDA memory pool, or customizing the flags
    used by the memory pool.

    The memory pool passed in must not be destroyed during the lifetime of this
    memory resource.

    Parameters
    ----------
    pool_handle : cudaMemPool_t or CUmemoryPool
        Handle to a CUDA memory pool which will be used to serve allocation
        requests.
    """
    def __cinit__(
        self,
        pool_handle
    ):
        # Convert the pool_handle to a cyruntime.cudaMemPool_t
        if not isinstance(pool_handle, (runtime.cudaMemPool_t, driver.CUmemoryPool)):
            raise ValueError("pool_handle must be a cudaMemPool_t or CUmemoryPool")

        cdef cyruntime.cudaMemPool_t c_pool_handle
        c_pool_handle = <cyruntime.cudaMemPool_t><uintptr_t>int(pool_handle)

        self.c_obj.reset(
            new cuda_async_view_memory_resource(c_pool_handle)
        )

    def pool_handle(self):
        cdef cuda_async_view_memory_resource* c_mr = \
            <cuda_async_view_memory_resource*>self.c_obj.get()
        return <uintptr_t>c_mr.pool_handle()


cdef class ManagedMemoryResource(DeviceMemoryResource):
    def __cinit__(self):
        self.c_obj.reset(
            new managed_memory_resource()
        )

    def __init__(self):
        """
        Memory resource that uses ``cudaMallocManaged``/``cudaFree`` for
        allocation/deallocation.
        """
        pass


cdef class SystemMemoryResource(DeviceMemoryResource):
    def __cinit__(self):
        self.c_obj.reset(
            new system_memory_resource()
        )

    def __init__(self):
        """
        Memory resource that uses ``malloc``/``free`` for
        allocation/deallocation.
        """
        pass


cdef class SamHeadroomMemoryResource(DeviceMemoryResource):
    def __cinit__(
        self,
        size_t headroom
    ):
        self.c_obj.reset(new sam_headroom_memory_resource(headroom))

    def __init__(
        self,
        size_t headroom
    ):
        """
        Memory resource that uses ``malloc``/``free`` for
        allocation/deallocation.

        Parameters
        ----------
        headroom : size_t
            Size of the reserved GPU memory as headroom
        """
        pass


cdef class PoolMemoryResource(UpstreamResourceAdaptor):

    def __cinit__(
            self,
            DeviceMemoryResource upstream_mr,
            initial_pool_size=None,
            maximum_pool_size=None
    ):
        cdef size_t c_initial_pool_size
        cdef optional[size_t] c_maximum_pool_size
        c_initial_pool_size = (
            c_percent_of_free_device_memory(50) if
            initial_pool_size is None
            else parse_bytes(initial_pool_size)
        )
        c_maximum_pool_size = (
            optional[size_t]() if
            maximum_pool_size is None
            else optional[size_t](<size_t> parse_bytes(maximum_pool_size))
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
        initial_pool_size : int | str, optional
            Initial pool size in bytes. By default, half the available memory
            on the device is used.
        maximum_pool_size : int | str, optional
            Maximum size in bytes, that the pool can grow to.
        """
        pass

    def pool_size(self):
        cdef pool_memory_resource[device_memory_resource]* c_mr = (
            <pool_memory_resource[device_memory_resource]*>(self.get_mr())
        )
        return c_mr.pool_size()

cdef class ArenaMemoryResource(UpstreamResourceAdaptor):
    def __cinit__(
        self, DeviceMemoryResource upstream_mr,
        arena_size=None,
        dump_log_on_failure=False
    ):
        cdef optional[size_t] c_arena_size = (
            optional[size_t]() if
            arena_size is None
            else optional[size_t](<size_t> parse_bytes(arena_size))
        )
        self.c_obj.reset(
            new arena_memory_resource[device_memory_resource](
                upstream_mr.get_mr(),
                c_arena_size,
                dump_log_on_failure,
            )
        )

    def __init__(
        self,
        DeviceMemoryResource upstream_mr,
        object arena_size=None,
        bool dump_log_on_failure=False
    ):
        """
        A suballocator that emphasizes fragmentation avoidance and scalable concurrency
        support.

        Parameters
        ----------
        upstream_mr : DeviceMemoryResource
            The DeviceMemoryResource from which to allocate memory for arenas.
        arena_size : int, optional
            Size in bytes of the global arena. Defaults to half of the available memory
            on the current device.
        dump_log_on_failure : bool, optional
            Whether to dump the arena on allocation failure.
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

        self._bin_mrs = []

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
        ``[2**min_size_exponent, 2**max_size_exponent]``.

        Call :py:meth:`~.add_bin` to add additional bin allocators.

        Parameters
        ----------
        upstream_mr : DeviceMemoryResource
            The memory resource to use for allocations larger than any of the
            bins.
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
        if bin_resource is None:
            (<binning_memory_resource[device_memory_resource]*>(
                self.c_obj.get()))[0].add_bin(allocation_size)
        else:
            # Save the ref to the new bin resource to ensure its lifetime
            self._bin_mrs.append(bin_resource)

            (<binning_memory_resource[device_memory_resource]*>(
                self.c_obj.get()))[0].add_bin(
                    allocation_size,
                    bin_resource.get_mr())

    @property
    def bin_mrs(self) -> list:
        """Get the list of binned memory resources."""
        return self._bin_mrs


cdef void* _allocate_callback_wrapper(
    size_t nbytes,
    cuda_stream_view stream,
    void* ctx
    # Note that this function is specifically designed to rethrow Python
    # exceptions as C++ exceptions when called as a callback from C++, so it is
    # noexcept from Cython's perspective.
) noexcept nogil:
    cdef CppExcept err
    with gil:
        try:
            return <void*><uintptr_t>((<object>ctx)(
                nbytes,
                Stream._from_cudaStream_t(stream.value())
            ))
        except BaseException as e:
            err = translate_python_except_to_cpp(e)
    throw_cpp_except(err)

cdef void _deallocate_callback_wrapper(
    void* ptr,
    size_t nbytes,
    cuda_stream_view stream,
    void* ctx
) except * with gil:
    (<object>ctx)(<uintptr_t>(ptr), nbytes, Stream._from_cudaStream_t(stream.value()))


cdef class CallbackMemoryResource(DeviceMemoryResource):
    """
    A memory resource that uses the user-provided callables to do
    memory allocation and deallocation.

    ``CallbackMemoryResource`` should really only be used for
    debugging memory issues, as there is a significant performance
    penalty associated with using a Python function for each memory
    allocation and deallocation.

    Parameters
    ----------
    allocate_func: callable
        The allocation function must accept two arguments. An integer
        representing the number of bytes to allocate and a Stream on
        which to perform the allocation, and return an integer
        representing the pointer to the allocated memory.
    deallocate_func: callable
        The deallocation function must accept three arguments. an integer
        representing the pointer to the memory to free, a second
        integer representing the number of bytes to free, and a Stream
        on which to perform the deallocation.

    Examples
    --------
    >>> import rmm
    >>> base_mr = rmm.mr.CudaMemoryResource()
    >>> def allocate_func(size, stream):
    ...     print(f"Allocating {size} bytes")
    ...     return base_mr.allocate(size, stream)
    ...
    >>> def deallocate_func(ptr, size, stream):
    ...     print(f"Deallocating {size} bytes")
    ...     return base_mr.deallocate(ptr, size, stream)
    ...
    >>> rmm.mr.set_current_device_resource(
        rmm.mr.CallbackMemoryResource(allocate_func, deallocate_func)
    )
    >>> dbuf = rmm.DeviceBuffer(size=256)
    Allocating 256 bytes
    >>> del dbuf
    Deallocating 256 bytes
    """
    def __init__(
        self,
        allocate_func,
        deallocate_func,
    ):
        self._allocate_func = allocate_func
        self._deallocate_func = deallocate_func
        self.c_obj.reset(
            new callback_memory_resource(
                <allocate_callback_t>(_allocate_callback_wrapper),
                <deallocate_callback_t>(_deallocate_callback_wrapper),
                <void*>(allocate_func),
                <void*>(deallocate_func)
            )
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


cdef class LimitingResourceAdaptor(UpstreamResourceAdaptor):

    def __cinit__(
        self,
        DeviceMemoryResource upstream_mr,
        size_t allocation_limit
    ):
        self.c_obj.reset(
            new limiting_resource_adaptor[device_memory_resource](
                upstream_mr.get_mr(),
                allocation_limit
            )
        )

    def __init__(
        self,
        DeviceMemoryResource upstream_mr,
        size_t allocation_limit
    ):
        """
        Memory resource that limits the total allocation amount possible
        performed by an upstream memory resource.

        Parameters
        ----------
        upstream_mr : DeviceMemoryResource
            The upstream memory resource.
        allocation_limit : size_t
            Maximum memory allowed for this allocator.
        """
        pass

    def get_allocated_bytes(self) -> size_t:
        """
        Query the number of bytes that have been allocated. Note that this can
        not be used to know how large of an allocation is possible due to both
        possible fragmentation and also internal page sizes and alignment that
        is not tracked by this allocator.
        """
        return (<limiting_resource_adaptor[device_memory_resource]*>(
            self.c_obj.get())
        )[0].get_allocated_bytes()

    def get_allocation_limit(self) -> size_t:
        """
        Query the maximum number of bytes that this allocator is allowed to
        allocate. This is the limit on the allocator and not a representation
        of the underlying device. The device may not be able to support this
        limit.
        """
        return (<limiting_resource_adaptor[device_memory_resource]*>(
            self.c_obj.get())
        )[0].get_allocation_limit()


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

        A stack of counters is maintained. Use :meth:`push_counters` and
        :meth:`pop_counters` to track statistics at different nesting levels.

        Parameters
        ----------
        upstream : DeviceMemoryResource
            The upstream memory resource.
        """
        pass

    @property
    def allocation_counts(self) -> Statistics:
        """
        Gets the current, peak, and total allocated bytes and number of
        allocations.

        The dictionary keys are ``current_bytes``, ``current_count``,
        ``peak_bytes``, ``peak_count``, ``total_bytes``, and ``total_count``.

        Returns:
            dict: Dictionary containing allocation counts and bytes.
        """
        cdef statistics_resource_adaptor[device_memory_resource]* mr = \
            <statistics_resource_adaptor[device_memory_resource]*> self.c_obj.get()

        counts = deref(mr).get_allocations_counter()
        byte_counts = deref(mr).get_bytes_counter()
        return Statistics(
            current_bytes=byte_counts.value,
            current_count=counts.value,
            peak_bytes=byte_counts.peak,
            peak_count=counts.peak,
            total_bytes=byte_counts.total,
            total_count=counts.total,
        )

    def pop_counters(self) -> Statistics:
        """
        Pop a counter pair (bytes and allocations) from the stack

        Returns
        -------
        The popped statistics
        """
        cdef statistics_resource_adaptor[device_memory_resource]* mr = \
            <statistics_resource_adaptor[device_memory_resource]*> self.c_obj.get()

        bytes_and_allocs = deref(mr).pop_counters()
        return Statistics(
            current_bytes=bytes_and_allocs.first.value,
            current_count=bytes_and_allocs.second.value,
            peak_bytes=bytes_and_allocs.first.peak,
            peak_count=bytes_and_allocs.second.peak,
            total_bytes=bytes_and_allocs.first.total,
            total_count=bytes_and_allocs.second.total,
        )

    def push_counters(self) -> Statistics:
        """
        Push a new counter pair (bytes and allocations) on the stack

        Returns
        -------
        The statistics _before_ the push
        """

        cdef statistics_resource_adaptor[device_memory_resource]* mr = \
            <statistics_resource_adaptor[device_memory_resource]*> self.c_obj.get()

        bytes_and_allocs = deref(mr).push_counters()
        return Statistics(
            current_bytes=bytes_and_allocs.first.value,
            current_count=bytes_and_allocs.second.value,
            peak_bytes=bytes_and_allocs.first.peak,
            peak_count=bytes_and_allocs.second.peak,
            total_bytes=bytes_and_allocs.first.total,
            total_count=bytes_and_allocs.second.total,
        )

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


# Note that this function is specifically designed to rethrow Python exceptions
# as C++ exceptions when called as a callback from C++, so it is noexcept from
# Cython's perspective.
cdef bool _oom_callback_function(size_t bytes, void *callback_arg) noexcept nogil:
    cdef CppExcept err
    with gil:
        try:
            return (<object>callback_arg)(bytes)
        except BaseException as e:
            err = translate_python_except_to_cpp(e)
    throw_cpp_except(err)


cdef class FailureCallbackResourceAdaptor(UpstreamResourceAdaptor):

    def __cinit__(
        self,
        DeviceMemoryResource upstream_mr,
        object callback,
    ):
        self._callback = callback
        self.c_obj.reset(
            new failure_callback_resource_adaptor[device_memory_resource](
                upstream_mr.get_mr(),
                <failure_callback_t>_oom_callback_function,
                <void*>callback
            )
        )

    def __init__(
        self,
        DeviceMemoryResource upstream_mr,
        object callback,
    ):
        """
        Memory resource that call callback when memory allocation fails.

        Parameters
        ----------
        upstream : DeviceMemoryResource
            The upstream memory resource.
        callback : callable
            Function called when memory allocation fails.
        """
        pass

cdef class PrefetchResourceAdaptor(UpstreamResourceAdaptor):

    def __cinit__(
        self,
        DeviceMemoryResource upstream_mr
    ):
        self.c_obj.reset(
            new prefetch_resource_adaptor[device_memory_resource](
                upstream_mr.get_mr()
            )
        )

    def __init__(
        self,
        DeviceMemoryResource upstream_mr
    ):
        """
        Memory resource that prefetches all allocations.

        Parameters
        ----------
        upstream : DeviceMemoryResource
            The upstream memory resource.
        """
        pass


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
            initial_pool_size=None if initial_pool_size is None
            else parse_bytes(initial_pool_size),
            maximum_pool_size=None if maximum_pool_size is None
            else parse_bytes(maximum_pool_size)
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
        if e.status == runtime.cudaError_t.cudaErrorNoDevice:
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

    cpp_set_per_device_resource(deref(device_id), mr.get_mr())


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


def available_device_memory():
    """
    Returns a tuple of free and total device memory memory.
    """
    cdef pair[size_t, size_t] res
    res = c_available_device_memory()
    return (res.first, res.second)
