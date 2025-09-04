# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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
import numpy as np

cimport cython
from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdint cimport uintptr_t
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from rmm.pylibrmm.stream cimport Stream

from rmm.pylibrmm.stream import DEFAULT_STREAM

from cuda.bindings.cyruntime cimport (
    cudaError,
    cudaError_t,
    cudaMemcpyAsync,
    cudaMemcpyKind,
    cudaStream_t,
)

from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.device_buffer cimport (
    cuda_device_id,
    device_buffer,
    get_current_cuda_device,
    prefetch,
)
from rmm.librmm.memory_resource cimport device_memory_resource
from rmm.pylibrmm.memory_resource cimport (
    DeviceMemoryResource,
    get_current_device_resource,
)


# The DeviceMemoryResource attribute could be released prematurely
# by the gc if the DeviceBuffer is in a reference cycle. Removing
# the tp_clear function with the no_gc_clear decoration prevents that.
# See https://github.com/rapidsai/rmm/pull/931 for details.
@cython.no_gc_clear
cdef class DeviceBuffer:

    def __cinit__(self, *,
                  uintptr_t ptr=0,
                  size_t size=0,
                  Stream stream=DEFAULT_STREAM,
                  DeviceMemoryResource mr=None):
        """Construct a ``DeviceBuffer`` with optional size and data pointer

        Parameters
        ----------
        ptr : int
            pointer to some data on host or device to copy over
        size : int
            size of the buffer to allocate
            (and possibly size of data to copy)
        stream : optional
            CUDA stream to use for construction and/or copying,
            defaults to the CUDA default stream. A reference to the
            stream is stored internally to ensure it doesn't go out of
            scope while the DeviceBuffer is in use. Destroying the
            underlying stream while the DeviceBuffer is in use will
            result in undefined behavior.
        mr : optional
           DeviceMemoryResource for the allocation, if not provided
           defaults to the current device resource.

        Note
        ----
        If the pointer passed is non-null and ``stream`` is the default stream,
        it is synchronized after the copy. However if a non-default ``stream``
        is provided, this function is fully asynchronous.

        Examples
        --------
        >>> import rmm
        >>> db = rmm.DeviceBuffer(size=5)
        """
        cdef const void* c_ptr
        cdef device_memory_resource * mr_ptr
        # Save a reference to the MR and stream used for allocation
        self.mr = get_current_device_resource() if mr is None else mr
        self.stream = stream

        mr_ptr = self.mr.get_mr()
        with nogil:
            c_ptr = <const void*>ptr

            if c_ptr == NULL or size == 0:
                self.c_obj.reset(new device_buffer(size, stream.view(), mr_ptr))
            else:
                self.c_obj.reset(new device_buffer(c_ptr, size, stream.view(), mr_ptr))

                if stream.c_is_default():
                    stream.c_synchronize()

    def __dealloc__(self):
        # Relying on the unique_ptr to call the destructor when Python reclaims
        # the object is unsafe because deconstructing the underlying C++
        # device_buffer may involve a cudaFree call, which involves grabbing a
        # lock from the CUDA runtime. If that lock is being held by another
        # thread which is simultaneously trying to acquire the GIL, then we
        # could have the thread deallocating the device buffer holding the GIL
        # while trying to claim the CUDA lock while the other thread is holding
        # the CUDA lock and trying to acquire the GIL, resulting in a deadlock.
        with nogil:
            self.c_obj.reset()

    def __len__(self):
        return self.size

    def __sizeof__(self):
        return self.size

    def __bytes__(self):
        return self.tobytes()

    @property
    def nbytes(self):
        """Gets the size of the buffer in bytes."""
        return self.size

    @property
    def ptr(self):
        """Gets a pointer to the underlying data."""
        return int(<uintptr_t>self.c_data())

    @property
    def size(self):
        """Gets the size of the buffer in bytes."""
        return int(self.c_size())

    def __reduce__(self):
        return to_device, (self.copy_to_host(),)

    @property
    def __cuda_array_interface__(self):
        cdef dict intf = {
            "data": (self.ptr, False),
            "shape": (self.size,),
            "strides": None,
            "typestr": "|u1",
            "version": 0
        }
        return intf

    def prefetch(self, device=None, stream=None):
        """Prefetch buffer data to the specified device on the specified stream.

        Assumes the storage for this DeviceBuffer is CUDA managed memory
        (unified memory). If it is not, this function is a no-op.

        Parameters
        ----------
        device : optional
            The CUDA device to which to prefetch the memory for this buffer.
            Defaults to the current CUDA device. To prefetch to the CPU, pass
            :py:attr:`~cuda.bindings.runtime.cudaCpuDeviceId` as the device.
        stream : optional
            CUDA stream to use for prefetching. Defaults to self.stream
        """
        cdef cuda_device_id dev = (get_current_cuda_device()
                                   if device is None
                                   else cuda_device_id(device))
        cdef Stream strm = self.stream if stream is None else stream
        with nogil:
            prefetch(self.c_obj.get()[0].data(),
                     self.c_obj.get()[0].size(),
                     dev,
                     strm.view())

    def copy(self):
        """Returns a copy of DeviceBuffer.

        Returns
        -------
        A deep copy of existing ``DeviceBuffer``

        Examples
        --------
        >>> import rmm
        >>> db = rmm.DeviceBuffer.to_device(b"abc")
        >>> db_copy = db.copy()
        >>> db.copy_to_host()
        array([97, 98, 99], dtype=uint8)
        >>> db_copy.copy_to_host()
        array([97, 98, 99], dtype=uint8)
        >>> assert db is not db_copy
        >>> assert db.ptr != db_copy.ptr
        """
        ret = DeviceBuffer(ptr=self.ptr, size=self.size, stream=self.stream)
        ret.mr = self.mr
        return ret

    def __copy__(self):
        return self.copy()

    @staticmethod
    cdef DeviceBuffer c_from_unique_ptr(
        unique_ptr[device_buffer] ptr,
        Stream stream=DEFAULT_STREAM,
        DeviceMemoryResource mr=None,
    ):
        cdef DeviceBuffer buf = DeviceBuffer.__new__(DeviceBuffer)
        if stream.c_is_default():
            stream.c_synchronize()
        buf.c_obj = move(ptr)
        buf.mr = get_current_device_resource() if mr is None else mr
        buf.stream = stream
        return buf

    @staticmethod
    cdef DeviceBuffer c_to_device(const unsigned char[::1] b,
                                  Stream stream=DEFAULT_STREAM) except *:
        """Calls ``to_device`` function on arguments provided"""
        return to_device(b, stream)

    @staticmethod
    def to_device(const unsigned char[::1] b,
                  Stream stream=DEFAULT_STREAM):
        """Calls ``to_device`` function on arguments provided."""
        return to_device(b, stream)

    cpdef copy_to_host(self, ary=None, Stream stream=DEFAULT_STREAM):
        """Copy from a ``DeviceBuffer`` to a buffer on host.

        Parameters
        ----------
        ary : ``bytes``-like buffer to write into
        stream : CUDA stream to use for copying, default the default stream

        Examples
        --------
        >>> import rmm
        >>> db = rmm.DeviceBuffer.to_device(b"abc")
        >>> hb = bytearray(db.nbytes)
        >>> db.copy_to_host(hb)
        >>> print(hb)
        bytearray(b'abc')
        >>> hb = db.copy_to_host()
        >>> print(hb)
        bytearray(b'abc')
        """
        cdef const device_buffer* dbp = self.c_obj.get()
        cdef size_t s = dbp.size()

        cdef unsigned char[::1] hb = ary
        if hb is None:
            # NumPy leverages huge pages under-the-hood,
            # which speeds up the copy from device to host.
            hb = ary = np.empty((s,), dtype="u1")
        elif len(hb) < s:
            raise ValueError(
                "Argument `ary` is too small. Need space for %i bytes." % s
            )

        copy_ptr_to_host(<uintptr_t>dbp.data(), hb[:s], stream)

        return ary

    cpdef copy_from_host(self, ary, Stream stream=DEFAULT_STREAM):
        """Copy from a buffer on host to ``self``

        Parameters
        ----------
        ary : ``bytes``-like buffer to copy from
        stream : CUDA stream to use for copying, default the default stream

        Examples
        --------
        >>> import rmm
        >>> db = rmm.DeviceBuffer(size=10)
        >>> hb = b"abcdef"
        >>> db.copy_from_host(hb)
        >>> hb = db.copy_to_host()
        >>> print(hb)
        array([97, 98, 99,  0,  0,  0,  0,  0,  0,  0], dtype=uint8)
        """
        cdef device_buffer* dbp = self.c_obj.get()

        cdef const unsigned char[::1] hb = ary
        cdef size_t s = len(hb)
        if s > self.size:
            raise ValueError(
                "Argument `ary` is too large. Need space for %i bytes." % s
            )

        copy_host_to_ptr(hb[:s], <uintptr_t>dbp.data(), stream)

    cpdef copy_from_device(self, cuda_ary,
                           Stream stream=DEFAULT_STREAM):
        """Copy from a buffer on host to ``self``

        Parameters
        ----------
        cuda_ary : object to copy from that has ``__cuda_array_interface__``
        stream : CUDA stream to use for copying, default the default stream

        Examples
        --------
        >>> import rmm
        >>> db = rmm.DeviceBuffer(size=5)
        >>> db2 = rmm.DeviceBuffer.to_device(b"abc")
        >>> db.copy_from_device(db2)
        >>> hb = db.copy_to_host()
        >>> print(hb)
        array([97, 98, 99,  0,  0], dtype=uint8)
        """
        if not hasattr(cuda_ary, "__cuda_array_interface__"):
            raise ValueError(
                "Expected object to support `__cuda_array_interface__` "
                "protocol"
            )

        cuda_ary_interface = cuda_ary.__cuda_array_interface__
        shape = cuda_ary_interface["shape"]
        strides = cuda_ary_interface.get("strides")
        dtype = np.dtype(cuda_ary_interface["typestr"])

        if len(shape) > 1:
            raise ValueError(
                "Only 1-D contiguous arrays are supported, got {}-D "
                "array".format(str(len(shape)))
            )

        if strides is not None:
            if strides[0] != dtype.itemsize:
                raise ValueError(
                    "Only 1-D contiguous arrays are supported, got a "
                    "non-contiguous array"
                )

        cdef uintptr_t src_ptr = cuda_ary_interface["data"][0]
        cdef size_t s = shape[0] * dtype.itemsize
        if s > self.size:
            raise ValueError(
                "Argument `hb` is too large. Need space for %i bytes." % s
            )

        cdef device_buffer* dbp = self.c_obj.get()

        copy_device_to_ptr(
            <uintptr_t>src_ptr,
            <uintptr_t>dbp.data(),
            s,
            stream
        )

    cpdef bytes tobytes(self, Stream stream=DEFAULT_STREAM):
        cdef const device_buffer* dbp = self.c_obj.get()
        cdef size_t s = dbp.size()

        cdef bytes b = PyBytes_FromStringAndSize(NULL, s)
        cdef unsigned char* p = b
        cdef unsigned char[::1] mv = (<unsigned char[:(s + 1):1]>p)[:s]
        self.copy_to_host(mv, stream)

        return b

    cdef size_t c_size(self) except *:
        return self.c_obj.get()[0].size()

    cpdef void reserve(self,
                       size_t new_capacity,
                       Stream stream=DEFAULT_STREAM) except *:
        self.c_obj.get()[0].reserve(new_capacity, stream.view())

    cpdef void resize(self,
                      size_t new_size,
                      Stream stream=DEFAULT_STREAM) except *:
        self.c_obj.get()[0].resize(new_size, stream.view())

    cpdef size_t capacity(self) except *:
        return self.c_obj.get()[0].capacity()

    cdef void* c_data(self) except *:
        return self.c_obj.get()[0].data()

    cdef device_buffer c_release(self) except *:
        """
        Releases ownership of the data held by this DeviceBuffer.
        """
        return move(cython.operator.dereference(self.c_obj))


@cython.boundscheck(False)
cpdef DeviceBuffer to_device(const unsigned char[::1] b,
                             Stream stream=DEFAULT_STREAM):
    """Return a new ``DeviceBuffer`` with a copy of the data.

    Parameters
    ----------
    b : ``bytes``-like data on host to copy to device
    stream : CUDA stream to use for copying, default the default stream

    Returns
    -------
    ``DeviceBuffer`` with copy of data from host

    Examples
    --------
    >>> import rmm
    >>> db = rmm.pylibrmm.device_buffer.to_device(b"abc")
    >>> print(bytes(db))
    b'abc'
    """

    if b is None:
        raise TypeError(
            "Argument 'b' has incorrect type"
            " (expected bytes-like, got NoneType)"
        )

    cdef uintptr_t p = <uintptr_t>&b[0]
    cdef size_t s = len(b)
    return DeviceBuffer(ptr=p, size=s, stream=stream)


@cython.boundscheck(False)
cdef void _copy_async(const void* src,
                      void* dst,
                      size_t count,
                      cudaMemcpyKind kind,
                      cuda_stream_view stream) except * nogil:
    """
    Asynchronously copy data between host and/or device pointers.

    This is a convenience wrapper around cudaMemcpyAsync that
    checks for errors. Only used for internal implementation.

    Parameters
    ----------
    src : pointer to ``bytes``-like host buffer or device data to copy from
    dst : pointer to ``bytes``-like host buffer or device data to copy into
    count : the size in bytes to copy
    kind : the kind of copy to perform
    stream : CUDA stream to use for copying, default the default stream
    """
    cdef cudaError_t err
    with nogil:
        err = cudaMemcpyAsync(dst, src, count, kind,
                              <cudaStream_t>stream)

    if err != cudaError.cudaSuccess:
        raise RuntimeError(f"Memcpy failed with error: {err}")


@cython.boundscheck(False)
cpdef void copy_ptr_to_host(uintptr_t db,
                            unsigned char[::1] hb,
                            Stream stream=DEFAULT_STREAM) except *:
    """Copy from a device pointer to a buffer on host

    Parameters
    ----------
    db : pointer to data on device to copy
    hb : ``bytes``-like buffer to write into
    stream : CUDA stream to use for copying, default the default stream

    Note
    ----
    If ``stream`` is the default stream, it is synchronized after the copy.
    However if a non-default ``stream`` is provided, this function is fully
    asynchronous.

    Examples
    --------
    >>> import rmm
    >>> db = rmm.DeviceBuffer.to_device(b"abc")
    >>> hb = bytearray(db.nbytes)
    >>> rmm.pylibrmm.device_buffer.copy_ptr_to_host(db.ptr, hb)
    >>> print(hb)
    bytearray(b'abc')
    """

    if hb is None:
        raise TypeError(
            "Argument `hb` has incorrect type"
            " (expected bytes-like, got NoneType)"
        )

    with nogil:
        _copy_async(<const void*>db, <void*>&hb[0], len(hb),
                    cudaMemcpyKind.cudaMemcpyDeviceToHost, stream.view())

    if stream.c_is_default():
        stream.c_synchronize()


@cython.boundscheck(False)
cpdef void copy_host_to_ptr(const unsigned char[::1] hb,
                            uintptr_t db,
                            Stream stream=DEFAULT_STREAM) except *:
    """Copy from a host pointer to a device pointer

    Parameters
    ----------
    hb : ``bytes``-like host buffer to copy
    db : pointer to data on device to write into
    stream : CUDA stream to use for copying, default the default stream

    Note
    ----
    If ``stream`` is the default stream, it is synchronized after the copy.
    However if a non-default ``stream`` is provided, this function is fully
    asynchronous.

    Examples
    --------
    >>> import rmm
    >>> db = rmm.DeviceBuffer(size=10)
    >>> hb = b"abc"
    >>> rmm.pylibrmm.device_buffer.copy_host_to_ptr(hb, db.ptr)
    >>> hb = db.copy_to_host()
    >>> print(hb)
    array([97, 98, 99,  0,  0,  0,  0,  0,  0,  0], dtype=uint8)
    """

    if hb is None:
        raise TypeError(
            "Argument `hb` has incorrect type"
            " (expected bytes-like, got NoneType)"
        )

    with nogil:
        _copy_async(<const void*>&hb[0], <void*>db, len(hb),
                    cudaMemcpyKind.cudaMemcpyHostToDevice, stream.view())

    if stream.c_is_default():
        stream.c_synchronize()


@cython.boundscheck(False)
cpdef void copy_device_to_ptr(uintptr_t d_src,
                              uintptr_t d_dst,
                              size_t count,
                              Stream stream=DEFAULT_STREAM) except *:
    """Copy from a device pointer to a device pointer

    Parameters
    ----------
    d_src : pointer to data on device to copy from
    d_dst : pointer to data on device to write into
    count : the size in bytes to copy
    stream : CUDA stream to use for copying, default the default stream

    Examples
    --------
    >>> import rmm
    >>> db = rmm.DeviceBuffer(size=5)
    >>> db2 = rmm.DeviceBuffer.to_device(b"abc")
    >>> rmm.pylibrmm.device_buffer.copy_device_to_ptr(db2.ptr, db.ptr, db2.size)
    >>> hb = db.copy_to_host()
    >>> hb
    array([97, 98, 99,  0,  0], dtype=uint8)
    """

    with nogil:
        _copy_async(<const void*>d_src, <void*>d_dst, count,
                    cudaMemcpyKind.cudaMemcpyDeviceToDevice, stream.view())
