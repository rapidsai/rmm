# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_FromStringAndSize
from libc.stdint cimport uintptr_t
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from rmm._cuda.gpu cimport cudaError, cudaError_t
from rmm._lib.lib cimport (
    cudaMemcpyDeviceToDevice,
    cudaMemcpyDeviceToHost,
    cudaMemcpyHostToDevice,
    cudaStream_t,
    cudaStreamSynchronize,
)


cdef class DeviceBuffer:

    def __cinit__(self, *,
                  uintptr_t ptr=0,
                  size_t size=0,
                  uintptr_t stream=0):
        """Construct a ``DeviceBuffer`` with optional size and data pointer

        Parameters
        ----------
        ptr : pointer to some data on host or device to copy over
        size : size of the buffer to allocate
               (and possibly size of data to copy)
        stream : CUDA stream to use for construction and/or copying, default 0

        Note
        ----

        If ``stream`` is the default stream, it is synchronized after the copy.
        However if a non-default ``stream`` is provided, this function is fully
        asynchronous.

        Examples
        --------
        >>> import rmm
        >>> db = rmm.DeviceBuffer(size=5)
        """
        cdef const void* c_ptr
        cdef cudaStream_t c_stream
        cdef cudaError_t err

        with nogil:
            c_ptr = <const void*>ptr
            c_stream = <cudaStream_t>stream

            if size == 0:
                self.c_obj.reset(new device_buffer())
            elif c_ptr == NULL:
                self.c_obj.reset(new device_buffer(size, c_stream))
            else:
                self.c_obj.reset(new device_buffer(c_ptr, size, c_stream))

                if c_stream == NULL:
                    err = cudaStreamSynchronize(c_stream)
                    if err != cudaError.cudaSuccess:
                        with gil:
                            raise RuntimeError(
                                f"Stream sync failed with error: {err}"
                            )

    def __len__(self):
        return self.size

    def __sizeof__(self):
        return self.size

    def __bytes__(self):
        return self.tobytes()

    @property
    def nbytes(self):
        return self.size

    @property
    def ptr(self):
        return int(<uintptr_t>self.c_data())

    @property
    def size(self):
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

    @staticmethod
    cdef DeviceBuffer c_from_unique_ptr(unique_ptr[device_buffer] ptr):
        cdef DeviceBuffer buf = DeviceBuffer.__new__(DeviceBuffer)
        buf.c_obj = move(ptr)
        return buf

    @staticmethod
    cdef DeviceBuffer c_to_device(const unsigned char[::1] b,
                                  uintptr_t stream=0):
        """Calls ``to_device`` function on arguments provided"""
        return to_device(b, stream)

    @staticmethod
    def to_device(const unsigned char[::1] b, uintptr_t stream=0):
        """Calls ``to_device`` function on arguments provided"""
        return to_device(b, stream)

    cpdef copy_to_host(self, ary=None, uintptr_t stream=0):
        """Copy from a ``DeviceBuffer`` to a buffer on host

        Parameters
        ----------
        hb : ``bytes``-like buffer to write into
        stream : CUDA stream to use for copying, default 0

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
                "Argument `hb` is too small. Need space for %i bytes." % s
            )

        with nogil:
            copy_ptr_to_host(<uintptr_t>dbp.data(), hb[:s], stream)

        return ary

    cpdef copy_from_host(self, ary, uintptr_t stream=0):
        """Copy from a buffer on host to ``self``

        Parameters
        ----------
        hb : ``bytes``-like buffer to copy from
        stream : CUDA stream to use for copying, default 0

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
                "Argument `hb` is too large. Need space for %i bytes." % s
            )

        with nogil:
            copy_host_to_ptr(hb[:s], <uintptr_t>dbp.data(), stream)

    cpdef copy_from_device(self, cuda_ary, uintptr_t stream=0):
        """Copy from a buffer on host to ``self``

        Parameters
        ----------
        cuda_ary : object to copy from that has ``__cuda_array_interface__``
        stream : CUDA stream to use for copying, default 0

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

        with nogil:
            copy_device_to_ptr(
                <uintptr_t>src_ptr,
                <uintptr_t>dbp.data(),
                s,
                stream
            )

    cpdef bytes tobytes(self, uintptr_t stream=0):
        cdef const device_buffer* dbp = self.c_obj.get()
        cdef size_t s = dbp.size()

        cdef bytes b = PyBytes_FromStringAndSize(NULL, s)
        cdef unsigned char* p = <unsigned char*>PyBytes_AS_STRING(b)
        cdef unsigned char[::1] mv = (<unsigned char[:(s + 1):1]>p)[:s]
        self.copy_to_host(mv, stream)

        return b

    cdef size_t c_size(self) except *:
        return self.c_obj.get()[0].size()

    cpdef void resize(self, size_t new_size) except *:
        self.c_obj.get()[0].resize(new_size)

    cpdef size_t capacity(self) except *:
        return self.c_obj.get()[0].capacity()

    cdef void* c_data(self) except *:
        return self.c_obj.get()[0].data()

    cdef device_buffer c_release(self) except *:
        """
        Releases ownership the data held by this DeviceBuffer.
        """
        return move(cython.operator.dereference(self.c_obj))


@cython.boundscheck(False)
cpdef DeviceBuffer to_device(const unsigned char[::1] b, uintptr_t stream=0):
    """Return a new ``DeviceBuffer`` with a copy of the data

    Parameters
    ----------
    b : ``bytes``-like data on host to copy to device
    stream : CUDA stream to use for copying, default 0

    Returns
    -------
    ``DeviceBuffer`` with copy of data from host

    Examples
    --------
    >>> import rmm
    >>> db = rmm._lib.device_buffer.to_device(b"abc")
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
cpdef void copy_ptr_to_host(uintptr_t db,
                            unsigned char[::1] hb,
                            uintptr_t stream=0) nogil except *:
    """Copy from a device pointer to a buffer on host

    Parameters
    ----------
    db : pointer to data on device to copy
    hb : ``bytes``-like buffer to write into
    stream : CUDA stream to use for copying, default 0

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
    >>> rmm._lib.device_buffer.copy_ptr_to_host(db.ptr, hb)
    >>> print(hb)
    bytearray(b'abc')
    """

    if hb is None:
        with gil:
            raise TypeError(
                "Argument `hb` has incorrect type"
                " (expected bytes-like, got NoneType)"
            )

    cdef cudaError_t err

    err = cudaMemcpyAsync(<void*>&hb[0], <const void*>db, len(hb),
                          cudaMemcpyDeviceToHost, <cudaStream_t>stream)
    if err != cudaError.cudaSuccess:
        with gil:
            raise RuntimeError(f"Memcpy failed with error: {err}")

    if stream == 0:
        err = cudaStreamSynchronize(<cudaStream_t>stream)
        if err != cudaError.cudaSuccess:
            with gil:
                raise RuntimeError(f"Stream sync failed with error: {err}")


@cython.boundscheck(False)
cpdef void copy_host_to_ptr(const unsigned char[::1] hb,
                            uintptr_t db,
                            uintptr_t stream=0) nogil except *:
    """Copy from a host pointer to a device pointer

    Parameters
    ----------
    hb : ``bytes``-like host buffer to copy
    db : pointer to data on device to write into
    stream : CUDA stream to use for copying, default 0

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
    >>> rmm._lib.device_buffer.copy_host_to_ptr(hb, db.ptr)
    >>> hb = db.copy_to_host()
    >>> print(hb)
    array([97, 98, 99,  0,  0,  0,  0,  0,  0,  0], dtype=uint8)
    """

    if hb is None:
        with gil:
            raise TypeError(
                "Argument `hb` has incorrect type"
                " (expected bytes-like, got NoneType)"
            )

    cdef cudaError_t err

    err = cudaMemcpyAsync(<void*>db, <const void*>&hb[0], len(hb),
                          cudaMemcpyHostToDevice, <cudaStream_t>stream)
    if err != cudaError.cudaSuccess:
        with gil:
            raise RuntimeError(f"Memcpy failed with error: {err}")

    if stream == 0:
        err = cudaStreamSynchronize(<cudaStream_t>stream)
        if err != cudaError.cudaSuccess:
            with gil:
                raise RuntimeError(f"Stream sync failed with error: {err}")


@cython.boundscheck(False)
cpdef void copy_device_to_ptr(uintptr_t d_src,
                              uintptr_t d_dst,
                              size_t count,
                              uintptr_t stream=0) nogil except *:
    """Copy from a host pointer to a device pointer

    Parameters
    ----------
    d_src : pointer to data on device to copy from
    d_dst : pointer to data on device to write into
    stream : CUDA stream to use for copying, default 0

    Note
    ----

    If ``stream`` is the default stream, it is synchronized after the copy.
    However if a non-default ``stream`` is provided, this function is fully
    asynchronous.

    Examples
    --------
    >>> import rmm
    >>> import numpy as np
    >>> db = rmm.DeviceBuffer(size=5)
    >>> db2 = rmm.DeviceBuffer.to_device(b"abc")
    >>> rmm._lib.device_buffer.copy_device_to_ptr(db2.ptr, db.ptr, db2.size)
    >>> hb = db.copy_to_host()
    >>> print(hb)
    array([10, 11, 12,  0,  0], dtype=uint8)
    """
    cdef cudaError_t err

    err = cudaMemcpyAsync(<void*>d_dst, <const void*>d_src, count,
                          cudaMemcpyDeviceToDevice, <cudaStream_t>stream)
    if err != cudaError.cudaSuccess:
        with gil:
            raise RuntimeError(f"Memcpy failed with error: {err}")

    if stream == 0:
        err = cudaStreamSynchronize(<cudaStream_t>stream)
        if err != cudaError.cudaSuccess:
            with gil:
                raise RuntimeError(f"Stream sync failed with error: {err}")
