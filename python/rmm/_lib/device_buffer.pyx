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

# cython: profile = False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3


import numpy as np

from libcpp.memory cimport unique_ptr
from libc.stdint cimport uintptr_t

from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AS_STRING

from rmm._lib.lib cimport (cudaError_t, cudaSuccess,
                           cudaStream_t, cudaStreamSynchronize)

cimport cython


cdef class DeviceBuffer:

    def __cinit__(self, *,
                  uintptr_t ptr=0,
                  size_t size=0,
                  uintptr_t stream=0):
        cdef const void* c_ptr
        cdef cudaStream_t c_stream

        with nogil:
            c_ptr = <const void*>ptr
            c_stream = <cudaStream_t>stream

            if c_ptr == NULL:
                self.c_obj.reset(new device_buffer(size, c_stream))
            else:
                self.c_obj.reset(new device_buffer(c_ptr, size, c_stream))

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

    def __getstate__(self):
        return self.tobytes()

    def __setstate__(self, state):
        cdef DeviceBuffer other = DeviceBuffer.c_to_device(state)
        self.c_obj = move(other.c_obj)

    @property
    def __cuda_array_interface__(self):
        cdef dict intf = {
            "data": (self.ptr, False),
            "shape": (self.size,),
            "strides": (1,),
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

    cpdef bytes tobytes(self, uintptr_t stream=0):
        cdef const device_buffer* dbp = self.c_obj.get()
        cdef size_t s = dbp.size()

        cdef bytes b = PyBytes_FromStringAndSize(NULL, s)
        cdef unsigned char* p = <unsigned char*>PyBytes_AS_STRING(b)
        cdef unsigned char[::1] mv = (<unsigned char[:(s + 1):1]>p)[:s]
        self.copy_to_host(mv, stream)

        return b

    cdef size_t c_size(self):
        return self.c_obj.get()[0].size()

    cpdef void resize(self, size_t new_size):
        self.c_obj.get()[0].resize(new_size)

    cpdef size_t capacity(self):
        return self.c_obj.get()[0].capacity()

    cdef void* c_data(self):
        return self.c_obj.get()[0].data()


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

    If ``stream`` is the default stream, we synchronize on it.
    However if a non-default ``stream`` is provided, we leave
    it up to the user to synchronize.

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
    if err != cudaSuccess:
        with gil:
            raise RuntimeError(f"Memcpy failed with error: {err}")

    if stream == 0:
        err = cudaStreamSynchronize(<cudaStream_t>stream)
        if err != cudaSuccess:
            with gil:
                raise RuntimeError(f"Stream sync failed with error: {err}")
