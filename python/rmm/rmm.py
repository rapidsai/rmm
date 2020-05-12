# Copyright (c) 2019, NVIDIA CORPORATION.
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

import ctypes
import os

import numpy as np
from numba import cuda

import rmm._lib as librmm


# Utility Functions
class RMMError(Exception):
    def __init__(self, errcode, msg):
        self.errcode = errcode
        super(RMMError, self).__init__(msg)


# API Functions
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
    if not pool_allocator:
        if not managed_memory:
            kind = "cuda"
        else:
            kind = "managed"
    else:
        if not managed_memory:
            kind = "cnmem"
        else:
            kind = "cnmem_managed"

        if initial_pool_size is None:
            initial_pool_size = 0

        if devices is None:
            devices = [0]
        elif isinstance(devices, int):
            devices = [devices]

    librmm.set_default_resource(
        kind, initial_pool_size, devices, logging, log_file_name
    )


def reinitialize(
    pool_allocator=False,
    managed_memory=False,
    initial_pool_size=None,
    devices=0,
    logging=False,
    log_file_name=None,
):
    """
    Finalizes and then initializes RMM using the options passed. Using memory
    from a previous initialization of RMM is undefined behavior and should be
    avoided.

    Parameters
    ----------
    pool_allocator : bool, default False
        If True, use a pool allocation strategy which can greatly improve
        performance.
    managed_memory : bool, default False
        If True, use managed memory for device memory allocation
    initial_pool_size : int, default None
        When `pool_allocator` is True, this indicates the initial pool size in
        bytes. None is used to indicate the default size of the underlying
        memorypool implementation, which currently is 1/2 total GPU memory.
    devices : int or List[int], default 0
        GPU device  IDs to register. By default registers only GPU 0.
    logging : bool, default False
        If True, enable run-time logging of all memory events
        (alloc, free, realloc).
        This has significant performance impact.
    log_file_name : str
        Name of the log file. If not specified, the environment variable
        RMM_LOG_FILE is used. A TypeError is thrown if neither is available.
    """
    if logging and not log_file_name:
        log_file_name = os.getenv("RMM_LOG_FILE")
        if not log_file_name:
            raise TypeError(
                "RMM log file must be specified either using "
                "log_file_name= argument or RMM_LOG_FILE "
                "environment variable"
            )
    _initialize(
        pool_allocator=pool_allocator,
        managed_memory=managed_memory,
        initial_pool_size=initial_pool_size,
        devices=devices,
        logging=logging,
        log_file_name=log_file_name,
    )


def is_initialized():
    """
    Returns true if RMM has been initialized, false otherwise
    """
    return librmm.is_initialized()


def device_array_from_ptr(ptr, nelem, dtype=np.float, finalizer=None):
    """
    device_array_from_ptr(ptr, size, dtype=np.float, stream=0)

    Create a Numba device array from a ptr, size, and dtype.
    """
    # Handle Datetime Column
    if dtype == np.datetime64:
        dtype = np.dtype("datetime64[ms]")
    else:
        dtype = np.dtype(dtype)

    elemsize = dtype.itemsize
    datasize = elemsize * nelem
    shape = (nelem,)
    strides = (elemsize,)
    # note no finalizer -- freed externally!
    ctx = cuda.current_context()
    ptr = ctypes.c_uint64(int(ptr))
    mem = cuda.driver.MemoryPointer(ctx, ptr, datasize, finalizer=finalizer)
    return cuda.cudadrv.devicearray.DeviceNDArray(
        shape, strides, dtype, gpu_data=mem
    )


def device_array(shape, dtype=np.float, strides=None, order="C", stream=0):
    """
    device_array(shape, dtype=np.float, strides=None, order='C',
                 stream=0)

    Allocate an empty Numba device array. Clone of Numba `cuda.device_array`,
    but uses RMM for device memory management.
    """
    shape, strides, dtype = cuda.api._prepare_shape_strides_dtype(
        shape, strides, dtype, order
    )
    datasize = cuda.driver.memory_size_from_info(
        shape, strides, dtype.itemsize
    )

    buf = librmm.DeviceBuffer(size=datasize, stream=stream)

    ctx = cuda.current_context()
    ptr = ctypes.c_uint64(int(buf.ptr))
    mem = cuda.driver.MemoryPointer(ctx, ptr, datasize, owner=buf)
    return cuda.cudadrv.devicearray.DeviceNDArray(
        shape, strides, dtype, gpu_data=mem
    )


def device_array_like(ary, stream=0):
    """
    device_array_like(ary, stream=0)

    Call rmmlib.device_array with information from `ary`. Clone of Numba
    `cuda.device_array_like`, but uses RMM for device memory management.
    """
    if ary.ndim == 0:
        ary = ary.reshape(1)

    return device_array(ary.shape, ary.dtype, ary.strides, stream=stream)


def to_device(ary, stream=0, copy=True, to=None):
    """
    to_device(ary, stream=0, copy=True, to=None)

    Allocate and transfer a numpy ndarray or structured scalar to the device.
    Clone of Numba `cuda.to_device`, but uses RMM for device memory management.
    """
    if to is None:
        to = device_array_like(ary, stream=stream)
        to.copy_to_device(ary, stream=stream)
        return to
    if copy:
        to.copy_to_device(ary, stream=stream)
    return to


def get_ipc_handle(ary, stream=0):
    """
    Get an IPC handle from the DeviceArray ary with offset modified by
    the RMM memory pool.
    """
    ipch = cuda.devices.get_context().get_ipc_handle(ary.gpu_data)
    ptr = ary.device_ctypes_pointer.value
    offset = librmm.rmm_getallocationoffset(ptr, stream)
    # replace offset with RMM's offset
    ipch.offset = offset
    desc = dict(shape=ary.shape, strides=ary.strides, dtype=ary.dtype)
    return cuda.cudadrv.devicearray.IpcArrayHandle(
        ipc_handle=ipch, array_desc=desc
    )


def get_info(stream=0):
    """
    Get the free and total bytes of memory managed by a manager associated with
    the stream as a namedtuple with members `free` and `total`.
    """
    return librmm.rmm_getinfo(stream)


try:
    import cupy
except Exception:
    cupy = None


def rmm_cupy_allocator(nbytes):
    """
    A CuPy allocator that make use of RMM.

    Examples
    --------
    >>> import rmm
    >>> import cupy
    >>> cupy.cuda.set_allocator(rmm.rmm_cupy_allocator)
    """
    if cupy is None:
        raise ModuleNotFoundError("No module named 'cupy'")

    buf = librmm.device_buffer.DeviceBuffer(size=nbytes)
    dev_id = -1 if buf.ptr else cupy.cuda.device.get_device_id()
    mem = cupy.cuda.UnownedMemory(
        ptr=buf.ptr, size=buf.size, owner=buf, device_id=dev_id
    )
    ptr = cupy.cuda.memory.MemoryPointer(mem, 0)

    return ptr
