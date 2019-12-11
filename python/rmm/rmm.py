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
from enum import IntEnum

import numpy as np
from numba import cuda

import rmm._lib as librmm


# Utility Functions
class RMMError(Exception):
    def __init__(self, errcode, msg):
        self.errcode = errcode
        super(RMMError, self).__init__(msg)


def _array_helper(addr, datasize, shape, strides, dtype, finalizer=None):
    ctx = cuda.current_context()
    ptr = ctypes.c_uint64(int(addr))
    mem = cuda.driver.MemoryPointer(ctx, ptr, datasize, finalizer=finalizer)
    return cuda.cudadrv.devicearray.DeviceNDArray(
        shape, strides, dtype, gpu_data=mem
    )


class rmm_allocation_mode(IntEnum):
    CudaDefaultAllocation = (0,)
    PoolAllocation = (1,)
    CudaManagedMemory = (2,)


# API Functions
def _initialize(
    pool_allocator=False,
    managed_memory=False,
    initial_pool_size=None,
    devices=0,
    logging=False,
):
    """
    Initializes RMM library using the options passed
    """
    allocation_mode = 0

    if pool_allocator:
        allocation_mode |= rmm_allocation_mode.PoolAllocation
    if managed_memory:
        allocation_mode |= rmm_allocation_mode.CudaManagedMemory

    if not pool_allocator:
        initial_pool_size = 0
    elif pool_allocator and initial_pool_size is None:
        initial_pool_size = 0
    elif pool_allocator and initial_pool_size == 0:
        initial_pool_size = 1

    if devices is None:
        devices = [0]
    elif isinstance(devices, int):
        devices = [devices]

    return librmm.rmm_initialize(
        allocation_mode, initial_pool_size, devices, logging
    )


def _finalize():
    """
    Finalizes the RMM library, freeing all allocated memory
    """
    return librmm.rmm_finalize()


def reinitialize(
    pool_allocator=False,
    managed_memory=False,
    initial_pool_size=None,
    devices=0,
    logging=False,
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
    """
    _finalize()
    return _initialize(
        pool_allocator=pool_allocator,
        managed_memory=managed_memory,
        initial_pool_size=initial_pool_size,
        devices=devices,
        logging=logging,
    )


def is_initialized():
    """
    Returns true if RMM has been initialized, false otherwise
    """
    return librmm.rmm_is_initialized()


def csv_log():
    """
    Returns a CSV log of all events logged by RMM, if logging is enabled
    """
    return librmm.rmm_csv_log()


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
    # note no finalizer -- freed externally!
    return _array_helper(
        addr=ptr,
        datasize=datasize,
        shape=(nelem,),
        strides=(elemsize,),
        dtype=dtype,
        finalizer=finalizer,
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

    addr = librmm.rmm_alloc(datasize, stream)

    # Note Numba will call the finalizer to free the device memory
    # allocated above
    return _array_helper(
        addr=addr,
        datasize=datasize,
        shape=shape,
        strides=strides,
        dtype=dtype,
        finalizer=_make_finalizer(addr, stream),
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


def auto_device(obj, stream=0, copy=True):
    """
    Create a DeviceRecord or DeviceArray like obj and optionally copy data from
    host to device. If obj already represents device memory, it is returned and
    no copy is made. Uses RMM for device memory allocation if necessary.
    """
    if cuda.driver.is_device_memory(obj):
        return obj, False
    if hasattr(obj, "__cuda_array_interface__"):
        new_dev_array = cuda.as_cuda_array(obj)
        # Allocate new output array using rmm and copy the numba device
        # array to an rmm owned device array
        out_dev_array = device_array_like(new_dev_array)
        out_dev_array.copy_to_device(new_dev_array)
        return out_dev_array, False
    else:
        if isinstance(obj, np.void):
            devobj = cuda.devicearray.from_record_like(obj, stream=stream)
        else:
            if not isinstance(obj, np.ndarray):
                obj = np.asarray(obj)
            cuda.devicearray.sentry_contiguous(obj)
            devobj = device_array_like(obj, stream=stream)

        if copy:
            devobj.copy_to_device(obj, stream=stream)
        return devobj, True


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


def _make_finalizer(handle, stream):
    """
    Factory to make the finalizer function.
    We need to bind *handle* and *stream* into the actual finalizer, which
    takes no args.
    """

    def finalizer():
        """
        Invoked when the MemoryPointer is freed
        """
        librmm.rmm_free(handle, stream)

    return finalizer


def _register_atexit_finalize():
    """
    Registers rmmFinalize() with ``std::atexit``.
    """
    librmm.register_atexit_finalize()
