# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
import inspect

from cuda.bindings.driver import CUdeviceptr, cuIpcGetMemHandle
from numba import config, cuda
from numba.cuda import HostOnlyCUDAMemoryManager, IpcHandle, MemoryPointer

from rmm import pylibrmm


def _make_emm_plugin_finalizer(handle, allocations):
    """
    Factory to make the finalizer function.
    We need to bind *handle* and *allocations* into the actual finalizer, which
    takes no args.
    """

    def finalizer():
        """
        Invoked when the MemoryPointer is freed
        """
        # At exit time (particularly in the Numba test suite) allocations may
        # have already been cleaned up by a call to Context.reset() for the
        # context, even if there are some DeviceNDArrays and their underlying
        # allocations lying around. Finalizers then get called by weakref's
        # atexit finalizer, at which point allocations[handle] no longer
        # exists. This is harmless, except that a traceback is printed just
        # prior to exit (without abnormally terminating the program), but is
        # worrying for the user. To avoid the traceback, we check if
        # allocations is already empty.
        #
        # In the case where allocations is not empty, but handle is not in
        # allocations, then something has gone wrong - so we only guard against
        # allocations being completely empty, rather than handle not being in
        # allocations.
        if allocations:
            del allocations[handle]

    return finalizer


class RMMNumbaManager(HostOnlyCUDAMemoryManager):
    """
    External Memory Management Plugin implementation for Numba. Provides
    on-device allocation only.

    See https://numba.readthedocs.io/en/stable/cuda/external-memory.html for
    details of the interface being implemented here.
    """

    def initialize(self):
        # No special initialization needed to use RMM within a given context.
        pass

    def memalloc(self, size):
        """
        Allocate an on-device array from the RMM pool.
        """
        buf = pylibrmm.DeviceBuffer(size=size)
        ctx = self.context

        if config.CUDA_USE_NVIDIA_BINDING:
            ptr = CUdeviceptr(int(buf.ptr))
        else:
            # expect ctypes bindings in numba
            ptr = ctypes.c_uint64(int(buf.ptr))

        finalizer = _make_emm_plugin_finalizer(int(buf.ptr), self.allocations)

        # self.allocations is initialized by the parent, HostOnlyCUDAManager,
        # and cleared upon context reset, so although we insert into it here
        # and delete from it in the finalizer, we need not do any other
        # housekeeping elsewhere.
        self.allocations[int(buf.ptr)] = buf

        return MemoryPointer(ctx, ptr, size, finalizer=finalizer)

    def get_ipc_handle(self, memory):
        """
        Get an IPC handle for the MemoryPointer memory with offset modified by
        the RMM memory pool.
        """
        start, end = cuda.cudadrv.driver.device_extents(memory)

        if config.CUDA_USE_NVIDIA_BINDING:
            _, ipc_handle = cuIpcGetMemHandle(start)
            offset = int(memory.handle) - int(start)
        else:
            ipc_handle = (ctypes.c_byte * 64)()  # IPC handle is 64 bytes
            cuda.cudadrv.driver.driver.cuIpcGetMemHandle(
                ctypes.byref(ipc_handle),
                start,
            )
            offset = memory.handle.value - start
        source_info = cuda.current_context().device.get_device_identity()

        return IpcHandle(
            memory, ipc_handle, memory.size, source_info, offset=offset
        )

    def get_memory_info(self):
        """Returns ``(free, total)`` memory in bytes in the context.

        This implementation raises `NotImplementedError` because the allocation
        will be performed using rmm's currently set default mr, which may be a
        pool allocator.
        """
        raise NotImplementedError()

    @property
    def interface_version(self):
        return 1


# The parent class docstrings contain references without fully qualified names,
# so we need to replace them here for our Sphinx docs to render properly.
for _, method in inspect.getmembers(RMMNumbaManager, inspect.isfunction):
    if method.__doc__ is not None:
        method.__doc__ = method.__doc__.replace(
            ":class:`BaseCUDAMemoryManager`",
            ":class:`numba.cuda.BaseCUDAMemoryManager`",
        )


# Enables the use of RMM for Numba via an environment variable setting,
# NUMBA_CUDA_MEMORY_MANAGER=rmm. See:
# https://numba.readthedocs.io/en/stable/cuda/external-memory.html#environment-variable
_numba_memory_manager = RMMNumbaManager
