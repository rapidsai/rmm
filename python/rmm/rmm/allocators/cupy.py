# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from rmm import pylibrmm
from rmm.pylibrmm.stream import Stream

try:
    import cupy
except ImportError:
    cupy = None


def rmm_cupy_allocator(nbytes):
    """
    A CuPy allocator that makes use of RMM.

    Examples
    --------
    >>> from rmm.allocators.cupy import rmm_cupy_allocator
    >>> import cupy
    >>> cupy.cuda.set_allocator(rmm_cupy_allocator)
    """
    if cupy is None:
        raise ModuleNotFoundError("No module named 'cupy'")

    stream = Stream(obj=cupy.cuda.get_current_stream())
    buf = pylibrmm.device_buffer.DeviceBuffer(size=nbytes, stream=stream)
    dev_id = -1 if buf.ptr else cupy.cuda.device.get_device_id()
    mem = cupy.cuda.UnownedMemory(
        ptr=buf.ptr, size=buf.size, owner=buf, device_id=dev_id
    )
    ptr = cupy.cuda.memory.MemoryPointer(mem, 0)

    return ptr
