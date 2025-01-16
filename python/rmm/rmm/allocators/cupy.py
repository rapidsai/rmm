# Copyright (c) 2023-2025, NVIDIA CORPORATION.
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
