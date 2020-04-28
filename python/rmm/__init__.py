# Copyright (c) 2018-2019, NVIDIA CORPORATION.
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

import weakref

from rmm.rmm import (
    RMMError,
    RMMNumbaManager,
    _finalize,
    _initialize,
    _numba_memory_manager,
    _register_atexit_finalize,
    csv_log,
    get_info,
    is_initialized,
    reinitialize,
    rmm_cupy_allocator,
)

from rmm._lib.device_buffer import DeviceBuffer
from rmm._lib.device_pointer import DevicePointer as _DevicePointer

# Initialize RMM on import, finalize RMM on process exit
_initialize()
_register_atexit_finalize()
