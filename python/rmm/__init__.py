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
    _finalize,
    _initialize,
    _make_finalizer,
    _register_atexit_finalize,
    auto_device,
    csv_log,
    device_array,
    device_array_from_ptr,
    device_array_like,
    get_ipc_handle,
    is_initialized,
    reinitialize,
    to_device,
)

from rmm._lib.device_buffer import DeviceBuffer

# Initialize RMM on import, finalize RMM on process exit
_initialize()
_register_atexit_finalize()
