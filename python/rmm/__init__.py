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

import atexit

from rmm import rmm_cfg
from rmm.rmm import (
    RMMError,
    auto_device,
    csv_log,
    device_array,
    device_array_from_ptr,
    device_array_like,
    finalize,
    get_ipc_handle,
    initialize,
    is_initialized,
    to_device,
)

initialize()
atexit.register(finalize)
