# Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

from rmm import mr
from rmm._lib.device_buffer import DeviceBuffer
from rmm.mr import disable_logging, enable_logging, get_log_filenames
from rmm.rmm import (
    RMMError,
    is_initialized,
    register_reinitialize_hook,
    reinitialize,
    unregister_reinitialize_hook,
)

__all__ = [
    "DeviceBuffer",
    "RMMError",
    "disable_logging",
    "enable_logging",
    "get_log_filenames",
    "is_initialized",
    "mr",
    "register_reinitialize_hook",
    "reinitialize",
    "unregister_reinitialize_hook",
]

__version__ = "23.04.00"


_deprecated_names = {
    "rmm_cupy_allocator": "cupy",
    "rmm_torch_allocator": "torch",
    "RMMNumbaManager": "numba",
    "_numba_memory_manager": "numba",
}


def __getattr__(name):
    if name in _deprecated_names:
        import importlib
        import warnings

        package = _deprecated_names[name]
        warnings.warn(
            f"Use of 'rmm.{name}' is deprecated and will be removed. "
            f"'{name}' now lives in the 'rmm.allocators.{package}' sub-module, "
            "please update your imports.",
            FutureWarning,
        )
        module = importlib.import_module(
            f".allocators.{package}", package=__name__
        )
        return getattr(module, name)
    else:
        raise AttributeError(f"Module '{__name__}' has no attribute '{name}'")
