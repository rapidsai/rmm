# Copyright (c) 2018-2024, NVIDIA CORPORATION.
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

import warnings

from rmm import mr
from rmm._version import __git_commit__, __version__
from rmm.mr import disable_logging, enable_logging, get_log_filenames
from rmm.pylibrmm.device_buffer import DeviceBuffer
from rmm.pylibrmm.logger import (
    flush_logger,
    get_flush_level,
    get_logging_level,
    set_flush_level,
    set_logging_level,
    should_log,
)

# TODO: Clean up after we remove legacy logging.
try:
    from rmm.pylibrmm.logger import logging_level

    logging_level_var = "logging_level"
except ImportError:
    from rmm.pylibrmm.logger import level_enum

    logging_level_var = "level_enum"

from rmm.rmm import (
    RMMError,
    is_initialized,
    register_reinitialize_hook,
    reinitialize,
    unregister_reinitialize_hook,
)

__all__ = [
    "DeviceBuffer",
    "disable_logging",
    "RMMError",
    "enable_logging",
    "flush_logger",
    "get_flush_level",
    "get_log_filenames",
    "get_logging_level",
    "is_initialized",
    logging_level_var,
    "mr",
    "register_reinitialize_hook",
    "reinitialize",
    "set_flush_level",
    "set_logging_level",
    "should_log",
    "unregister_reinitialize_hook",
]


def __getattr__(name):
    if name == "_lib":
        import importlib

        warnings.warn(
            "The `rmm._lib` module is deprecated in will be removed in a future release. Use `rmm.pylibrmm` instead.",
            FutureWarning,
            stacklevel=2,
        )

        module = importlib.import_module("rmm.pylibrmm")
        return module
    else:
        raise AttributeError(f"Module '{__name__}' has no attribute '{name}'")
