# SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import warnings

# This path is only taken for wheels where librmm is a separate Python package.
try:
    import librmm
except ModuleNotFoundError:
    pass
else:
    librmm.load_library()
    del librmm

from rmm import mr
from rmm._version import __git_commit__, __version__
from rmm.mr import disable_logging, enable_logging, get_log_filenames
from rmm.pylibrmm.device_buffer import DeviceBuffer
from rmm.pylibrmm.logger import (
    flush_logger,
    get_flush_level,
    get_logging_level,
    level_enum,
    set_flush_level,
    set_logging_level,
    should_log,
)
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
    "level_enum",
    "mr",
    "register_reinitialize_hook",
    "reinitialize",
    "set_flush_level",
    "set_logging_level",
    "should_log",
    "unregister_reinitialize_hook",
]
