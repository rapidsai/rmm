# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.librmm._logger import level_enum

def should_log(level: level_enum) -> bool: ...
def set_logging_level(level: level_enum) -> None: ...
def get_logging_level() -> level_enum: ...
def flush_logger() -> None: ...
def set_flush_level(level: level_enum) -> None: ...
def get_flush_level() -> level_enum: ...

# explicitly export level_enum

__all__ = [
    "flush_logger",
    "get_flush_level",
    "get_logging_level",
    "level_enum",
    "set_flush_level",
    "set_logging_level",
    "should_log",
]
