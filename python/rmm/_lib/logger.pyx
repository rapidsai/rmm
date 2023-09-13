# Copyright (c) 2023, NVIDIA CORPORATION.
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


cdef extern from "spdlog/common.h" namespace "spdlog::level" nogil:
    cdef enum level_enum:
        trace = 0
        debug = 1
        info = 2
        warn = 3
        err = 4
        critical = 5
        off = 6


cdef extern from "spdlog/spdlog.h" namespace "spdlog" nogil:
    cdef cppclass spdlog_logger "spdlog::logger":
        spdlog_logger() except +
        void set_level(level_enum log_level) except +
        level_enum level() except +
        void flush() except +
        void flush_on(level_enum log_level) except +
        level_enum flush_level() except +


cdef extern from "rmm/logger.hpp" namespace "rmm" nogil:
    cdef spdlog_logger& logger() except +


logging_levels = [
    "trace",
    "debug",
    "info",
    "warn",
    "err",
    "critical",
    "off",
]


def _normalize_logging_level(level):
    if isinstance(level, str):
        try:
            level = logging_levels.index(level.lower())
        except ValueError:
            raise ValueError(
                f"Invalid logging level '{level}'. Valid levels are "
                f"{logging_levels}."
            )
    elif not isinstance(level, int):
        raise TypeError(
            f"Logging level must be an integer or string, not {type(level)}"
        )
    else:
        if level < 0 or level >= len(logging_levels):
            raise ValueError(
                f"Logging level must be between 0 and {len(logging_levels)-1}"
            )

    return level


def set_logging_level(level):
    """
    Set the debug logging level for the RMM library.

    Parameters
    ----------
    level : int or str
        The debug logging level. Valid string names are (in decreasing order
        of verbosity) "trace", "debug", "info", "warn", "err", "critical",
        and "off", corresponding to valid integer levels 0 through 6. Default
        is 2 (info).

    See Also
    --------
    get_logging_level : Get the current debug logging level for the RMM

    Examples
    --------
    >>> import rmm
    >>> rmm.set_logging_level("debug") # set logging level to debug
    >>> rmm.set_logging_level(3) # set logging level to warn
    """
    logger().set_level(_normalize_logging_level(level))


def get_logging_level():
    """
    Get the current debug logging level for the RMM library.

    Returns
    -------
    level : int
    The current debug logging level. Valid values are 0 through 6 inclusive,
    where 0 is the most verbose and 6 is the least verbose. Default is 2
    (info).

    See Also
    --------
    set_logging_level : Set the debug logging level for the RMM library.

    Examples
    --------
    >>> import rmm
    >>> rmm.get_logging_level() # get current logging level
    2
    """
    return logger().level()


def flush_logger():
    """
    Flush the RMM logger.

    See Also
    --------
    set_logging_level : Set the debug logging level for the RMM library.

    Examples
    --------
    >>> import rmm
    >>> rmm.flush_logger() # flush the logger
    """
    logger().flush()


def set_flush_level(level):
    """
    Set the flush level for the RMM logger.

    Parameters
    ----------
    level : int or str
        The flush level. Valid string names are (in decreasing order
        of verbosity) "trace", "debug", "info", "warn", "err", "critical",
        and "off", corresponding to valid integer levels 0 through 6. Default
        is 2 (info).

    See Also
    --------
    get_logging_level : Get the current debug logging level for the RMM

    Examples
    --------
    >>> import rmm
    >>> rmm.flush_on("debug") # set flush level to debug
    >>> rmm.flush_on(3) # set flush level to warn
    """
    logger().flush_on(_normalize_logging_level(level))


def get_flush_level():
    """
    Get the current flush level for the RMM library.

    Returns
    -------
    level : int
    The current flush level. Valid values are 0 through 6 inclusive,
    where 0 is the most verbose and 6 is the least verbose. Default is 2
    (info).

    See Also
    --------
    set_logging_level : Set the debug logging level for the RMM library.

    Examples
    --------
    >>> import rmm
    >>> rmm.flush_level() # get current flush level
    2
    """
    return logger().flush_level()
