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

import warnings
from enum import Enum, auto

from libcpp cimport bool


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
        void set_level(level_enum log_level)
        level_enum level()
        void flush() except +
        void flush_on(level_enum log_level)
        level_enum flush_level()
        bool should_log(level_enum msg_level)


cdef extern from "rmm/logger.hpp" namespace "rmm" nogil:
    cdef spdlog_logger& logger() except +


class logging_level(Enum):
    """
    The debug logging level for the RMM library.

    Debug logging prints messages to a log file. See
    `Debug Logging https://github.com/rapidsai/rmm#debug-logging`_ for more
    information.

    Valid levels, in decreasing order of verbosity, are TRACE, DEBUG, INFO,
    WARN, ERR, CRITICAL, and OFF. Default is INFO.

    Examples
    --------
    >>> import rmm
    >>> rmm.logging_level.DEBUG
    <logging_level.DEBUG: 1>
    >>> rmm.logging_level.DEBUG.value
    1
    >>> rmm.logging_level.DEBUG.name
    'DEBUG'

    See Also
    --------
    set_logging_level : Set the debug logging level for the RMM library.
    get_logging_level : Get the current debug logging level for the RMM
    """
    TRACE=0
    DEBUG=auto()
    INFO=auto()
    WARN=auto()
    ERR=auto()
    CRITICAL=auto()
    OFF=auto()


def _normalize_logging_level(level):
    """
    Normalize the logging level to an integer.

    Parameters
    ----------
    level : logging_level, int or str
        The debug logging level. Valid values are instances of the
        ``logging_level`` enum or any name (case-insensitive) or value of the
        enum.

    Returns
    -------
    level : int
        The logging level as an integer.

    Raises
    ------
    ValueError
        If the logging level is invalid.
    TypeError
        If the logging level is not an instance of the ``logging_level`` enum,
        an integer or a string.
    """
    if (isinstance(level, str)):
        return logging_level[level.upper()].value
    else:
        return logging_level(level).value


def should_log(level):
    """
    Check if a message at the given level would be logged.

    Debug logging prints messages to a log file. See
    `Debug Logging https://github.com/rapidsai/rmm#debug-logging`_ for more
    information.

    Parameters
    ----------
    level : logging_level, int or str
        The debug logging level. Valid values are instances of the
        ``logging_level`` enum or any name (case-insensitive) or value of the
        enum.

    Returns
    -------
    should_log : bool
        True if a message at the given level would be logged, False otherwise.

    Raises
    ------
    ValueError
        If the logging level is invalid.
    TypeError
        If the logging level is not an instance of the ``logging_level`` enum,
        an integer or a string.
    """
    return logger().should_log(_normalize_logging_level(level))


def set_logging_level(level):
    """
    Set the debug logging level for the RMM library.

    Debug logging prints messages to a log file. See
    `Debug Logging https://github.com/rapidsai/rmm#debug-logging`_ for more
    information.

    Parameters
    ----------
    level : logging_level, int or str
        The debug logging level. Valid values are instances of the
        ``logging_level`` enum or any name (case-insensitive) or value of the
        enum.

    Raises
    ------
    ValueError
        If the logging level is invalid.
    TypeError
        If the logging level is not an instance of the ``logging_level`` enum,
        an integer or a string.

    See Also
    --------
    get_logging_level : Get the current debug logging level for RMM.

    Examples
    --------
    >>> import rmm
    >>> rmm.set_logging_level("debug") # set logging level to debug
    >>> rmm.set_logging_level(3) # set logging level to warn
    >>> rmm.set_logging_level(rmm.logging_level.WARN) # set logging level to warn
    """
    level = logging_level(_normalize_logging_level(level))
    logger().set_level(level.value)

    if not should_log(level):
        warnings.warn(f"RMM will not log warning level {level.name}. This "
                      "may be because the C++ library is compiled for a "
                      "less-verbose logging level.")


def get_logging_level():
    """
    Get the current debug logging level for the RMM library.

    Debug logging prints messages to a log file. See
    `Debug Logging https://github.com/rapidsai/rmm#debug-logging`_ for more
    information.

    Returns
    -------
    level : logging_level
        The current debug logging level, an instance of the ``logging_level``
        enum.

    See Also
    --------
    set_logging_level : Set the debug logging level for the RMM library.

    Examples
    --------
    >>> import rmm
    >>> rmm.get_logging_level() # get current logging level
    <logging_level.INFO: 2>
    """
    return logging_level(logger().level())


def flush_logger():
    """
    Flush the RMM debug logger. This will cause any buffered log messages to be
    written immediately to the log file.

    Debug logging prints messages to a log file. See
    `Debug Logging https://github.com/rapidsai/rmm#debug-logging`_ for more
    information.

    Examples
    --------
    >>> import rmm
    >>> rmm.flush_logger() # flush the logger
    """
    logger().flush()


def set_flush_level(level):
    """
    Set the debug logging flush level for the RMM logger. Messages of this
    level or higher will automatically flush to the file.

    Debug logging prints messages to a log file. See
    `Debug Logging https://github.com/rapidsai/rmm#debug-logging`_ for more
    information.

    Parameters
    ----------
    level : int or str
        The flush level. Valid values are instances of the
        ``logging_level`` enum or any name (case-insensitive) or value of the
        enum.

    Raises
    ------
    ValueError
        If the logging level is invalid.
    TypeError
        If the logging level is not an instance of the ``logging_level`` enum,
        an integer or a string.

    See Also
    --------
    get_logging_level : Get the current debug logging level for the RMM
    set_logging_level : Set the debug logging level for the RMM library.

    Examples
    --------
    >>> import rmm
    >>> rmm.flush_on("debug") # set flush level to debug
    >>> rmm.flush_on(3) # set flush level to warn
    >>> rmm.flush_on(rmm.logging_level.WARN) # set flush level to warn
    """
    level = logging_level(_normalize_logging_level(level))
    logger().flush_on(level.value)

    if not should_log(level):
        warnings.warn(f"RMM will not log warning level {level.name}. This "
                      "may be because the C++ library is compiled for a "
                      "less-verbose logging level.")


def get_flush_level():
    """
    Get the current debug logging flush level for the RMM logger. Messages of
    this level or higher will automatically flush to the file.

    Debug logging prints messages to a log file. See
    `Debug Logging https://github.com/rapidsai/rmm#debug-logging`_ for more
    information.

    Returns
    -------
    level : logging_level
        The current flush level, an instance of the ``logging_level``
        enum.

    See Also
    --------
    set_flush_level : Set the flush level for the RMM logger.
    set_logging_level : Set the debug logging level for the RMM library.

    Examples
    --------
    >>> import rmm
    >>> rmm.flush_level() # get current flush level
    <logging_level.INFO: 2>
    """
    return logging_level(logger().flush_level())
