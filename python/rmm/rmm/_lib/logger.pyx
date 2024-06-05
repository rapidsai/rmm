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

from libcpp cimport bool


cdef extern from "spdlog/common.h" namespace "spdlog::level" nogil:
    cpdef enum logging_level "spdlog::level::level_enum":
        """
        The debug logging level for RMM.

        Debug logging prints messages to a log file. See
        `Debug Logging <https://github.com/rapidsai/rmm#debug-logging>`_
        for more information.

        Valid levels, in decreasing order of verbosity, are TRACE, DEBUG,
        INFO, WARN, ERR, CRITICAL, and OFF. Default is INFO.

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
        set_logging_level : Set the debug logging level
        get_logging_level : Get the current debug logging level
        """
        TRACE "spdlog::level::trace"
        DEBUG "spdlog::level::debug"
        INFO "spdlog::level::info"
        WARN "spdlog::level::warn"
        ERR "spdlog::level::err"
        CRITICAL "spdlog::level::critical"
        OFF "spdlog::level::off"


cdef extern from "spdlog/spdlog.h" namespace "spdlog" nogil:
    cdef cppclass spdlog_logger "spdlog::logger":
        spdlog_logger() except +
        void set_level(logging_level level)
        logging_level level()
        void flush() except +
        void flush_on(logging_level level)
        logging_level flush_level()
        bool should_log(logging_level msg_level)


cdef extern from "rmm/logger.hpp" namespace "rmm" nogil:
    cdef spdlog_logger& logger() except +


def _validate_level_type(level):
    if not isinstance(level, logging_level):
        raise TypeError("level must be an instance of the logging_level enum")


def should_log(level):
    """
    Check if a message at the given level would be logged.

    A message at the given level would be logged if the current debug logging
    level is set to a level that is at least as verbose than the given level,
    *and* the RMM module is compiled for a logging level at least as verbose.
    If these conditions are not both met, this function will return false.

    Debug logging prints messages to a log file. See
    `Debug Logging <https://github.com/rapidsai/rmm#debug-logging>`_
    for more information.

    Parameters
    ----------
    level : logging_level
        The debug logging level. Valid values are instances of the
        ``logging_level`` enum.

    Returns
    -------
    should_log : bool
        True if a message at the given level would be logged, False otherwise.

    Raises
    ------
    TypeError
        If the logging level is not an instance of the ``logging_level`` enum.
    """
    _validate_level_type(level)
    return logger().should_log(level)


def set_logging_level(level):
    """
    Set the debug logging level.

    Debug logging prints messages to a log file. See
    `Debug Logging <https://github.com/rapidsai/rmm#debug-logging>`_
    for more information.

    Parameters
    ----------
    level : logging_level
        The debug logging level. Valid values are instances of the
        ``logging_level`` enum.

    Raises
    ------
    TypeError
        If the logging level is not an instance of the ``logging_level`` enum.

    See Also
    --------
    get_logging_level : Get the current debug logging level.

    Examples
    --------
    >>> import rmm
    >>> rmm.set_logging_level(rmm.logging_level.WARN) # set logging level to warn
    """
    _validate_level_type(level)
    logger().set_level(level)

    if not should_log(level):
        warnings.warn(f"RMM will not log logging_level.{level.name}. This "
                      "may be because the C++ library is compiled for a "
                      "less-verbose logging level.")


def get_logging_level():
    """
    Get the current debug logging level.

    Debug logging prints messages to a log file. See
    `Debug Logging <https://github.com/rapidsai/rmm#debug-logging>`_
    for more information.

    Returns
    -------
    level : logging_level
        The current debug logging level, an instance of the ``logging_level``
        enum.

    See Also
    --------
    set_logging_level : Set the debug logging level.

    Examples
    --------
    >>> import rmm
    >>> rmm.get_logging_level() # get current logging level
    <logging_level.INFO: 2>
    """
    return logging_level(logger().level())


def flush_logger():
    """
    Flush the debug logger. This will cause any buffered log messages to
    be written to the log file.

    Debug logging prints messages to a log file. See
    `Debug Logging <https://github.com/rapidsai/rmm#debug-logging>`_
    for more information.

    See Also
    --------
    set_flush_level : Set the flush level for the debug logger.
    get_flush_level : Get the current debug logging flush level.

    Examples
    --------
    >>> import rmm
    >>> rmm.flush_logger() # flush the logger
    """
    logger().flush()


def set_flush_level(level):
    """
    Set the flush level for the debug logger. Messages of this level or higher
    will automatically flush to the file.

    Debug logging prints messages to a log file. See
    `Debug Logging <https://github.com/rapidsai/rmm#debug-logging>`_
    for more information.

    Parameters
    ----------
    level : logging_level
        The debug logging level. Valid values are instances of the
        ``logging_level`` enum.

    Raises
    ------
    TypeError
        If the logging level is not an instance of the ``logging_level`` enum.

    See Also
    --------
    get_flush_level : Get the current debug logging flush level.
    flush_logger : Flush the logger.

    Examples
    --------
    >>> import rmm
    >>> rmm.flush_on(rmm.logging_level.WARN) # set flush level to warn
    """
    _validate_level_type(level)
    logger().flush_on(level)

    if not should_log(level):
        warnings.warn(f"RMM will not log logging_level.{level.name}. This "
                      "may be because the C++ library is compiled for a "
                      "less-verbose logging level.")


def get_flush_level():
    """
    Get the current debug logging flush level for the RMM logger. Messages of
    this level or higher will automatically flush to the file.

    Debug logging prints messages to a log file. See
    `Debug Logging <https://github.com/rapidsai/rmm#debug-logging>`_
    for more information.

    Returns
    -------
    logging_level
        The current flush level, an instance of the ``logging_level``
        enum.

    See Also
    --------
    set_flush_level : Set the flush level for the logger.
    flush_logger : Flush the logger.

    Examples
    --------
    >>> import rmm
    >>> rmm.flush_level() # get current flush level
    <logging_level.INFO: 2>
    """
    return logging_level(logger().flush_level())
