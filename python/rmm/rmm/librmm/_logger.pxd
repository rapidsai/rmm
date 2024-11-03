# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

from libcpp cimport bool
from libcpp.string cimport string

# Note: This must match the LOGGING_COMPATIBILITY setting in CMakeLists.txt for
# the C++ build or the build will fail since the expected symbols will not exist.
DEF LOGGING_COMPATIBILITY = False

# Conditional compilation in Cython is deprecated, but we will remove the need
# for it here before that becomes an issue; this conditional just exists to
# smooth the transition.
# https://docs.cython.org/en/latest/src/userguide/language_basics.html#conditional-compilation
IF LOGGING_COMPATIBILITY:
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

    cdef extern from "rmm/logger.hpp" namespace "rmm::detail" nogil:
        cdef spdlog_logger& logger() except +
ELSE:
    cdef extern from "rmm/logger.hpp" namespace "rmm" nogil:
        cpdef enum class level_enum:
            trace
            debug
            info
            warn
            error
            critical
            off
            n_levels

        cdef cppclass logger:
            logger(string name, string filename) except +
            void set_level(level_enum log_level) except +
            level_enum level() except +
            void flush() except +
            void flush_on(level_enum level) except +
            level_enum flush_level() except +
            bool should_log(level_enum msg_level) except +

        cdef logger& default_logger() except +
