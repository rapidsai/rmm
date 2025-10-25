# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from libcpp.string cimport string


cdef extern from "rapids_logger/logger.hpp" namespace "rapids_logger" nogil:
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


cdef extern from "rmm/logger.hpp" namespace "rmm" nogil:
    cdef logger& default_logger() except +
