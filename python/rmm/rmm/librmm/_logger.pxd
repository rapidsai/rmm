# Copyright (c) 2023-2025, NVIDIA CORPORATION.
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
