# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

cdef extern from "logger.hpp" nogil:
  ctypedef enum logging_level "logging_level":
      trace = 0,
      debug = 1,
      info = 2,
      warn = 3,
      err = 4,
      critical = 5,
      off = 6

  void set_logging_level(logging_level level) except +
  logging_level get_logging_level()
