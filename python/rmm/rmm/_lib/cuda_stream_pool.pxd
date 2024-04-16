# Copyright (c) 2021, NVIDIA CORPORATION.
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

cimport cython

from rmm._lib.cuda_stream_view cimport cuda_stream_view


cdef extern from "rmm/cuda_stream_pool.hpp" namespace "rmm" nogil:
    cdef cppclass cuda_stream_pool:
        cuda_stream_pool(size_t pool_size)
        cuda_stream_view get_stream()
        cuda_stream_view get_stream(size_t stream_id) except +
        size_t get_pool_size()
