# Copyright (c) 2020, NVIDIA CORPORATION.
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
from libcpp.memory cimport unique_ptr

from rmm._lib.lib cimport cudaStream_t

cdef extern from "rmm/cuda_stream_view.hpp" namespace "rmm" nogil:
    cdef cppclass cuda_stream_view:
        cuda_stream_view()
        cuda_stream_view(cudaStream_t)
        bool operator==(cuda_stream_view const)
        void synchronize() except +


cdef class CudaStreamView:
    cdef unique_ptr[cuda_stream_view] c_obj
