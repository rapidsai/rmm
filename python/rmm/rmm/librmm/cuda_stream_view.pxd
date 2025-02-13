# Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

from cuda.bindings.cyruntime cimport cudaStream_t
from libcpp cimport bool


cdef extern from "rmm/cuda_stream_view.hpp" namespace "rmm" nogil:
    cdef cppclass cuda_stream_view:
        cuda_stream_view()
        cuda_stream_view(cudaStream_t)
        cudaStream_t value()
        bool is_default()
        bool is_per_thread_default()
        void synchronize() except +

    cdef bool operator==(cuda_stream_view const, cuda_stream_view const)

    const cuda_stream_view cuda_stream_default
    const cuda_stream_view cuda_stream_legacy
    const cuda_stream_view cuda_stream_per_thread
