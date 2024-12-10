# Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
from cuda.bindings.cyruntime cimport cudaStream_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr

from rmm.librmm.cuda_stream cimport cuda_stream


@cython.final
cdef class CudaStream:
    cdef unique_ptr[cuda_stream] c_obj
    cdef cudaStream_t value(self) except * nogil
    cdef bool is_valid(self) except * nogil
