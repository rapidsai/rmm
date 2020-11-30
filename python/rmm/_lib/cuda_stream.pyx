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

from libc.stdint cimport uintptr_t
from libcpp cimport bool


cdef class CudaStream:
    """
    Wraps rmm::cuda_stream (an owning class).
    """
    def __cinit__(self):
        self.c_obj.reset(new cuda_stream())

    cpdef bool is_valid(self) except *:
        return self.c_obj.get()[0].is_valid()

    def value(self):
        return <uintptr_t><void *>self.c_obj.get()[0].value()
