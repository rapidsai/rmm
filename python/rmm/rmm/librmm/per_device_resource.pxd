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
from rmm.librmm.memory_resource cimport device_memory_resource


cdef extern from "rmm/mr/device/per_device_resource.hpp" namespace "rmm" nogil:
    cdef cppclass cuda_device_id:
        ctypedef int value_type

        cuda_device_id(value_type id)

        value_type value()

cdef extern from "rmm/mr/device/per_device_resource.hpp" \
        namespace "rmm::mr" nogil:
    cdef device_memory_resource* set_current_device_resource(
        device_memory_resource* new_mr
    )
    cdef device_memory_resource* get_current_device_resource()
    cdef device_memory_resource* set_per_device_resource(
        cuda_device_id id, device_memory_resource* new_mr
    )
    cdef device_memory_resource* get_per_device_resource (
        cuda_device_id id
    )
