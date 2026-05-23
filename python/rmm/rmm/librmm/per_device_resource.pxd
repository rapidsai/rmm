# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from rmm.librmm.memory_resource cimport any_resource, device_accessible


cdef extern from "rmm/mr/per_device_resource.hpp" namespace "rmm" nogil:
    cdef cppclass cuda_device_id:
        ctypedef int value_type

        cuda_device_id(value_type id)

        value_type value()

cdef extern from "rmm/mr/per_device_resource.hpp" \
        namespace "rmm::mr" nogil:
    cdef void set_per_device_resource(
        cuda_device_id id, any_resource[device_accessible] new_resource
    )
    cdef void set_current_device_resource(
        any_resource[device_accessible] new_resource
    )
