# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from rmm.librmm.memory_resource cimport device_memory_resource


cdef extern from "rmm/resource_ref.hpp" namespace "rmm" nogil:
    cdef cppclass device_async_resource_ref:
        device_async_resource_ref(device_memory_resource&)


cdef extern from "rmm/mr/per_device_resource.hpp" namespace "rmm" nogil:
    cdef cppclass cuda_device_id:
        ctypedef int value_type

        cuda_device_id(value_type id)

        value_type value()

cdef extern from "rmm/mr/per_device_resource.hpp" \
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

    # resource_ref-based APIs
    cdef device_async_resource_ref set_current_device_resource_ref(
        device_async_resource_ref new_resource_ref
    )
    cdef device_async_resource_ref get_current_device_resource_ref()
    cdef device_async_resource_ref set_per_device_resource_ref(
        cuda_device_id device_id, device_async_resource_ref new_resource_ref
    )
    cdef device_async_resource_ref get_per_device_resource_ref(
        cuda_device_id device_id
    )
