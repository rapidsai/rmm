from rmm._lib.memory_resource cimport device_memory_resource


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
