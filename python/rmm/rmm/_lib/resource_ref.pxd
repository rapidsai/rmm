cdef extern from "rmm/resource_ref.hpp" namespace "rmm" nogil:
    cdef cppclass device_resource_ref:
        pass
    cdef cppclass device_async_resource_ref:
        pass
    cdef cppclass host_resource_ref:
        pass
    cdef cppclass host_async_resource_ref:
        pass
    cdef cppclass host_device_resource_ref:
        pass
    cdef cppclass host_device_async_resource_ref:
        pass