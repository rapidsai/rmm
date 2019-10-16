cdef extern from "rmm/device_scalar.hpp" namespace "rmm" nogil:
    cdef cppclass device_scalar[T]:
        device_scalar(T initial_value)
        device_scalar()
        device_scalar(device_scalar&)
        T value()
        T* get()
