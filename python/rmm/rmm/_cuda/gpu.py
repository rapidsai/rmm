# Copyright (c) 2020, NVIDIA CORPORATION.

from cuda import cuda, cudart


class CUDARuntimeError(RuntimeError):
    def __init__(self, status: cudart.cudaError_t):
        self.status = status

        _, name = cudart.cudaGetErrorName(status)
        _, msg = cudart.cudaGetErrorString(status)

        super(CUDARuntimeError, self).__init__(
            f"{name.decode()}: {msg.decode()}"
        )

    def __reduce__(self):
        return (type(self), (self.status,))


class CUDADriverError(RuntimeError):
    def __init__(self, status: cuda.CUresult):
        self.status = status

        _, name = cuda.cuGetErrorName(status)
        _, msg = cuda.cuGetErrorString(status)

        super(CUDADriverError, self).__init__(
            f"{name.decode()}: {msg.decode()}"
        )

    def __reduce__(self):
        return (type(self), (self.status,))


def driverGetVersion():
    """
    Returns in the latest version of CUDA supported by the driver.
    The version is returned as (1000 major + 10 minor). For example,
    CUDA 9.2 would be represented by 9020. If no driver is installed,
    then 0 is returned as the driver version.

    This function automatically raises CUDARuntimeError with error message
    and status code.
    """
    status, version = cudart.cudaDriverGetVersion()
    if status != cudart.cudaError_t.cudaSuccess:
        raise CUDARuntimeError(status)
    return version


def getDevice():
    """
    Get the current CUDA device
    """
    status, device = cudart.cudaGetDevice()
    if status != cudart.cudaError_t.cudaSuccess:
        raise CUDARuntimeError(status)
    return device


def setDevice(device: int):
    """
    Set the current CUDA device
    Parameters
    ----------
    device : int
        The ID of the device to set as current
    """
    (status,) = cudart.cudaSetDevice(device)
    if status != cudart.cudaError_t.cudaSuccess:
        raise CUDARuntimeError(status)


def runtimeGetVersion():
    """
    Returns the version number of the current CUDA Runtime instance.
    The version is returned as (1000 major + 10 minor). For example,
    CUDA 9.2 would be represented by 9020.

    This calls numba.cuda.runtime.get_version() rather than cuda-python due to
    current limitations in cuda-python.
    """
    # TODO: Replace this with `cuda.cudart.cudaRuntimeGetVersion()` when the
    # limitation is fixed.
    import numba.cuda

    major, minor = numba.cuda.runtime.get_version()
    return major * 1000 + minor * 10


def getDeviceCount():
    """
    Returns the number of devices with compute capability greater or
    equal to 2.0 that are available for execution.

    This function automatically raises CUDARuntimeError with error message
    and status code.
    """
    status, count = cudart.cudaGetDeviceCount()
    if status != cudart.cudaError_t.cudaSuccess:
        raise CUDARuntimeError(status)
    return count


def getDeviceAttribute(attr: cudart.cudaDeviceAttr, device: int):
    """
    Returns information about the device.

    Parameters
    ----------
        attr : cudaDeviceAttr
            Device attribute to query
        device : int
            Device number to query

    This function automatically raises CUDARuntimeError with error message
    and status code.
    """
    status, value = cudart.cudaDeviceGetAttribute(attr, device)
    if status != cudart.cudaError_t.cudaSuccess:
        raise CUDARuntimeError(status)
    return value


def getDeviceProperties(device: int):
    """
    Returns information about the compute-device.

    Parameters
    ----------
        device : int
            Device number to query

    This function automatically raises CUDARuntimeError with error message
    and status code.
    """
    status, prop = cudart.cudaGetDeviceProperties(device)
    if status != cudart.cudaError_t.cudaSuccess:
        raise CUDARuntimeError(status)
    return prop


def deviceGetName(device: int):
    """
    Returns an identifier string for the device.

    Parameters
    ----------
        device : int
            Device number to query

    This function automatically raises CUDADriverError with error message
    and status code.
    """

    status, device_name = cuda.cuDeviceGetName(256, cuda.CUdevice(device))
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise CUDADriverError(status)
    return device_name.decode()
