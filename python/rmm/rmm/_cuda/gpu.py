# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from cuda.bindings import driver, runtime


class CUDARuntimeError(RuntimeError):
    def __init__(self, status: runtime.cudaError_t):
        self.status = status

        _, name = runtime.cudaGetErrorName(status)
        _, msg = runtime.cudaGetErrorString(status)

        super(CUDARuntimeError, self).__init__(
            f"{name.decode()}: {msg.decode()}"
        )

    def __reduce__(self):
        return (type(self), (self.status,))


class CUDADriverError(RuntimeError):
    def __init__(self, status: driver.CUresult):
        self.status = status

        _, name = driver.cuGetErrorName(status)
        _, msg = driver.cuGetErrorString(status)

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
    status, version = runtime.cudaDriverGetVersion()
    if status != runtime.cudaError_t.cudaSuccess:
        raise CUDARuntimeError(status)
    return version


def getDevice():
    """
    Get the current CUDA device
    """
    status, device = runtime.cudaGetDevice()
    if status != runtime.cudaError_t.cudaSuccess:
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
    (status,) = runtime.cudaSetDevice(device)
    if status != runtime.cudaError_t.cudaSuccess:
        raise CUDARuntimeError(status)


def runtimeGetVersion():
    """
    Returns the version number of the local CUDA runtime.

    The version is returned as ``(1000 * major + 10 * minor)``. For example,
    CUDA 12.5 would be represented by 12050.

    This function automatically raises CUDARuntimeError with error message
    and status code.
    """
    status, version = runtime.getLocalRuntimeVersion()
    if status != runtime.cudaError_t.cudaSuccess:
        raise CUDARuntimeError(status)
    return version


def getDeviceCount():
    """
    Returns the number of devices with compute capability greater or
    equal to 2.0 that are available for execution.

    This function automatically raises CUDARuntimeError with error message
    and status code.
    """
    status, count = runtime.cudaGetDeviceCount()
    if status != runtime.cudaError_t.cudaSuccess:
        raise CUDARuntimeError(status)
    return count


def getDeviceAttribute(attr: runtime.cudaDeviceAttr, device: int):
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
    status, value = runtime.cudaDeviceGetAttribute(attr, device)
    if status != runtime.cudaError_t.cudaSuccess:
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
    status, prop = runtime.cudaGetDeviceProperties(device)
    if status != runtime.cudaError_t.cudaSuccess:
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

    status, device_name = driver.cuDeviceGetName(256, driver.CUdevice(device))
    if status != driver.CUresult.CUDA_SUCCESS:
        raise CUDADriverError(status)
    return device_name.decode()
