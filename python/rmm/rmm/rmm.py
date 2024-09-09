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
from rmm import mr


# Utility Functions
class RMMError(Exception):
    def __init__(self, errcode, msg):
        self.errcode = errcode
        super(RMMError, self).__init__(msg)


_reinitialize_hooks = []


def reinitialize(
    pool_allocator=False,
    managed_memory=False,
    initial_pool_size=None,
    maximum_pool_size=None,
    devices=0,
    logging=False,
    log_file_name=None,
):
    """
    Finalizes and then initializes RMM using the options passed. Using memory
    from a previous initialization of RMM is undefined behavior and should be
    avoided.

    Parameters
    ----------
    pool_allocator : bool, default False
        If True, use a pool allocation strategy which can greatly improve
        performance.
    managed_memory : bool, default False
        If True, use managed memory for device memory allocation
    initial_pool_size : int | str, default None
        When `pool_allocator` is True, this indicates the initial pool size in
        bytes. By default, 1/2 of the total GPU memory is used.
        When `pool_allocator` is False, this argument is ignored if provided.
        A string argument is parsed using `parse_bytes`.
    maximum_pool_size : int | str, default None
        When `pool_allocator` is True, this indicates the maximum pool size in
        bytes. By default, the total available memory on the GPU is used.
        When `pool_allocator` is False, this argument is ignored if provided.
        A string argument is parsed using `parse_bytes`.
    devices : int or List[int], default 0
        GPU device  IDs to register. By default registers only GPU 0.
    logging : bool, default False
        If True, enable run-time logging of all memory events
        (alloc, free, realloc).
        This has a significant performance impact.
    log_file_name : str
        Name of the log file. If not specified, the environment variable
        ``RMM_LOG_FILE`` is used. A ``ValueError`` is thrown if neither is
        available. A separate log file is produced for each device, and the
        suffix `".dev{id}"` is automatically added to the log file name.

    Notes
    -----
    Note that if you use the environment variable ``CUDA_VISIBLE_DEVICES`` with
    logging enabled, the suffix may not be what you expect. For example, if you
    set ``CUDA_VISIBLE_DEVICES=1``, the log file produced will still have
    suffix ``0``. Similarly, if you set ``CUDA_VISIBLE_DEVICES=1,0`` and use
    devices 0 and 1, the log file with suffix ``0`` will correspond to the GPU
    with device ID ``1``. Use `rmm.get_log_filenames()` to get the log file
    names corresponding to each device.
    """
    for func, args, kwargs in reversed(_reinitialize_hooks):
        func(*args, **kwargs)

    mr._initialize(
        pool_allocator=pool_allocator,
        managed_memory=managed_memory,
        initial_pool_size=initial_pool_size,
        maximum_pool_size=maximum_pool_size,
        devices=devices,
        logging=logging,
        log_file_name=log_file_name,
    )


def is_initialized():
    """
    Returns True if RMM has been initialized, False otherwise.
    """
    return mr.is_initialized()


def register_reinitialize_hook(func, *args, **kwargs):
    """
    Add a function to the list of functions ("hooks") that will be
    called before :py:func:`~rmm.reinitialize()`.

    A user or library may register hooks to perform any necessary
    cleanup before RMM is reinitialized. For example, a library with
    an internal cache of objects that use device memory allocated by
    RMM can register a hook to release those references before RMM is
    reinitialized, thus ensuring that the relevant device memory
    resource can be deallocated.

    Hooks are called in the *reverse* order they are registered. This
    is useful, for example, when a library registers multiple hooks
    and needs them to run in a specific order for cleanup to be safe.
    Hooks cannot rely on being registered in a particular order
    relative to hooks registered by other packages, since that is
    determined by package import ordering.

    Parameters
    ----------
    func : callable
        Function to be called before :py:func:`~rmm.reinitialize()`
    args, kwargs
        Positional and keyword arguments to be passed to `func`
    """
    global _reinitialize_hooks
    _reinitialize_hooks.append((func, args, kwargs))
    return func


def unregister_reinitialize_hook(func):
    """
    Remove `func` from list of hooks that will be called before
    :py:func:`~rmm.reinitialize()`.

    If `func` was registered more than once, every instance of it will
    be removed from the list of hooks.
    """
    global _reinitialize_hooks
    _reinitialize_hooks = [x for x in _reinitialize_hooks if x[0] != func]
