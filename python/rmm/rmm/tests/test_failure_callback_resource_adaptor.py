# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Tests for FailureCallbackResourceAdaptor."""

import pytest

import rmm


def test_failure_callback_resource_adaptor():
    retried = [False]

    def callback(nbytes: int) -> bool:
        if retried[0]:
            return False
        else:
            retried[0] = True
            return True

    def allocate_func(size, stream):
        raise MemoryError("Intentional allocation failure")

    def deallocate_func(ptr, size, stream):
        pass

    failing_mr = rmm.mr.CallbackMemoryResource(allocate_func, deallocate_func)
    mr = rmm.mr.FailureCallbackResourceAdaptor(failing_mr, callback)
    rmm.mr.set_current_device_resource(mr)

    with pytest.raises(MemoryError):
        rmm.DeviceBuffer(size=256)
    assert retried[0]


def test_failure_callback_resource_adaptor_error():
    def callback(nbytes: int) -> bool:
        raise RuntimeError("MyError")

    def allocate_func(size, stream):
        raise MemoryError("Intentional allocation failure")

    def deallocate_func(ptr, size, stream):
        pass

    failing_mr = rmm.mr.CallbackMemoryResource(allocate_func, deallocate_func)
    mr = rmm.mr.FailureCallbackResourceAdaptor(failing_mr, callback)
    rmm.mr.set_current_device_resource(mr)

    with pytest.raises(RuntimeError, match="MyError"):
        rmm.DeviceBuffer(size=256)
