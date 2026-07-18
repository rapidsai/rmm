# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for FailureCallbackResourceAdaptor."""

import pytest

import rmm


def test_failure_callback_resource_adaptor():
    allocation_calls = 0
    callback_calls = 0

    def callback(nbytes: int) -> bool:
        nonlocal callback_calls
        callback_calls += 1
        return callback_calls == 1

    def allocate_func(stream, size, alignment):
        nonlocal allocation_calls
        allocation_calls += 1
        raise MemoryError("Intentional allocation failure")

    def deallocate_func(stream, ptr, size, alignment):
        pass

    failing_mr = rmm.mr.CallbackMemoryResource(allocate_func, deallocate_func)
    mr = rmm.mr.FailureCallbackResourceAdaptor(failing_mr, callback)
    rmm.mr.set_current_device_resource(mr)

    with pytest.raises(MemoryError):
        rmm.DeviceBuffer(size=256)
    assert allocation_calls == 2
    assert callback_calls == 2


def test_failure_callback_resource_adaptor_error():
    def callback(nbytes: int) -> bool:
        raise RuntimeError("MyError")

    def allocate_func(stream, size, alignment):
        raise MemoryError("Intentional allocation failure")

    def deallocate_func(stream, ptr, size, alignment):
        pass

    failing_mr = rmm.mr.CallbackMemoryResource(allocate_func, deallocate_func)
    mr = rmm.mr.FailureCallbackResourceAdaptor(failing_mr, callback)
    rmm.mr.set_current_device_resource(mr)

    with pytest.raises(RuntimeError, match="MyError"):
        rmm.DeviceBuffer(size=256)
