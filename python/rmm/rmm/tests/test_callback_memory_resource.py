# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CallbackMemoryResource."""

import functools
import gc

import pytest

import rmm


def test_custom_mr(capsys):
    base_mr = rmm.mr.CudaMemoryResource()

    def allocate_func(size, stream):
        print(f"Allocating {size} bytes")
        return base_mr.allocate(size, stream)

    def deallocate_func(ptr, size, stream):
        print(f"Deallocating {size} bytes")
        return base_mr.deallocate(ptr, size, stream)

    rmm.mr.set_current_device_resource(
        rmm.mr.CallbackMemoryResource(allocate_func, deallocate_func)
    )

    rmm.DeviceBuffer(size=256)

    captured = capsys.readouterr()
    assert captured.out == "Allocating 256 bytes\nDeallocating 256 bytes\n"


@pytest.mark.parametrize(
    "err_raise,err_catch",
    [
        (MemoryError, MemoryError),
        (RuntimeError, RuntimeError),
        (Exception, RuntimeError),
        (BaseException, RuntimeError),
    ],
)
def test_callback_mr_error(err_raise, err_catch):
    base_mr = rmm.mr.CudaMemoryResource()

    def allocate_func(size, stream):
        raise err_raise("My alloc error")

    def deallocate_func(ptr, size, stream):
        return base_mr.deallocate(ptr, size)

    rmm.mr.set_current_device_resource(
        rmm.mr.CallbackMemoryResource(allocate_func, deallocate_func)
    )

    with pytest.raises(err_catch, match="My alloc error"):
        rmm.DeviceBuffer(size=256)


def test_device_buffer_with_mr():
    allocations = []
    base = rmm.mr.CudaMemoryResource()
    rmm.mr.set_current_device_resource(base)

    def alloc_cb(size, stream, *, base):
        allocations.append(size)
        return base.allocate(size, stream)

    def dealloc_cb(ptr, size, stream, *, base):
        return base.deallocate(ptr, size, stream)

    cb_mr = rmm.mr.CallbackMemoryResource(
        functools.partial(alloc_cb, base=base),
        functools.partial(dealloc_cb, base=base),
    )
    rmm.DeviceBuffer(size=10)
    assert len(allocations) == 0
    buf = rmm.DeviceBuffer(size=256, mr=cb_mr)
    assert len(allocations) == 1
    assert allocations[0] == 256
    del cb_mr
    gc.collect()
    del buf
