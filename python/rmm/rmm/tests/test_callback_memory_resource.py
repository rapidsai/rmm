# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CallbackMemoryResource."""

import functools
import gc
import sys

import pytest

import rmm


def test_custom_mr(capsys):
    allocation_size = 257
    base_mr = rmm.mr.CudaMemoryResource()
    allocations = []
    deallocations = []

    def allocate_func(stream, size, alignment):
        print(f"Allocating {size} bytes")
        ptr = base_mr.allocate(size, stream)
        allocations.append((stream, ptr, size, alignment))
        return ptr

    def deallocate_func(stream, ptr, size, alignment):
        print(f"Deallocating {size} bytes")
        deallocations.append((stream, ptr, size, alignment))
        return base_mr.deallocate(ptr, size, stream)

    rmm.mr.set_current_device_resource(
        rmm.mr.CallbackMemoryResource(allocate_func, deallocate_func)
    )

    rmm.DeviceBuffer(size=allocation_size)

    captured = capsys.readouterr()
    assert captured.out == "Allocating 257 bytes\nDeallocating 257 bytes\n"
    assert len(allocations) == 1
    assert allocations[0][2] == allocation_size
    assert allocations[0][3] == 256
    assert deallocations == allocations


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

    def allocate_func(stream, size, alignment):
        raise err_raise("My alloc error")

    def deallocate_func(stream, ptr, size, alignment):
        return base_mr.deallocate(ptr, size)

    rmm.mr.set_current_device_resource(
        rmm.mr.CallbackMemoryResource(allocate_func, deallocate_func)
    )

    with pytest.raises(err_catch, match="My alloc error"):
        rmm.DeviceBuffer(size=256)


def test_callback_mr_python_api_uses_default_alignment():
    """Python allocation APIs do not expose alignment, so C++ covers values above 256."""
    alignments = []
    base_mr = rmm.mr.CudaMemoryResource()

    def allocate_func(stream, size, alignment):
        alignments.append(alignment)
        return base_mr.allocate(size, stream)

    def deallocate_func(stream, ptr, size, alignment):
        alignments.append(alignment)
        return base_mr.deallocate(ptr, size, stream)

    cb_mr = rmm.mr.CallbackMemoryResource(allocate_func, deallocate_func)
    buf = rmm.DeviceBuffer(size=1, mr=cb_mr)
    del buf

    assert alignments == [256, 256]


def test_callback_mr_reports_deallocation_errors(monkeypatch):
    base_mr = rmm.mr.CudaMemoryResource()
    unraisable = []

    def allocate_func(stream, size, alignment):
        return base_mr.allocate(size, stream)

    def deallocate_func(stream, ptr, size, alignment):
        base_mr.deallocate(ptr, size, stream)
        raise RuntimeError("My dealloc error")

    monkeypatch.setattr(sys, "unraisablehook", unraisable.append)
    cb_mr = rmm.mr.CallbackMemoryResource(allocate_func, deallocate_func)
    buf = rmm.DeviceBuffer(size=1, mr=cb_mr)
    del buf

    assert len(unraisable) == 1
    assert isinstance(unraisable[0].exc_value, RuntimeError)
    assert str(unraisable[0].exc_value) == "My dealloc error"


def test_device_buffer_with_mr():
    allocations = []
    base = rmm.mr.CudaMemoryResource()
    rmm.mr.set_current_device_resource(base)

    def alloc_cb(stream, size, alignment, *, base):
        allocations.append(size)
        return base.allocate(size, stream)

    def dealloc_cb(stream, ptr, size, alignment, *, base):
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
