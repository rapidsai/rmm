# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Tests for TrackingResourceAdaptor."""

import gc

import rmm


def test_tracking_resource_adaptor():
    cuda_mr = rmm.mr.CudaMemoryResource()

    mr = rmm.mr.TrackingResourceAdaptor(cuda_mr, capture_stacks=True)

    rmm.mr.set_current_device_resource(mr)

    buffers = [rmm.DeviceBuffer(size=1000) for _ in range(10)]

    for i in range(9, 0, -2):
        del buffers[i]

    assert mr.get_allocated_bytes() == 5000

    # Push a new Tracking adaptor
    mr2 = rmm.mr.TrackingResourceAdaptor(mr, capture_stacks=True)
    rmm.mr.set_current_device_resource(mr2)

    for _ in range(2):
        buffers.append(rmm.DeviceBuffer(size=1000))

    assert mr2.get_allocated_bytes() == 2000
    assert mr.get_allocated_bytes() == 7000

    # Ensure we get back a non-empty string for the allocations
    assert len(mr.get_outstanding_allocations_str()) > 0

    del buffers
    gc.collect()

    assert mr2.get_allocated_bytes() == 0
    assert mr.get_allocated_bytes() == 0

    # make sure the allocations string is now empty
    assert len(mr2.get_outstanding_allocations_str()) == 0
    assert len(mr.get_outstanding_allocations_str()) == 0


def test_mr_allocate_deallocate():
    mr = rmm.mr.TrackingResourceAdaptor(rmm.mr.get_current_device_resource())
    size = 1 << 23  # 8 MiB
    ptr = mr.allocate(size)
    assert mr.get_allocated_bytes() == 1 << 23
    mr.deallocate(ptr, size)
    assert mr.get_allocated_bytes() == 0
