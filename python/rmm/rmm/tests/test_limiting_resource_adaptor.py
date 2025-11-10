# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Tests for LimitingResourceAdaptor."""

import pytest

import rmm


@pytest.mark.parametrize(
    "mr",
    [
        rmm.mr.CudaMemoryResource,
        pytest.param(rmm.mr.CudaAsyncMemoryResource),
    ],
)
def test_limiting_resource_adaptor(mr):
    cuda_mr = mr()

    allocation_limit = 1 << 20
    num_buffers = 2
    buffer_size = allocation_limit // num_buffers

    mr = rmm.mr.LimitingResourceAdaptor(
        cuda_mr, allocation_limit=allocation_limit
    )
    assert mr.get_allocation_limit() == allocation_limit

    rmm.mr.set_current_device_resource(mr)

    buffers = [rmm.DeviceBuffer(size=buffer_size) for _ in range(num_buffers)]

    assert mr.get_allocated_bytes() == sum(b.size for b in buffers)

    with pytest.raises(MemoryError):
        rmm.DeviceBuffer(size=1)
