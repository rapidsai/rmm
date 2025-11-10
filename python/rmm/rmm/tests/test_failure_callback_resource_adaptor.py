# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Tests for FailureCallbackResourceAdaptor."""

import pytest
from test_helpers import _IS_INTEGRATED_MEMORY_SYSTEM

import rmm


@pytest.mark.skipif(
    _IS_INTEGRATED_MEMORY_SYSTEM,
    reason="Integrated memory systems may kill the process when attempting allocations larger than available memory",
)
def test_failure_callback_resource_adaptor():
    retried = [False]

    def callback(nbytes: int) -> bool:
        if retried[0]:
            return False
        else:
            retried[0] = True
            return True

    cuda_mr = rmm.mr.CudaMemoryResource()
    mr = rmm.mr.FailureCallbackResourceAdaptor(cuda_mr, callback)
    rmm.mr.set_current_device_resource(mr)

    with pytest.raises(MemoryError):
        rmm.DeviceBuffer(size=1024**5)  # 1 petabyte
    assert retried[0]


@pytest.mark.skipif(
    _IS_INTEGRATED_MEMORY_SYSTEM,
    reason="Integrated memory systems may kill the process when attempting allocations larger than available memory",
)
def test_failure_callback_resource_adaptor_error():
    def callback(nbytes: int) -> bool:
        raise RuntimeError("MyError")

    cuda_mr = rmm.mr.CudaMemoryResource()
    mr = rmm.mr.FailureCallbackResourceAdaptor(cuda_mr, callback)
    rmm.mr.set_current_device_resource(mr)

    with pytest.raises(RuntimeError, match="MyError"):
        rmm.DeviceBuffer(size=1024**5)  # 1 petabyte
