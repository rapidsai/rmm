# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Tests for PrefetchResourceAdaptor."""

import numpy as np
import pytest
from cuda.bindings import runtime
from test_helpers import (
    _CONCURRENT_MANAGED_ACCESS_SUPPORTED,
    assert_prefetched,
)

import rmm


@pytest.mark.parametrize("managed", [True, False])
def test_prefetch_resource_adaptor(managed):
    if managed:
        upstream_mr = rmm.mr.ManagedMemoryResource()
    else:
        upstream_mr = rmm.mr.CudaMemoryResource()
    mr = rmm.mr.PrefetchResourceAdaptor(upstream_mr)
    rmm.mr.set_current_device_resource(mr)

    # This allocation should be prefetched
    db = rmm.DeviceBuffer.to_device(np.zeros(256, dtype="u1"))

    err, device_id = runtime.cudaGetDevice()
    assert err == runtime.cudaError_t.cudaSuccess

    if managed and _CONCURRENT_MANAGED_ACCESS_SUPPORTED:
        assert_prefetched(db, device_id)
    db.prefetch()  # just test that it doesn't throw
    if managed and _CONCURRENT_MANAGED_ACCESS_SUPPORTED:
        assert_prefetched(db, device_id)
