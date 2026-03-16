# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Tests for BinningMemoryResource."""

import numpy as np
import pytest
from test_helpers import (
    _SYSTEM_MEMORY_SUPPORTED,
    _allocs,
    _dtypes,
    array_tester,
)

import rmm

# Sized so float64 allocations (1 KiB - 128 KiB) hit distinct auto bins.
_BINNING_NELEMS = [128, 256, 512, 1024, 4096, 16384]

# 16777216 float64 elements = 128 MiB; exercises the explicit CudaMR bin.
_LARGE_NELEM = 16777216


def _make_binning_mr(upstream_mr):
    """Build a BinningMemoryResource from the given upstream factory."""
    upstream = upstream_mr()

    # Auto-create fixed-size bins: 1 KiB, 2 KiB, 4 KiB, … 128 KiB
    mr = rmm.mr.BinningMemoryResource(upstream, 10, 17)

    # Test adding explicit bin MRs
    fixed_mr = rmm.mr.FixedSizeMemoryResource(upstream, 1 << 10)
    cuda_mr = rmm.mr.CudaMemoryResource()
    mr.add_bin(1 << 10, fixed_mr)  # 1 KiB bin (replaces auto bin)
    mr.add_bin(1 << 27, cuda_mr)  # 128 MiB bin
    return mr


_UPSTREAM_MRS = [
    lambda: rmm.mr.CudaMemoryResource(),
    lambda: rmm.mr.ManagedMemoryResource(),
    lambda: rmm.mr.PoolMemoryResource(rmm.mr.CudaMemoryResource(), 1 << 28),
] + (
    [
        lambda: rmm.mr.SystemMemoryResource(),
        lambda: rmm.mr.SamHeadroomMemoryResource(headroom=1 << 20),
    ]
    if _SYSTEM_MEMORY_SUPPORTED
    else []
)


# Create the BinningMemoryResource once per upstream_mr (class-scoped),
# avoiding the expensive ManagedMemoryResource/FixedSizeMemoryResource slab
# allocation on every test invocation.
@pytest.fixture(scope="class", params=_UPSTREAM_MRS)
def binning_mr(request):
    """Create the BinningMemoryResource once per upstream_mr."""
    return _make_binning_mr(request.param)


class TestBinningMemoryResource:
    @pytest.fixture(autouse=True)
    def _apply_mr(self, binning_mr):
        """Set the device resource before each test."""
        rmm.mr.set_current_device_resource(binning_mr)

    @pytest.mark.parametrize("dtype", _dtypes)
    @pytest.mark.parametrize("nelem", _BINNING_NELEMS)
    @pytest.mark.parametrize("alloc", _allocs)
    def test_binning_memory_resource(self, binning_mr, dtype, nelem, alloc):
        assert (
            rmm.mr.get_current_device_resource_type()
            is rmm.mr.BinningMemoryResource
        )
        array_tester(dtype, nelem, alloc)

    @pytest.mark.parametrize("alloc", _allocs)
    def test_binning_large_allocation(self, binning_mr, alloc):
        """Allocate 128 MiB to exercise the explicit CudaMemoryResource bin."""
        assert (
            rmm.mr.get_current_device_resource_type()
            is rmm.mr.BinningMemoryResource
        )
        array_tester(np.float64, _LARGE_NELEM, alloc)
