# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Tests for BinningMemoryResource."""

import pytest
from test_helpers import (
    _SYSTEM_MEMORY_SUPPORTED,
    _allocs,
    _dtypes,
    _nelems,
    array_tester,
)

import rmm


@pytest.mark.parametrize("dtype", _dtypes)
@pytest.mark.parametrize("nelem", _nelems)
@pytest.mark.parametrize("alloc", _allocs)
@pytest.mark.parametrize(
    "upstream_mr",
    [
        lambda: rmm.mr.CudaMemoryResource(),
        lambda: rmm.mr.ManagedMemoryResource(),
        lambda: rmm.mr.PoolMemoryResource(
            rmm.mr.CudaMemoryResource(), 1 << 20
        ),
    ]
    + (
        [
            lambda: rmm.mr.SystemMemoryResource(),
            lambda: rmm.mr.SamHeadroomMemoryResource(headroom=1 << 20),
        ]
        if _SYSTEM_MEMORY_SUPPORTED
        else []
    ),
)
def test_binning_memory_resource(dtype, nelem, alloc, upstream_mr):
    upstream = upstream_mr()

    # Add fixed-size bins 256KiB, 512KiB, 1MiB, 2MiB, 4MiB
    mr = rmm.mr.BinningMemoryResource(upstream, 18, 22)

    # Test adding some explicit bin MRs
    fixed_mr = rmm.mr.FixedSizeMemoryResource(upstream, 1 << 10)
    cuda_mr = rmm.mr.CudaMemoryResource()
    mr.add_bin(1 << 10, fixed_mr)  # 1KiB bin
    mr.add_bin(1 << 23, cuda_mr)  # 8MiB bin

    rmm.mr.set_current_device_resource(mr)
    assert rmm.mr.get_current_device_resource_type() is type(mr)
    array_tester(dtype, nelem, alloc)
