# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Tests for FixedSizeMemoryResource."""

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
    "upstream",
    [
        lambda: rmm.mr.CudaMemoryResource(),
        lambda: rmm.mr.ManagedMemoryResource(),
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
def test_fixed_size_memory_resource(dtype, nelem, alloc, upstream):
    mr = rmm.mr.FixedSizeMemoryResource(
        upstream(), block_size=1 << 20, blocks_to_preallocate=128
    )
    rmm.mr.set_current_device_resource(mr)
    assert rmm.mr.get_current_device_resource_type() is type(mr)
    array_tester(dtype, nelem, alloc)
