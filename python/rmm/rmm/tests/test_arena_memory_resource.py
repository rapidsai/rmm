# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ArenaMemoryResource."""

import pytest
from test_helpers import _allocs, _dtypes, _nelems, array_tester

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
    ],
)
def test_arena_memory_resource(dtype, nelem, alloc, upstream_mr):
    upstream = upstream_mr()
    mr = rmm.mr.ArenaMemoryResource(upstream)

    rmm.mr.set_current_device_resource(mr)
    assert rmm.mr.get_current_device_resource_type() is type(mr)
    array_tester(dtype, nelem, alloc)
