# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Tests for basic RMM allocations."""

from itertools import product

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
def test_rmm_alloc(dtype, nelem, alloc):
    array_tester(dtype, nelem, alloc)


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize("dtype", _dtypes)
@pytest.mark.parametrize("nelem", _nelems)
@pytest.mark.parametrize("alloc", _allocs)
@pytest.mark.parametrize(
    "managed, pool", list(product([False, True], [False, True]))
)
def test_rmm_modes(dtype, nelem, alloc, managed, pool):
    assert rmm.is_initialized()
    array_tester(dtype, nelem, alloc)

    rmm.reinitialize(pool_allocator=pool, managed_memory=managed)

    assert rmm.is_initialized()

    array_tester(dtype, nelem, alloc)


@pytest.mark.skipif(
    not _SYSTEM_MEMORY_SUPPORTED,
    reason="System memory not supported",
)
@pytest.mark.parametrize("dtype", _dtypes)
@pytest.mark.parametrize("nelem", _nelems)
@pytest.mark.parametrize("alloc", _allocs)
@pytest.mark.parametrize(
    "system, pool, headroom",
    list(product([False, True], [False, True], [False, True])),
)
def test_rmm_modes_system_memory(dtype, nelem, alloc, system, pool, headroom):
    assert rmm.is_initialized()
    array_tester(dtype, nelem, alloc)

    if system:
        if headroom:
            base_mr = rmm.mr.SamHeadroomMemoryResource(headroom=1 << 20)
        else:
            base_mr = rmm.mr.SystemMemoryResource()
    else:
        base_mr = rmm.mr.CudaMemoryResource()
    if pool:
        mr = rmm.mr.PoolMemoryResource(base_mr)
    else:
        mr = base_mr
    rmm.mr.set_current_device_resource(mr)

    assert rmm.is_initialized()

    array_tester(dtype, nelem, alloc)
