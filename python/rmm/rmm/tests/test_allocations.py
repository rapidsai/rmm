# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Tests for basic RMM allocations."""

from itertools import product
from typing import Any

import numpy as np
import pytest
from test_helpers import (
    _SYSTEM_MEMORY_SUPPORTED,
    _TEST_POOL_SIZE,
    _allocs,
    _dtypes,
    _nelems,
    array_tester,
)

import rmm

# Type aliases
NUMPY_DTYPE_T = type[
    np.generic
]  # NumPy dtype classes like np.int8, np.float32, etc.


@pytest.mark.parametrize("dtype", _dtypes)
@pytest.mark.parametrize("nelem", _nelems)
@pytest.mark.parametrize("alloc", _allocs)
def test_rmm_alloc(dtype: NUMPY_DTYPE_T, nelem: int, alloc: Any) -> None:
    array_tester(dtype, nelem, alloc)


def _make_mr(managed: bool, pool: bool) -> rmm.mr.DeviceMemoryResource:
    """Build a memory resource for the given managed/pool mode."""
    base_mr: rmm.mr.DeviceMemoryResource
    if managed:
        base_mr = rmm.mr.ManagedMemoryResource()
    else:
        base_mr = rmm.mr.CudaMemoryResource()
    if pool:
        return rmm.mr.PoolMemoryResource(
            base_mr, initial_pool_size=_TEST_POOL_SIZE
        )
    return base_mr


def _make_system_memory_mr(
    system: bool, pool: bool, headroom: bool
) -> rmm.mr.DeviceMemoryResource:
    """Build a memory resource for the given system memory test mode."""
    base_mr: rmm.mr.DeviceMemoryResource
    if system:
        if headroom:
            base_mr = rmm.mr.SamHeadroomMemoryResource(headroom=1 << 20)
        else:
            base_mr = rmm.mr.SystemMemoryResource()
    else:
        base_mr = rmm.mr.CudaMemoryResource()
    if pool:
        return rmm.mr.PoolMemoryResource(
            base_mr, initial_pool_size=_TEST_POOL_SIZE
        )
    return base_mr


# Test all combinations of default/managed and pooled/non-pooled allocation.
# The mode fixture is class-scoped so that the (potentially expensive) memory
# resource is created once per mode combo rather than once per dtype/nelem.
@pytest.mark.parametrize(
    "managed, pool",
    list(product([False, True], [False, True])),
    scope="class",
)
class TestRmmModes:
    @pytest.fixture(autouse=True)
    def _apply_mode(self, managed: bool, pool: bool) -> None:
        mr = _make_mr(managed, pool)
        rmm.mr.set_current_device_resource(mr)

    @pytest.mark.parametrize("dtype", _dtypes)
    @pytest.mark.parametrize("nelem", _nelems)
    @pytest.mark.parametrize("alloc", _allocs)
    def test_rmm_modes(
        self, dtype: NUMPY_DTYPE_T, nelem: int, alloc: Any
    ) -> None:
        assert rmm.is_initialized()
        array_tester(dtype, nelem, alloc)


@pytest.mark.skipif(
    not _SYSTEM_MEMORY_SUPPORTED,
    reason="System memory not supported",
)
@pytest.mark.parametrize(
    "system, pool, headroom",
    list(product([False, True], [False, True], [False, True])),
    scope="class",
)
class TestRmmModesSystemMemory:
    @pytest.fixture(autouse=True)
    def _apply_mode(self, system: bool, pool: bool, headroom: bool) -> None:
        mr = _make_system_memory_mr(system, pool, headroom)
        rmm.mr.set_current_device_resource(mr)

    @pytest.mark.parametrize("dtype", _dtypes)
    @pytest.mark.parametrize("nelem", _nelems)
    @pytest.mark.parametrize("alloc", _allocs)
    def test_rmm_modes_system_memory(
        self, dtype: NUMPY_DTYPE_T, nelem: int, alloc: Any
    ) -> None:
        assert rmm.is_initialized()
        array_tester(dtype, nelem, alloc)
