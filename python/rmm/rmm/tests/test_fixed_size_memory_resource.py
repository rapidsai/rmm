# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
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


def _make_fixed_size_mr(upstream):
    """Build a FixedSizeMemoryResource from the given upstream factory."""
    return rmm.mr.FixedSizeMemoryResource(
        upstream(), block_size=1 << 20, blocks_to_preallocate=128
    )


_UPSTREAMS = [
    lambda: rmm.mr.CudaMemoryResource(),
    lambda: rmm.mr.ManagedMemoryResource(),
] + (
    [
        lambda: rmm.mr.SystemMemoryResource(),
        lambda: rmm.mr.SamHeadroomMemoryResource(headroom=1 << 20),
    ]
    if _SYSTEM_MEMORY_SUPPORTED
    else []
)


# Create the FixedSizeMemoryResource once per upstream (class-scoped),
# avoiding the expensive ManagedMemoryResource slab allocation on every
# test invocation.
@pytest.fixture(scope="class", params=_UPSTREAMS)
def fixed_size_mr(request):
    """Create the FixedSizeMemoryResource once per upstream."""
    return _make_fixed_size_mr(request.param)


class TestFixedSizeMemoryResource:
    @pytest.fixture(autouse=True)
    def _apply_mr(self, fixed_size_mr):
        """Set the device resource before each test."""
        rmm.mr.set_current_device_resource(fixed_size_mr)

    @pytest.mark.parametrize("dtype", _dtypes)
    @pytest.mark.parametrize("nelem", _nelems)
    @pytest.mark.parametrize("alloc", _allocs)
    def test_fixed_size_memory_resource(
        self, fixed_size_mr, dtype, nelem, alloc
    ):
        assert (
            rmm.mr.get_current_device_resource_type()
            is rmm.mr.FixedSizeMemoryResource
        )
        array_tester(dtype, nelem, alloc)
