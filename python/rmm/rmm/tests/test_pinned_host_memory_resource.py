# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Tests for PinnedHostMemoryResource."""

import numpy as np
import pytest
from test_helpers import _allocs, _dtypes, _nelems, array_tester

import rmm


@pytest.mark.parametrize("dtype", _dtypes)
@pytest.mark.parametrize("nelem", _nelems)
@pytest.mark.parametrize("alloc", _allocs)
def test_pinned_host_memory_resource(dtype, nelem, alloc):
    """Test PinnedHostMemoryResource as a basic memory resource."""
    mr = rmm.mr.PinnedHostMemoryResource()
    rmm.mr.set_current_device_resource(mr)
    assert rmm.mr.get_current_device_resource_type() is type(mr)
    array_tester(dtype, nelem, alloc)


@pytest.mark.parametrize("dtype", _dtypes)
@pytest.mark.parametrize("nelem", _nelems)
@pytest.mark.parametrize("alloc", _allocs)
def test_pinned_host_memory_resource_with_pool(dtype, nelem, alloc):
    """Test PinnedHostMemoryResource with PoolMemoryResource."""
    base_mr = rmm.mr.PinnedHostMemoryResource()
    mr = rmm.mr.PoolMemoryResource(
        base_mr,
        initial_pool_size="4MiB",
        maximum_pool_size="8MiB",
    )
    rmm.mr.set_current_device_resource(mr)
    assert rmm.mr.get_current_device_resource_type() is type(mr)
    array_tester(dtype, nelem, alloc)


def test_pinned_host_memory_resource_allocate_deallocate():
    """Test direct allocation and deallocation with PinnedHostMemoryResource."""
    mr = rmm.mr.PinnedHostMemoryResource()

    # Test various allocation sizes
    sizes = [256, 1024, 4096, 1024 * 1024]
    ptrs = []

    for size in sizes:
        ptr = mr.allocate(size)
        assert ptr != 0, f"Allocation of {size} bytes returned null pointer"
        ptrs.append((ptr, size))

    # Deallocate all
    for ptr, size in ptrs:
        mr.deallocate(ptr, size)


def test_pinned_host_memory_resource_with_device_buffer():
    """Test PinnedHostMemoryResource with DeviceBuffer."""
    mr = rmm.mr.PinnedHostMemoryResource()
    rmm.mr.set_current_device_resource(mr)

    # Test creating various sized buffers
    sizes = [0, 256, 1024, 4096, 1024 * 1024]
    for size in sizes:
        buf = rmm.DeviceBuffer(size=size)
        assert buf.size == size
        if size > 0:
            assert buf.ptr != 0
            assert buf.capacity() >= size
        else:
            assert buf.ptr == 0


def test_pinned_host_memory_resource_host_device_access():
    """Test that pinned memory is accessible from both host and device."""
    mr = rmm.mr.PinnedHostMemoryResource()
    rmm.mr.set_current_device_resource(mr)

    # Create test data
    test_data = np.arange(100, dtype=np.float32)

    # Copy to device using pinned memory
    device_buf = rmm.DeviceBuffer.to_device(test_data.tobytes())

    # Copy back from device
    result = np.frombuffer(device_buf.tobytes(), dtype=np.float32)

    # Verify data integrity
    np.testing.assert_array_equal(test_data, result)


def test_pinned_host_memory_resource_type_check():
    """Test PinnedHostMemoryResource type and inheritance."""
    mr = rmm.mr.PinnedHostMemoryResource()

    # Check type
    assert isinstance(mr, rmm.mr.PinnedHostMemoryResource)
    assert isinstance(mr, rmm.mr.DeviceMemoryResource)

    # Check that it's not an upstream resource adaptor
    assert not isinstance(mr, rmm.mr.UpstreamResourceAdaptor)


@pytest.mark.parametrize(
    "adaptor_factory",
    [
        lambda mr: rmm.mr.StatisticsResourceAdaptor(mr),
        lambda mr: rmm.mr.TrackingResourceAdaptor(mr),
        lambda mr: rmm.mr.LimitingResourceAdaptor(
            mr, allocation_limit=1024 * 1024 * 10
        ),
    ],
)
def test_pinned_host_memory_resource_with_adaptors(adaptor_factory):
    """Test PinnedHostMemoryResource with various resource adaptors."""
    base_mr = rmm.mr.PinnedHostMemoryResource()
    mr = adaptor_factory(base_mr)
    rmm.mr.set_current_device_resource(mr)

    # Test with a simple allocation
    buf = rmm.DeviceBuffer(size=1024)
    assert buf.size == 1024
    assert buf.ptr != 0
