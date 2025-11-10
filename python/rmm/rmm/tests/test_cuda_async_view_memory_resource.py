# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CudaAsyncViewMemoryResource."""

import pytest
from cuda.bindings import runtime
from test_helpers import _allocs, _dtypes, _nelems, array_tester

import rmm


@pytest.mark.parametrize("dtype", _dtypes)
@pytest.mark.parametrize("nelem", _nelems)
@pytest.mark.parametrize("alloc", _allocs)
def test_cuda_async_view_memory_resource_default_pool(dtype, nelem, alloc):
    # Get the default memory pool handle
    current_device = rmm._cuda.gpu.getDevice()
    err, pool = runtime.cudaDeviceGetDefaultMemPool(current_device)
    assert err == runtime.cudaError_t.cudaSuccess

    mr = rmm.mr.CudaAsyncViewMemoryResource(pool)
    rmm.mr.set_current_device_resource(mr)
    assert rmm.mr.get_current_device_resource_type() is type(mr)
    array_tester(dtype, nelem, alloc)


@pytest.mark.parametrize("dtype", _dtypes)
@pytest.mark.parametrize("nelem", _nelems)
@pytest.mark.parametrize("alloc", _allocs)
def test_cuda_async_view_memory_resource_custom_pool(dtype, nelem, alloc):
    # Create a memory pool handle
    props = runtime.cudaMemPoolProps()
    props.allocType = runtime.cudaMemAllocationType.cudaMemAllocationTypePinned
    props.location.id = rmm._cuda.gpu.getDevice()
    props.location.type = runtime.cudaMemLocationType.cudaMemLocationTypeDevice
    err, pool = runtime.cudaMemPoolCreate(props)
    assert err == runtime.cudaError_t.cudaSuccess

    mr = rmm.mr.CudaAsyncViewMemoryResource(pool)
    rmm.mr.set_current_device_resource(mr)
    assert rmm.mr.get_current_device_resource_type() is type(mr)
    array_tester(dtype, nelem, alloc)

    # After the pool is destroyed, new allocations should raise
    (err,) = runtime.cudaMemPoolDestroy(pool)
    assert err == runtime.cudaError_t.cudaSuccess
    with pytest.raises(MemoryError):
        array_tester(dtype, nelem, alloc)
