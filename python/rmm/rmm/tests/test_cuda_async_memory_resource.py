# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CudaAsyncMemoryResource."""

import numpy as np
import pytest
from test_helpers import _allocs, _dtypes, _nelems, array_tester

import rmm
from rmm.pylibrmm.stream import Stream


@pytest.mark.parametrize("dtype", _dtypes)
@pytest.mark.parametrize("nelem", _nelems)
@pytest.mark.parametrize("alloc", _allocs)
def test_cuda_async_memory_resource(dtype, nelem, alloc):
    mr = rmm.mr.CudaAsyncMemoryResource()
    rmm.mr.set_current_device_resource(mr)
    assert rmm.mr.get_current_device_resource_type() is type(mr)
    array_tester(dtype, nelem, alloc)


def test_cuda_async_memory_resource_ipc():
    # CUDA 11.3+ is required for IPC memory handle support
    mr = rmm.mr.CudaAsyncMemoryResource(enable_ipc=True)
    rmm.mr.set_current_device_resource(mr)
    assert rmm.mr.get_current_device_resource_type() is type(mr)


def test_cuda_async_memory_resource_fabric():
    # TODO: We don't have a great way to check if fabric is supported in Python,
    # without using the C++ function
    # rmm::detail::runtime_async_alloc::is_export_handle_type_supported.
    # We can't accurately test this via Python because
    # cuda-python always has the fabric handle enum defined (which normally
    # requires a CUDA 12.3 runtime) and the cuda-compat package in Docker
    # containers prevents us from assuming that the driver we see actually
    # supports fabric handles even if its reported version is new enough (we may
    # see a newer driver than what is present on the host). We can only know
    # the expected behavior by checking the C++ function mentioned above, which
    # is then a redundant check because the CudaAsyncMemoryResource constructor
    # follows the same logic. Therefore, we cannot easily ensure this test
    # passes in certain expected configurations -- we can only ensure that if
    # it fails, it fails in a predictable way.
    try:
        mr = rmm.mr.CudaAsyncMemoryResource(enable_fabric=True)
    except RuntimeError as e:
        # CUDA 12.3 is required for fabric memory handle support
        assert str(e).endswith(
            "Requested IPC memory handle type not supported"
        )
    else:
        rmm.mr.set_current_device_resource(mr)
        assert rmm.mr.get_current_device_resource_type() is type(mr)


@pytest.mark.parametrize("nelems", _nelems)
def test_cuda_async_memory_resource_stream(nelems):
    # test that using CudaAsyncMemoryResource
    # with a non-default stream works
    mr = rmm.mr.CudaAsyncMemoryResource()
    rmm.mr.set_current_device_resource(mr)
    stream = Stream()
    expected = np.full(nelems, 5, dtype="u1")
    dbuf = rmm.DeviceBuffer.to_device(expected, stream=stream)
    result = np.asarray(dbuf.copy_to_host())
    np.testing.assert_equal(expected, result)


@pytest.mark.parametrize("nelem", _nelems)
@pytest.mark.parametrize("alloc", _allocs)
def test_cuda_async_memory_resource_threshold(nelem, alloc):
    # initial pool size == 0
    mr = rmm.mr.CudaAsyncMemoryResource(
        initial_pool_size=0, release_threshold=nelem
    )
    rmm.mr.set_current_device_resource(mr)
    array_tester("u1", nelem, alloc)  # should not trigger release
    array_tester("u1", 2 * nelem, alloc)  # should trigger release
