# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Tests for object lifetime and garbage collection."""

import gc

import rmm


def test_mr_devicebuffer_lifetime():
    # Test ensures MR/Stream lifetime is longer than DeviceBuffer. Even if all
    # references go out of scope
    # It is necessary to verify that it also works when using an upstream :
    # here a Pool MR with the current MR as upstream
    rmm.mr.set_current_device_resource(
        rmm.mr.PoolMemoryResource(rmm.mr.get_current_device_resource())
    )

    # Creates a new non-default stream
    stream = rmm.pylibrmm.stream.Stream()

    # Allocate DeviceBuffer with Pool and Stream
    a = rmm.DeviceBuffer(size=10, stream=stream)

    # Change current MR. Will cause Pool to go out of scope
    rmm.mr.set_current_device_resource(rmm.mr.CudaMemoryResource())

    # Force collection to ensure objects are cleaned up
    gc.collect()

    # Delete a. Used to crash before. Pool MR should still be alive
    del a


def test_dev_buf_circle_ref_dealloc():
    # This test creates a reference cycle containing a `DeviceBuffer`
    # and ensures that the garbage collector does not clear it, i.e.,
    # that the GC does not remove all references to other Python
    # objects from it. The `DeviceBuffer` needs to keep its reference
    # to the `DeviceMemoryResource` that was used to create it in
    # order to be cleaned up properly. See GH #931.

    rmm.mr.set_current_device_resource(rmm.mr.CudaMemoryResource())

    dbuf1 = rmm.DeviceBuffer(size=1_000_000)

    # Make dbuf1 part of a reference cycle:
    l1 = [dbuf1]
    l1.append(l1)

    # due to the reference cycle, the device buffer doesn't actually get
    # cleaned up until after `gc.collect()` is called.
    del dbuf1, l1

    rmm.mr.set_current_device_resource(rmm.mr.CudaMemoryResource())

    # test that after the call to `gc.collect()`, the `DeviceBuffer`
    # is deallocated successfully (i.e., without a segfault).
    gc.collect()


def test_upstream_mr_circle_ref_dealloc():
    # This test is just like the one above, except it tests that
    # instances of `UpstreamResourceAdaptor` (such as
    # `PoolMemoryResource`) are not cleared by the GC.

    rmm.mr.set_current_device_resource(rmm.mr.CudaMemoryResource())
    mr = rmm.mr.PoolMemoryResource(rmm.mr.get_current_device_resource())
    l1 = [mr]
    l1.append(l1)
    del mr, l1
    rmm.mr.set_current_device_resource(rmm.mr.CudaMemoryResource())
    gc.collect()
