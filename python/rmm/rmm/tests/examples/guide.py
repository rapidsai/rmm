# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Code examples for docs/user_guide/guide.md


def explicit_resource() -> None:
    # [explicit-resource]
    import rmm

    mr = rmm.mr.CudaAsyncMemoryResource()

    # Pass the resource explicitly
    buffer = rmm.DeviceBuffer(size=1024, mr=mr)
    # [/explicit-resource]

    assert buffer.size == 1024


def current_device_resource() -> None:
    # [current-device-resource]
    import rmm

    async_mr = rmm.mr.CudaAsyncMemoryResource()
    rmm.mr.set_current_device_resource(async_mr)

    # Allocations that don't specify a resource use the current device resource
    mr = rmm.mr.get_current_device_resource()
    # [/current-device-resource]

    assert mr is not None


def device_buffer_example() -> None:
    # [device-buffer]
    import rmm

    # Allocate 1024 bytes
    buffer = rmm.DeviceBuffer(size=1024)

    # Access pointer and size
    ptr = buffer.ptr
    size = buffer.size

    # Resize (may reallocate)
    buffer.resize(2048)

    # Copy construct (deep copy)
    buffer2 = buffer.copy()
    # [/device-buffer]

    assert buffer.size == 2048
    assert buffer2.size == 2048
    _ = ptr, size


def statistics_tracking() -> None:
    # [statistics-tracking]
    import rmm

    # Wrap base resource with statistics adaptor
    cuda_mr = rmm.mr.CudaAsyncMemoryResource()
    stats_mr = rmm.mr.StatisticsResourceAdaptor(cuda_mr)

    # Allocate using the statistics-wrapped resource
    buffer = rmm.DeviceBuffer(size=1024, mr=stats_mr)

    # Get statistics
    stats = stats_mr.allocation_counts
    print(f"Current bytes: {stats.current_bytes}")
    print(f"Peak bytes: {stats.peak_bytes}")
    print(f"Total bytes: {stats.total_bytes}")
    # [/statistics-tracking]

    assert stats.current_bytes >= 1024
    _ = buffer


def logging_example() -> None:
    # [logging]
    import rmm

    base_mr = rmm.mr.CudaAsyncMemoryResource()
    log_mr = rmm.mr.LoggingResourceAdaptor(
        base_mr, log_file_name="allocations.csv"
    )

    # Allocations through log_mr are logged to CSV
    buffer = rmm.DeviceBuffer(size=1024, mr=log_mr)
    # [/logging]

    assert buffer.size == 1024

    import os

    if os.path.exists("allocations.csv"):
        os.remove("allocations.csv")


def composing_resources() -> None:
    # [composing-resources]
    import rmm

    # Base resource
    cuda_mr = rmm.mr.CudaMemoryResource()

    # Add pool
    pool_mr = rmm.mr.PoolMemoryResource(cuda_mr, initial_pool_size=2**30)

    # Add statistics
    stats_mr = rmm.mr.StatisticsResourceAdaptor(pool_mr)

    # Add logging
    log_mr = rmm.mr.LoggingResourceAdaptor(stats_mr, log_file_name="log.csv")

    # Use log_mr for allocations — all allocations are pooled, tracked, and logged
    buffer = rmm.DeviceBuffer(size=1024, mr=log_mr)
    # [/composing-resources]

    assert buffer.size == 1024

    import os

    if os.path.exists("log.csv"):
        os.remove("log.csv")


def cupy_example() -> None:
    try:
        import cupy as cp
    except ImportError:
        print("CuPy not available, skipping cupy_example")
        return

    # isort: off
    # [cupy]
    import rmm
    import cupy as cp
    from rmm.allocators.cupy import rmm_cupy_allocator

    # Configure RMM
    mr = rmm.mr.CudaAsyncMemoryResource()
    rmm.mr.set_current_device_resource(mr)

    # Set CuPy to use RMM
    cp.cuda.set_allocator(rmm_cupy_allocator)

    # All CuPy arrays now use RMM
    array = cp.zeros(1000)
    # [/cupy]
    # isort: on

    assert array.shape == (1000,)


def numba_example() -> None:
    try:
        from numba import cuda
    except ImportError:
        print("Numba not available, skipping numba_example")
        return

    # isort: off
    # [numba]
    from numba import cuda
    from rmm.allocators.numba import RMMNumbaManager
    import rmm

    # Configure RMM
    mr = rmm.mr.CudaAsyncMemoryResource()
    rmm.mr.set_current_device_resource(mr)

    # Set Numba to use RMM
    cuda.set_memory_manager(RMMNumbaManager)
    # [/numba]
    # isort: on


def pytorch_example() -> None:
    try:
        import torch
    except ImportError:
        print("PyTorch not available, skipping pytorch_example")
        return

    # isort: off
    # [pytorch]
    import rmm
    import torch
    from rmm.allocators.torch import rmm_torch_allocator

    # Configure RMM
    mr = rmm.mr.CudaAsyncMemoryResource()
    rmm.mr.set_current_device_resource(mr)

    # Set PyTorch to use RMM
    torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

    # All PyTorch tensors now use RMM
    tensor = torch.zeros(1000, device="cuda")
    # [/pytorch]
    # isort: on

    assert tensor.shape == (1000,)


def multi_device_example() -> None:
    try:
        from cuda.bindings import runtime
    except ImportError:
        print("cuda.bindings not available, skipping multi_device_example")
        return

    _, num_devices = runtime.cudaGetDeviceCount()
    if num_devices < 1:
        print("No CUDA devices, skipping multi_device_example")
        return

    # isort: off
    # [multi-device]
    import rmm
    from cuda.bindings import runtime

    _, num_devices = runtime.cudaGetDeviceCount()

    # Store resources to maintain lifetime
    resources = []

    for device_id in range(num_devices):
        # Create resource for this device
        mr = rmm.mr.CudaAsyncMemoryResource()
        resources.append(mr)

        # Set as per-device resource
        rmm.mr.set_per_device_resource(device_id, mr)

    # Use device 0
    buffer = rmm.DeviceBuffer(size=1024)  # Uses device 0's resource
    # [/multi-device]
    # isort: on

    assert buffer.size == 1024


if __name__ == "__main__":
    explicit_resource()
    current_device_resource()
    device_buffer_example()
    statistics_tracking()
    logging_example()
    composing_resources()
    cupy_example()
    numba_example()
    pytorch_example()
    multi_device_example()

    print("All guide examples passed.")
