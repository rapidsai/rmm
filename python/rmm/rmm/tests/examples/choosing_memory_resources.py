# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Code examples for docs/user_guide/choosing_memory_resources.md
# ruff: noqa: RUF059


def recommended_default() -> None:
    # [recommended-default]
    import rmm

    mr = rmm.mr.CudaAsyncMemoryResource()
    buffer = rmm.DeviceBuffer(size=1024, mr=mr)
    # [/recommended-default]

    assert buffer.size == 1024


def managed_pool_prefetch() -> None:
    # [managed-pool-prefetch]
    import rmm

    # Use 80% of GPU memory, rounded down to nearest 256 bytes
    free_memory, total_memory = rmm.mr.available_device_memory()
    pool_size = int(total_memory * 0.8) // 256 * 256

    mr = rmm.mr.PrefetchResourceAdaptor(
        rmm.mr.PoolMemoryResource(
            rmm.mr.ManagedMemoryResource(),
            initial_pool_size=pool_size,
        )
    )
    # [/managed-pool-prefetch]

    buffer = rmm.DeviceBuffer(size=1024, mr=mr)
    assert buffer.size == 1024


def managed_memory_example() -> None:
    # [managed-memory-example]
    import rmm

    # Combine managed memory with a pool and prefetching for performance.
    # Without prefetching, page faults cause significant overhead.
    base = rmm.mr.ManagedMemoryResource()
    pool = rmm.mr.PoolMemoryResource(base, initial_pool_size=2**30)
    prefetch_mr = rmm.mr.PrefetchResourceAdaptor(pool)
    buffer = rmm.DeviceBuffer(size=1024, mr=prefetch_mr)
    # [/managed-memory-example]

    assert buffer.size == 1024


def pool_memory_example() -> None:
    # [pool-memory-example]
    import rmm

    pool = rmm.mr.PoolMemoryResource(
        rmm.mr.CudaMemoryResource(),
        initial_pool_size=2**32,  #  4 GiB
        maximum_pool_size=2**34,  # 16 GiB
    )
    buffer = rmm.DeviceBuffer(size=1024, mr=pool)
    # [/pool-memory-example]

    assert buffer.size == 1024


def composing_adaptor() -> None:
    # [composing-adaptor]
    # Adaptor wrapping a base resource
    import rmm

    adaptor = rmm.mr.StatisticsResourceAdaptor(
        rmm.mr.CudaAsyncMemoryResource()
    )
    # [/composing-adaptor]

    _ = adaptor


def prefetch_composition() -> None:
    # [prefetch-composition]
    import rmm

    # Prefetch adaptor wrapping managed memory pool
    base = rmm.mr.ManagedMemoryResource()
    pool = rmm.mr.PoolMemoryResource(base, initial_pool_size=2**30)
    prefetch = rmm.mr.PrefetchResourceAdaptor(pool)
    buffer = rmm.DeviceBuffer(size=1024, mr=prefetch)
    # [/prefetch-composition]

    assert buffer.size == 1024


def statistics_composition() -> None:
    # [statistics-composition]
    import rmm

    # Track allocation statistics (counts, peak, and total bytes)
    base = rmm.mr.CudaAsyncMemoryResource()
    stats_mr = rmm.mr.StatisticsResourceAdaptor(base)
    buffer = rmm.DeviceBuffer(size=1024, mr=stats_mr)
    # [/statistics-composition]

    assert buffer.size == 1024


def logging_composition() -> None:
    # [logging-composition]
    import rmm

    # Log every allocation and deallocation to a file
    base = rmm.mr.CudaAsyncMemoryResource()
    logging_mr = rmm.mr.LoggingResourceAdaptor(
        base, log_file_name="allocations.csv"
    )
    buffer = rmm.DeviceBuffer(size=1024, mr=logging_mr)
    # [/logging-composition]

    assert buffer.size == 1024

    import os

    if os.path.exists("allocations.csv"):
        os.remove("allocations.csv")


def multi_library_pytorch() -> None:
    try:
        import torch
    except ImportError:
        print("PyTorch not available, skipping multi_library_pytorch")
        return

    # isort: off
    # [multi-library-pytorch]
    import rmm
    import torch
    from rmm.allocators.torch import rmm_torch_allocator

    # Configure RMM
    rmm.mr.set_current_device_resource(rmm.mr.CudaAsyncMemoryResource())

    # Configure PyTorch to allocate through RMM
    torch.cuda.memory.change_current_allocator(rmm_torch_allocator)
    # [/multi-library-pytorch]
    # isort: on


def best_practices_set_early() -> None:
    # [best-practices-set-early]
    import rmm

    # Do this first, before any GPU allocations
    rmm.mr.set_current_device_resource(rmm.mr.CudaAsyncMemoryResource())
    # [/best-practices-set-early]


if __name__ == "__main__":
    recommended_default()
    managed_pool_prefetch()
    managed_memory_example()
    pool_memory_example()
    composing_adaptor()
    prefetch_composition()
    statistics_composition()
    logging_composition()
    multi_library_pytorch()
    best_practices_set_early()

    print("All choosing_memory_resources examples passed.")
