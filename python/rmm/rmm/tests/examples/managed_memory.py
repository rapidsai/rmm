# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Code examples for docs/user_guide/managed_memory.md

import os


def prefetch_on_allocate() -> None:
    # [prefetch-on-allocate]
    import rmm

    managed_mr = rmm.mr.ManagedMemoryResource()
    prefetch_mr = rmm.mr.PrefetchResourceAdaptor(managed_mr)

    # This allocation is prefetched to the GPU automatically
    buffer = rmm.DeviceBuffer(size=1000000, mr=prefetch_mr)
    # [/prefetch-on-allocate]

    assert buffer.size == 1000000


def prefetch_with_pool() -> None:
    # [prefetch-with-pool]
    import rmm

    managed_mr = rmm.mr.ManagedMemoryResource()
    pool_mr = rmm.mr.PoolMemoryResource(managed_mr, initial_pool_size=2**30)
    prefetch_mr = rmm.mr.PrefetchResourceAdaptor(pool_mr)

    # Logging and statistics don't allocate, so they can go on the outside
    stats_mr = rmm.mr.StatisticsResourceAdaptor(prefetch_mr)
    log_mr = rmm.mr.LoggingResourceAdaptor(stats_mr, log_file_name="log.csv")

    buffer = rmm.DeviceBuffer(size=1000000, mr=log_mr)
    # [/prefetch-with-pool]

    assert buffer.size == 1000000

    if os.path.exists("log.csv"):
        os.remove("log.csv")


def prefetch_on_access() -> None:
    # [prefetch-on-access]
    import rmm
    from rmm.pylibrmm.stream import Stream

    managed_mr = rmm.mr.ManagedMemoryResource()
    buffer = rmm.DeviceBuffer(size=1000000, mr=managed_mr)

    # Prefetch to device 0 on this stream
    stream = Stream()
    buffer.prefetch(device=0, stream=stream)

    # Kernel on the same stream finds the data already resident
    # [/prefetch-on-access]


if __name__ == "__main__":
    prefetch_on_allocate()
    prefetch_with_pool()
    prefetch_on_access()

    print("All managed_memory examples passed.")
