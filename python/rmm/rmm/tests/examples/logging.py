# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Code examples for docs/user_guide/logging.md

import os


def logging_adaptor() -> None:
    # [logging-adaptor]
    import rmm

    base_mr = rmm.mr.CudaAsyncMemoryResource()
    log_mr = rmm.mr.LoggingResourceAdaptor(
        base_mr, log_file_name="memory_log.csv"
    )

    # Allocations through log_mr are logged to CSV
    buf1 = rmm.DeviceBuffer(size=1024, mr=log_mr)
    buf2 = rmm.DeviceBuffer(size=2048, mr=log_mr)
    # [/logging-adaptor]

    assert buf1.size == 1024
    assert buf2.size == 2048

    if os.path.exists("memory_log.csv"):
        os.remove("memory_log.csv")


def statistics_adaptor() -> None:
    # [statistics-adaptor]
    import rmm

    cuda_mr = rmm.mr.CudaAsyncMemoryResource()
    stats_mr = rmm.mr.StatisticsResourceAdaptor(cuda_mr)

    # Allocate using the statistics-wrapped resource
    buf1 = rmm.DeviceBuffer(size=1024, mr=stats_mr)
    buf2 = rmm.DeviceBuffer(size=2048, mr=stats_mr)

    # Get statistics
    stats = stats_mr.allocation_counts
    print(f"Current bytes: {stats.current_bytes}")
    print(f"Peak bytes: {stats.peak_bytes}")
    print(f"Total allocations: {stats.total_count}")
    # [/statistics-adaptor]

    assert stats.current_bytes >= 1024
    _ = buf1, buf2


def statistics_global() -> None:
    # [statistics-global]
    import rmm

    # Enable statistics globally
    rmm.statistics.enable_statistics()

    # Or use context manager for specific code blocks
    with rmm.statistics.statistics():
        buffer = rmm.DeviceBuffer(size=1024)

        stats = rmm.statistics.get_statistics()
        assert stats is not None
        print(f"Current bytes: {stats.current_bytes}")
        print(f"Peak bytes: {stats.peak_bytes}")
        print(f"Total allocations: {stats.total_count}")
    # [/statistics-global]

    _ = buffer


def tracking_memory_growth() -> None:
    # [tracking-memory-growth]
    import rmm

    rmm.statistics.enable_statistics()

    def checkpoint(label) -> None:
        stats = rmm.statistics.get_statistics()
        assert stats is not None
        print(f"{label}:")
        print(
            f"  Current: {stats.current_bytes:,} bytes ({stats.current_count} allocations)"
        )
        print(f"  Peak: {stats.peak_bytes:,} bytes")

    checkpoint("Start")

    # Allocate
    buffers = [rmm.DeviceBuffer(size=1024 * 1024) for _ in range(10)]
    checkpoint("After 10x1MB allocations")

    # Free some
    buffers = buffers[:5]
    checkpoint("After freeing 5")

    # Allocate more
    buffers.extend([rmm.DeviceBuffer(size=2 * 1024 * 1024) for _ in range(5)])
    checkpoint("After 5x2MB allocations")
    # [/tracking-memory-growth]


def profiling_functions() -> None:
    # [profiling-functions]
    import rmm

    # Enable statistics first
    rmm.statistics.enable_statistics()

    # Profile a function
    @rmm.statistics.profiler()
    def process_data(size):
        buffer = rmm.DeviceBuffer(size=size)
        # ... processing ...
        return buffer

    # Run function
    process_data(1000000)

    # View report
    print(rmm.statistics.default_profiler_records.report())
    # [/profiling-functions]


def profiling_code_blocks() -> None:
    # [profiling-code-blocks]
    import rmm

    rmm.statistics.enable_statistics()

    # Profile specific code blocks
    with rmm.statistics.profiler(name="data loading"):
        data = rmm.DeviceBuffer(size=1000000)

    with rmm.statistics.profiler(name="processing"):
        buffer1 = rmm.DeviceBuffer(size=500000)
        buffer2 = rmm.DeviceBuffer(size=500000)

    # View report
    print(rmm.statistics.default_profiler_records.report())
    # [/profiling-code-blocks]

    _ = data, buffer1, buffer2


def nested_profiling() -> None:
    # [nested-profiling]
    import rmm

    rmm.statistics.enable_statistics()

    with rmm.statistics.profiler(name="outer"):
        buffer1 = rmm.DeviceBuffer(size=1000)

        with rmm.statistics.profiler(name="inner"):
            buffer2 = rmm.DeviceBuffer(size=2000)

        buffer3 = rmm.DeviceBuffer(size=500)

    print(rmm.statistics.default_profiler_records.report())
    # [/nested-profiling]

    _ = buffer1, buffer2, buffer3


def custom_profiler_records() -> None:
    # [custom-profiler-records]
    import rmm

    rmm.statistics.enable_statistics()

    # Create custom profiler records
    custom_records = rmm.statistics.ProfilerRecords()

    # Use with context manager
    with rmm.statistics.profiler(name="my operation", records=custom_records):
        buffer = rmm.DeviceBuffer(size=1024)

    # View only custom records
    print(custom_records.report())
    # [/custom-profiler-records]

    _ = buffer


def debug_log_level() -> None:
    # [debug-log-level]
    import rmm

    # Available levels: trace, debug, info, warn, error, critical, off
    rmm.set_logging_level(rmm.level_enum.trace)
    # [/debug-log-level]

    # Reset to default
    rmm.set_logging_level(rmm.level_enum.info)


def combining_features() -> None:
    # [combining-features]
    import rmm

    # Set debug log level
    rmm.set_logging_level(rmm.level_enum.debug)

    # Build resource stack: statistics + logging
    cuda_mr = rmm.mr.CudaAsyncMemoryResource()
    stats_mr = rmm.mr.StatisticsResourceAdaptor(cuda_mr)
    log_mr = rmm.mr.LoggingResourceAdaptor(
        stats_mr, log_file_name="events.csv"
    )

    # All allocations through log_mr are tracked and logged
    buffer = rmm.DeviceBuffer(size=1024, mr=log_mr)

    # Get statistics
    stats = stats_mr.allocation_counts
    print(f"Peak bytes: {stats.peak_bytes}")

    # Profiling can also be used alongside event logging
    rmm.statistics.enable_statistics()

    @rmm.statistics.profiler()
    def my_function():
        return rmm.DeviceBuffer(size=1024, mr=log_mr)

    my_function()
    print(rmm.statistics.default_profiler_records.report())
    # [/combining-features]

    # Reset to default
    rmm.set_logging_level(rmm.level_enum.info)
    _ = buffer

    if os.path.exists("events.csv"):
        os.remove("events.csv")


def debugging_oom() -> None:
    # [debugging-oom]
    import rmm

    # Enable detailed logging
    base_mr = rmm.mr.CudaAsyncMemoryResource()
    stats_mr = rmm.mr.StatisticsResourceAdaptor(base_mr)
    log_mr = rmm.mr.LoggingResourceAdaptor(
        stats_mr, log_file_name="oom_debug.csv"
    )
    rmm.set_logging_level(rmm.level_enum.debug)

    # Run problematic code
    try:
        large_buffer = rmm.DeviceBuffer(size=100 * 2**30, mr=log_mr)  # noqa: F841
    except MemoryError:
        stats = stats_mr.allocation_counts
        print(f"Peak before OOM: {stats.peak_bytes / 2**30:.2f} GiB")
        print("Check oom_debug.csv for allocation history")
        raise
    # [/debugging-oom]


def profiling_pipeline() -> None:
    # [profiling-pipeline]
    import rmm

    rmm.statistics.enable_statistics()

    @rmm.statistics.profiler()
    def load_data():
        return rmm.DeviceBuffer(size=1000000)

    @rmm.statistics.profiler()
    def process_data(buffer):
        temp = rmm.DeviceBuffer(size=2000000)  # noqa: F841
        result = rmm.DeviceBuffer(size=500000)
        return result

    @rmm.statistics.profiler()
    def save_data(buffer):
        pass

    # Run pipeline
    data = load_data()
    result = process_data(data)
    save_data(result)

    # Identify memory hotspots
    print(rmm.statistics.default_profiler_records.report())
    # [/profiling-pipeline]


def benchmarking_resources() -> None:
    # isort: off
    # [benchmarking-resources]
    import rmm
    import time

    def benchmark_allocations(mr_name, mr) -> None:
        start = time.time()
        buffers = []
        for _ in range(1000):
            buffers.append(rmm.DeviceBuffer(size=1024, mr=mr))
        end = time.time()

        print(f"{mr_name}: {(end - start) * 1000:.2f} ms for 1000 allocations")

    # Compare resources
    benchmark_allocations("CudaMemoryResource", rmm.mr.CudaMemoryResource())
    benchmark_allocations(
        "CudaAsyncMemoryResource", rmm.mr.CudaAsyncMemoryResource()
    )
    pool_mr = rmm.mr.PoolMemoryResource(
        rmm.mr.CudaMemoryResource(), initial_pool_size=2**20
    )
    benchmark_allocations("PoolMemoryResource", pool_mr)
    # [/benchmarking-resources]
    # isort: on


def analyzing_logs() -> None:
    # Generate a sample log file for the example
    import rmm

    base_mr = rmm.mr.CudaAsyncMemoryResource()
    log_mr = rmm.mr.LoggingResourceAdaptor(
        base_mr, log_file_name="memory_log.csv"
    )
    buf1 = rmm.DeviceBuffer(size=1024, mr=log_mr)
    buf2 = rmm.DeviceBuffer(size=2048, mr=log_mr)
    del buf1

    try:
        import pandas as pd  # type: ignore[import-untyped]
    except ImportError:
        print("pandas not available, skipping analyzing_logs")
        if os.path.exists("memory_log.csv"):
            os.remove("memory_log.csv")
        return

    # [analyzing-logs]
    import pandas as pd

    # Read log file
    df = pd.read_csv("memory_log.csv")

    # Total bytes allocated
    total_allocated = df[df["Action"] == "allocate"]["Size"].sum()
    print(f"Total allocated: {total_allocated:,} bytes")

    # Allocation size distribution
    print(df[df["Action"] == "allocate"]["Size"].describe())

    # Peak memory usage (simple analysis)
    df["Delta"] = df.apply(
        lambda row: row["Size"]
        if row["Action"] == "allocate"
        else -row["Size"],
        axis=1,
    )
    df["Cumulative"] = df["Delta"].cumsum()
    peak = df["Cumulative"].max()
    print(f"Peak usage: {peak:,} bytes")
    # [/analyzing-logs]

    _ = buf2

    if os.path.exists("memory_log.csv"):
        os.remove("memory_log.csv")


if __name__ == "__main__":
    logging_adaptor()
    statistics_adaptor()
    statistics_global()
    tracking_memory_growth()
    profiling_functions()
    profiling_code_blocks()
    nested_profiling()
    custom_profiler_records()
    debug_log_level()
    combining_features()
    # debugging_oom() — intentionally skipped (raises MemoryError)
    profiling_pipeline()
    benchmarking_resources()
    analyzing_logs()

    print("All logging examples passed.")
