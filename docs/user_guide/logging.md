# Logging and Profiling

RMM provides adaptors for tracking memory allocations and deallocations.

The {cpp:class}`~rmm::mr::logging_resource_adaptor` / {py:class}`~rmm.mr.LoggingResourceAdaptor` will produce a CSV file of all allocations/deallocations with timestamps and stream IDs.

The {cpp:class}`~rmm::mr::statistics_resource_adaptor` / {py:class}`~rmm.mr.StatisticsResourceAdaptor`, and {py:mod}`rmm.statistics`, can be used to track allocation statistics such as peak memory and total memory.

## Memory Event Logging

Memory event logging writes details of every allocation and deallocation to a CSV file. This is useful for:
- Debugging memory issues
- Understanding allocation patterns
- Profiling memory usage
- Replaying workloads for benchmarking

### Using the Logging Adaptor

Wrap any memory resource with the logging adaptor to record allocations and deallocations to a CSV file:

`````{tabs}
````{group-tab} C++
```{literalinclude} ../../cpp/examples/docs/src/logging.cpp
---
language: cpp
start-after: "// [logging-adaptor]"
end-before: "// [/logging-adaptor]"
dedent:
---
```
````
````{group-tab} Python
```{literalinclude} ../../python/rmm/rmm/tests/examples/logging.py
---
language: python
start-after: "# [logging-adaptor]"
end-before: "# [/logging-adaptor]"
dedent:
---
```
````
`````

If no filename is provided, the `RMM_LOG_FILE` environment variable is used:

```bash
export RMM_LOG_FILE="allocations.csv"
```

### CSV Log Format

Each row represents an allocation or deallocation with the following columns:

```
Thread,Time,Action,Pointer,Size,Stream
```

Example:
```
Thread,Time,Action,Pointer,Size,Stream
140573312345856,1634567890.123456,allocate,0x7f8a40000000,1024,0x7f8a38001020
140573312345856,1634567890.234567,allocate,0x7f8a40000400,2048,0x7f8a38001020
140573312345856,1634567890.345678,deallocate,0x7f8a40000000,1024,0x7f8a38001020
```

- **Thread**: Thread ID performing the operation
- **Time**: Timestamp (seconds since epoch)
- **Action**: `allocate` or `deallocate`
- **Pointer**: Memory address
- **Size**: Allocation size in bytes
- **Stream**: CUDA stream pointer

### Analyzing Logs

You can parse and analyze logs with Python:

```{literalinclude} ../../python/rmm/rmm/tests/examples/logging.py
---
language: python
start-after: "# [analyzing-logs]"
end-before: "# [/analyzing-logs]"
dedent:
---
```

### Replay Benchmark

When building RMM from source, logs can be used with `REPLAY_BENCHMARK`:

```bash
cd build/gbenchmarks
./REPLAY_BENCHMARK --log_file=memory_log.csv
```

This replays the allocation pattern from the log, useful for:
- Benchmarking different memory resources
- Testing allocator implementations
- Profiling allocation overhead

## Memory Statistics

RMM provides statistics tracking for allocations using `statistics_resource_adaptor`. The adaptor tracks current, peak, and total allocation bytes and counts.

### Using the Statistics Adaptor

`````{tabs}
````{group-tab} C++
```{literalinclude} ../../cpp/examples/docs/src/logging.cpp
---
language: cpp
start-after: "// [statistics-adaptor]"
end-before: "// [/statistics-adaptor]"
dedent:
---
```
````
````{group-tab} Python
```{literalinclude} ../../python/rmm/rmm/tests/examples/logging.py
---
language: python
start-after: "# [statistics-adaptor]"
end-before: "# [/statistics-adaptor]"
dedent:
---
```
````
`````

Python also provides a convenience API for enabling statistics globally:

```{literalinclude} ../../python/rmm/rmm/tests/examples/logging.py
---
language: python
start-after: "# [statistics-global]"
end-before: "# [/statistics-global]"
dedent:
---
```

### Tracking Memory Growth

Monitor memory usage over time:

```{literalinclude} ../../python/rmm/rmm/tests/examples/logging.py
---
language: python
start-after: "# [tracking-memory-growth]"
end-before: "# [/tracking-memory-growth]"
dedent:
---
```

## Memory Profiling (Python)

The memory profiler tracks allocations by function/code block.

### Profiling Functions

```{literalinclude} ../../python/rmm/rmm/tests/examples/logging.py
---
language: python
start-after: "# [profiling-functions]"
end-before: "# [/profiling-functions]"
dedent:
---
```

The report shows the number of calls, peak memory, and total memory for each profiled function.

### Profiling Code Blocks

```{literalinclude} ../../python/rmm/rmm/tests/examples/logging.py
---
language: python
start-after: "# [profiling-code-blocks]"
end-before: "# [/profiling-code-blocks]"
dedent:
---
```

### Nested Profiling

```{literalinclude} ../../python/rmm/rmm/tests/examples/logging.py
---
language: python
start-after: "# [nested-profiling]"
end-before: "# [/nested-profiling]"
dedent:
---
```

The report includes entries for both the outer and inner profiling scopes.

### Custom Profiler Records

Use custom profiler records for separate tracking:

```{literalinclude} ../../python/rmm/rmm/tests/examples/logging.py
---
language: python
start-after: "# [custom-profiler-records]"
end-before: "# [/custom-profiler-records]"
dedent:
---
```

## Debug Logging

RMM uses [rapids-logger](https://github.com/rapidsai/rapids-logger) for debug output.

### Enabling Debug Logging

Debug logs show internal RMM behavior, errors, and warnings.

#### Output Location

By default, logs go to stderr. Set `RMM_DEBUG_LOG_FILE` to write to a file:

```bash
export RMM_DEBUG_LOG_FILE=/path/to/rmm_debug.log
```

#### Log Levels

Set at **compile time** with CMake:

```bash
cmake .. -DRMM_LOGGING_LEVEL=DEBUG
```

Available levels (increasing verbosity):
- `OFF` - No logging
- `CRITICAL` - Only critical errors
- `ERROR` - Errors
- `WARN` - Warnings and errors
- `INFO` - Informational messages (default)
- `DEBUG` - Detailed debug info
- `TRACE` - Very verbose tracing

#### Runtime Log Level

Even with verbose logging compiled in, you must enable it at runtime:

`````{tabs}
````{group-tab} C++
```{literalinclude} ../../cpp/examples/docs/src/logging.cpp
---
language: cpp
start-after: "// [debug-log-level]"
end-before: "// [/debug-log-level]"
dedent:
---
```
````
````{group-tab} Python
```{literalinclude} ../../python/rmm/rmm/tests/examples/logging.py
---
language: python
start-after: "# [debug-log-level]"
end-before: "# [/debug-log-level]"
dedent:
---
```
````
`````

### What Gets Logged

Debug logging shows:
- Memory resource initialization
- Allocation failures and errors
- Pool growth and shrinkage
- Stream synchronization events
- Multi-device operations
- Internal state changes

Example debug output:
```
[2024-01-15 10:30:45.123] [info] Initializing cuda_async_memory_resource
[2024-01-15 10:30:45.234] [debug] pool_memory_resource: allocated 1 GiB from upstream
[2024-01-15 10:30:45.345] [warn] Allocation of 10 GiB failed, pool exhausted
[2024-01-15 10:30:45.456] [debug] Growing pool by 2 GiB
```

## Combining Logging Features

Multiple logging features can be composed together by stacking adaptors:

`````{tabs}
````{group-tab} C++
```{literalinclude} ../../cpp/examples/docs/src/logging.cpp
---
language: cpp
start-after: "// [combining-features]"
end-before: "// [/combining-features]"
dedent:
---
```
````
````{group-tab} Python
```{literalinclude} ../../python/rmm/rmm/tests/examples/logging.py
---
language: python
start-after: "# [combining-features]"
end-before: "# [/combining-features]"
dedent:
---
```
````
`````

## Use Cases

### Debugging OOM Errors

```{literalinclude} ../../python/rmm/rmm/tests/examples/logging.py
---
language: python
start-after: "# [debugging-oom]"
end-before: "# [/debugging-oom]"
dedent:
---
```

### Profiling Memory in Data Pipeline

```{literalinclude} ../../python/rmm/rmm/tests/examples/logging.py
---
language: python
start-after: "# [profiling-pipeline]"
end-before: "# [/profiling-pipeline]"
dedent:
---
```

### Benchmarking Memory Resources

```{literalinclude} ../../python/rmm/rmm/tests/examples/logging.py
---
language: python
start-after: "# [benchmarking-resources]"
end-before: "# [/benchmarking-resources]"
dedent:
---
```

## Best Practices

1. **Use event logging for debugging** - CSV logs help understand allocation patterns
2. **Enable statistics for profiling** - Track memory usage over time
3. **Use profiler for hotspot analysis** - Identify which functions allocate most memory
4. **Set appropriate debug level** - Use `INFO` normally, `DEBUG`/`TRACE` when troubleshooting
5. **Disable logging in production** - Logging has overhead; only enable when needed
6. **Analyze logs with tools** - Use pandas, REPLAY_BENCHMARK, or custom scripts
7. **Combine with NVIDIA tools** - Use [NVIDIA Nsight™ Systems](https://developer.nvidia.com/nsight-systems) alongside RMM logging for a complete picture
