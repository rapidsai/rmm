# Logging and Profiling

RMM provides two types of logging: **memory event logging** for tracking allocations and deallocations, and **debug logging** for troubleshooting internal behavior.

## Memory Event Logging

Memory event logging writes details of every allocation and deallocation to a CSV file. This is useful for:
- Debugging memory issues
- Understanding allocation patterns
- Profiling memory usage
- Replaying workloads for benchmarking

### Python: Using Memory Event Logging

Enable logging by wrapping your memory resource with `LoggingResourceAdaptor`:

```python
import rmm

# Wrap the current resource with logging adaptor
base = rmm.mr.CudaAsyncMemoryResource()
log_mr = rmm.mr.LoggingResourceAdaptor(base, log_file_name="memory_log.csv")
rmm.mr.set_current_device_resource(log_mr)

# Allocations are now logged
buffer1 = rmm.DeviceBuffer(size=1024)
buffer2 = rmm.DeviceBuffer(size=2048)

# All allocations/deallocations written to memory_log.csv
```

If `log_file_name` is not provided, the environment variable `RMM_LOG_FILE` is used:

```bash
export RMM_LOG_FILE="allocations.csv"
python script.py
```

### C++: Using logging_resource_adaptor

Wrap any memory resource with `logging_resource_adaptor`:

```cpp
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/logging_resource_adaptor.hpp>

int main() {
    // Create upstream resource
    auto cuda_mr = rmm::mr::cuda_async_memory_resource{};

    // Wrap with logging adaptor
    auto log_mr = rmm::mr::logging_resource_adaptor{&cuda_mr, "memory_log.csv"};

    // Set as current resource
    rmm::mr::set_current_device_resource(&log_mr);

    // All allocations logged to CSV
    rmm::cuda_stream stream;
    rmm::device_buffer buffer(1024, stream.view());

    return 0;
}
```

If filename is not provided, `RMM_LOG_FILE` environment variable is checked:

```bash
export RMM_LOG_FILE="allocations.csv"
./my_app
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

```python
import pandas as pd

# Read log file
df = pd.read_csv("memory_log.csv")

# Total bytes allocated
total_allocated = df[df['Action'] == 'allocate']['Size'].sum()
print(f"Total allocated: {total_allocated:,} bytes")

# Allocation size distribution
print(df[df['Action'] == 'allocate']['Size'].describe())

# Peak memory usage (simple analysis)
df['Delta'] = df.apply(
    lambda row: row['Size'] if row['Action'] == 'allocate' else -row['Size'],
    axis=1
)
df['Cumulative'] = df['Delta'].cumsum()
peak = df['Cumulative'].max()
print(f"Peak usage: {peak:,} bytes")
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

RMM provides statistics tracking for allocations using `statistics_resource_adaptor`.

### Python: Enabling Statistics

```python
import rmm

# Enable statistics globally
rmm.statistics.enable_statistics()

# Or use context manager for specific code blocks
with rmm.statistics.statistics():
    buffer = rmm.DeviceBuffer(size=1024)

    # Get current statistics
    stats = rmm.statistics.get_statistics()
    print(f"Current bytes: {stats.current_bytes}")
    print(f"Peak bytes: {stats.peak_bytes}")
    print(f"Total allocations: {stats.total_count}")
```

Available statistics:

```python
class Statistics:
    current_bytes: int    # Currently allocated bytes
    current_count: int    # Number of active allocations
    peak_bytes: int       # Peak bytes allocated
    peak_count: int       # Peak number of allocations
    total_bytes: int      # Total bytes ever allocated
    total_count: int      # Total number of allocations
```

### C++: Using statistics_resource_adaptor

```cpp
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/statistics_resource_adaptor.hpp>
#include <iostream>

int main() {
    auto cuda_mr = rmm::mr::cuda_async_memory_resource{};
    auto stats_mr = rmm::mr::statistics_resource_adaptor{&cuda_mr};
    rmm::mr::set_current_device_resource(&stats_mr);

    // Allocate
    rmm::cuda_stream stream;
    rmm::device_buffer buffer1(1024, stream.view());
    rmm::device_buffer buffer2(2048, stream.view());

    // Get statistics
    auto stats = stats_mr.get_statistics();
    std::cout << "Allocated bytes: " << stats.allocated_bytes << "\n";
    std::cout << "Allocation count: " << stats.num_allocations << "\n";

    return 0;
}
```

### Tracking Memory Growth

Monitor memory usage over time:

```python
import rmm
import time

rmm.statistics.enable_statistics()

def checkpoint(label):
    stats = rmm.statistics.get_statistics()
    print(f"{label}:")
    print(f"  Current: {stats.current_bytes:,} bytes ({stats.current_count} allocations)")
    print(f"  Peak: {stats.peak_bytes:,} bytes")

checkpoint("Start")

# Allocate
buffers = [rmm.DeviceBuffer(size=1024*1024) for _ in range(10)]
checkpoint("After 10x1MB allocations")

# Free some
buffers = buffers[:5]
checkpoint("After freeing 5")

# Allocate more
buffers.extend([rmm.DeviceBuffer(size=2*1024*1024) for _ in range(5)])
checkpoint("After 5x2MB allocations")
```

## Memory Profiling

The memory profiler tracks allocations by function/code block.

### Python: Using the Profiler

#### Profiling Functions

```python
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
```

Output:
```
Memory Profiling
================

Legends:
  ncalls       - number of times the function or code block was called
  memory_peak  - peak memory allocated in function or code block (in bytes)
  memory_total - total memory allocated in function or code block (in bytes)

Ordered by: memory_peak

ncalls     memory_peak    memory_total  filename:lineno(function)
     1       1,000,016       1,000,016  script.py:5(process_data)
```

#### Profiling Code Blocks

```python
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
```

Output:
```
ncalls     memory_peak    memory_total  filename:lineno(function)
     1       1,000,016       1,000,016  data loading
     1       1,000,032       1,000,032  processing
```

#### Nested Profiling

```python
import rmm

rmm.statistics.enable_statistics()

with rmm.statistics.profiler(name="outer"):
    buffer1 = rmm.DeviceBuffer(size=1000)

    with rmm.statistics.profiler(name="inner"):
        buffer2 = rmm.DeviceBuffer(size=2000)

    buffer3 = rmm.DeviceBuffer(size=500)

print(rmm.statistics.default_profiler_records.report())
```

Output shows both nested and total allocations:
```
ncalls     memory_peak    memory_total  filename:lineno(function)
     1           3,520           3,520  outer
     1           2,016           2,016  inner
```

### Custom Profiler Records

Use custom profiler records for separate tracking:

```python
import rmm

rmm.statistics.enable_statistics()

# Create custom profiler records
custom_records = rmm.statistics.profiler_records()

# Use with context manager
with rmm.statistics.profiler(name="my operation", records=custom_records):
    buffer = rmm.DeviceBuffer(size=1024)

# View only custom records
print(custom_records.report())
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

#### Runtime Log Level (Python)

Even with verbose logging compiled in, you must enable it at runtime:

```python
import rmm

# Enable all logging down to TRACE level
rmm.set_logging_level("trace")

# Now you'll see TRACE and DEBUG messages
```

Available Python levels: `"trace"`, `"debug"`, `"info"`, `"warn"`, `"error"`, `"critical"`, `"off"`

#### Runtime Log Level (C++)

```cpp
#include <rmm/logger.hpp>

int main() {
    // Enable all logging down to TRACE level
    rmm::default_logger().set_level(rapids_logger::level_enum::trace);

    // Your code here

    return 0;
}
```

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

Use multiple logging features together:

```python
import rmm

# Enable memory event logging by wrapping with adaptor
base = rmm.mr.CudaAsyncMemoryResource()
log_mr = rmm.mr.LoggingResourceAdaptor(base, log_file_name="events.csv")
rmm.mr.set_current_device_resource(log_mr)

# Enable statistics and profiling
rmm.statistics.enable_statistics()

# Set debug log level
rmm.set_logging_level("debug")

# Now all logging is active
@rmm.statistics.profiler()
def my_function():
    buffer = rmm.DeviceBuffer(size=1024)
    return buffer

my_function()

# Get statistics
stats = rmm.statistics.get_statistics()
print(f"Peak bytes: {stats.peak_bytes}")

# View profiler report
print(rmm.statistics.default_profiler_records.report())
```

C++ equivalent:

```cpp
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/logging_resource_adaptor.hpp>
#include <rmm/mr/device/statistics_resource_adaptor.hpp>
#include <rmm/logger.hpp>

int main() {
    // Set debug log level
    rmm::default_logger().set_level(rapids_logger::level_enum::debug);

    // Build resource stack
    auto cuda_mr = rmm::mr::cuda_async_memory_resource{};
    auto stats_mr = rmm::mr::statistics_resource_adaptor{&cuda_mr};
    auto log_mr = rmm::mr::logging_resource_adaptor{&stats_mr, "events.csv"};

    rmm::mr::set_current_device_resource(&log_mr);

    // Now all logging is active
    rmm::cuda_stream stream;
    rmm::device_buffer buffer(1024, stream.view());

    // Get statistics
    auto stats = stats_mr.get_statistics();
    std::cout << "Peak bytes: " << stats.peak_bytes << "\n";

    return 0;
}
```

## Use Cases

### Debugging OOM Errors

```python
import rmm

# Enable detailed logging
base = rmm.mr.CudaAsyncMemoryResource()
log_mr = rmm.mr.LoggingResourceAdaptor(base, log_file_name="oom_debug.csv")
rmm.mr.set_current_device_resource(log_mr)
rmm.set_logging_level("debug")
rmm.statistics.enable_statistics()

# Run problematic code
try:
    large_buffer = rmm.DeviceBuffer(size=100 * 2**30)  # 100 GiB
except MemoryError as e:
    stats = rmm.statistics.get_statistics()
    print(f"Peak before OOM: {stats.peak_bytes / 2**30:.2f} GiB")
    print(f"Check oom_debug.csv for allocation history")
    raise
```

### Profiling Memory in Data Pipeline

```python
import rmm

rmm.statistics.enable_statistics()

@rmm.statistics.profiler()
def load_data():
    return rmm.DeviceBuffer(size=1000000)

@rmm.statistics.profiler()
def process_data(buffer):
    temp = rmm.DeviceBuffer(size=2000000)
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
```

### Benchmarking Memory Resources

```python
import rmm
import time

def benchmark_allocations(mr_name, mr):
    rmm.mr.set_current_device_resource(mr)

    start = time.time()
    buffers = []
    for _ in range(1000):
        buffers.append(rmm.DeviceBuffer(size=1024))
    end = time.time()

    print(f"{mr_name}: {(end - start) * 1000:.2f} ms for 1000 allocations")

# Compare resources
benchmark_allocations("CudaMemoryResource", rmm.mr.CudaMemoryResource())
benchmark_allocations("CudaAsyncMemoryResource", rmm.mr.CudaAsyncMemoryResource())
pool = rmm.mr.PoolMemoryResource(rmm.mr.CudaAsyncMemoryResource(), initial_pool_size=2**20)
benchmark_allocations("PoolMemoryResource", pool)
```

## Best Practices

1. **Use event logging for debugging** - CSV logs help understand allocation patterns
2. **Enable statistics for profiling** - Track memory usage over time
3. **Use profiler for hotspot analysis** - Identify which functions allocate most memory
4. **Set appropriate debug level** - Use `INFO` normally, `DEBUG`/`TRACE` when troubleshooting
5. **Disable logging in production** - Logging has overhead; only enable when needed
6. **Analyze logs with tools** - Use pandas, REPLAY_BENCHMARK, or custom scripts
7. **Combine with NVIDIA tools** - Use Nsight Systems alongside RMM logging for complete picture

## See Also

- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)
