# Record and Replay Memory Events

RMM can write each allocation and deallocation to a CSV file. This file is
called a memory event log. The `REPLAY_BENCH` benchmark reads the log and
replays the allocations and frees against the chosen memory resource.

Use this workflow to reproduce an allocator failure, such as an out of memory
error from `pool_memory_resource`, without the original workload, the original
data, or the original GPU.

This page has six steps. Complete them in order.

## 1. Record the log

### C++

To record a log in C++, wrap a memory resource with
`rmm::mr::logging_resource_adaptor` and set it as the current device resource.

```c++
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/logging_resource_adaptor.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda/stream>

int main()
{
  rmm::mr::cuda_memory_resource upstream{};
  rmm::mr::logging_resource_adaptor log_mr{upstream, "log.csv"};
  rmm::mr::set_current_device_resource(log_mr);

  cuda::stream_ref stream{cudaStream_t{cudaStreamDefault}};

  {
    rmm::device_buffer buf{1 << 20, stream};
  }

  // Write the tail of the log to disk before the process exits.
  log_mr.flush();
  return 0;
}
```

In C++, the file name is used as given. This program writes `log.csv`.

### Python

Set `logging=True` in `rmm.reinitialize()` to record a log.

```python
import rmm

rmm.reinitialize(logging=True, log_file_name="log.csv")

buf = rmm.DeviceBuffer(size=1 << 20)
del buf

rmm.mr.get_current_device_resource().flush()
print(rmm.get_log_filenames())
```

Output:

```text
{0: '/path/to/your/directory/log.dev0.csv'}
```

The requested name `log.csv` becomes `log.dev0.csv`. The number is the CUDA
device ordinal.

Call `rmm.get_log_filenames()` to get the file name. It returns a dictionary.
The key is the device ordinal and the value is the full path.

### Environment variable

`RMM_LOG_FILE` supplies only the file name. It does not turn logging on. When
no file name is provided, RMM reads the file name from `RMM_LOG_FILE`.
Logging must still be enabled separately. In Python, use
`rmm.reinitialize(logging=True)` and omit `log_file_name`. In C++, construct
`logging_resource_adaptor` without a file name.

```bash
RMM_LOG_FILE=mylog.csv python my_workload.py
```

The same device suffix rule applies. `RMM_LOG_FILE=mylog.csv` writes
`mylog.dev0.csv`.

In C++, if no file name is provided and `RMM_LOG_FILE` is not set, the
`logging_resource_adaptor` constructor throws an exception.

### Flush the log

The log is buffered. If the process exits or crashes before the buffer is
written, the end of the log is lost. The end of the log is usually the required
part.

Call `flush()` before normal exit and in every handled error path:

- C++: `log_mr.flush()`
- Python: `rmm.mr.get_current_device_resource().flush()`

An abrupt crash, such as a segfault or abort, kills the process before any
flush call runs. The buffered tail is still lost.

C++ has an `auto_flush` constructor argument. When enabled, it writes every
event at once at a performance cost. The Python API does not expose
`auto_flush`.

### The CSV columns

The log has one header row and one row for each event.

```text
Thread,Time,Action,Pointer,Size,Stream
2342727,15:47:02.841961,allocate,0x10012400000,1048576,0
2342727,15:47:02.841977,free,0x10012400000,1048576,0
```

| Column | Meaning |
| --- | --- |
| `Thread` | Operating system thread id. Events are grouped by thread for replay. |
| `Time` | Wall clock time of the event. Not used by the replay tool. |
| `Action` | `allocate`, `free`, or `allocate failure`. |
| `Pointer` | Address that was returned or freed. Used to match a free to its allocation. |
| `Size` | Size in bytes. |
| `Stream` | The stream of the event. `0` is the default stream. |

## 2. Build the replay tool

`REPLAY_BENCH` is a benchmark. It is not in any conda package or pip wheel.
Build RMM from source to get it.

From the RMM source root, run:

```bash
./build.sh librmm benchmarks
```

CMake can also be configured directly with `-DBUILD_BENCHMARKS=ON`.

The binary is written to:

```text
cpp/build/gbenchmarks/REPLAY_BENCH
```

## 3. Replay the log

```bash
./cpp/build/gbenchmarks/REPLAY_BENCH -f rmm-log.txt -r pool -s 78 --benchmark_min_time=1x
```

### Flags

| Flag | Meaning |
| --- | --- |
| `-f <file>` | The memory event log to replay. Required. |
| `-r <name>` | Memory resource to replay against: `pool`, `arena`, `binning`, `cuda`, or `managed`. |
| `-s <GiB>` | Size of the simulated GPU in GiB. Supported by `pool` and `binning` only. |
| `-v` | Print every event before the replay starts. |
| `--benchmark_min_time=1x` | A Google Benchmark flag. Replay the log one time only. |

`REPLAY_BENCH --help` prints the Google Benchmark flags only. To see the flags
in the table above, run the tool with no `-f`.

### Rule 1: Always pass `-r`

The help text says that the default resource is `pool`. That is not what
happens. Without `-r`, the tool replays the log against all five
resources, one after the other.

This can break the run:

- `pool` and `binning` use `-s` to simulate a large GPU.
- `arena` ignores the simulation and does a real `cudaMalloc` of `-s` GiB. If
  the GPU is smaller than `-s`, this fails at once.
- `cuda` and `managed` ignore `-s` and run against the real GPU.

The run stops at the first resource that fails. With `-s 80` and no `-r` on a
16 GiB GPU, the pool result is correct, and then `arena` kills the run:

```text
Pool Resource/threads:1          1.56 ms         1.56 ms            1
[info  ] ------ Start of Benchmark -----
Exception caught: std::bad_alloc: out_of_memory: CUDA error (failed to allocate 85899345920 bytes) at: .../cuda_memory_resource.cpp:26: cudaErrorMemoryAllocation out of memory
```

Always pass `-r pool` or `-r binning`.

### Rule 2: Use whole numbers for `-s`

`-s` is a float in GiB. RMM converts it to bytes. A fractional value can give a
size that is not a multiple of 256 bytes, and the pool rejects it. For example,
`-s 0.05` gives:

```text
Exception caught: RMM failure at: .../pool_memory_resource_impl.cpp:40: Error, Initial pool size required to be a multiple of 256 bytes
```

Use whole numbers.

### Rule 3: Always pass `--benchmark_min_time=1x`

Google Benchmark repeats the benchmark until it reaches a minimum time. For a
2115 event log, this results in hundreds of replays:

```text
Benchmark                        Time             CPU   Iterations
Pool Resource/threads:1      0.744 ms        0.742 ms          937
```

Use `--benchmark_min_time=1x` to run one replay instead of 937.

### Rule 4: Check the first line of the output

Unknown positional arguments are ignored. There is no warning. Typing `0s 78`
instead of `-s 78` runs the tool with no simulated GPU and replays the log
against the real GPU. The failure then looks like a real
`cudaErrorMemoryAllocation`, and it is easy to report it as the bug.

Before trusting a result, check the first line of the output:

```text
Simulating GPU with memory size of 83751862272 bytes.
```

If that line is not there, the simulation is off and the result is not the
intended result.

## 4. Read the result

> :warning: The process exits with code 0 in every case. A failure, a missing
> log file, and a clean run all give exit code 0. Do not use the exit code. Read
> the output text.

### Failure

Two lines show a failure. The `[error]` line comes first:

```text
[error ] [A][Stream 0x1][Upstream 11136000000B][FAILURE maximum pool size exceeded: Not enough room to grow, current/max/try size = 78.000000 GiB, 78.000000 GiB, 10.371208 GiB]
Exception caught: std::bad_alloc: out_of_memory: RMM failure at:.../pool_memory_resource_impl.cpp:73: Maximum pool size exceeded (failed to allocate 10.371208 GiB): Not enough room to grow, current/max/try size = 78.000000 GiB, 78.000000 GiB, 10.371208 GiB
```

The text `Maximum pool size exceeded` means that the pool could not grow. The
three sizes are the current pool size, the maximum pool size, and the size of
the allocation that failed.

There is no results table after a failure.

### Success

A successful replay prints the `End of Benchmark` line and a results table:

```text
[info  ] ------ End of Benchmark -----
------------------------------------------------------------------
Benchmark                        Time             CPU   Iterations
------------------------------------------------------------------
Pool Resource/threads:1       1.59 ms         1.58 ms            1
```

The `Pool Resource/threads:1` row is proof that the replay finished.

### Missing or unreadable log file

```text
Failed to parse events: basic_ios::clear: iostream error
Simulating GPU with memory size of 83751862272 bytes.
Total Events: 0
...
Exception caught: cannot create std::vector larger than max_size()
```

`Total Events: 0` means that the tool read no events. Check the path of the log
file. This is not an allocator bug.

## 5. Worked example: issue #1969

[Issue #1969](https://github.com/rapidsai/rmm/issues/1969) reports pool
fragmentation. The reporter attached a memory event log. Download that log from
the issue and save it in the RMM source root as `rmm-log.txt`.

The log contains 2115 events on one thread.

### The failing size

```bash
./cpp/build/gbenchmarks/REPLAY_BENCH -f rmm-log.txt -r pool -s 78 --benchmark_min_time=1x
```

```text
Simulating GPU with memory size of 83751862272 bytes.
Total Events: 2115
Thread 0: 2115 events
[info  ] ------ Start of Benchmark -----
[error ] [A][Stream 0x1][Upstream 11136000000B][FAILURE maximum pool size exceeded: Not enough room to grow, current/max/try size = 78.000000 GiB, 78.000000 GiB, 10.371208 GiB]
Exception caught: std::bad_alloc: out_of_memory: RMM failure at:.../pool_memory_resource_impl.cpp:73: Maximum pool size exceeded (failed to allocate 10.371208 GiB): Not enough room to grow, current/max/try size = 78.000000 GiB, 78.000000 GiB, 10.371208 GiB
```

### The passing size

```bash
./cpp/build/gbenchmarks/REPLAY_BENCH -f rmm-log.txt -r pool -s 79 --benchmark_min_time=1x
```

```text
Simulating GPU with memory size of 84825604096 bytes.
Total Events: 2115
Thread 0: 2115 events
[info  ] ------ Start of Benchmark -----
[info  ] ------ End of Benchmark -----
------------------------------------------------------------------
Benchmark                        Time             CPU   Iterations
------------------------------------------------------------------
Pool Resource/threads:1       1.10 ms         1.09 ms            1
```

The log needs 73.2 GiB of live memory at its peak, but it fails at a simulated
78 GiB. The gap is fragmentation. This is the bug in the issue.

### The threshold moves between versions

The exact pass and fail sizes depend on the RMM version. The pool changes, so
the fragmentation changes.

| RMM version | Fails at | Passes at |
| --- | --- | --- |
| 25.x (reported in the issue) | `-s 82` | `-s 83` |
| main at 904372c1 (VERSION 26.12.00) | `-s 78` | `-s 79` |

Do not treat 78 and 79 as fixed numbers. Find the threshold for the version
under test, and report the version with the result.

## 6. For AI agents

Follow these steps in order.

1. Download the memory event log from the issue. Save it in the RMM source root
   as `rmm-log.txt`. Copy the attachment link from the issue page and run
   `curl -L -o rmm-log.txt <link>`.
2. Build the tool with `./build.sh librmm benchmarks`. The binary is
   `cpp/build/gbenchmarks/REPLAY_BENCH`.
3. Replay at the GiB size that the reporter gave. Use a whole number. If the
   reporter gave a GPU size and a percent, multiply them and round down to a
   whole GiB:
   `./cpp/build/gbenchmarks/REPLAY_BENCH -f rmm-log.txt -r pool -s <N> --benchmark_min_time=1x`
4. Replay again at `-s <N+1>`. If both fail, go up. If both pass, go down.
   Repeat until one size fails and one size passes. Step by 1 GiB.
5. Decide pass or fail from the output text, as defined in section 4. The exit
   code is always 0 and provides no useful information. Check that the first line of the
   output says `Simulating GPU with memory size of`. If it does not, the flags
   were wrong. Run the command again.
6. Report these four things:
   - The largest size that fails.
   - The smallest size that passes.
   - The complete `Maximum pool size exceeded` line.
   - The RMM version and the git commit used for the build. Run `cat VERSION`
     and `git rev-parse --short HEAD` in the source root and report both.

## Limitations

These are known properties of the tool today.

- The process always exits with code 0. Every exception is caught and printed.
  Read the output text.
- Google Benchmark repeats the replay. Use `--benchmark_min_time=1x` for one
  replay.
- Events on a non-default stream are replayed on the default stream. The tool
  has a check for this, but the check does not fire, because the `Stream` column
  in the log is hexadecimal and the parser reads it as 0. The replay is still
  useful for reproduction, but it is not a copy of the original stream order.
- Rows with the action `allocate failure` are parsed and then skipped. They are
  not replayed.
- The error message gives the sizes, but not the index of the failed event and
  not the live byte count at that point. Searching the log for the failed
  allocation size narrows the search, but it can match several rows. The log
  from issue #1969 has 12 rows with the failed size.
