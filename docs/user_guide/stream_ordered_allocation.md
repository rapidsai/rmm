# Stream-Ordered Memory Allocation

RMM containers ({cpp:class}`~rmm::device_buffer`, {py:class}`~rmm.DeviceBuffer`) and [memory resources](../python/mr.md) are stream-ordered: allocations and deallocations are enqueued on a CUDA stream rather than blocking the CPU. This lets memory operations overlap with kernel execution and avoids the synchronization cost of `cudaMalloc`/`cudaFree`. For background on CUDA streams and asynchronous execution, see the [CUDA Programming Guide: Asynchronous Concurrent Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#what-is-asynchronous-concurrent-execution).

## How It Works

When you allocate from a stream-ordered resource, the call returns a pointer immediately. The pointer value is available on the CPU right away — you can store it, pass it to kernel launch arguments, or hand it to another API. The memory backing behind the pointer becomes available for GPU operations enqueued on the same stream after the allocation:

`````{tabs}
````{group-tab} C++
```{literalinclude} ../../cpp/examples/docs/src/stream_ordered_allocation.cu
---
language: cuda
start-after: "// [how-it-works]"
end-before: "// [/how-it-works]"
dedent:
---
```
````
````{group-tab} Python
```{literalinclude} ../../python/rmm/rmm/tests/examples/stream_ordered_allocation.py
---
language: python
start-after: "# [how-it-works]"
end-before: "# [/how-it-works]"
dedent:
---
```
````
`````

Deallocations are also stream-ordered: when a buffer is destroyed, the deallocation is enqueued on the stream, so the memory is not actually freed until all prior work on that stream completes.

## When to Synchronize

### Reading results on the host

The pointer returned by a stream-ordered allocation is a CPU value — you can store it or pass it to other APIs without synchronization. However, the stream must be synchronized before the CPU reads data that was written by GPU operations on that stream. The most common case is a device-to-host copy followed by a sync:

`````{tabs}
````{group-tab} C++
```{literalinclude} ../../cpp/examples/docs/src/stream_ordered_allocation.cu
---
language: cuda
start-after: "// [reading-results]"
end-before: "// [/reading-results]"
dedent:
---
```
````
````{group-tab} Python
```{literalinclude} ../../python/rmm/rmm/tests/examples/stream_ordered_allocation.py
---
language: python
start-after: "# [reading-results]"
end-before: "# [/reading-results]"
dedent:
---
```
````
`````

### Cross-stream usage

Memory allocated on one stream can only be safely used on a different stream after the allocation is known to have completed. The simplest approach is to synchronize the allocating stream, but that stalls the CPU. A lighter-weight alternative is to record a CUDA event on the allocating stream and have the consuming stream wait on it:

`````{tabs}
````{group-tab} C++
```{literalinclude} ../../cpp/examples/docs/src/stream_ordered_allocation.cu
---
language: cuda
start-after: "// [cross-stream]"
end-before: "// [/cross-stream]"
dedent:
---
```
````
````{group-tab} Python
```{literalinclude} ../../python/rmm/rmm/tests/examples/stream_ordered_allocation.py
---
language: python
start-after: "# [cross-stream]"
end-before: "# [/cross-stream]"
dedent:
---
```
````
`````

### Buffer lifetime across streams

If a buffer is allocated and used on the same stream, deallocation is safe — stream ordering guarantees prior work completes first. The problem arises when a buffer is used on a *different* stream from the one it will be deallocated on. In that case, you need to ensure the consuming stream's work finishes before the buffer is destroyed. The same event pattern works here — record an event on the consuming stream and have the deallocating stream wait on it:

`````{tabs}
````{group-tab} C++
```{literalinclude} ../../cpp/examples/docs/src/stream_ordered_allocation.cu
---
language: cuda
start-after: "// [buffer-lifetime]"
end-before: "// [/buffer-lifetime]"
dedent:
---
```
````
````{group-tab} Python
```{literalinclude} ../../python/rmm/rmm/tests/examples/stream_ordered_allocation.py
---
language: python
start-after: "# [buffer-lifetime]"
end-before: "# [/buffer-lifetime]"
dedent:
---
```
````
`````

## Which Resources Support Stream Ordering?

- **{py:class}`~rmm.mr.CudaAsyncMemoryResource`**: Fully stream-ordered (recommended)
- **{py:class}`~rmm.mr.PoolMemoryResource`**: Internally stream-safe — suballocations are mutex-protected, independent of upstream
- **{py:class}`~rmm.mr.ArenaMemoryResource`**: Internally stream-safe — uses per-stream arenas, independent of upstream
- **{py:class}`~rmm.mr.CudaMemoryResource`**: NOT stream-ordered (`cudaMalloc` is synchronous)
- **{py:class}`~rmm.mr.ManagedMemoryResource`**: NOT stream-ordered (`cudaMallocManaged` is synchronous)

## Example: Numba Kernel with RMM Stream

This example allocates an RMM buffer and launches a Numba kernel on the same stream, so the allocation is guaranteed to complete before the kernel accesses the memory:

```{literalinclude} ../../python/rmm/rmm/tests/examples/stream_ordered_allocation.py
---
language: python
start-after: "# [numba-stream]"
end-before: "# [/numba-stream]"
dedent:
---
```

## See Also

- [Choosing a Memory Resource](choosing_memory_resources.md) - Which resources support stream ordering
- [CUDA Programming Guide: Asynchronous Concurrent Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#what-is-asynchronous-concurrent-execution)
