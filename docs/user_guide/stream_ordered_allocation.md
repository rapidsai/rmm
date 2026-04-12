# Stream-Ordered Memory Allocation

RMM containers and memory resources are stream-ordered: allocations and deallocations are enqueued on a CUDA stream rather than blocking the CPU. This lets memory operations overlap with kernel execution and avoids the synchronization cost of `cudaMalloc`/`cudaFree`. For background on CUDA streams and asynchronous execution, see the [CUDA Programming Guide: Asynchronous Concurrent Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#what-is-asynchronous-concurrent-execution).

## How It Works

When you allocate from a stream-ordered resource, the call returns a pointer immediately. The pointer value is available on the CPU right away — you can store it, pass it to kernel launch arguments, or hand it to another API. The memory backing behind the pointer becomes available for GPU operations enqueued on the same stream after the allocation:

`````{tabs}
````{code-tab} c++
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/device_buffer.hpp>

rmm::mr::cuda_async_memory_resource mr;
rmm::cuda_stream stream;
rmm::device_buffer buffer(1000, stream.view(), mr);

// buffer.data() is usable immediately in stream-ordered operations
launch_kernel<<<grid, block, 0, stream.value()>>>(buffer.data());
````
````{code-tab} python
import rmm
from rmm.pylibrmm.stream import Stream

mr = rmm.mr.CudaAsyncMemoryResource()
stream = Stream()
buffer = rmm.DeviceBuffer(size=1000, stream=stream, mr=mr)

# buffer.ptr is usable immediately in stream-ordered operations
````
`````

Deallocations are also stream-ordered: when a buffer is destroyed, the deallocation is enqueued on the stream, so the memory is not actually freed until all prior work on that stream completes.

## When to Synchronize

### Reading results on the host

The pointer returned by a stream-ordered allocation is a CPU value — you can store it or pass it to other APIs without synchronization. However, the stream must be synchronized before the CPU reads data that was written by GPU operations on that stream. The most common case is a device-to-host copy followed by a sync:

`````{tabs}
````{code-tab} c++
rmm::mr::cuda_async_memory_resource mr;
rmm::cuda_stream stream;
rmm::device_buffer d_buf(1000 * sizeof(float), stream.view(), mr);

// Launch kernel that writes to d_buf on stream ...

// Copy results to host on the same stream
std::vector<float> h_buf(1000);
cudaMemcpyAsync(h_buf.data(), d_buf.data(), d_buf.size(),
                cudaMemcpyDeviceToHost, stream.value());

// Synchronize before reading h_buf on the CPU
stream.synchronize();
````
````{code-tab} python
import rmm
from rmm.pylibrmm.stream import Stream

mr = rmm.mr.CudaAsyncMemoryResource()
stream = Stream()
d_buf = rmm.DeviceBuffer(size=1000, stream=stream, mr=mr)

# ... GPU work writes to d_buf on stream ...

# Async copy to host on the same stream, then sync before reading
h_buf = bytearray(d_buf.size)
d_buf.copy_to_host(h_buf, stream)
stream.synchronize()
````
`````

### Cross-stream usage

Memory allocated on one stream can only be safely used on a different stream after the allocation is known to have completed. The simplest approach is to synchronize the allocating stream, but that stalls the CPU. A lighter-weight alternative is to record a CUDA event on the allocating stream and have the consuming stream wait on it:

`````{tabs}
````{code-tab} c++
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/device_buffer.hpp>

rmm::mr::cuda_async_memory_resource mr;
rmm::cuda_stream stream_a;
rmm::cuda_stream stream_b;

rmm::device_buffer buffer(1000, stream_a.view(), mr);

// Record an event after the allocation on stream_a
cudaEvent_t event;
cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
cudaEventRecord(event, stream_a.value());

// stream_b waits for the event — no CPU synchronization needed
cudaStreamWaitEvent(stream_b.value(), event);

// Now safe to use buffer.data() in operations on stream_b
launch_kernel<<<grid, block, 0, stream_b.value()>>>(buffer.data());

cudaEventDestroy(event);
````
````{code-tab} python
import rmm
from rmm.pylibrmm.stream import Stream
from cuda.core import Device

dev = Device()
dev.set_current()

mr = rmm.mr.CudaAsyncMemoryResource()
stream_a = dev.create_stream()
stream_b = dev.create_stream()

buffer = rmm.DeviceBuffer(size=1000, stream=Stream(obj=stream_a), mr=mr)

# Record an event after the allocation on stream_a
alloc_event = dev.create_event(options={"enable_timing": False})
stream_a.record(alloc_event)

# stream_b waits for the event — no CPU synchronization needed
stream_b.wait(alloc_event)

# Now safe to use buffer.ptr in operations on stream_b
````
`````

### Buffer lifetime across streams

If a buffer is allocated and used on the same stream, deallocation is safe — stream ordering guarantees prior work completes first. The problem arises when a buffer is used on a *different* stream from the one it will be deallocated on. In that case, you need to ensure the consuming stream's work finishes before the buffer is destroyed. The same event pattern works here — record an event on the consuming stream and have the deallocating stream wait on it:

`````{tabs}
````{code-tab} c++
rmm::mr::cuda_async_memory_resource mr;
rmm::cuda_stream stream_a;
rmm::cuda_stream stream_b;

rmm::device_buffer buffer(1000, stream_a.view(), mr);

// Make stream_b wait for the allocation on stream_a
cudaEvent_t alloc_event;
cudaEventCreateWithFlags(&alloc_event, cudaEventDisableTiming);
cudaEventRecord(alloc_event, stream_a.value());
cudaStreamWaitEvent(stream_b.value(), alloc_event);

// Use buffer on stream_b
launch_kernel<<<grid, block, 0, stream_b.value()>>>(buffer.data());

// Before destroying buffer, make stream_a wait for stream_b's work
cudaEvent_t done_event;
cudaEventCreateWithFlags(&done_event, cudaEventDisableTiming);
cudaEventRecord(done_event, stream_b.value());
cudaStreamWaitEvent(stream_a.value(), done_event);

// Now safe to destroy buffer — deallocation on stream_a is ordered after the kernel on stream_b
buffer = rmm::device_buffer{};

cudaEventDestroy(alloc_event);
cudaEventDestroy(done_event);
````
````{code-tab} python
import rmm
from rmm.pylibrmm.stream import Stream
from cuda.core import Device

dev = Device()
dev.set_current()

mr = rmm.mr.CudaAsyncMemoryResource()
stream_a = dev.create_stream()
stream_b = dev.create_stream()

buffer = rmm.DeviceBuffer(size=1000, stream=Stream(obj=stream_a), mr=mr)

# Make stream_b wait for the allocation on stream_a
alloc_event = dev.create_event(options={"enable_timing": False})
stream_a.record(alloc_event)
stream_b.wait(alloc_event)

# Use buffer on stream_b ...

# Before destroying buffer, make stream_a wait for stream_b's work
done_event = dev.create_event(options={"enable_timing": False})
stream_b.record(done_event)
stream_a.wait(done_event)

# Now safe to destroy buffer
del buffer
````
`````

## Which Resources Support Stream Ordering?

- **`CudaAsyncMemoryResource`**: Fully stream-ordered (recommended)
- **`PoolMemoryResource`**: Internally stream-safe — suballocations are mutex-protected, independent of upstream
- **`ArenaMemoryResource`**: Internally stream-safe — uses per-stream arenas, independent of upstream
- **`CudaMemoryResource`**: NOT stream-ordered (`cudaMalloc` is synchronous)
- **`ManagedMemoryResource`**: NOT stream-ordered (`cudaMallocManaged` is synchronous)

## Example: Numba Kernel with RMM Stream

This example allocates an RMM buffer and launches a Numba kernel on the same stream, so the allocation is guaranteed to complete before the kernel accesses the memory:

```python
import rmm
from rmm.pylibrmm.stream import Stream
from cuda.core import Device
from numba import cuda

dev = Device()
dev.set_current()

@cuda.jit
def kernel(data, n):
    idx = cuda.grid(1)
    if idx < n:
        data[idx] = idx * 2

mr = rmm.mr.CudaAsyncMemoryResource()
stream = dev.create_stream()

buffer = rmm.DeviceBuffer(size=1000 * 4, stream=Stream(obj=stream), mr=mr)

numba_stream = cuda.external_stream(int(stream.handle))
kernel[100, 10, numba_stream](cuda.as_cuda_array(buffer).view('float32'), 1000)

stream.sync()
```

## See Also

- [Choosing a Memory Resource](choosing_memory_resources.md) - Which resources support stream ordering
- [CUDA Programming Guide: Asynchronous Concurrent Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#what-is-asynchronous-concurrent-execution)
