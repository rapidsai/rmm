# Stream-Ordered Memory Allocation

RMM provides **stream-ordered memory allocation**, which means that memory allocations and deallocations are ordered with respect to operations on a CUDA stream. This is a fundamental concept for achieving optimal performance in asynchronous CUDA applications.

## What is Stream-Ordered Allocation?

In stream-ordered allocation:

1. **Allocations are asynchronous**: Calling `allocate()` schedules the allocation on a stream and returns immediately
2. **Memory is available after stream synchronization**: The allocated memory is guaranteed to be available for use by operations scheduled after the allocation on the same stream
3. **Deallocations are also stream-ordered**: Memory is not actually freed until all prior operations on the stream complete

This allows memory operations to be interleaved with kernel launches and other CUDA operations without explicit synchronization.

## Why Stream-Ordered Allocation Matters

Traditional memory allocation (e.g., `cudaMalloc`) is **synchronous** - it blocks until the allocation completes. This creates bubbles in the execution pipeline where the CPU waits for GPU operations to complete.

Stream-ordered allocation enables:
- **Overlapping compute and memory operations**: Allocations can be scheduled while kernels are running
- **Reduced synchronization overhead**: No need to synchronize the stream before allocating
- **Better multi-stream performance**: Different streams can allocate independently

## How It Works

Consider the following example of allocating memory from a stream-ordered memory resource.

C++:

```cpp
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/device_buffer.hpp>

rmm::cuda_stream_view stream;
auto buffer = rmm::device_buffer(1000, stream);
```

Python:

```python
import rmm

# Allocate on a specific stream
stream = rmm.cuda_stream()
buffer = rmm.DeviceBuffer(size=1000, stream=stream)
```

The following happens:

1. The allocation request is **scheduled** on `stream`
2. The function returns immediately (asynchronous)
3. The memory is **guaranteed to be available** for operations enqueued on `stream` after the allocation
4. You can use `buffer.data()` (the pointer) immediately in subsequent stream operations

## Key Semantics

### Safe to Use the Pointer Immediately

**You can use the returned pointer in stream-ordered operations without synchronization:**

```python
import rmm
import cupy as cp

stream = rmm.cuda_stream()

# Allocate memory on the stream
buffer = rmm.DeviceBuffer(size=1000, stream=stream)

# Use the pointer immediately in a CuPy operation on the same stream
# This is SAFE - no synchronization needed
with stream:
    array = cp.ndarray(shape=(250,), dtype=cp.float32,
                       memptr=cp.cuda.MemoryPointer(
                           cp.cuda.UnownedMemory(buffer.ptr, buffer.size, buffer),
                           0))
    # Kernel launches on this stream will see the allocated memory
    array[:] = 42
```

The allocation is guaranteed to complete before the kernel that uses it, as long as both are on the same stream.

### Deallocations Are Also Stream-Ordered

When you deallocate (e.g., a buffer goes out of scope), the deallocation is also stream-ordered:

```python
import rmm

stream = rmm.cuda_stream()

# Allocate
buffer = rmm.DeviceBuffer(size=1000, stream=stream)

# Schedule some work on the stream
# ... kernels using buffer.ptr ...

# When buffer is destroyed, deallocation is scheduled on the stream
# The memory won't actually be freed until all prior work completes
buffer = None  # triggers deallocation
```

This ensures that:
- Memory is not freed while still in use by a kernel
- Deallocations don't block waiting for kernels to complete

### Stream Synchronization

To guarantee that an allocation has completed (for example, if you need to access it from the CPU), synchronize the stream:

```python
import rmm

stream = rmm.cuda_stream()
buffer = rmm.DeviceBuffer(size=1000, stream=stream)

# Synchronize to ensure allocation completes
stream.synchronize()

# Now safe to do CPU operations with buffer.ptr
# (though accessing GPU memory from CPU usually requires managed memory)
```

## Memory Resources and Stream Ordering

### Which Resources Support Stream Ordering?

- **`CudaAsyncMemoryResource`**: Fully stream-ordered (recommended)
- **`PoolMemoryResource`**: Can be stream-ordered when wrapping a stream-ordered upstream
- **`ArenaMemoryResource`**: Stream-ordered when wrapping a stream-ordered upstream
- **`CudaMemoryResource`**: NOT stream-ordered (synchronous `cudaMalloc`)
- **`ManagedMemoryResource`**: NOT stream-ordered (synchronous `cudaMallocManaged`)

### Example: Pool Wrapping Async MR

```python
import rmm

# Create a pool that maintains stream-ordered semantics
pool = rmm.mr.PoolMemoryResource(
    rmm.mr.CudaAsyncMemoryResource(),  # stream-ordered upstream
    initial_pool_size=2**30
)
rmm.mr.set_current_device_resource(pool)

# Allocations from this pool are stream-ordered
stream = rmm.cuda_stream()
buffer = rmm.DeviceBuffer(size=1000, stream=stream)
```

## Common Patterns

### Pattern 1: Allocate and Use in Kernel

```python
import rmm
from numba import cuda

@cuda.jit
def kernel(data, n):
    idx = cuda.grid(1)
    if idx < n:
        data[idx] = idx * 2

stream = rmm.cuda_stream()

# Allocate
buffer = rmm.DeviceBuffer(size=1000 * 4, stream=stream)  # 1000 float32s

# Use immediately
with stream:
    kernel[100, 10](cuda.as_cuda_array(buffer).view('float32'), 1000)

# Synchronize to wait for kernel
stream.synchronize()
```

### Pattern 2: Allocate, Compute, Deallocate, Repeat

```python
import rmm

stream = rmm.cuda_stream()

for i in range(100):
    # Allocate
    buffer = rmm.DeviceBuffer(size=1000000, stream=stream)

    # Use buffer in computations
    # ... launch kernels on stream ...

    # Deallocate (automatic, or explicitly set buffer = None)
    buffer = None

# All allocations and deallocations are stream-ordered
# No need to synchronize between iterations
```

### Pattern 3: Multi-Stream Allocation

```python
import rmm

# Create multiple streams
streams = [rmm.cuda_stream() for _ in range(4)]

# Allocate on different streams independently
buffers = []
for stream in streams:
    # Each allocation is independent
    buffer = rmm.DeviceBuffer(size=1000000, stream=stream)
    buffers.append(buffer)

    # Launch work on this stream
    # ... kernels using buffer ...

# Synchronize all streams
for stream in streams:
    stream.synchronize()
```

## Performance Implications

### Benefits

1. **Reduced CPU-GPU synchronization**: No blocking on allocations
2. **Better pipeline utilization**: Memory operations overlap with compute
3. **Multi-stream scalability**: Streams can allocate independently

### Pitfalls to Avoid

1. **Don't mix streams**: Using memory allocated on stream A in operations on stream B requires synchronization:

   ```python
   stream_a = rmm.cuda_stream()
   stream_b = rmm.cuda_stream()

   # Allocate on stream A
   buffer = rmm.DeviceBuffer(size=1000, stream=stream_a)

   # To use on stream B, synchronize stream A first
   stream_a.synchronize()

   # Now safe to use on stream B
   with stream_b:
       # ... operations using buffer ...
   ```

2. **Don't access from CPU without sync**: Stream-ordered allocations are asynchronous - accessing from CPU requires synchronization:

   ```python
   stream = rmm.cuda_stream()
   buffer = rmm.DeviceBuffer(size=1000, stream=stream)

   # BAD: May access uninitialized memory
   # some_function(buffer.ptr)

   # GOOD: Synchronize first
   stream.synchronize()
   some_function(buffer.ptr)
   ```

3. **Resource lifetime**: Ensure buffers live until all stream operations complete:

   ```python
   stream = rmm.cuda_stream()

   def allocate_and_use():
       buffer = rmm.DeviceBuffer(size=1000, stream=stream)
       # Launch kernel using buffer
       kernel[...](buffer.ptr)
       # BAD: buffer is deallocated when function returns
       # but kernel may still be running!

   allocate_and_use()
   stream.synchronize()  # May crash - buffer already freed
   ```

   Fix: Keep buffer alive until synchronization:

   ```python
   stream = rmm.cuda_stream()
   buffer = allocate_and_use()  # Return the buffer
   stream.synchronize()  # Now safe
   buffer = None  # Explicit cleanup after sync
   ```

## C++ API

In C++, stream-ordered allocation is the default for most RMM containers:

```cpp
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/cuda_stream_view.hpp>

// Set async MR as default
auto async_mr = rmm::mr::cuda_async_memory_resource{};
rmm::mr::set_current_device_resource(&async_mr);

// Create a stream
rmm::cuda_stream stream;

// Allocate stream-ordered memory
rmm::device_buffer buffer(1000, stream.view());
rmm::device_uvector<float> vec(1000, stream.view());

// Use immediately in stream-ordered operations
launch_kernel<<<grid, block, 0, stream.value()>>>(buffer.data(), vec.data());

// Synchronize
stream.synchronize();
```

## Summary

- Stream-ordered allocation enables asynchronous, non-blocking memory operations
- Allocated pointers can be used immediately in subsequent operations on the same stream
- Deallocations are also stream-ordered, preventing use-after-free
- `CudaAsyncMemoryResource` provides the best stream-ordered allocation support
- Always synchronize before accessing memory from the CPU
- Ensure buffer lifetimes extend until all stream operations complete

For more details on choosing memory resources, see [Choosing a Memory Resource](choosing_memory_resources.md).
