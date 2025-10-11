# Managed Memory and Prefetching

CUDA Managed Memory (also called Unified Memory) allows memory to be accessed from both CPU and GPU, with automatic page migration managed by the CUDA driver. RMM provides `ManagedMemoryResource` to leverage this capability.

## What is Managed Memory?

Managed memory creates a single address space accessible from both CPU and GPU:

- Allocations can be accessed using the same pointer from host or device code
- The CUDA driver automatically migrates pages between CPU and GPU as needed
- Enables working with datasets **larger than GPU memory**

## When to Use Managed Memory

Managed memory is ideal for:

1. **Datasets larger than GPU memory**: When your data doesn't fit in VRAM
2. **Prototyping**: Simplifies development by removing explicit memory transfers
3. **CPU-GPU interoperability**: When you need to access the same data from both host and device

**Important**: Managed memory has performance implications. Always combine with prefetching for production workloads.

## Basic Usage

### Python

```python
import rmm

# Use managed memory as the default resource
rmm.mr.set_current_device_resource(rmm.mr.ManagedMemoryResource())

# Or use rmm.reinitialize
rmm.reinitialize(managed_memory=True)

# Allocations now use managed memory
buffer = rmm.DeviceBuffer(size=1000000)
```

### C++

```cpp
#include <rmm/mr/device/managed_memory_resource.hpp>

auto managed_mr = rmm::mr::managed_memory_resource{};
rmm::mr::set_current_device_resource(&managed_mr);

// Allocations use managed memory
rmm::device_buffer buffer(1000000);
```

## Performance Considerations

### Page Faults and Migration

When the GPU accesses managed memory that is not resident on the GPU, a **page fault** occurs:

1. GPU execution pauses
2. The driver migrates the page from CPU to GPU
3. GPU execution resumes

These page faults can significantly impact performance, especially for:
- First-touch access patterns
- Random memory access
- Large datasets that don't fit in GPU memory

### The Prefetching Solution

**Prefetching** explicitly migrates data to the GPU before it's accessed, eliminating page faults.

## Prefetching Strategies

There are two main strategies for prefetching:

### 1. Prefetch on Allocate (Eager Prefetching)

Automatically prefetch memory to the GPU when it's allocated. This is useful when you know the data will be used on the GPU immediately after allocation.

**Implementation: Use `PrefetchResourceAdaptor`**

```python
import rmm

# Wrap managed memory with prefetch adaptor
base = rmm.mr.ManagedMemoryResource()
prefetch_mr = rmm.mr.PrefetchResourceAdaptor(base)
rmm.mr.set_current_device_resource(prefetch_mr)

# Every allocation is automatically prefetched to the GPU
buffer = rmm.DeviceBuffer(size=1000000)
# Buffer is already on the GPU, no page faults on first access
```

**With a pool:**

```python
import rmm

# Combine managed memory, pool, and prefetching
base = rmm.mr.ManagedMemoryResource()
pool = rmm.mr.PoolMemoryResource(base, initial_pool_size=2**30)
prefetch_mr = rmm.mr.PrefetchResourceAdaptor(pool)
rmm.mr.set_current_device_resource(prefetch_mr)
```

**When to use:**
- Allocations are immediately used on the GPU
- You want automatic prefetching without code changes

### 2. Prefetch on Access (Lazy Prefetching)

Explicitly prefetch data just before it's used in a kernel. This gives finer control and can optimize for specific access patterns.

**Implementation: Manual prefetch calls**

```python
import rmm

rmm.mr.set_current_device_resource(rmm.mr.ManagedMemoryResource())

# Allocate managed memory (not prefetched yet)
buffer = rmm.DeviceBuffer(size=1000000)

# ... later, just before using on GPU ...
stream = rmm.cuda_stream()
buffer.prefetch(device=0, stream=stream)  # Prefetch to device 0

# Launch kernel on the same stream
# ... kernel will not page fault ...
```

**In C++:**

```cpp
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/prefetch.hpp>

auto managed_mr = rmm::mr::managed_memory_resource{};
rmm::mr::set_current_device_resource(&managed_mr);

rmm::cuda_stream stream;
rmm::device_buffer buffer(1000000, stream.view());

// Prefetch before using
rmm::prefetch(buffer.data(), buffer.size(),
              rmm::get_current_cuda_device(), stream.view());

// Launch kernel
launch_kernel<<<grid, block, 0, stream.value()>>>(buffer.data());
```

**When to use:**
- You need fine-grained control over when data is prefetched
- Access patterns are complex or dynamic
- You're optimizing for specific workload characteristics

## Practical Example: PyTorch with Larger-Than-VRAM Models

Here's how to use managed memory with PyTorch to work with models or data larger than GPU memory:

```python
import rmm
import torch
from rmm.allocators.torch import rmm_torch_allocator

# Use managed memory with prefetching
base = rmm.mr.ManagedMemoryResource()
pool = rmm.mr.PoolMemoryResource(base, initial_pool_size=2**30, maximum_pool_size=2**34)
prefetch_mr = rmm.mr.PrefetchResourceAdaptor(pool)
rmm.mr.set_current_device_resource(prefetch_mr)

# Configure PyTorch to use RMM
torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

# Now you can work with larger-than-VRAM data
# Example: Large tensor that doesn't fit in VRAM
large_tensor = torch.randn(100000, 100000, device='cuda')  # ~40 GB

# Operations will automatically page as needed
result = large_tensor @ large_tensor.T
```

**What happens:**
1. RMM allocates managed memory for tensors
2. The prefetch adaptor prefetches to GPU on allocation
3. If memory exceeds GPU capacity, pages migrate between CPU and GPU
4. Performance is better than without prefetching

## Prefetching Best Practices

### 1. Prefetch Adaptor Should Be Outermost

When composing memory resources, always make the prefetch adaptor the outermost layer:

```python
# Correct: Prefetch is outermost
base = rmm.mr.ManagedMemoryResource()
pool = rmm.mr.PoolMemoryResource(base, initial_pool_size=2**30)
stats = rmm.mr.StatisticsResourceAdaptor(pool)
prefetch_mr = rmm.mr.PrefetchResourceAdaptor(stats)  # Outermost
rmm.mr.set_current_device_resource(prefetch_mr)

# Incorrect: Prefetch is not outermost
base = rmm.mr.ManagedMemoryResource()
prefetch_mr = rmm.mr.PrefetchResourceAdaptor(base)
pool = rmm.mr.PoolMemoryResource(prefetch_mr, initial_pool_size=2**30)  # Wrong!
```

### 2. Prefetch on the Correct Stream

When manually prefetching, use the same stream as the subsequent kernel:

```python
stream = rmm.cuda_stream()

# Prefetch on stream
buffer.prefetch(device=0, stream=stream)

# Use on the same stream
with stream:
    # ... operations using buffer ...
```

### 3. Prefetch Size Considerations

Prefetching is most effective when:
- The prefetch size is large enough to amortize the migration cost
- Data is used shortly after prefetching
- Access patterns are predictable

### 4. Profile and Measure

Always profile to verify that prefetching improves performance:

```python
import rmm
import time

# Without prefetching
rmm.mr.set_current_device_resource(rmm.mr.ManagedMemoryResource())
buffer = rmm.DeviceBuffer(size=10**9)
start = time.time()
# ... run workload ...
print(f"Without prefetch: {time.time() - start:.2f}s")

# With prefetching
base = rmm.mr.ManagedMemoryResource()
prefetch_mr = rmm.mr.PrefetchResourceAdaptor(base)
rmm.mr.set_current_device_resource(prefetch_mr)
buffer = rmm.DeviceBuffer(size=10**9)
start = time.time()
# ... run workload ...
print(f"With prefetch: {time.time() - start:.2f}s")
```

Use NVIDIA Nsight Systems to visualize page faults and data migration:

```bash
nsys profile -o output python your_script.py
```

## Managed Memory Limitations

### 1. Not Stream-Ordered

`ManagedMemoryResource` uses `cudaMallocManaged`, which is **synchronous**. Allocations block until complete, unlike stream-ordered resources.

For better performance in multi-stream applications, use `CudaAsyncMemoryResource` instead.

### 2. Performance Overhead

Even with prefetching, managed memory has overhead compared to explicit memory management:
- Page fault handling
- Driver page migration
- Potential CPU-GPU transfer latency

For performance-critical code with data that fits in GPU memory, prefer `CudaAsyncMemoryResource`.

### 3. PCIe Bandwidth Limitation

If your workload constantly migrates data between CPU and GPU, you're limited by PCIe bandwidth:
- PCIe Gen3 x16: ~12 GB/s
- PCIe Gen4 x16: ~24 GB/s
- PCIe Gen5 x16: ~48 GB/s

For such workloads, consider:
- Algorithmic changes to reduce data movement
- Using system memory as a staging area
- Streaming data in smaller chunks

## Comparison: Prefetch Strategies

| Strategy | Advantages | Disadvantages | Use Case |
|----------|-----------|---------------|----------|
| **PrefetchResourceAdaptor** | Automatic, no code changes | Prefetches everything, even if not needed | General-purpose, allocate-and-use patterns |
| **Manual prefetch** | Fine-grained control, can optimize specific patterns | Requires code changes | Complex access patterns, performance tuning |
| **No prefetching** | Simple | High page fault overhead | Prototyping only, not for production |

## Multi-GPU Considerations

When using managed memory with multiple GPUs:

```python
import rmm

# Set up managed memory on each device
for device_id in [0, 1]:
    with cuda.Device(device_id):
        base = rmm.mr.ManagedMemoryResource()
        prefetch_mr = rmm.mr.PrefetchResourceAdaptor(base)
        rmm.mr.set_per_device_resource(device_id, prefetch_mr)

# Prefetch to specific devices
buffer = rmm.DeviceBuffer(size=1000000)
buffer.prefetch(device=0, stream=stream_0)  # Prefetch to GPU 0
buffer.prefetch(device=1, stream=stream_1)  # Prefetch to GPU 1
```

## Summary

- Managed memory enables larger-than-VRAM workloads and simplifies CPU-GPU interoperability
- Always use prefetching in production to avoid page fault overhead
- Use `PrefetchResourceAdaptor` for automatic, eager prefetching
- Use manual `prefetch()` calls for fine-grained control
- Profile with Nsight Systems to measure page fault overhead
- For best performance with data that fits in VRAM, use `CudaAsyncMemoryResource` instead

## See Also

- [Choosing a Memory Resource](choosing_memory_resources.md) - When to use managed memory vs. other resources
- [Stream-Ordered Allocation](stream_ordered_allocation.md) - Understanding asynchronous allocation semantics
- [NVIDIA Developer Blog: Unified Memory](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)
- [NVIDIA Developer Blog: Memory Oversubscription](https://developer.nvidia.com/blog/improving-gpu-memory-oversubscription-performance/)
