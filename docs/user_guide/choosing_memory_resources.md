# Choosing a Memory Resource

One of the most common questions when using RMM is: "Which memory resource should I use?"

This guide recommends memory resources based on optimal allocation performance for common workloads.

## Recommended Defaults

For most applications, the CUDA async memory pool provides the best allocation performance with no tuning required.

`````{tabs}
````{code-tab} c++
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

rmm::mr::cuda_async_memory_resource mr;
rmm::mr::set_current_device_resource_ref(mr);
````
````{code-tab} python
import rmm

mr = rmm.mr.CudaAsyncMemoryResource()
rmm.mr.set_current_device_resource(mr)
````
`````

For applications that require GPU memory oversubscription (allocating more memory than physically available on the GPU), use a pooled managed memory resource with prefetching. This uses [CUDA Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html) (`cudaMallocManaged`) to enable automatic page migration between CPU and GPU at the cost of slower allocation performance. Coupling the managed memory "base" allocator with adaptors for pool allocation and prefetching to device on allocation recovers some of the performance lost to the overhead of managed allocations. Note: Managed memory has [limited support on WSL2](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#unified-memory-on-windows-wsl-and-tegra).

`````{tabs}
````{code-tab} c++
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/prefetch_resource_adaptor.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/cuda_device.hpp>

// Use 80% of GPU memory, rounded down to nearest 256 bytes
auto [free_memory, total_memory] = rmm::available_device_memory();
std::size_t pool_size = (static_cast<std::size_t>(total_memory * 0.8) / 256) * 256;

rmm::mr::managed_memory_resource managed_mr;
rmm::mr::pool_memory_resource pool_mr{managed_mr, pool_size};
rmm::mr::prefetch_resource_adaptor prefetch_mr{pool_mr};
rmm::mr::set_current_device_resource_ref(prefetch_mr);
````
````{code-tab} python
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
rmm.mr.set_current_device_resource(mr)
````
`````

## Memory Resource Considerations

Resources that use the CUDA driver's pool suballocation (`cudaMallocFromPoolAsync`) provide fast allocation performance because the driver can manage virtual address space efficiently and reduce fragmentation.

### CudaAsyncMemoryResource

The `CudaAsyncMemoryResource` allocates from a custom CUDA memory pool using `cudaMallocFromPoolAsync`. This is the **recommended default** for most applications.

Note: This creates a *custom* mempool, not the default device mempool. A custom pool is used to enable features like Blackwell decompression engine support and custom release thresholds.

**Advantages:**
- **Fast allocation**: Driver-managed pool reuses previously allocated memory
- **Reduced fragmentation**: Virtual addressing allows non-contiguous physical memory to back contiguous allocations, unlike `PoolMemoryResource` which requires contiguous free regions
- **Stream-ordered semantics**: Allocations and deallocations are stream-ordered by default, avoiding pipeline stalls in multi-stream workloads
- **Low configuration**: The driver manages pool growth automatically, though release threshold and maximum size may need tuning in some environments (e.g., when co-existing with libraries that allocate outside the pool)

**When to use:**
- Default choice for GPU-accelerated applications
- Multi-stream or multi-threaded applications
- Most production workloads

### CudaMemoryResource

The `CudaMemoryResource` uses the legacy `cudaMalloc`/`cudaFree` APIs directly with no pooling or stream-ordering support. It is generally not recommended.

**When to use:**
- Debugging memory issues (to isolate allocator-related problems)
- Benchmarking baseline allocation overhead

### PoolMemoryResource

The `PoolMemoryResource` maintains a pool of memory allocated from an upstream resource. It provides fast suballocation but requires manual tuning for pool sizes and does not match the performance of `CudaAsyncMemoryResource` in multi-stream workloads.

**Advantages:**
- Fast suballocation from pre-allocated pool
- Configurable initial and maximum pool sizes for explicit memory budgeting

**Disadvantages:**
- **Slower than async MR** in multi-stream workloads due to internal locking
- Can suffer from fragmentation (async MR reduces this with virtual addressing)
- Pool cannot be shared across CUDA applications unless all applications are using RMM
- May require tuning of pool size for optimal performance

**When to use:**
- Explicit memory budgeting with fixed pool sizes
- Wrapping non-CUDA memory sources (e.g., managed memory)
- Prefer `CudaAsyncMemoryResource` for new code unless you need explicit pool size control

**Note**: `PoolMemoryResource` does not return memory to the upstream resource on deallocation. Once the pool grows, that memory stays allocated until the resource is destroyed. Set `maximum_pool_size` to limit growth.

**Example:**
```python
import rmm

pool = rmm.mr.PoolMemoryResource(
    rmm.mr.CudaMemoryResource(),
    initial_pool_size=2**32,  #  4 GiB
    maximum_pool_size=2**34   # 16 GiB
)
rmm.mr.set_current_device_resource(pool)
```

### ManagedMemoryResource

The `ManagedMemoryResource` allocates [CUDA Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html) via `cudaMallocManaged`. Unified Memory creates a single address space accessible from both CPU and GPU, with the CUDA driver migrating pages between processors on demand. This enables [GPU memory oversubscription](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html) — allocating more memory than physically available on the GPU — but generally comes with a performance cost.

**Advantages:**
- Enables GPU memory oversubscription for datasets larger than GPU memory
- Automatic page migration between CPU and GPU

**Disadvantages:**
- **Slower than device memory** due to page faults and migration overhead, especially in multi-stream workloads (see [Performance Tuning](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#performance-tuning) in the CUDA Programming Guide)
- Requires prefetching to achieve acceptable performance (see [Managed Memory guide](managed_memory.md))

**Example:**
```python
import rmm

# Always combine managed memory with a pool and prefetching for acceptable
# performance. Without prefetching, page faults cause significant overhead,
# especially in multi-stream workloads.
base = rmm.mr.ManagedMemoryResource()
pool = rmm.mr.PoolMemoryResource(base, initial_pool_size=2**30)
prefetch_mr = rmm.mr.PrefetchResourceAdaptor(pool)
rmm.mr.set_current_device_resource(prefetch_mr)
```

**When to use:**
- Datasets larger than available GPU memory
- Always combine with a pool and prefetching (see [Managed Memory guide](managed_memory.md))

### ArenaMemoryResource

The `ArenaMemoryResource` divides a large allocation into size-binned arenas, reducing fragmentation.

**Advantages:**
- Better fragmentation characteristics than basic pool
- Good for mixed allocation sizes
- Predictable performance

**Disadvantages:**
- More complex configuration
- May waste memory if bin sizes don't match allocation patterns

**Example:**
```python
import rmm

arena = rmm.mr.ArenaMemoryResource(
    rmm.mr.CudaMemoryResource(),
    arena_size=2**28  # 256 MiB arenas
)
rmm.mr.set_current_device_resource(arena)
```

**When to use:**
- Applications with diverse allocation sizes
- Long-running services with complex allocation patterns
- When fragmentation is observed with pool allocators

## Composing Memory Resources

Memory resources can be composed (wrapped) to combine their properties. The general pattern is:

```python
# Adaptor wrapping a base resource
adaptor = rmm.mr.SomeAdaptor(base_resource)
```

### Common Compositions

**Prefetching with managed memory:**
```python
import rmm

# Prefetch adaptor wrapping managed memory pool
base = rmm.mr.ManagedMemoryResource()
pool = rmm.mr.PoolMemoryResource(base, initial_pool_size=2**30)
prefetch = rmm.mr.PrefetchResourceAdaptor(pool)
rmm.mr.set_current_device_resource(prefetch)
```

**Statistics tracking:**
```python
import rmm

# Track allocation statistics (counts, peak, and total bytes)
base = rmm.mr.CudaAsyncMemoryResource()
stats = rmm.mr.StatisticsResourceAdaptor(base)
rmm.mr.set_current_device_resource(stats)
```

**Allocation logging:**
```python
import rmm

# Log every allocation and deallocation to a file
base = rmm.mr.CudaAsyncMemoryResource()
logged = rmm.mr.LoggingResourceAdaptor(base, log_file_name="allocations.csv")
rmm.mr.set_current_device_resource(logged)
```

## Multi-Library Applications

When using RMM with multiple GPU libraries (e.g., cuDF, PyTorch, CuPy), configuring each library to allocate through RMM ensures all allocations flow through the same resource. This avoids memory partitioning where each library holds its own pool, leaving less memory available for the others.

Each library must be explicitly configured to use RMM. RMM provides allocator integrations for common libraries:

**Example: RMM + PyTorch**
```python
import rmm
import torch
from rmm.allocators.torch import rmm_torch_allocator

# Configure RMM
rmm.mr.set_current_device_resource(rmm.mr.CudaAsyncMemoryResource())

# Configure PyTorch to allocate through RMM
torch.cuda.memory.change_current_allocator(rmm_torch_allocator)
```

With this setup, both PyTorch and any other RMM-configured library (like cuDF) allocate from the same resource.

## Best Practices

1. **Set the memory resource before any allocations**: Changing the resource after allocations have been made can lead to crashes.

   ```python
   import rmm

   # Do this first, before any GPU allocations
   rmm.mr.set_current_device_resource(rmm.mr.CudaAsyncMemoryResource())
   ```

2. **Use adaptors for diagnostics**: Wrap with `StatisticsResourceAdaptor` to track allocation counts and peak usage, or `LoggingResourceAdaptor` to log every allocation and deallocation (see [Logging and Profiling](logging.md)).

## See Also

- [Managed Memory](managed_memory.md) - Guide to using managed memory and prefetching
- [Stream-Ordered Allocation](stream_ordered_allocation.md) - Understanding stream-ordered semantics
