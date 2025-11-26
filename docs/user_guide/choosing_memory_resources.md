# Choosing a Memory Resource

One of the most common questions when using RMM is: "Which memory resource should I use?"

This guide provides recommendations for selecting the appropriate memory resource based on your application's needs.

## Quick Recommendations

**For most applications**: Use `CudaAsyncMemoryResource` (the CUDA async memory pool).

**For applications larger than GPU memory**: Use `ManagedMemoryResource` with prefetching strategies.

**For specific allocation patterns**: Consider `ArenaMemoryResource` or custom configurations.

## Understanding Memory Resources

RMM provides several memory resource types, each optimized for different use cases:

### CudaAsyncMemoryResource (Recommended Default)

The `CudaAsyncMemoryResource` uses CUDA's driver-managed memory pool (via `cudaMallocAsync`). This is the **recommended default** for most applications.

**Advantages:**
- **Driver-managed pool**: Uses efficient suballocation with virtual addressing to avoid fragmentation
- **Cross-library sharing**: The pool can be shared across multiple applications and libraries, even those not using RMM directly
- **Stream-ordered semantics**: Allocations and deallocations are stream-ordered by default
- **Performance**: Similar or better performance compared to RMM's pool implementations

**Example:**
```python
import rmm

# Set async memory resource as default
rmm.mr.set_current_device_resource(rmm.mr.CudaAsyncMemoryResource())
```

**When to use:**
- Default choice for GPU-accelerated applications
- Multi-stream or multi-threaded applications
- Applications using multiple GPU libraries (e.g., cuDF + PyTorch)
- Most production workloads

### CudaMemoryResource

The `CudaMemoryResource` uses `cudaMalloc` directly for each allocation, with no pooling.

**Advantages:**
- Simple and predictable
- No fragmentation concerns
- Memory is immediately returned to the system on deallocation

**Disadvantages:**
- Slower than pooled allocators due to synchronization overhead
- Each allocation requires a driver call

**Example:**
```python
import rmm

rmm.mr.set_current_device_resource(rmm.mr.CudaMemoryResource())
```

**When to use:**
- Simple applications with infrequent allocations
- Debugging memory issues
- Testing or benchmarking baseline performance

### PoolMemoryResource

The `PoolMemoryResource` maintains a pool of memory allocated from an upstream resource.

**Advantages:**
- Fast suballocation from pre-allocated pool
- Configurable initial and maximum pool sizes

**Disadvantages:**
- Can suffer from fragmentation (unlike async MR)
- Pool is not shared across applications
- Requires careful tuning of pool sizes

**Example:**
```python
import rmm

pool = rmm.mr.PoolMemoryResource(
    rmm.mr.CudaMemoryResource(),  # upstream resource
    initial_pool_size=2**30,  # 1 GiB
    maximum_pool_size=2**32   # 4 GiB
)
rmm.mr.set_current_device_resource(pool)
```

**When to use:**
- Legacy applications (prefer `CudaAsyncMemoryResource` for new code)
- Specific tuning requirements not met by async MR
- Wrapping non-CUDA memory sources

**Important**: If using `PoolMemoryResource`, prefer wrapping `CudaAsyncMemoryResource` as the upstream rather than `CudaMemoryResource`:

```python
# Better: Pool wrapping async MR
pool = rmm.mr.PoolMemoryResource(
    rmm.mr.CudaAsyncMemoryResource(),
    initial_pool_size=2**30
)
```

This combines the benefits of both: fast suballocation from RMM's pool and the driver's virtual addressing capabilities.

### ManagedMemoryResource

The `ManagedMemoryResource` uses CUDA unified memory (via `cudaMallocManaged`), allowing memory to be accessible from both CPU and GPU.

**Advantages:**
- Enables working with datasets larger than GPU memory
- Automatic page migration between CPU and GPU
- Simplifies memory management for host/device code

**Disadvantages:**
- Performance overhead due to page faults and migration
- Requires careful prefetching for optimal performance

**Example:**
```python
import rmm

rmm.mr.set_current_device_resource(rmm.mr.ManagedMemoryResource())
```

**When to use:**
- Datasets larger than available GPU memory
- Prototyping or applications where performance is not critical
- Always combine with prefetching strategies (see [Managed Memory guide](managed_memory.md))

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

# Track allocation statistics
base = rmm.mr.CudaAsyncMemoryResource()
stats = rmm.mr.StatisticsResourceAdaptor(base)
rmm.mr.set_current_device_resource(stats)
```

## Multi-Library Applications

When using RMM with multiple GPU libraries (e.g., cuDF, PyTorch, CuPy), `CudaAsyncMemoryResource` is especially important because:

1. The driver-managed pool is shared automatically across all libraries
2. You don't need to configure every library to use RMM
3. Memory is not artificially partitioned between libraries

**Example: RMM + PyTorch**
```python
import rmm
import torch
from rmm.allocators.torch import rmm_torch_allocator

# Use async MR as the base
rmm.mr.set_current_device_resource(rmm.mr.CudaAsyncMemoryResource())

# Configure PyTorch to use RMM
torch.cuda.memory.change_current_allocator(rmm_torch_allocator)
```

With this setup, both PyTorch and any other RMM-using code (like cuDF) will share the same driver-managed pool.

## Decision Tree

```
Start: What are your requirements?
│
├─ Working with data larger than GPU memory?
│  └─ Use ManagedMemoryResource + PrefetchResourceAdaptor
│     (See managed_memory.md for details)
│
├─ Using multiple GPU libraries (cuDF, PyTorch, etc.)?
│  └─ Use CudaAsyncMemoryResource (enables cross-library sharing)
│
├─ Need statistics or profiling?
│  └─ Wrap your resource with StatisticsResourceAdaptor
│     Example: StatisticsResourceAdaptor(CudaAsyncMemoryResource())
│
├─ Legacy application or specific tuning needs?
│  └─ Use PoolMemoryResource or ArenaMemoryResource
│     (Consider wrapping CudaAsyncMemoryResource as upstream)
│
└─ Default case / Not sure?
   └─ Use CudaAsyncMemoryResource
```

## Performance Considerations

### Async MR vs. Pool MR

In most cases, `CudaAsyncMemoryResource` provides similar or better performance than `PoolMemoryResource`:

- Both use pooling for fast suballocation
- Async MR uses virtual addressing to avoid fragmentation
- Async MR shares memory across applications

**When Pool MR might be faster:**
- Very specific allocation patterns that align well with pool design
- Custom upstream resources (not CUDA memory)

### Multi-stream Applications

For applications using multiple CUDA streams or threads:

- `CudaAsyncMemoryResource` is **strongly recommended**
- Pool allocators can create "pipeline bubbles" where streams wait for allocations
- The async MR handles stream synchronization efficiently

## Best Practices

1. **Set the memory resource before any allocations**: Once memory is allocated, changing the resource can lead to crashes

   ```python
   import rmm

   # Do this first, before any GPU allocations
   rmm.mr.set_current_device_resource(rmm.mr.CudaAsyncMemoryResource())
   ```

2. **Prefer async MR by default**: Unless you have specific requirements, start with `CudaAsyncMemoryResource`

3. **Use statistics for tuning**: If you need to understand allocation patterns, wrap with `StatisticsResourceAdaptor`

4. **Don't over-engineer**: Start simple, profile, and optimize only if needed

## See Also

- [Pool Allocators](pool_allocators.md) - Detailed guide on pool and arena allocators
- [Managed Memory](managed_memory.md) - Guide to using managed memory and prefetching
- [Stream-Ordered Allocation](stream_ordered_allocation.md) - Understanding stream-ordered semantics
