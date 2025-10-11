# Pool Memory Allocators

Pool allocators maintain a "pool" of pre-allocated memory to enable fast suballocation without repeatedly calling the underlying memory allocation API. RMM provides several pool-based memory resources, each with different characteristics and use cases.

## Why Use Pool Allocators?

Direct allocation (e.g., `cudaMalloc`) has overhead:
- Requires driver synchronization
- Can be slow for small, frequent allocations
- Forces serialization of allocation requests

Pool allocators address this by:
- Pre-allocating large blocks of memory
- Suballocating from the pool without driver calls
- Reusing freed memory for new allocations

## RMM's Pool Allocators

RMM provides three main pool-like allocators:

1. **`CudaAsyncMemoryResource`**: Driver-managed pool (recommended default)
2. **`PoolMemoryResource`**: RMM-managed coalescing pool
3. **`ArenaMemoryResource`**: Size-binned arena pool

## CudaAsyncMemoryResource (Recommended)

The `CudaAsyncMemoryResource` uses CUDA's driver-managed memory pool via `cudaMallocAsync`.

**Advantages:**
- Virtual address space management (avoids fragmentation)
- Shared across all applications using the same GPU
- Stream-ordered allocation
- No manual tuning of pool sizes

**Example:**
```python
import rmm

rmm.mr.set_current_device_resource(rmm.mr.CudaAsyncMemoryResource())
```

**When to use:** Default choice for most applications. See [Choosing a Memory Resource](choosing_memory_resources.md) for details.

## PoolMemoryResource

The `PoolMemoryResource` wraps an upstream memory resource and maintains a pool using a coalescing best-fit allocator.

### Configuration

```python
import rmm

pool = rmm.mr.PoolMemoryResource(
    upstream=rmm.mr.CudaMemoryResource(),  # or CudaAsyncMemoryResource
    initial_pool_size=2**30,  # 1 GiB - initial allocation
    maximum_pool_size=2**32   # 4 GiB - max the pool can grow to
)
rmm.mr.set_current_device_resource(pool)
```

### Parameters

- **`upstream`**: The underlying memory resource to allocate from
  - Use `CudaAsyncMemoryResource()` for best results
  - `CudaMemoryResource()` for basic CUDA memory
  - Can be any memory resource (including another pool!)

- **`initial_pool_size`**: Size of the initial allocation
  - Larger values reduce early-stage growth overhead
  - Should be based on your typical memory usage
  - Use string notation: `"1GiB"`, `"512MiB"`, etc.
  - Or use powers of 2: `2**30` (1 GiB)

- **`maximum_pool_size`**: Maximum size the pool can grow to
  - Acts as a limit on total GPU memory usage
  - `None` means no limit (pool can grow until GPU memory is exhausted)
  - Useful for multi-tenant or multi-process scenarios

### How It Works

1. **Initial allocation**: On first use, allocates `initial_pool_size` from upstream
2. **Suballocation**: Subsequent allocations are served from the pool
3. **Growth**: If pool is exhausted, allocates more from upstream
4. **Coalescing**: Adjacent freed blocks are merged to reduce fragmentation
5. **Shrinking**: The pool does **not** automatically return memory to upstream

### Best Practices

#### 1. Choose Appropriate Pool Sizes

**Initial pool size:**
- Profile your application to understand memory usage
- Set initial size to ~80% of typical peak usage
- Too small: frequent growth overhead
- Too large: wastes memory, longer startup

**Example:**
```python
import rmm

# For an application that typically uses 2 GiB
pool = rmm.mr.PoolMemoryResource(
    rmm.mr.CudaAsyncMemoryResource(),
    initial_pool_size=int(1.6 * 2**30),  # 1.6 GiB
    maximum_pool_size=int(4 * 2**30)     # 4 GiB max
)
rmm.mr.set_current_device_resource(pool)
```

#### 2. Prefer Async MR as Upstream

Wrapping `CudaAsyncMemoryResource` combines benefits:

```python
# Good: Pool wrapping async MR
pool = rmm.mr.PoolMemoryResource(
    rmm.mr.CudaAsyncMemoryResource(),
    initial_pool_size=2**30
)
```

This gives:
- Fast suballocation from RMM pool
- Driver's virtual addressing for fragmentation resistance
- Shared memory pool across libraries

#### 3. Avoid Double Pooling

Don't wrap a pool in another pool:

```python
# Bad: Double pooling
inner_pool = rmm.mr.PoolMemoryResource(rmm.mr.CudaMemoryResource(), 2**30)
outer_pool = rmm.mr.PoolMemoryResource(inner_pool, 2**30)  # Wasteful!
```

#### 4. Monitor Fragmentation

Pool allocators can suffer from fragmentation:

```python
import rmm

# Enable statistics to monitor fragmentation
pool = rmm.mr.PoolMemoryResource(rmm.mr.CudaAsyncMemoryResource(), 2**30)
stats_mr = rmm.mr.StatisticsResourceAdaptor(pool)
rmm.mr.set_current_device_resource(stats_mr)

# Run workload
# ...

# Check statistics
stats = rmm.statistics.get_statistics()
print(f"Peak bytes: {stats.peak_bytes}")
print(f"Current bytes: {stats.current_bytes}")
```

If `peak_bytes` is much larger than needed, fragmentation may be occurring.

### Common Issues

#### Issue 1: Out of Memory (OOM) Before Max Pool Size

**Symptom:** OOM errors even though allocated memory is less than `maximum_pool_size`

**Cause:** Fragmentation. The pool has free memory, but not in contiguous blocks.

**Solutions:**
1. Use `ArenaMemoryResource` instead (better fragmentation characteristics)
2. Use `CudaAsyncMemoryResource` (virtual addressing prevents fragmentation)
3. Adjust allocation patterns to reduce fragmentation

#### Issue 2: Pool Doesn't Shrink

**Symptom:** Memory remains allocated even after deallocations

**Cause:** By design, pools don't return memory to the upstream resource.

**Solutions:**
1. Destroy and recreate the pool (not recommended for long-running applications)
2. Set appropriate `maximum_pool_size` to limit growth
3. Use `CudaAsyncMemoryResource` if memory should be returned to the system

## ArenaMemoryResource

The `ArenaMemoryResource` divides memory into size-binned arenas to reduce fragmentation.

### Configuration

```python
import rmm

arena = rmm.mr.ArenaMemoryResource(
    upstream=rmm.mr.CudaMemoryResource(),
    arena_size=2**28,  # 256 MiB per arena
    dump_log_on_failure=False
)
rmm.mr.set_current_device_resource(arena)
```

### How It Works

1. Allocates memory in fixed-size "arenas"
2. Each arena is divided into size-binned "superblocks"
3. Allocations are served from the appropriate bin
4. Reduces fragmentation by isolating allocation sizes

### When to Use

- Applications with diverse allocation sizes
- Long-running services with complex allocation patterns
- When `PoolMemoryResource` suffers from fragmentation

### Example: Mixed Allocation Sizes

```python
import rmm

# Application allocates small (KB), medium (MB), and large (GB) buffers
arena = rmm.mr.ArenaMemoryResource(
    rmm.mr.CudaAsyncMemoryResource(),
    arena_size=2**28  # 256 MiB arenas
)
rmm.mr.set_current_device_resource(arena)

# Allocations are binned by size
small = rmm.DeviceBuffer(size=1024)      # Small bin
medium = rmm.DeviceBuffer(size=1024**2)  # Medium bin
large = rmm.DeviceBuffer(size=1024**3)   # Large bin
```

## BinningMemoryResource

The `BinningMemoryResource` routes allocations to different memory resources based on size.

### Configuration

```python
import rmm

# Create resources for different size ranges
small_mr = rmm.mr.FixedSizeMemoryResource(
    rmm.mr.CudaMemoryResource(),
    block_size=256  # 256 bytes
)
large_mr = rmm.mr.PoolMemoryResource(
    rmm.mr.CudaMemoryResource(),
    initial_pool_size=2**30
)

# Bin allocations by size
binning_mr = rmm.mr.BinningMemoryResource(
    upstream=large_mr,  # Default for allocations not in bins
)

# Add bins: allocations of size <= threshold go to this resource
binning_mr.add_bin(256, small_mr)    # <= 256 bytes -> small_mr
binning_mr.add_bin(1024, None)       # <= 1 KiB -> upstream (large_mr)
# Anything > 1 KiB goes to upstream (large_mr)

rmm.mr.set_current_device_resource(binning_mr)
```

### How It Works

Allocations are routed based on size:
```
Allocation size <= bin1_threshold -> bin1_resource
Allocation size <= bin2_threshold -> bin2_resource
...
Allocation size > largest_threshold -> upstream
```

### Best Practices for Binning

#### 1. Profile Allocation Sizes

Before configuring bins, understand your allocation patterns:

```python
import rmm

# Enable statistics to see allocation sizes
base = rmm.mr.CudaMemoryResource()
stats_mr = rmm.mr.StatisticsResourceAdaptor(base)
rmm.mr.set_current_device_resource(stats_mr)

# Run workload
# ...

# Analyze allocation patterns
stats = rmm.statistics.get_statistics()
print(stats)
```

#### 2. Optimize for Common Sizes

Configure bins to match your most common allocation sizes:

```python
import rmm

# Based on profiling, we know:
# - Many small allocations (< 1 KiB)
# - Medium allocations (1 KiB - 1 MiB)
# - Large allocations (> 1 MiB)

# Fixed-size resource for small allocations
small_mr = rmm.mr.FixedSizeMemoryResource(
    rmm.mr.CudaAsyncMemoryResource(),
    block_size=1024  # 1 KiB
)

# Pool for medium allocations
medium_mr = rmm.mr.PoolMemoryResource(
    rmm.mr.CudaAsyncMemoryResource(),
    initial_pool_size=2**28  # 256 MiB
)

# Pool for large allocations
large_mr = rmm.mr.PoolMemoryResource(
    rmm.mr.CudaAsyncMemoryResource(),
    initial_pool_size=2**30  # 1 GiB
)

# Configure binning
binning_mr = rmm.mr.BinningMemoryResource(upstream=large_mr)
binning_mr.add_bin(1024, small_mr)      # <= 1 KiB
binning_mr.add_bin(1024**2, medium_mr)  # <= 1 MiB
# > 1 MiB goes to large_mr

rmm.mr.set_current_device_resource(binning_mr)
```

#### 3. Consider Using ArenaMemoryResource Instead

For many use cases, `ArenaMemoryResource` provides similar benefits with simpler configuration:

```python
# Simpler: Arena handles size-binning automatically
arena = rmm.mr.ArenaMemoryResource(
    rmm.mr.CudaAsyncMemoryResource(),
    arena_size=2**28
)
rmm.mr.set_current_device_resource(arena)
```

### Example: PyTorch with Binning

From issue #1958, here's a practical example for PyTorch workloads:

```python
import rmm
import torch
from rmm.allocators.torch import rmm_torch_allocator

# Use managed memory as base (for larger-than-VRAM scenarios)
upstream = rmm.mr.ManagedMemoryResource()

# Create a pool wrapping managed memory
pool = rmm.mr.PoolMemoryResource(
    upstream,
    initial_pool_size=2**20,  # 1 MiB
    maximum_pool_size=int(80 * 2**30)  # 80 GiB max
)

# Fixed-size resource for small allocations
fixed_mr = rmm.mr.FixedSizeMemoryResource(pool, block_size=1024)  # 1 KiB blocks

# Binning resource
binning_mr = rmm.mr.BinningMemoryResource(upstream=pool)

# Add bins for common PyTorch tensor sizes
binning_mr.add_bin(256 * 1024, fixed_mr)      # <= 256 KiB
binning_mr.add_bin(512 * 1024, None)          # <= 512 KiB -> pool
binning_mr.add_bin(1024 * 1024, None)         # <= 1 MiB -> pool
binning_mr.add_bin(2 * 1024 * 1024, None)     # <= 2 MiB -> pool
binning_mr.add_bin(4 * 1024 * 1024, None)     # <= 4 MiB -> pool
# > 4 MiB goes to pool

rmm.mr.set_current_device_resource(binning_mr)

# Configure PyTorch
torch.cuda.memory.change_current_allocator(rmm_torch_allocator)
```

**Note:** For production PyTorch workloads, prefer `CudaAsyncMemoryResource` unless you specifically need managed memory for larger-than-VRAM scenarios.

## Choosing Between Pool Allocators

| Resource | Best For | Fragmentation Handling | Complexity |
|----------|----------|------------------------|------------|
| **CudaAsyncMemoryResource** | General purpose, multi-stream apps | Excellent (virtual addressing) | Low |
| **PoolMemoryResource** | Simple pooling needs | Fair (coalescing) | Low |
| **ArenaMemoryResource** | Diverse allocation sizes | Good (size binning) | Medium |
| **BinningMemoryResource** | Custom size-based routing | Depends on configuration | High |

## Debugging Pool Issues

### Enable Logging

```python
import rmm

arena = rmm.mr.ArenaMemoryResource(
    rmm.mr.CudaMemoryResource(),
    arena_size=2**28,
    dump_log_on_failure=True  # Log on allocation failure
)
rmm.mr.set_current_device_resource(arena)
```

### Track Statistics

```python
import rmm

pool = rmm.mr.PoolMemoryResource(rmm.mr.CudaAsyncMemoryResource(), 2**30)
stats_mr = rmm.mr.StatisticsResourceAdaptor(pool)
rmm.mr.set_current_device_resource(stats_mr)

# Run workload
buffer = rmm.DeviceBuffer(size=1000000)

# Check usage
stats = rmm.statistics.get_statistics()
print(f"Current bytes: {stats.current_bytes:,}")
print(f"Peak bytes: {stats.peak_bytes:,}")
print(f"Total allocations: {stats.total_count}")
```

### Profile with Nsight Systems

```bash
nsys profile -o output python your_script.py
```

Look for:
- Allocation frequency and sizes
- Memory usage over time
- Fragmentation indicators

## Summary

- **For most cases**: Use `CudaAsyncMemoryResource` (driver-managed pool)
- **For simple pooling**: Use `PoolMemoryResource` wrapping `CudaAsyncMemoryResource`
- **For fragmentation issues**: Try `ArenaMemoryResource`
- **For size-based routing**: Use `BinningMemoryResource` (or `ArenaMemoryResource`)
- **Always profile**: Use statistics and Nsight Systems to understand allocation patterns
- **Set appropriate pool sizes**: Too small causes growth overhead, too large wastes memory

## See Also

- [Choosing a Memory Resource](choosing_memory_resources.md) - High-level guidance on selecting resources
- [Stream-Ordered Allocation](stream_ordered_allocation.md) - Understanding async allocation
- [RMM Statistics Documentation](https://docs.rapids.ai/api/rmm/stable/guide/#memory-statistics-and-profiling)
