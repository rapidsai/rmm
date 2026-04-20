# Choosing a Memory Resource

One of the most common questions when using RMM is: "Which memory resource should I use?"

This guide recommends memory resources based on optimal allocation performance for common workloads. See the API references for the full list of available resources.

## Recommended Defaults

For most applications, the CUDA async memory pool provides the best allocation performance with no tuning required.

`````{tabs}
````{group-tab} C++
```{literalinclude} ../../cpp/examples/docs/src/choosing_memory_resources.cpp
---
language: cpp
start-after: "// [recommended-default]"
end-before: "// [/recommended-default]"
dedent:
---
```
````
````{group-tab} Python
```{literalinclude} ../../python/rmm/rmm/tests/examples/choosing_memory_resources.py
---
language: python
start-after: "# [recommended-default]"
end-before: "# [/recommended-default]"
dedent:
---
```
````
`````

For applications that require GPU memory oversubscription (allocating more memory than physically available on the GPU), use a pooled managed memory resource with prefetching. This uses [CUDA Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html) (`cudaMallocManaged`) to enable automatic page migration between CPU and GPU at the cost of slower allocation performance. Coupling the managed memory "base" allocator with adaptors for pool allocation and prefetching to device on allocation recovers some of the performance lost to the overhead of managed allocations. Note: Managed memory has [limited support on WSL2](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#unified-memory-on-windows-wsl-and-tegra).

`````{tabs}
````{group-tab} C++
```{literalinclude} ../../cpp/examples/docs/src/choosing_memory_resources.cpp
---
language: cpp
start-after: "// [managed-pool-prefetch]"
end-before: "// [/managed-pool-prefetch]"
dedent:
---
```
````
````{group-tab} Python
```{literalinclude} ../../python/rmm/rmm/tests/examples/choosing_memory_resources.py
---
language: python
start-after: "# [managed-pool-prefetch]"
end-before: "# [/managed-pool-prefetch]"
dedent:
---
```
````
`````

## Memory Resource Considerations

Resources that use the CUDA driver's pool suballocation (`cudaMallocFromPoolAsync`) provide fast allocation performance because the driver can manage virtual address space efficiently and reduce fragmentation.

### CUDA Async Memory Resource

{cpp:class}`~rmm::mr::cuda_async_memory_resource` (C++) / {py:class}`~rmm.mr.CudaAsyncMemoryResource` (Python) allocates from a custom CUDA memory pool using `cudaMallocFromPoolAsync`. This is the **recommended default** for most applications.

Note: This creates a *custom* mempool, not the default device mempool. A custom pool is used to enable features like Blackwell decompression engine support and custom release thresholds.

**Features:**
- **Fast allocation**: Driver-managed pool reuses previously allocated memory
- **Reduced fragmentation**: Virtual addressing allows non-contiguous physical memory to back contiguous allocations, unlike `PoolMemoryResource` which requires contiguous free regions
- **Stream-ordered semantics**: Allocations and deallocations are stream-ordered by default, avoiding pipeline stalls in multi-stream workloads
- **Low configuration**: The driver manages pool growth automatically, though release threshold and maximum size may need tuning in some environments (e.g., when co-existing with libraries that allocate outside the pool)

**When to use:**
- Default choice for GPU-accelerated applications
- Multi-stream or multi-threaded applications
- Most production workloads

### CUDA Memory Resource

{cpp:class}`~rmm::mr::cuda_memory_resource` (C++) / {py:class}`~rmm.mr.CudaMemoryResource` (Python) uses the legacy `cudaMalloc`/`cudaFree` APIs directly with no pooling or stream-ordering support. It is generally not recommended.

**When to use:**
- Debugging memory issues (to isolate allocator-related problems)
- Benchmarking baseline allocation overhead

### Managed Memory Resource

{cpp:class}`~rmm::mr::managed_memory_resource` (C++) / {py:class}`~rmm.mr.ManagedMemoryResource` (Python) allocates [CUDA Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html) via `cudaMallocManaged`. Unified Memory creates a single address space accessible from both CPU and GPU, with the CUDA driver migrating pages between processors on demand. This enables [GPU memory oversubscription](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html) — allocating more memory than physically available on the GPU — but generally comes with a performance cost.

**Features:**
- Enables GPU memory oversubscription for datasets larger than GPU memory
- Automatic page migration between CPU and GPU

**Caution:**
By default, managed memory adds overhead for page faults and migration (see [Performance Tuning](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#performance-tuning) in the CUDA Programming Guide). See the [Managed Memory guide](managed_memory.md) for a recommended solution with a pool and prefetching adaptor.

**When to use:**
- Datasets larger than available GPU memory
- Typically combined with a pool and prefetching (see [Managed Memory guide](managed_memory.md))

**Example:**

```{literalinclude} ../../python/rmm/rmm/tests/examples/choosing_memory_resources.py
---
language: python
start-after: "# [managed-memory-example]"
end-before: "# [/managed-memory-example]"
dedent:
---
```

### Pool Memory Resource

{cpp:class}`~rmm::mr::pool_memory_resource` (C++) / {py:class}`~rmm.mr.PoolMemoryResource` (Python) maintains a pool of memory allocated from an upstream resource, providing fast suballocation.

**Features:**
- Fast suballocation from pre-allocated pool
- Configurable initial and maximum pool sizes for explicit memory budgeting

**When to use:**
- The [Managed Memory guide](managed_memory.md) provides a good example of usage, because initial allocations of managed memory can be slow. The pool resource amortizes that initial cost over the lifetime of the pool.

**Caution:**
There are pool implementations in both RMM (this memory resource) and in the CUDA driver (leveraging `cudaMallocFromPoolAsync` and `cudaMemPool_t`).
The RMM pool implementation is not as good at handling fragmentation compared to the CUDA driver.
Also, RMM's pool can be slower than the CUDA driver's pool implementation in heavy multi-stream workloads depending on application details.

**Note**: `PoolMemoryResource` does not return memory to the upstream resource on deallocation. Once the pool grows, that memory stays allocated until the resource is destroyed. Set `maximum_pool_size` to limit growth.

**Example:**

```{literalinclude} ../../python/rmm/rmm/tests/examples/choosing_memory_resources.py
---
language: python
start-after: "# [pool-memory-example]"
end-before: "# [/pool-memory-example]"
dedent:
---
```

## Composing Memory Resources

Memory resources can be composed (wrapped) to combine their properties. The general pattern is:

```{literalinclude} ../../python/rmm/rmm/tests/examples/choosing_memory_resources.py
---
language: python
start-after: "# [composing-adaptor]"
end-before: "# [/composing-adaptor]"
dedent:
---
```

### Common Compositions

**Prefetching with managed memory:**

```{literalinclude} ../../python/rmm/rmm/tests/examples/choosing_memory_resources.py
---
language: python
start-after: "# [prefetch-composition]"
end-before: "# [/prefetch-composition]"
dedent:
---
```

**Statistics tracking** (see [Logging and Profiling](logging.md)):

```{literalinclude} ../../python/rmm/rmm/tests/examples/choosing_memory_resources.py
---
language: python
start-after: "# [statistics-composition]"
end-before: "# [/statistics-composition]"
dedent:
---
```

**Allocation logging** (see [Logging and Profiling](logging.md)):

```{literalinclude} ../../python/rmm/rmm/tests/examples/choosing_memory_resources.py
---
language: python
start-after: "# [logging-composition]"
end-before: "# [/logging-composition]"
dedent:
---
```

## Multi-Library Applications

When using RMM with multiple GPU libraries (e.g., cuDF, PyTorch, CuPy), configuring each library to allocate through RMM ensures all allocations flow through the same resource. This avoids memory partitioning where each library holds its own pool, leaving less memory available for the others.

Each library must be explicitly configured to use RMM. RMM provides allocator integrations for common libraries:

**Example: RMM + PyTorch**

```{literalinclude} ../../python/rmm/rmm/tests/examples/choosing_memory_resources.py
---
language: python
start-after: "# [multi-library-pytorch]"
end-before: "# [/multi-library-pytorch]"
dedent:
---
```

With this setup, both PyTorch and any other RMM-configured library (like cuDF) allocate from the same resource.

## Best Practices

1. **Set the memory resource before any allocations**: Changing the resource after allocations have been made can lead to crashes.

   ```{literalinclude} ../../python/rmm/rmm/tests/examples/choosing_memory_resources.py
   ---
   language: python
   start-after: "# [best-practices-set-early]"
   end-before: "# [/best-practices-set-early]"
   dedent:
   ---
   ```

2. **Use adaptors for diagnostics**: Wrap with {cpp:class}`~rmm::mr::statistics_resource_adaptor` (C++) / {py:class}`~rmm.mr.StatisticsResourceAdaptor` (Python) to track allocation counts and peak usage, or {cpp:class}`~rmm::mr::logging_resource_adaptor` (C++) / {py:class}`~rmm.mr.LoggingResourceAdaptor` (Python) to log every allocation and deallocation (see [Logging and Profiling](logging.md)).

## See Also

- [Managed Memory](managed_memory.md) - Guide to using managed memory and prefetching
- [Stream-Ordered Allocation](stream_ordered_allocation.md) - Understanding stream-ordered semantics
