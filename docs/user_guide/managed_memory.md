# Managed Memory and Prefetching

CUDA Managed Memory (also called Unified Memory) provides a single address space accessible from both CPU and GPU. The CUDA driver migrates pages between host and device memory on demand, which means you can work with datasets larger than GPU memory or share data between host and device code without explicit copies.

RMM's {cpp:class}`~rmm::mr::managed_memory_resource` (C++) / {py:class}`~rmm.mr.ManagedMemoryResource` (Python) allocates managed memory via `cudaMallocManaged`. For background on how Unified Memory works at the driver level, see the [CUDA Programming Guide: Unified Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#unified-memory-programming).

The main trade-off is performance: on-demand page migration introduces latency from page faults. For production workloads, combining managed memory with prefetching (described below) is essential to avoid this overhead.

## Prefetching

Without prefetching, the first GPU access to a managed allocation triggers a page fault that stalls execution while the driver migrates data from host memory. If the working set exceeds GPU memory, pages get evicted and re-faulted repeatedly, which can degrade performance severely. The [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#performance-tuning) covers page fault behavior and tuning in detail.

Prefetching migrates data to the GPU ahead of time so that kernels find it already resident. RMM supports two approaches.

### Prefetch on Allocate (Eager)

{cpp:class}`~rmm::mr::prefetch_resource_adaptor` (C++) / {py:class}`~rmm.mr.PrefetchResourceAdaptor` (Python) wraps another resource and prefetches each allocation to the current device as soon as it's made. This works well when data is used on the GPU shortly after allocation, such as when copying or writing to the new allocation:

```python
import rmm

managed_mr = rmm.mr.ManagedMemoryResource()
prefetch_mr = rmm.mr.PrefetchResourceAdaptor(managed_mr)

# This allocation is prefetched to the GPU automatically
buffer = rmm.DeviceBuffer(size=1000000, mr=prefetch_mr)
```

Adding a pool between the managed resource and the prefetch adaptor avoids calling `cudaMallocManaged` on every allocation. The pool grabs large chunks of managed memory upfront, and the prefetch adaptor ensures each suballocation is migrated to the GPU before use. Non-allocating adaptors like logging or statistics can safely wrap the prefetch adaptor on the outside:

```python
import rmm

managed_mr = rmm.mr.ManagedMemoryResource()
pool_mr = rmm.mr.PoolMemoryResource(managed_mr, initial_pool_size=2**30)
prefetch_mr = rmm.mr.PrefetchResourceAdaptor(pool_mr)

# Logging and statistics don't allocate, so they can go on the outside
stats_mr = rmm.mr.StatisticsResourceAdaptor(prefetch_mr)
log_mr = rmm.mr.LoggingResourceAdaptor(stats_mr, log_file_name="log.csv")

buffer = rmm.DeviceBuffer(size=1000000, mr=log_mr)
```

### Prefetch on Access (Lazy)

When you need control over exactly when data moves to the GPU — for instance because the allocation happens long before the kernel that consumes it — you can prefetch manually:

`````{tabs}
````{code-tab} c++
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/prefetch.hpp>

rmm::mr::managed_memory_resource managed_mr;
rmm::cuda_stream stream;
rmm::device_buffer buffer(1000000, stream.view(), managed_mr);

// Prefetch to the current device on this stream
rmm::prefetch(buffer.data(), buffer.size(),
              rmm::get_current_cuda_device(), stream.view());

// Kernel on the same stream finds the data already resident
launch_kernel<<<grid, block, 0, stream.value()>>>(buffer.data());
````
````{code-tab} python
import rmm
from rmm.pylibrmm.stream import Stream

managed_mr = rmm.mr.ManagedMemoryResource()
buffer = rmm.DeviceBuffer(size=1000000, mr=managed_mr)

# Prefetch to device 0 on this stream
stream = Stream()
buffer.prefetch(device=0, stream=stream)

# Kernel on the same stream finds the data already resident
````
`````

## Prefetching Best Practices

### Stream ordering

When prefetching manually, issue the prefetch on the same stream as the kernel that will consume the data. This guarantees the migration completes before the kernel launches.

### Profiling

[NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) can visualize page faults and data migration to help you decide where prefetching is needed:

```bash
nsys profile -o output python your_script.py
```

When using `compute-sanitizer` with managed memory, enable page fault tracking:

```bash
compute-sanitizer --tool memcheck \
    --cuda-um-cpu-page-faults=true \
    --cuda-um-gpu-page-faults=true \
    python your_script.py
```

## Limitations

- **Not stream-ordered**: `ManagedMemoryResource` uses `cudaMallocManaged`, which is synchronous — the call blocks until the allocation is complete. For multi-stream applications where allocation latency matters, prefer `CudaAsyncMemoryResource`.
- **Migration overhead**: Even with prefetching, managed memory carries overhead from driver-managed page migration. If your data fits comfortably in GPU memory, `CudaAsyncMemoryResource` avoids this cost entirely.
- **Interconnect bandwidth**: Workloads that constantly migrate data between CPU and GPU are bounded by the throughput of the CPU-GPU interconnect (PCIe, NVLink-C2C, etc.).

## See Also

- [Choosing a Memory Resource](choosing_memory_resources.md) - When to use managed memory vs. other resources
- [Stream-Ordered Allocation](stream_ordered_allocation.md) - Understanding asynchronous allocation semantics
- [CUDA Programming Guide: Unified Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#unified-memory-programming)
- [NVIDIA Developer Blog: Unified Memory](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)
