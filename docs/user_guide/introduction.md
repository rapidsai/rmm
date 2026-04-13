# Introduction to RMM

RMM (RAPIDS Memory Manager) is a C++ and Python library for GPU memory allocation. It provides a common interface — the **memory resource** — that lets you swap allocation strategies at runtime without recompiling, and a set of containers that manage device memory lifetime automatically.

GPU applications often benefit from customizing how memory is allocated. For example, pooling reduces the overhead of frequent small allocations, managed memory enables working with datasets larger than GPU memory, and pinned host memory speeds up CPU-GPU transfers compared to pageable host memory. RMM provides these and other features as interchangeable memory resources, so you can experiment with different strategies and measure their impact on your workload.

RMM provides integrations with GPU libraries including cuDF, cuML, cuGraph, PyTorch, and CuPy, enabling uniform memory handling across your application.

## Key Concepts

### Memory Resources

A memory resource is an object that knows how to allocate and deallocate memory. The choice of resource determines the kind of memory (device, host, managed, pinned) and the allocation strategy (pooled, stream-ordered, etc.). RMM's resources implement the `cuda::mr::resource` concept defined by [CCCL](https://github.com/NVIDIA/cccl) (CUDA Core Compute Libraries), so they interoperate directly with any library that accepts CCCL resources.

For most applications, the CUDA async memory resource (`rmm::mr::cuda_async_memory_resource` in C++, `rmm.mr.CudaAsyncMemoryResource` in Python) is a good starting point — it uses a CUDA driver-managed pool and supports stream-ordered (asynchronous) allocations. See [Choosing a Memory Resource](choosing_memory_resources.md) for guidance on when to use other resources.

### Resource Adaptors

Resource adaptors wrap an existing resource to add functionality. For example, `StatisticsResourceAdaptor` tracks allocation statistics, and `LoggingResourceAdaptor` logs allocations to a CSV file. Adaptors are composable — you can stack several to get combined functionality. See [Logging and Profiling](logging.md) for details.

### Containers

RMM provides [RAII](https://en.cppreference.com/w/cpp/language/raii.html) containers that manage device memory lifetime, avoiding common problems like memory leaks or improper stream ordering:

- C++: `device_buffer` (untyped), `device_uvector<T>` (typed, uninitialized), `device_scalar<T>` (single element)
- Python: `DeviceBuffer` (untyped)

All containers accept a stream and a memory resource, and use stream-ordered allocation.

## Basic Example

`````{tabs}
````{code-tab} c++
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/device_buffer.hpp>

rmm::mr::cuda_async_memory_resource mr;
rmm::cuda_stream stream;
rmm::device_buffer buffer(1024, stream.view(), mr);
````
````{code-tab} python
import rmm

mr = rmm.mr.CudaAsyncMemoryResource()
buffer = rmm.DeviceBuffer(size=1024, mr=mr)
````
`````

## Resources and Support

- [RMM GitHub Repository](https://github.com/rapidsai/rmm): Source code and development
- [RMM Issue Tracker](https://github.com/rapidsai/rmm/issues): Report bugs or request features
- [RAPIDS Documentation](https://docs.rapids.ai): RAPIDS ecosystem docs
- [RAPIDS Installation Guide](https://docs.rapids.ai/install): Installation instructions
- [Developer Blog: Fast, Flexible Allocation](https://developer.nvidia.com/blog/fast-flexible-allocation-for-cuda-with-rapids-memory-manager/): RMM design walkthrough
- [Developer Blog: Stream-Ordered Allocation](https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/): Deep dive into stream-ordered semantics
