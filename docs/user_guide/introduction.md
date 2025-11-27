# Introduction to RMM

**RMM (RAPIDS Memory Manager)** is a library for allocating and managing GPU memory in C++ and Python. It provides a flexible interface for customizing how device memory is allocated, along with efficient implementations and containers.

## Purpose

Achieving optimal performance in GPU-accelerated applications frequently requires customizing memory allocation strategies. For example:

- Using **memory pools** to reduce the overhead of dynamic allocation
- Using **managed memory** to work with datasets larger than GPU memory
- Using **pinned host memory** for faster asynchronous CPU ↔ GPU transfers
- Customizing allocation strategies for specific workload patterns

RMM provides a unified interface, called a **memory resource**, which is a building block for GPU-accelerated applications.

Memory resources provide a **minimal-overhead abstraction** over memory allocation that is **pluggable at runtime**, making it possible to debug, measure performance, and optimize a CUDA application without recompiling.
Memory resources aim to serve the needs of a wide range of applications, from data science and machine learning to high-performance simulation.

RMM's memory resources leverage CUDA features like **stream-ordered** (asynchronous) pipeline parallelism, **managed** memory (also known as unified virtual memory, UVM), and **pinned** memory, making it easier to write complex workflows that optimally use both device and host memory.
The integrations provided in RMM allow memory resources to benefit memory management across libraries frequently used together, such as **PyTorch** and **RAPIDS**.

## Key Features

RMM is built around three main concepts.

### 1. Memory Resources

Memory resources provide a common abstraction for device memory allocation.
The API of RMM's memory resources is based on the CCCL memory resource design to facilitate interoperability.

The choice of resource determines the underlying type of memory and thus its accessibility from host or device.
For example, the `cuda_async_memory_resource` uses a pool of memory managed by the CUDA driver.
This resource is recommended for most applications, because of its performance and support for asynchrous (stream-ordered) allocations. See [Stream-Ordered Allocation](stream_ordered_allocation.md) for details.
As another example, the `managed_memory_resource` provides unified memory for CPU+GPU, and is recommended for applications exceeding the available GPU memory.

See [Choosing a Memory Resource](choosing_memory_resources.md) for guidance on the available memory resources, performance considerations, and how they fit into efficient CUDA application design strategies.
[NVIDIA Nsight™ Systems](https://developer.nvidia.com/nsight-systems) can be used to profile memory resource performance.

### 2. Resource Adaptors

Resource adaptors wrap and add functionality to existing resources.
For example, the `statistics_resource_adaptor` can be used to track allocation statistics.
The `logging_resource_adaptor` logs allocations to a CSV file.
Adaptors are composable - wrap multiple adaptors for combined functionality.

### 3. Containers

RMM provides [RAII](https://en.cppreference.com/w/cpp/language/raii.html) container classes that manage memory lifetime.
Using these containers avoids common problems with performing raw allocation such as memory leaks or improper stream ordering.
- `device_buffer`: Untyped device memory
- `device_uvector<T>`: Typed, uninitialized vector of device memory (trivially copyable types)
- `device_scalar<T>`: Single typed element

All containers use stream-ordered allocation and work with any memory resource.

## Basic Example

### C++

```cpp
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/device_buffer.hpp>

// Use CUDA async memory pool
auto async_mr = rmm::mr::cuda_async_memory_resource{};
rmm::mr::set_current_device_resource(&async_mr);

// Allocate device memory asynchronously
rmm::cuda_stream stream;
rmm::device_buffer buffer(1024, stream.view());
stream.synchronize();
```

### Python

```python
import rmm
import cupy as cp

# Create a CUDA async memory resource
mr = rmm.mr.CudaAsyncMemoryResource()

# Set the current device memory resource
rmm.mr.set_current_device_resource(mr)

# Allocating device memory uses the current device resource by default
buffer = rmm.DeviceBuffer(size=1024)

# Use the current device resource with CuPy
cp.cuda.set_allocator(rmm.allocators.cupy.rmm_cupy_allocator)
array = cp.zeros(1000)  # Now uses RMM for allocation
```

## Integration with GPU Libraries

RMM integrates seamlessly with popular GPU libraries:

### PyTorch

Set the PyTorch allocator to use the current device resource:

```python
import rmm
import torch
from rmm.allocators.torch import rmm_torch_allocator

mr = rmm.mr.CudaAsyncMemoryResource()
rmm.mr.set_current_device_resource(mr)
torch.cuda.memory.change_current_allocator(rmm_torch_allocator)
```

### CuPy

Set the CuPy allocator to use the current device resource:

```python
import rmm
import cupy
from rmm.allocators.cupy import rmm_cupy_allocator

mr = rmm.mr.CudaAsyncMemoryResource()
rmm.mr.set_current_device_resource(mr)
cupy.cuda.set_allocator(rmm_cupy_allocator)
```

### Numba

When launching a script:
```bash
NUMBA_CUDA_MEMORY_MANAGER=rmm.allocators.numba python script.py
```

Or from Python:

```python
import rmm
from numba import cuda
from rmm.allocators.numba import RMMNumbaManager

mr = rmm.mr.CudaAsyncMemoryResource()
rmm.mr.set_current_device_resource(mr)
cuda.set_memory_manager(RMMNumbaManager)
```

## Resources and Support

- [RMM GitHub Repository](https://github.com/rapidsai/rmm): Source code and development
- [RMM Issue Tracker](https://github.com/rapidsai/rmm/issues): Report bugs or request features
- [RAPIDS Documentation](https://docs.rapids.ai): RAPIDS ecosystem docs
- [RAPIDS Installation Guide](https://docs.rapids.ai/install): Installation instructions
- [Developer Blog: Fast, Flexible Allocation](https://developer.nvidia.com/blog/fast-flexible-allocation-for-cuda-with-rapids-memory-manager/): RMM design walkthrough
- [Developer Blog: Stream-Ordered Allocation](https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/): Deep dive into stream-ordered semantics
