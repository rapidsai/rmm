# Introduction to RMM

**RMM (RAPIDS Memory Manager)** is a library for allocating and managing GPU memory in Python and C++. It provides a flexible interface for customizing how device memory is allocated, along with efficient implementations and data structures.

## Why RMM?

Achieving optimal performance in GPU-accelerated applications frequently requires customizing memory allocation strategies. For example:

- Using **memory pools** to reduce the overhead of dynamic allocation
- Using **managed memory** to work with datasets larger than GPU memory
- Using **pinned host memory** for faster asynchronous CPU ↔ GPU transfers
- Customizing allocation strategies for specific workload patterns

RMM provides the building blocks to implement these optimizations through a unified interface.

## Key Features

### Unified Interface
- Common abstraction for device memory allocation based on CCCL's memory resource design
- Stream-ordered allocation for asynchronous GPU workflows

### Flexible Memory Resources
- Multiple built-in base memory resource implementations
- Composable design - wrap resources with *adaptors* to add functionality
- Easy integration with CUDA libraries (cuDF, PyTorch, CuPy, Numba)

### Efficient Containers
- RAII-friendly containers avoid problems arising from managing raw allocations such as memory leaks or improper stream ordering
- `device_buffer`: Untyped device memory allocation
- `device_uvector<T>`: Typed vector of device memory
- `device_scalar<T>`: Single element in device memory

### Memory Profiling
- Statistics tracking for allocations
- CSV logging for debugging
- Integration with NVIDIA Nsight Systems

## Quick Example

### Python

```python
import rmm
import cupy as cp

# Use CUDA async memory pool (recommended)
rmm.reinitialize(pool_allocator=False)

# Allocate device memory
buffer = rmm.DeviceBuffer(size=1024)

# Integrate with CuPy
cp.cuda.set_allocator(rmm.allocators.cupy.rmm_cupy_allocator)
array = cp.zeros(1000)  # Now uses RMM for allocation
```

### C++

```cpp
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/device_buffer.hpp>

// Use CUDA async memory pool (recommended)
auto async_mr = rmm::mr::cuda_async_memory_resource{};
rmm::mr::set_current_device_resource(&async_mr);

// Allocate device memory
rmm::cuda_stream stream;
rmm::device_buffer buffer(1024, stream.view());
```

## Architecture

RMM is built around three main concepts:

### 1. Memory Resources

Abstract interfaces for memory allocation:
- `device_memory_resource`: GPU memory allocation

Implementations include:
- `cuda_async_memory_resource`: Driver-managed pool (recommended default)
- `cuda_memory_resource`: Direct CUDA allocation
- `pool_memory_resource`: Custom pool allocator
- `managed_memory_resource`: Unified memory for CPU+GPU
- `arena_memory_resource`: Size-binned allocation
- Many more...

See [Choosing a Memory Resource](choosing_memory_resources.md) for guidance.

### 2. Resource Adaptors

Wrappers that add functionality to existing resources:
- `statistics_resource_adaptor`: Track allocation statistics
- `logging_resource_adaptor`: Log allocations to CSV
- `prefetch_resource_adaptor`: Automatically prefetch managed memory
- `failure_callback_resource_adaptor`: Custom error handling

Adaptors are composable - wrap multiple adaptors for combined functionality.

### 3. Data Structures

RAII classes that manage memory lifetime:
- `device_buffer`: Untyped device memory
- `device_uvector<T>`: Typed device vector (trivially copyable types)
- `device_scalar<T>`: Single typed element

All data structures use stream-ordered allocation and work with any memory resource.

## Stream-Ordered Allocation

RMM provides **stream-ordered memory allocation**, meaning allocations and deallocations are ordered with respect to CUDA streams. This enables:

- Asynchronous allocation without blocking
- Safe memory reuse within a stream
- Better multi-stream performance

```cpp
rmm::cuda_stream stream;

// Allocate on stream
auto buffer = rmm::device_buffer(1024, stream.view());

// Use immediately on same stream (no synchronization needed)
kernel<<<grid, block, 0, stream.value()>>>(buffer.data());
```

See [Stream-Ordered Allocation](stream_ordered_allocation.md) for details.

## Integration with Other Libraries

RMM integrates seamlessly with popular GPU libraries:

### PyTorch

```python
import rmm
import torch
from rmm.allocators.torch import rmm_torch_allocator

rmm.reinitialize(pool_allocator=False)
torch.cuda.memory.change_current_allocator(rmm_torch_allocator)
```

### CuPy

```python
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator

rmm.reinitialize(pool_allocator=False)
cupy.cuda.set_allocator(rmm_cupy_allocator)
```

### Numba

```bash
NUMBA_CUDA_MEMORY_MANAGER=rmm.allocators.numba python script.py
```

Or programmatically:

```python
from numba import cuda
from rmm.allocators.numba import RMMNumbaManager

cuda.set_memory_manager(RMMNumbaManager)
```

## Use Cases

### High-Performance Computing
- Pool allocators reduce allocation overhead
- Stream-ordered allocation enables pipeline parallelism
- Custom memory resources for specific hardware

### Data Science and Machine Learning
- Managed memory for larger-than-GPU datasets
- Integration with PyTorch, CuPy, and RAPIDS libraries
- Memory profiling to optimize data pipelines

### Multi-GPU Applications
- Per-device memory resources
- Shared memory pools across libraries
- Efficient inter-GPU memory management

### Memory-Constrained Environments
- Managed memory with prefetching
- Pool size limits to control memory usage
- Statistics tracking for optimization

## Performance Considerations

### Recommended Defaults

For most applications, use `cuda_async_memory_resource`:

```python
import rmm
rmm.reinitialize(pool_allocator=False)  # Uses async MR
```

**Why?**
- Driver-managed memory pool with virtual addressing
- Avoids fragmentation issues
- Shared across all libraries using the GPU
- Stream-ordered allocation semantics

### When to Use Other Resources

- **`managed_memory_resource`**: Datasets larger than GPU memory
- **`pool_memory_resource`**: Specific tuning needs or wrapping custom upstream
- **`arena_memory_resource`**: Applications with diverse allocation sizes
- **`cuda_memory_resource`**: Debugging or baseline comparison

See [Choosing a Memory Resource](choosing_memory_resources.md) for detailed guidance.

## Resources and Support

### Documentation
- [User Guide](choosing_memory_resources.md): Detailed guides and best practices
- [C++ API Reference](../cpp/index.md)
- [Python API Reference](../python/index.md)

### External Resources
- [RAPIDS Documentation](https://docs.rapids.ai): Full RAPIDS ecosystem docs
- [RAPIDS Installation Guide](https://docs.rapids.ai/install): Installation instructions
- [Developer Blog: Fast, Flexible Allocation](https://developer.nvidia.com/blog/fast-flexible-allocation-for-cuda-with-rapids-memory-manager/): RMM design walkthrough
- [Developer Blog: Stream-Ordered Allocation](https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/): Deep dive into stream-ordered semantics

### Community
- [GitHub Repository](https://github.com/rapidsai/rmm): Source code and development
- [Issue Tracker](https://github.com/rapidsai/rmm/issues): Report bugs or request features
- [RAPIDS Community](https://rapids.ai/learn-more/#get-involved): Get help and contribute
