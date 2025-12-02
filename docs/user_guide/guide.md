# Programming Guide

This guide covers using RMM in C++ and Python applications, including memory resources, containers, and library integrations.

## Basic Example

`````{tabs}
````{code-tab} c++
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/device_buffer.hpp>
#include <iostream>

int main() {
    // Use async MR (recommended)
    rmm::mr::cuda_async_memory_resource async_mr;
    rmm::mr::set_current_device_resource_ref(async_mr);

    // Allocate device memory
    rmm::cuda_stream stream;
    rmm::device_buffer buffer(1024, stream.view());

    std::cout << "Allocated " << buffer.size() << " bytes\n";

    return 0;
}
````
````{code-tab} python
import rmm

# Use async MR (recommended)
mr = rmm.mr.CudaAsyncMemoryResource()
rmm.mr.set_current_device_resource(mr)

# Allocate device memory
buffer = rmm.DeviceBuffer(size=1024)

print(f"Allocated {buffer.size} bytes at {hex(buffer.ptr)}")
````
`````

## Memory Resources

Memory resources control how device memory is allocated. RMM provides several resource types optimized for different use cases.

### Setting the Current Resource

The current device resource is used by default for all allocations:

`````{tabs}
````{code-tab} c++
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

// Get current device resource ref
rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref();

// Set current device resource ref
rmm::mr::cuda_async_memory_resource async_mr;
rmm::mr::set_current_device_resource_ref(async_mr);
````
````{code-tab} python
import rmm

# Get current device resource
mr = rmm.mr.get_current_device_resource()

# Set current device resource
async_mr = rmm.mr.CudaAsyncMemoryResource()
rmm.mr.set_current_device_resource(async_mr)
````
`````

> **Warning**: The default resource must be set **before** allocating any device memory on that device. Setting or changing the resource after device allocations have been made can lead to unexpected behavior or crashes.

### Available Resources

RMM provides several memory resource implementations:

| Resource | Description | Use Case |
|----------|-------------|----------|
| `CudaAsyncMemoryResource` | Uses `cudaMallocAsync` (driver-managed pool) | **Recommended default** |
| `CudaMemoryResource` | Uses `cudaMalloc`/`cudaFree` | Simple, no pooling |
| `ManagedMemoryResource` | Uses `cudaMallocManaged` (unified memory) | Datasets larger than GPU memory |
| `PoolMemoryResource` | Coalescing pool over upstream resource | Custom pool configuration |
| `ArenaMemoryResource` | Size-binned arenas | Mixed allocation sizes |

`````{tabs}
````{code-tab} c++
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

// CudaMemoryResource - uses cudaMalloc/cudaFree
auto cuda_mr = rmm::mr::cuda_memory_resource{};

// CudaAsyncMemoryResource - uses cudaMallocAsync (recommended)
auto async_mr = rmm::mr::cuda_async_memory_resource{};

// ManagedMemoryResource - uses cudaMallocManaged
auto managed_mr = rmm::mr::managed_memory_resource{};

// PoolMemoryResource - coalescing pool with 1 GiB initial size
rmm::mr::pool_memory_resource pool_mr{cuda_mr, 1ULL << 30};
````
````{code-tab} python
import rmm

# CudaMemoryResource - uses cudaMalloc/cudaFree
cuda_mr = rmm.mr.CudaMemoryResource()

# CudaAsyncMemoryResource - uses cudaMallocAsync (recommended)
async_mr = rmm.mr.CudaAsyncMemoryResource()

# ManagedMemoryResource - uses cudaMallocManaged
managed_mr = rmm.mr.ManagedMemoryResource()

# PoolMemoryResource - coalescing pool with 1 GiB initial size
pool_mr = rmm.mr.PoolMemoryResource(
    rmm.mr.CudaMemoryResource(),
    initial_pool_size=2**30  # 1 GiB
)
````
`````

See [Choosing a Memory Resource](choosing_memory_resources.md) for detailed guidance.

### Per-Device Resources

For multi-GPU systems, each device can have its own resource:

`````{tabs}
````{code-tab} c++
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/cuda_device.hpp>

// Get per-device resource ref
rmm::device_async_resource_ref mr0 = rmm::mr::get_per_device_resource_ref(rmm::cuda_device_id{0});

// Set per-device resource ref
rmm::mr::cuda_async_memory_resource async_mr;
rmm::mr::set_per_device_resource_ref(rmm::cuda_device_id{0}, async_mr);
````
````{code-tab} python
import rmm

# Get per-device resource
mr0 = rmm.mr.get_per_device_resource(0)

# Set per-device resource
async_mr = rmm.mr.CudaAsyncMemoryResource()
rmm.mr.set_per_device_resource(0, async_mr)
````
`````

## Containers

RMM provides RAII containers that automatically manage device memory lifetime.

### DeviceBuffer

Untyped, uninitialized device memory:

`````{tabs}
````{code-tab} c++
#include <rmm/device_buffer.hpp>

rmm::cuda_stream stream;

// Allocate 1024 bytes
rmm::device_buffer buffer(1024, stream.view());

// Access pointer and size
void* ptr = buffer.data();
std::size_t size = buffer.size();

// Resize (may reallocate)
buffer.resize(2048, stream.view());

// Copy construct (deep copy)
rmm::device_buffer buffer2(buffer, stream.view());
````
````{code-tab} python
import rmm

# Allocate 1024 bytes
buffer = rmm.DeviceBuffer(size=1024)

# Access pointer and size
ptr = buffer.ptr
size = buffer.size

# Resize (may reallocate)
buffer.resize(2048)

# Copy construct (deep copy)
buffer2 = buffer.copy()
````
`````

### device_uvector (C++)

Typed, uninitialized device vector for trivially copyable types:

```cpp
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <thrust/fill.h>

rmm::cuda_stream stream;

// Allocate 100 elements
rmm::device_uvector<int> vec(100, stream.view());

// Access as pointer
int* ptr = vec.data();

// Access as iterators
auto begin = vec.begin();
auto end = vec.end();

// Initialize with Thrust
thrust::fill(rmm::exec_policy(stream.view()), vec.begin(), vec.end(), 42);

// Resize
vec.resize(200, stream.view());
```

### device_scalar (C++)

Single typed element with host-device transfer convenience:

```cpp
#include <rmm/device_scalar.hpp>

rmm::cuda_stream stream;

// Allocate single int
rmm::device_scalar<int> scalar(stream.view());

// Set value from host (async on stream)
scalar.set_value(42, stream.view());

// Get value to host (async on stream)
int value = scalar.value(stream.view());

// Access device pointer
int* d_ptr = scalar.data();

// Pass to kernel
launch_kernel<<<..., stream.value()>>>(scalar.data());
```

## Resource Adaptors

Adaptors wrap resources to add functionality like statistics tracking and logging.

### Statistics Tracking

`````{tabs}
````{code-tab} c++
#include <rmm/mr/device/statistics_resource_adaptor.hpp>
#include <rmm/mr/per_device_resource.hpp>

rmm::mr::cuda_async_memory_resource cuda_mr;
rmm::mr::statistics_resource_adaptor stats_mr{cuda_mr};
rmm::mr::set_current_device_resource_ref(stats_mr);

// Allocate
rmm::cuda_stream stream;
rmm::device_buffer buffer(1024, stream.view());

// Get statistics
auto bytes = stats_mr.get_bytes_counter();
std::cout << "Current bytes: " << bytes.value << "\n";
std::cout << "Peak bytes: " << bytes.peak << "\n";
std::cout << "Total bytes: " << bytes.total << "\n";
````
````{code-tab} python
import rmm

# Wrap base resource with statistics adaptor
cuda_mr = rmm.mr.CudaAsyncMemoryResource()
stats_mr = rmm.mr.StatisticsResourceAdaptor(cuda_mr)
rmm.mr.set_current_device_resource(stats_mr)

# Allocate
buffer = rmm.DeviceBuffer(size=1024)

# Get statistics
stats = stats_mr.allocation_counts
print(f"Current bytes: {stats.current_bytes}")
print(f"Peak bytes: {stats.peak_bytes}")
print(f"Total bytes: {stats.total_bytes}")
````
`````

### Logging

`````{tabs}
````{code-tab} c++
#include <rmm/mr/device/logging_resource_adaptor.hpp>
#include <rmm/mr/per_device_resource.hpp>

rmm::mr::cuda_async_memory_resource cuda_mr;
rmm::mr::logging_resource_adaptor log_mr{cuda_mr, "allocations.csv"};
rmm::mr::set_current_device_resource_ref(log_mr);

// All allocations logged to CSV
rmm::device_buffer buffer(1024, rmm::cuda_stream_default);
````
````{code-tab} python
import rmm

# Wrap the current resource with logging adaptor
base = rmm.mr.CudaAsyncMemoryResource()
log_mr = rmm.mr.LoggingResourceAdaptor(base, log_file_name="allocations.csv")
rmm.mr.set_current_device_resource(log_mr)

# All allocations logged to CSV
buffer = rmm.DeviceBuffer(size=1024)
````
`````

CSV format: `Thread,Time,Action,Pointer,Size,Stream`

See [Logging and Profiling](logging.md) for more details.

### Composing Resources

Adaptors can be stacked to combine functionality:

`````{tabs}
````{code-tab} c++
// Base resource
rmm::mr::cuda_async_memory_resource cuda_mr;

// Add pool
rmm::mr::pool_memory_resource pool_mr{cuda_mr, 1ULL << 30};

// Add statistics
rmm::mr::statistics_resource_adaptor stats_mr{pool_mr};

// Add logging
rmm::mr::logging_resource_adaptor log_mr{stats_mr, "log.csv"};

// Set as current
rmm::mr::set_current_device_resource_ref(log_mr);
````
````{code-tab} python
import rmm

# Base resource
cuda_mr = rmm.mr.CudaAsyncMemoryResource()

# Add pool
pool_mr = rmm.mr.PoolMemoryResource(cuda_mr, initial_pool_size=2**30)

# Add statistics
stats_mr = rmm.mr.StatisticsResourceAdaptor(pool_mr)

# Add logging
log_mr = rmm.mr.LoggingResourceAdaptor(stats_mr, log_file_name="log.csv")

# Set as current
rmm.mr.set_current_device_resource(log_mr)
````
`````

Order matters: outer adaptors see all allocations from inner resources.

## Library Integrations

### Thrust (C++)

Use `rmm::exec_policy` to make Thrust algorithms use RMM for temporary storage:

```cpp
#include <rmm/exec_policy.hpp>
#include <rmm/device_uvector.hpp>
#include <thrust/sequence.h>
#include <thrust/sort.h>

rmm::cuda_stream stream;
rmm::device_uvector<int> vec(1000, stream.view());

// Fill with descending values
thrust::sequence(rmm::exec_policy(stream.view()),
                 vec.begin(), vec.end(), vec.size() - 1, -1);

// Sort using current device resource for temporary storage
thrust::sort(rmm::exec_policy(stream.view()), vec.begin(), vec.end());

// Or use a specific memory resource for temporary storage
rmm::mr::cuda_async_memory_resource custom_mr;
thrust::sort(rmm::exec_policy(stream.view(), custom_mr), vec.begin(), vec.end());

stream.synchronize();
```

### CuPy (Python)

```python
import rmm
import cupy as cp
from rmm.allocators.cupy import rmm_cupy_allocator

# Configure RMM
mr = rmm.mr.CudaAsyncMemoryResource()
rmm.mr.set_current_device_resource(mr)

# Set CuPy to use RMM
cp.cuda.set_allocator(rmm_cupy_allocator)

# All CuPy arrays now use RMM
array = cp.zeros(1000)
```

### Numba (Python)

```python
from numba import cuda
from rmm.allocators.numba import RMMNumbaManager
import rmm

# Configure RMM
mr = rmm.mr.CudaAsyncMemoryResource()
rmm.mr.set_current_device_resource(mr)

# Set Numba to use RMM
cuda.set_memory_manager(RMMNumbaManager)
```

Or use the environment variable:

```bash
NUMBA_CUDA_MEMORY_MANAGER=rmm.allocators.numba python script.py
```

### PyTorch (Python)

```python
import rmm
import torch
from rmm.allocators.torch import rmm_torch_allocator

# Configure RMM
mr = rmm.mr.CudaAsyncMemoryResource()
rmm.mr.set_current_device_resource(mr)

# Set PyTorch to use RMM
torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

# All PyTorch tensors now use RMM
tensor = torch.zeros(1000, device='cuda')
```

## Multi-Device Usage

`````{tabs}
````{code-tab} c++
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/cuda_device.hpp>
#include <memory>
#include <vector>

int num_devices;
cudaGetDeviceCount(&num_devices);

// Store resources to maintain lifetime
std::vector<std::unique_ptr<rmm::mr::cuda_async_memory_resource>> resources;

for (int i = 0; i < num_devices; ++i) {
    // Set device BEFORE creating resource
    cudaSetDevice(i);

    // Create resource for this device
    resources.push_back(std::make_unique<rmm::mr::cuda_async_memory_resource>());

    // Set as per-device resource ref
    rmm::mr::set_per_device_resource_ref(rmm::cuda_device_id{i}, *resources.back());
}

// Use device 0
cudaSetDevice(0);
rmm::cuda_stream stream;
rmm::device_buffer buffer(1024, stream.view());  // Uses device 0's resource
````
````{code-tab} python
import rmm
from cuda import cuda

num_devices = cuda.cuDeviceGetCount()[1]

# Store resources to maintain lifetime
resources = []

for device_id in range(num_devices):
    # Create resource for this device
    mr = rmm.mr.CudaAsyncMemoryResource()
    resources.append(mr)

    # Set as per-device resource
    rmm.mr.set_per_device_resource(device_id, mr)

# Use device 0
buffer = rmm.DeviceBuffer(size=1024)  # Uses device 0's resource
````
`````

## Best Practices

1. **Use `CudaAsyncMemoryResource` by default** - best performance for most workloads

2. **Set resources before any allocations** - changing resources after allocation can cause crashes

3. **Maintain resource lifetime** - resources must outlive any allocations from them

4. **Use RAII containers** - prefer `device_buffer` over raw pointers

5. **Profile and measure** - use statistics and logging to understand allocation patterns

## See Also

- [Choosing a Memory Resource](choosing_memory_resources.md)
- [Stream-Ordered Allocation](stream_ordered_allocation.md)
- [Managed Memory and Prefetching](managed_memory.md)
- [Pool Allocators](pool_allocators.md)
- [Logging and Profiling](logging.md)
