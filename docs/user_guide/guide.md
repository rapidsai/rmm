# Programming Guide

This guide covers using RMM in C++ and Python applications, including memory resources, containers, and library integrations.

## Basic Example

`````{tabs}
````{code-tab} c++
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/device_buffer.hpp>
#include <iostream>

int main() {
    // Create a memory resource
    rmm::mr::cuda_async_memory_resource async_mr;

    // Allocate device memory using the resource
    rmm::cuda_stream stream;
    rmm::device_buffer buffer(1024, stream.view(), async_mr);

    std::cout << "Allocated " << buffer.size() << " bytes\n";

    return 0;
}
````
````{code-tab} python
import rmm

# Create a memory resource
mr = rmm.mr.CudaAsyncMemoryResource()

# Allocate device memory using the resource
buffer = rmm.DeviceBuffer(size=1024, mr=mr)

print(f"Allocated {buffer.size} bytes at {hex(buffer.ptr)}")
````
`````

## Memory Resources

Memory resources control how device memory is allocated. RMM provides several resource types optimized for different use cases.

### Explicit Resource Passing

The preferred way to use a memory resource is to pass it explicitly when allocating memory. This makes it clear which resource handles each allocation:

`````{tabs}
````{code-tab} c++
rmm::mr::cuda_async_memory_resource async_mr;
rmm::cuda_stream stream;

// Pass the resource explicitly
rmm::device_buffer buffer(1024, stream.view(), async_mr);
````
````{code-tab} python
mr = rmm.mr.CudaAsyncMemoryResource()

# Pass the resource explicitly
buffer = rmm.DeviceBuffer(size=1024, mr=mr)
````
`````

### Setting the Current Device Resource

RMM also provides a global "current device resource" that is used when no resource is passed explicitly:

`````{tabs}
````{code-tab} c++
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

rmm::mr::cuda_async_memory_resource async_mr;
rmm::mr::set_current_device_resource_ref(async_mr);

// Allocations that don't specify a resource use the current device resource
rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref();
````
````{code-tab} python
import rmm

async_mr = rmm.mr.CudaAsyncMemoryResource()
rmm.mr.set_current_device_resource(async_mr)

# Allocations that don't specify a resource use the current device resource
mr = rmm.mr.get_current_device_resource()
````
`````

> **Warning**: The default resource must be set **before** allocating any device memory on that device. Setting or changing the resource after device allocations have been made can lead to unexpected behavior or crashes.

### Available Resources

RMM provides base memory resources (e.g., `CudaAsyncMemoryResource`, `ManagedMemoryResource`) and resource adaptors (e.g., `PoolMemoryResource`, `StatisticsResourceAdaptor`) that wrap an upstream resource to add functionality. See [Choosing a Memory Resource](choosing_memory_resources.md) for recommendations and the API references ([C++](../cpp/memory_resources/index.md), [Python](../python/index.md)) for the full list.

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
#include <rmm/mr/statistics_resource_adaptor.hpp>

rmm::mr::cuda_async_memory_resource cuda_mr;
rmm::mr::statistics_resource_adaptor stats_mr{cuda_mr};

// Allocate using the statistics-wrapped resource
rmm::cuda_stream stream;
rmm::device_buffer buffer(1024, stream.view(), stats_mr);

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

# Allocate using the statistics-wrapped resource
buffer = rmm.DeviceBuffer(size=1024, mr=stats_mr)

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
#include <rmm/mr/logging_resource_adaptor.hpp>

rmm::mr::cuda_async_memory_resource cuda_mr;
rmm::mr::logging_resource_adaptor log_mr{cuda_mr, "allocations.csv"};

// Allocations through log_mr are logged to CSV
rmm::cuda_stream stream;
rmm::device_buffer buffer(1024, stream.view(), log_mr);
````
````{code-tab} python
import rmm

base_mr = rmm.mr.CudaAsyncMemoryResource()
log_mr = rmm.mr.LoggingResourceAdaptor(base_mr, log_file_name="allocations.csv")

# Allocations through log_mr are logged to CSV
buffer = rmm.DeviceBuffer(size=1024, mr=log_mr)
````
`````

CSV format: `Thread,Time,Action,Pointer,Size,Stream`

See [Logging and Profiling](logging.md) for more details.

### Composing Resources

Adaptors can be stacked to combine functionality:

`````{tabs}
````{code-tab} c++
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <rmm/mr/logging_resource_adaptor.hpp>

// Base resource
rmm::mr::cuda_memory_resource cuda_mr;

// Add pool
rmm::mr::pool_memory_resource pool_mr{cuda_mr, 1ULL << 30};

// Add statistics
rmm::mr::statistics_resource_adaptor stats_mr{pool_mr};

// Add logging
rmm::mr::logging_resource_adaptor log_mr{stats_mr, "log.csv"};

// Use log_mr for allocations — all allocations are pooled, tracked, and logged
rmm::cuda_stream stream;
rmm::device_buffer buffer(1024, stream.view(), log_mr);
````
````{code-tab} python
import rmm

# Base resource
cuda_mr = rmm.mr.CudaMemoryResource()

# Add pool
pool_mr = rmm.mr.PoolMemoryResource(cuda_mr, initial_pool_size=2**30)

# Add statistics
stats_mr = rmm.mr.StatisticsResourceAdaptor(pool_mr)

# Add logging
log_mr = rmm.mr.LoggingResourceAdaptor(stats_mr, log_file_name="log.csv")

# Use log_mr for allocations — all allocations are pooled, tracked, and logged
buffer = rmm.DeviceBuffer(size=1024, mr=log_mr)
````
`````

Order matters: outer adaptors see all allocations from inner resources.

## Library Integrations

### Thrust (C++)

Use `rmm::exec_policy_nosync` to make Thrust algorithms use RMM for temporary storage. Passing the resource explicitly makes it clear which resource handles temporaries:

```cpp
#include <rmm/exec_policy.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/device_uvector.hpp>
#include <thrust/sequence.h>
#include <thrust/sort.h>

rmm::mr::cuda_async_memory_resource mr;
rmm::cuda_stream stream;
rmm::device_uvector<int> vec(1000, stream.view(), mr);

// Fill with descending values
thrust::sequence(rmm::exec_policy_nosync(stream.view(), mr),
                 vec.begin(), vec.end(), vec.size() - 1, -1);

// Sort — temporaries allocated from mr
thrust::sort(rmm::exec_policy_nosync(stream.view(), mr), vec.begin(), vec.end());

stream.synchronize();
```

`exec_policy_nosync` allows the Thrust backend to skip stream synchronizations that are not required for correctness, improving performance. Stream-ordered applications using RMM should always prefer `exec_policy_nosync`. If stream synchronizations are required, the application should insert them explicitly before reading device data from the host.

### CuPy (Python)

Configure CuPy to use RMM for all device memory allocations:

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

Configure Numba to use RMM for device memory in CUDA JIT-compiled functions:

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

Configure PyTorch to use RMM for CUDA tensor allocations:

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

For multi-GPU systems, each device can have its own memory resource. Use `set_per_device_resource_ref` (C++) or `set_per_device_resource` (Python) to configure each device before allocating memory on it:

`````{tabs}
````{code-tab} c++
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/cuda_device.hpp>
#include <vector>

int num_devices;
cudaGetDeviceCount(&num_devices);

// Store resources to maintain lifetime (resources are copyable value types)
std::vector<rmm::mr::cuda_async_memory_resource> resources;

for (int i = 0; i < num_devices; ++i) {
    // Set device BEFORE creating resource
    cudaSetDevice(i);

    // Create resource for this device
    resources.emplace_back();

    // Set as per-device resource ref
    rmm::mr::set_per_device_resource_ref(rmm::cuda_device_id{i}, resources.back());
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
