# Quick Start Guide

This guide provides a quick introduction to using RMM in Python and C++. For more detailed information, see the [User Guide](choosing_memory_resources.md).

## Python Quick Start

### Basic Usage

```python
import rmm

# RMM initializes with CudaMemoryResource by default
# For better performance, use the async memory resource
rmm.reinitialize(pool_allocator=False)  # Uses CudaAsyncMemoryResource

# Allocate device memory
buffer = rmm.DeviceBuffer(size=1024)  # 1024 bytes

# Get pointer and size
print(f"Allocated {buffer.size} bytes at {hex(buffer.ptr)}")

# Copy data from host to device
import numpy as np
host_data = np.array([1, 2, 3, 4], dtype=np.float32)
buffer = rmm.DeviceBuffer.to_device(host_data.view('uint8'))

# Copy data back to host
host_copy = np.frombuffer(buffer.tobytes(), dtype=np.float32)
print(host_copy)  # [1. 2. 3. 4.]
```

### Using a Pool Allocator

```python
import rmm

# Create a memory pool with 1 GiB initial size
rmm.reinitialize(
    pool_allocator=True,
    initial_pool_size=2**30  # 1 GiB
)

# All allocations now use the pool
buffer = rmm.DeviceBuffer(size=1024)
```

### Configuring Memory Resources

```python
import rmm

# Option 1: Use rmm.reinitialize (simple)
rmm.reinitialize(
    pool_allocator=False,  # Use async MR (recommended)
    managed_memory=False,  # Don't use managed memory
    devices=[0]  # Configure device 0
)

# Option 2: Set memory resource directly (more control)
mr = rmm.mr.CudaAsyncMemoryResource()
rmm.mr.set_current_device_resource(mr)

# Option 3: Use a pool wrapping async MR
pool = rmm.mr.PoolMemoryResource(
    rmm.mr.CudaAsyncMemoryResource(),
    initial_pool_size=2**30,  # 1 GiB
    maximum_pool_size=2**32   # 4 GiB
)
rmm.mr.set_current_device_resource(pool)
```

### Integration with CuPy

```python
import rmm
import cupy as cp
from rmm.allocators.cupy import rmm_cupy_allocator

# Configure RMM
rmm.reinitialize(pool_allocator=False)

# Set CuPy to use RMM
cp.cuda.set_allocator(rmm_cupy_allocator)

# All CuPy arrays now use RMM
array = cp.zeros(1000)
result = cp.sqrt(array)
```

### Integration with Numba

```python
from numba import cuda
from rmm.allocators.numba import RMMNumbaManager
import rmm

# Configure RMM
rmm.reinitialize(pool_allocator=False)

# Set Numba to use RMM
cuda.set_memory_manager(RMMNumbaManager)

# Numba device arrays now use RMM
@cuda.jit
def kernel(x):
    idx = cuda.grid(1)
    if idx < x.size:
        x[idx] = idx * 2

x = cuda.device_array(100)
kernel[10, 10](x)
```

Alternatively, use the environment variable:

```bash
NUMBA_CUDA_MEMORY_MANAGER=rmm.allocators.numba python script.py
```

### Integration with PyTorch

```python
import rmm
import torch
from rmm.allocators.torch import rmm_torch_allocator

# Configure RMM
rmm.reinitialize(pool_allocator=False)

# Set PyTorch to use RMM
torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

# All PyTorch tensors now use RMM
tensor = torch.zeros(1000, device='cuda')
result = tensor + 1
```

### Memory Statistics

```python
import rmm

# Enable statistics tracking
rmm.statistics.enable_statistics()

# Run some allocations
buffer1 = rmm.DeviceBuffer(size=1000)
buffer2 = rmm.DeviceBuffer(size=2000)

# Get statistics
stats = rmm.statistics.get_statistics()
print(f"Current bytes: {stats.current_bytes}")
print(f"Peak bytes: {stats.peak_bytes}")
print(f"Total allocations: {stats.total_count}")

# Or use context manager
with rmm.statistics.statistics():
    buffer = rmm.DeviceBuffer(size=5000)
    stats = rmm.statistics.get_statistics()
    print(f"Allocated: {stats.current_bytes} bytes")
```

### Memory Profiling

```python
import rmm

# Enable statistics first
rmm.statistics.enable_statistics()

# Profile a function
@rmm.statistics.profiler()
def my_function(size):
    return rmm.DeviceBuffer(size=size)

# Run the function
my_function(10000)

# View profiling report
print(rmm.statistics.default_profiler_records.report())
```

## C++ Quick Start

### Basic Usage

```cpp
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/device_buffer.hpp>
#include <iostream>

int main() {
    // Use async MR (recommended)
    auto async_mr = rmm::mr::cuda_async_memory_resource{};
    rmm::mr::set_current_device_resource(&async_mr);

    // Allocate device memory
    rmm::cuda_stream stream;
    rmm::device_buffer buffer(1024, stream.view());

    std::cout << "Allocated " << buffer.size() << " bytes\n";

    return 0;
}
```

### Using a Pool

```cpp
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/device_buffer.hpp>

int main() {
    // Create upstream resource
    auto cuda_mr = rmm::mr::cuda_async_memory_resource{};

    // Create pool with 1 GiB initial size
    auto pool_mr = rmm::mr::pool_memory_resource<rmm::mr::cuda_async_memory_resource>{
        &cuda_mr,
        1ULL << 30  // 1 GiB
    };

    // Set as current resource
    rmm::mr::set_current_device_resource(&pool_mr);

    // Allocate from pool
    rmm::cuda_stream stream;
    rmm::device_buffer buffer(1024, stream.view());

    return 0;
}
```

### Using device_uvector

```cpp
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/device_uvector.hpp>
#include <thrust/device_vector.h>
#include <thrust/fill.h>

int main() {
    auto async_mr = rmm::mr::cuda_async_memory_resource{};
    rmm::mr::set_current_device_resource(&async_mr);

    rmm::cuda_stream stream;

    // Allocate typed device vector
    rmm::device_uvector<int> vec(100, stream.view());

    // Initialize with Thrust
    thrust::fill(thrust::cuda::par.on(stream.value()),
                 vec.begin(), vec.end(), 42);

    stream.synchronize();

    return 0;
}
```

### Using device_scalar

```cpp
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/device_scalar.hpp>

int main() {
    auto async_mr = rmm::mr::cuda_async_memory_resource{};
    rmm::mr::set_current_device_resource(&async_mr);

    rmm::cuda_stream stream;

    // Allocate single value
    rmm::device_scalar<int> scalar(stream.view());

    // Set value from host
    scalar.set_value(42, stream.view());

    // Get value to host
    int value = scalar.value(stream.view());

    return 0;
}
```

### Stream-Ordered Allocation

```cpp
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/cuda_stream.hpp>

int main() {
    auto async_mr = rmm::mr::cuda_async_memory_resource{};
    rmm::mr::set_current_device_resource(&async_mr);

    // Create CUDA stream
    rmm::cuda_stream stream;

    // Allocate on stream (asynchronous)
    rmm::device_buffer buffer(1024, stream.view());

    // Can use immediately on same stream
    // launch_kernel<<<grid, block, 0, stream.value()>>>(buffer.data());

    // Synchronize when needed
    stream.synchronize();

    return 0;
}
```

### Using with Thrust

```cpp
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/exec_policy.hpp>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

int main() {
    auto async_mr = rmm::mr::cuda_async_memory_resource{};
    rmm::mr::set_current_device_resource(&async_mr);

    rmm::cuda_stream stream;

    thrust::device_vector<int> vec(100);
    // ... fill vec with data ...

    // Use RMM execution policy (uses RMM for temp allocations)
    thrust::sort(rmm::exec_policy(stream.view()), vec.begin(), vec.end());

    stream.synchronize();

    return 0;
}
```

### Multi-Device Usage

```cpp
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/device_buffer.hpp>
#include <cuda_runtime.h>
#include <memory>
#include <vector>

int main() {
    int num_devices;
    cudaGetDeviceCount(&num_devices);

    // Create resources for each device
    std::vector<std::unique_ptr<rmm::mr::cuda_async_memory_resource>> resources;

    for (int i = 0; i < num_devices; ++i) {
        cudaSetDevice(i);

        // Create resource for this device
        resources.push_back(std::make_unique<rmm::mr::cuda_async_memory_resource>());

        // Set as per-device resource
        rmm::mr::set_per_device_resource(rmm::cuda_device_id{i}, resources.back().get());
    }

    // Allocate on device 0
    cudaSetDevice(0);
    rmm::cuda_stream stream;
    rmm::device_buffer buffer(1024, stream.view());

    return 0;
}
```

### Composing Memory Resources

```cpp
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/statistics_resource_adaptor.hpp>
#include <rmm/device_buffer.hpp>
#include <iostream>

int main() {
    // Create base resource
    auto cuda_mr = rmm::mr::cuda_async_memory_resource{};

    // Wrap in pool
    auto pool_mr = rmm::mr::pool_memory_resource{&cuda_mr, 1ULL << 30};

    // Wrap in statistics adaptor
    auto stats_mr = rmm::mr::statistics_resource_adaptor{&pool_mr};

    // Set as current resource
    rmm::mr::set_current_device_resource(&stats_mr);

    // Allocate
    rmm::cuda_stream stream;
    rmm::device_buffer buffer(1024, stream.view());

    // Check statistics
    auto stats = stats_mr.get_statistics();
    std::cout << "Allocated: " << stats.allocated_bytes << " bytes\n";

    return 0;
}
```

## Common Patterns

### Pattern: Allocate, Process, Deallocate in Loop

**Python:**
```python
import rmm

rmm.reinitialize(pool_allocator=False)

for i in range(100):
    # Allocate
    buffer = rmm.DeviceBuffer(size=1000000)

    # Process
    # ... GPU operations using buffer ...

    # Deallocate (automatic when buffer goes out of scope)
    buffer = None
```

**C++:**
```cpp
rmm::cuda_stream stream;

for (int i = 0; i < 100; ++i) {
    // Allocate
    rmm::device_buffer buffer(1000000, stream.view());

    // Process
    // launch_kernel<<<..., stream.value()>>>(buffer.data());

    // Deallocate (automatic when buffer goes out of scope)
}

stream.synchronize();
```

### Pattern: Multiple Streams

**Python:**
```python
import rmm

rmm.reinitialize(pool_allocator=False)

# Create streams
streams = [rmm.cuda_stream() for _ in range(4)]

# Allocate on different streams
buffers = []
for stream in streams:
    buffer = rmm.DeviceBuffer(size=1000000, stream=stream)
    buffers.append(buffer)

    # Launch work on this stream
    # ...

# Synchronize all
for stream in streams:
    stream.synchronize()
```

**C++:**
```cpp
std::vector<rmm::cuda_stream> streams(4);
std::vector<rmm::device_buffer> buffers;

for (auto& stream : streams) {
    // Allocate on stream
    buffers.emplace_back(1000000, stream.view());

    // Launch work
    // launch_kernel<<<..., stream.value()>>>(buffers.back().data());
}

// Synchronize all
for (auto& stream : streams) {
    stream.synchronize();
}
```

## Next Steps

- **Choosing a memory resource**: See [Choosing a Memory Resource](choosing_memory_resources.md)
- **Understanding stream-ordered allocation**: Read [Stream-Ordered Allocation](stream_ordered_allocation.md)
- **Using managed memory**: Check out [Managed Memory Guide](managed_memory.md)
- **Optimizing with pools**: See [Pool Allocators Guide](pool_allocators.md)
- **C++ details**: Read the [C++ Guide](cpp_guide.md)
- **Profiling memory**: Learn about [Logging and Profiling](logging.md)
