# C++ Quick Start

## Basic Usage

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

## Using a Pool

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

## Using device_uvector

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

## Using device_scalar

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

## Stream-Ordered Allocation

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

## Using with Thrust

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

## Multi-Device Usage

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

## Composing Memory Resources

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
