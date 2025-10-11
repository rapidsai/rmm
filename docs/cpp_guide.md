# C++ Programming Guide

This guide covers using RMM in C++ applications, including memory resources, data structures, allocators, and advanced topics.

## Memory Resources

### device_memory_resource Interface

`rmm::mr::device_memory_resource` is the base class for all device memory resources. It provides two key functions:

```cpp
void* allocate(std::size_t bytes, cuda_stream_view stream);
void deallocate(void* ptr, std::size_t bytes, cuda_stream_view stream);
```

All allocations are **stream-ordered** and aligned to at least 256 bytes.

### Available Resources

```cpp
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/arena_memory_resource.hpp>
#include <rmm/mr/device/binning_memory_resource.hpp>
#include <rmm/mr/device/fixed_size_memory_resource.hpp>

// CudaMemoryResource - uses cudaMalloc/cudaFree
auto cuda_mr = rmm::mr::cuda_memory_resource{};

// CudaAsyncMemoryResource - uses cudaMallocAsync (recommended)
auto async_mr = rmm::mr::cuda_async_memory_resource{};

// ManagedMemoryResource - uses cudaMallocManaged
auto managed_mr = rmm::mr::managed_memory_resource{};

// PoolMemoryResource - coalescing pool
auto pool_mr = rmm::mr::pool_memory_resource{&cuda_mr, 1ULL << 30};

// ArenaMemoryResource - size-binned arenas
auto arena_mr = rmm::mr::arena_memory_resource{&cuda_mr};

// BinningMemoryResource - route by size
auto binning_mr = rmm::mr::binning_memory_resource{&cuda_mr};

// FixedSizeMemoryResource - single fixed size
auto fixed_mr = rmm::mr::fixed_size_memory_resource{&cuda_mr, 1024};
```

### Default and Per-Device Resources

```cpp
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

// Get current device resource
auto* mr = rmm::mr::get_current_device_resource();

// Set current device resource
auto async_mr = rmm::mr::cuda_async_memory_resource{};
rmm::mr::set_current_device_resource(&async_mr);

// Get/set per-device resource explicitly
auto* mr0 = rmm::mr::get_per_device_resource(rmm::cuda_device_id{0});
rmm::mr::set_per_device_resource(rmm::cuda_device_id{0}, &async_mr);
```

**Important**: The resource must remain valid as long as it's set as the current/per-device resource.

## Stream-Ordered Allocation Semantics

Memory allocated on a stream is valid for use on that stream:

```cpp
rmm::cuda_stream stream_a;

// Allocate on stream_a
void* ptr = mr->allocate(1024, stream_a.view());

// Safe: Use on stream_a
launch_kernel<<<..., stream_a.value()>>>(ptr);

// Unsafe: Use on different stream without synchronization
// launch_kernel<<<..., stream_b.value()>>>(ptr);  // UB!

// To use on stream_b, synchronize first
stream_a.synchronize();
// OR use event-based synchronization
cudaEvent_t event;
cudaEventCreate(&event);
cudaEventRecord(event, stream_a.value());
cudaStreamWaitEvent(stream_b.value(), event, 0);

// Now safe to use on stream_b
launch_kernel<<<..., stream_b.value()>>>(ptr);

// Deallocate on the stream it was last used on
mr->deallocate(ptr, 1024, stream_b.view());
```

**Key rules**:
1. Allocate returns a pointer valid on the specified stream
2. Using on a different stream requires synchronization
3. Deallocate on the stream where the memory was last used
4. Never destroy a stream passed to `deallocate` until deallocation completes

See [Stream-Ordered Allocation](user_guide/stream_ordered_allocation.md) for more details.

## Data Structures

### device_buffer

Untyped, uninitialized device memory with RAII semantics:

```cpp
#include <rmm/device_buffer.hpp>

rmm::cuda_stream stream;

// Allocate 1024 bytes
rmm::device_buffer buffer(1024, stream.view());

// Access pointer and size
void* ptr = buffer.data();
std::size_t size = buffer.size();

// Resize (may reallocate)
buffer.resize(2048, stream.view());

// Shrink to fit
buffer.shrink_to_fit(stream.view());

// Copy from another buffer
rmm::device_buffer buffer2 = buffer;  // Deep copy

// Move
rmm::device_buffer buffer3 = std::move(buffer);  // buffer is now empty
```

### device_uvector<T>

Typed, uninitialized device vector for trivially copyable types:

```cpp
#include <rmm/device_uvector.hpp>
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
thrust::fill(thrust::cuda::par.on(stream.value()),
             vec.begin(), vec.end(), 42);

// Resize
vec.resize(200, stream.view());

// Element access (device pointer)
int* elem_ptr = &vec.element(50, stream.view());
```

### device_scalar<T>

Single typed element with host-device transfer convenience:

```cpp
#include <rmm/device_scalar.hpp>

rmm::cuda_stream stream;

// Allocate single int
rmm::device_scalar<int> scalar(stream.view());

// Set value from host (async on stream)
scalar.set_value(42, stream.view());

// Get value to host (async on stream, returns immediately)
int value = scalar.value(stream.view());

// Access device pointer
int* d_ptr = scalar.data();

// Pass to kernel
launch_kernel<<<..., stream.value()>>>(scalar.data());
```

## CUDA Streams

### cuda_stream_view

Non-owning wrapper around `cudaStream_t`:

```cpp
#include <rmm/cuda_stream_view.hpp>

cudaStream_t raw_stream;
cudaStreamCreate(&raw_stream);

// Wrap in view
rmm::cuda_stream_view stream_view{raw_stream};

// Get underlying stream
cudaStream_t s = stream_view.value();

// Check if default stream
bool is_default = stream_view.is_default();

// Use in RMM APIs
rmm::device_buffer buffer(1024, stream_view);

cudaStreamDestroy(raw_stream);
```

Special streams:

```cpp
// Default stream
auto default_stream = rmm::cuda_stream_default;

// Per-thread default stream
auto per_thread_stream = rmm::cuda_stream_per_thread;
```

### cuda_stream

Owning RAII wrapper:

```cpp
#include <rmm/cuda_stream.hpp>

// Create stream (calls cudaStreamCreate)
rmm::cuda_stream stream;

// Access underlying stream
cudaStream_t s = stream.value();

// Use as view
rmm::cuda_stream_view view = stream.view();

// Synchronize
stream.synchronize();

// Movable, not copyable
rmm::cuda_stream stream2 = std::move(stream);

// Destroys stream on scope exit
```

### cuda_stream_pool

Pool of streams for reuse:

```cpp
#include <rmm/cuda_stream_pool.hpp>

// Create pool with 4 streams
rmm::cuda_stream_pool pool(4);

// Get stream from pool (may return same stream multiple times)
rmm::cuda_stream_view stream1 = pool.get_stream();
rmm::cuda_stream_view stream2 = pool.get_stream();

// Get stream for specific index
rmm::cuda_stream_view stream3 = pool.get_stream(2);

// Pool size
std::size_t size = pool.get_pool_size();
```

## Allocators

### polymorphic_allocator

Stream-ordered allocator similar to `std::pmr::polymorphic_allocator`:

```cpp
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>

rmm::cuda_stream stream;

// Create allocator from memory resource
auto async_mr = rmm::mr::cuda_async_memory_resource{};
rmm::mr::polymorphic_allocator<int> alloc{&async_mr};

// Allocate (returns raw pointer)
int* ptr = alloc.allocate(100, stream.view());

// Deallocate
alloc.deallocate(ptr, 100, stream.view());
```

### stream_allocator_adaptor

Adapts stream-ordered allocator to standard C++ allocator interface:

```cpp
#include <rmm/mr/device/polymorphic_allocator.hpp>
#include <rmm/mr/device/stream_allocator_adaptor.hpp>
#include <vector>

rmm::cuda_stream stream;
rmm::mr::polymorphic_allocator<int> stream_alloc;

// Adapt to standard allocator (binds stream)
auto adapted = rmm::mr::make_stream_allocator_adaptor(stream_alloc, stream.view());

// Use with standard containers (allocations use bound stream)
std::vector<int, decltype(adapted)> vec(adapted);
vec.push_back(42);  // Allocates on stream
```

### thrust_allocator

Allocator for use with Thrust:

```cpp
#include <rmm/mr/device/thrust_allocator.hpp>
#include <rmm/device_vector.hpp>

rmm::cuda_stream stream;

// Create Thrust allocator
rmm::mr::thrust_allocator<int> alloc(stream.view());

// Use with device_vector (via rmm::device_vector alias)
rmm::device_vector<int> vec(100, alloc);
```

## Using RMM with Thrust

### Temporary Allocations in Thrust Algorithms

Use `rmm::exec_policy` to make Thrust algorithms use RMM for temporary storage:

```cpp
#include <rmm/exec_policy.hpp>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

rmm::cuda_stream stream;
thrust::device_vector<int> vec(1000);
// ... fill vec ...

// Sort using RMM for temporary storage
thrust::sort(rmm::exec_policy(stream.view()), vec.begin(), vec.end());

stream.synchronize();
```

The first argument is the stream for RMM allocations, which must match the stream used for algorithm execution.

### device_vector with RMM

`rmm::device_vector` is an alias for `thrust::device_vector` with `rmm::mr::thrust_allocator`:

```cpp
#include <rmm/device_vector.hpp>

rmm::cuda_stream stream;

// Creates vector using current device resource
rmm::device_vector<int> vec(100, rmm::mr::thrust_allocator<int>(stream.view()));

// Use with Thrust algorithms
thrust::fill(thrust::cuda::par.on(stream.value()),
             vec.begin(), vec.end(), 42);
```

## Multi-Device Usage

### Creating Per-Device Resources

```cpp
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/cuda_device_id.hpp>
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

    // Set as per-device resource
    rmm::mr::set_per_device_resource(rmm::cuda_device_id{i}, resources.back().get());
}

// Use device 0
cudaSetDevice(0);
rmm::cuda_stream stream;
rmm::device_buffer buffer(1024, stream.view());  // Uses device 0's resource
```

### Important Multi-Device Rules

1. **Device must match at creation and use**:
   - The active device when creating a resource must match when using it
   - Data structures store the device ID and ensure it's active during (de)allocation

2. **Correct usage**:
   ```cpp
   cudaSetDevice(0);
   auto mr = rmm::mr::cuda_async_memory_resource{};
   rmm::device_buffer buf(16, rmm::cuda_stream_default, &mr);

   // Safe: can switch devices after creation
   cudaSetDevice(1);
   // ... destructor will automatically switch back to device 0
   ```

3. **Incorrect usage**:
   ```cpp
   cudaSetDevice(0);
   auto mr = rmm::mr::cuda_async_memory_resource{};

   cudaSetDevice(1);
   // Wrong! MR created on device 0, but used on device 1
   rmm::device_buffer buf(16, rmm::cuda_stream_default, &mr);  // UB!
   ```

## Host Memory Resources

### host_memory_resource Interface

```cpp
#include <rmm/mr/host/host_memory_resource.hpp>

void* allocate(std::size_t bytes, std::size_t alignment);
void deallocate(void* ptr, std::size_t bytes, std::size_t alignment);
```

Interface matches `std::pmr::memory_resource` (no stream argument).

### Available Host Resources

```cpp
#include <rmm/mr/host/new_delete_resource.hpp>
#include <rmm/mr/host/pinned_memory_resource.hpp>

// Uses operator new/delete
auto new_delete_mr = rmm::mr::new_delete_resource{};

// Uses cudaMallocHost/cudaFreeHost (pinned memory)
auto pinned_mr = rmm::mr::pinned_memory_resource{};

// Set as host resource (if supported in future)
// Currently no default host resource mechanism
```

## Thread Safety

All device memory resources are **thread-safe** with respect to concurrent `allocate()` and `deallocate()` calls. They are **not** thread-safe with respect to resource construction/destruction.

For resources that are not thread-safe, use `thread_safe_resource_adaptor`:

```cpp
#include <rmm/mr/device/thread_safe_resource_adaptor.hpp>

// Hypothetical non-thread-safe resource
my_custom_resource upstream;

// Make it thread-safe
rmm::mr::thread_safe_resource_adaptor<my_custom_resource> thread_safe_mr{&upstream};
```

Note: All current RMM resources are already thread-safe, so this adaptor is rarely needed.

## Resource Adaptors

Adaptors wrap resources to add functionality:

### statistics_resource_adaptor

Track allocation statistics:

```cpp
#include <rmm/mr/device/statistics_resource_adaptor.hpp>

auto cuda_mr = rmm::mr::cuda_async_memory_resource{};
auto stats_mr = rmm::mr::statistics_resource_adaptor{&cuda_mr};
rmm::mr::set_current_device_resource(&stats_mr);

// Allocate
rmm::cuda_stream stream;
rmm::device_buffer buffer(1024, stream.view());

// Get statistics
auto stats = stats_mr.get_statistics();
std::cout << "Allocated bytes: " << stats.allocated_bytes << "\n";
std::cout << "Allocation count: " << stats.num_allocations << "\n";
```

### logging_resource_adaptor

Log allocations to CSV:

```cpp
#include <rmm/mr/device/logging_resource_adaptor.hpp>

auto cuda_mr = rmm::mr::cuda_async_memory_resource{};
auto log_mr = rmm::mr::logging_resource_adaptor{&cuda_mr, "allocations.csv"};
rmm::mr::set_current_device_resource(&log_mr);

// All allocations logged to CSV
rmm::device_buffer buffer(1024, rmm::cuda_stream_default);
```

CSV format: `Thread,Time,Action,Pointer,Size,Stream`

### failure_callback_resource_adaptor

Handle allocation failures:

```cpp
#include <rmm/mr/device/failure_callback_resource_adaptor.hpp>
#include <iostream>

auto cuda_mr = rmm::mr::cuda_async_memory_resource{};

auto failure_callback = [](std::size_t bytes, void* callback_arg) {
    std::cerr << "Allocation of " << bytes << " bytes failed!\n";
    return false;  // false = rethrow exception, true = retry
};

auto callback_mr = rmm::mr::failure_callback_resource_adaptor{
    &cuda_mr, failure_callback, nullptr
};

rmm::mr::set_current_device_resource(&callback_mr);
```

### prefetch_resource_adaptor

Automatically prefetch managed memory:

```cpp
#include <rmm/mr/device/prefetch_resource_adaptor.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>

auto managed_mr = rmm::mr::managed_memory_resource{};
auto prefetch_mr = rmm::mr::prefetch_resource_adaptor{&managed_mr};
rmm::mr::set_current_device_resource(&prefetch_mr);

// All allocations automatically prefetched to GPU
rmm::device_buffer buffer(1024, rmm::cuda_stream_default);
```

## Advanced Patterns

### Composing Resources

Adaptors can be stacked:

```cpp
// Base resource
auto cuda_mr = rmm::mr::cuda_async_memory_resource{};

// Add pool
auto pool_mr = rmm::mr::pool_memory_resource{&cuda_mr, 1ULL << 30};

// Add statistics
auto stats_mr = rmm::mr::statistics_resource_adaptor{&pool_mr};

// Add logging
auto log_mr = rmm::mr::logging_resource_adaptor{&stats_mr, "log.csv"};

// Set as current
rmm::mr::set_current_device_resource(&log_mr);
```

Order matters: outer adaptors see all allocations from inner resources.

### Size-Based Binning

Route small and large allocations to different resources:

```cpp
auto cuda_mr = rmm::mr::cuda_async_memory_resource{};

// Small allocations go to fixed-size resource
auto small_mr = rmm::mr::fixed_size_memory_resource{&cuda_mr, 256};

// Large allocations go to pool
auto large_mr = rmm::mr::pool_memory_resource{&cuda_mr, 1ULL << 30};

// Create binning resource
auto binning_mr = rmm::mr::binning_memory_resource{&large_mr};

// Add bins
binning_mr.add_bin(256, &small_mr);   // <= 256 bytes
binning_mr.add_bin(1024, nullptr);    // <= 1KB (use upstream)
// > 1KB goes to large_mr

rmm::mr::set_current_device_resource(&binning_mr);
```

### Custom Memory Resource

Implement your own memory resource:

```cpp
#include <rmm/mr/device/device_memory_resource.hpp>

class my_memory_resource final : public rmm::mr::device_memory_resource {
    void* do_allocate(std::size_t bytes, cuda_stream_view stream) override {
        // Your allocation logic
        void* ptr = /* ... */;
        return ptr;
    }

    void do_deallocate(void* ptr, std::size_t bytes, cuda_stream_view stream) override {
        // Your deallocation logic
        // ...
    }

    bool do_is_equal(device_memory_resource const& other) const noexcept override {
        return dynamic_cast<my_memory_resource const*>(&other) != nullptr;
    }
};
```

## Best Practices

1. **Use `cuda_async_memory_resource` by default** - best performance for most workloads

2. **Set resources before any allocations** - changing resources after allocation can cause crashes

3. **Maintain resource lifetime** - resources must outlive any allocations from them

4. **Match device at resource creation and use** - especially important for multi-GPU

5. **Synchronize streams correctly** - follow stream-ordered allocation rules

6. **Use RAII data structures** - prefer `device_buffer` over raw pointers

7. **Profile and measure** - use statistics and logging to understand allocation patterns

## See Also

- [Choosing a Memory Resource](user_guide/choosing_memory_resources.md)
- [Stream-Ordered Allocation](user_guide/stream_ordered_allocation.md)
- [Pool Allocators](user_guide/pool_allocators.md)
- [C++ API Reference](cpp.rst)
