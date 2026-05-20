---
slug: api-reference/cpp-api-memory-resources
---

# Memory Resources

Generated from RMM C++ headers.

## `cpp/include/rmm/mr/arena_memory_resource.hpp`

### Arena Memory Resource Class

A suballocator that emphasizes fragmentation avoidance and scalable concurrency support.

Allocation and deallocation are thread-safe. Also, this class is compatible with CUDA per-thread default stream.

GPU memory is divided into a global arena, per-thread arenas for default streams, and per-stream arenas for non-default streams. Each arena allocates memory from the global arena in chunks called superblocks.

Blocks in each arena are allocated using address-ordered first fit. When a block is freed, it is coalesced with neighbouring free blocks if the addresses are contiguous. Free superblocks are returned to the global arena.

In real-world applications, allocation sizes tend to follow a power law distribution in which large allocations are rare, but small ones quite common. By handling small allocations in the per-thread arena, adequate performance can be achieved without introducing excessive memory fragmentation under high concurrency.

This design is inspired by several existing CPU memory allocators targeting multi-threaded applications (glibc malloc, Hoard, jemalloc, TCMalloc), albeit in a simpler form. Possible future improvements include using size classes, allocation caches, and more fine-grained locking or lock-free approaches.

This class is copyable and shares ownership of its internal state via `cuda::mr::shared_resource`.

allocation: A survey and critical review. In International Workshop on Memory Management (pp. 1-116). Springer, Berlin, Heidelberg.

memory allocator for multithreaded applications. ACM Sigplan Notices, 35(11), 117-128.

Proc. of the bsdcan conference, ottawa, canada.

```cpp
class RMM_EXPORT arena_memory_resource : public cuda::mr::shared_resource<detail::arena_memory_resource_impl>
```

_Source: `cpp/include/rmm/mr/arena_memory_resource.hpp:63`_

### Get Property (arena_memory_resource.hpp:71)

Enables the `cuda::mr::device_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(arena_memory_resource const&, cuda::mr::device_accessible) noexcept
```

_Source: `cpp/include/rmm/mr/arena_memory_resource.hpp:71`_

### Arena Memory Resource Constructor

Construct an `arena_memory_resource`.

**Parameters:**

- `upstream`: The resource from which to allocate blocks for the global arena.
- `arena_size`: Size in bytes of the global arena. Defaults to half of the available memory on the current device.
- `dump_log_on_failure`: If true, dump memory log when running out of memory.

```cpp
explicit arena_memory_resource(cuda::mr::any_resource<cuda::mr::device_accessible> upstream, std::optional<std::size_t> arena_size = std::nullopt, bool dump_log_on_failure = false);
```

_Source: `cpp/include/rmm/mr/arena_memory_resource.hpp:84`_

## `cpp/include/rmm/mr/binning_memory_resource.hpp`

### Binning Memory Resource Class

Allocates memory from upstream resources associated with bin sizes.

This class is copyable and shares ownership of its internal state, allowing multiple instances to safely reference the same underlying bins.

```cpp
class RMM_EXPORT binning_memory_resource : public cuda::mr::shared_resource<detail::binning_memory_resource_impl>
```

_Source: `cpp/include/rmm/mr/binning_memory_resource.hpp:30`_

### Get Property (binning_memory_resource.hpp:40)

Enables the `cuda::mr::device_accessible` property

This property declares that a `binning_memory_resource` provides device accessible memory

```cpp
RMM_CONSTEXPR_FRIEND void get_property(binning_memory_resource const&, cuda::mr::device_accessible) noexcept
```

_Source: `cpp/include/rmm/mr/binning_memory_resource.hpp:40`_

### Binning Memory Resource Constructor (binning_memory_resource.hpp:53)

Construct a new binning memory resource object.

Initially has no bins, so simply uses the upstream resource until bin resources are added with `add_bin`.

**Parameters:**

- `upstream`: The resource used to allocate bin pools.

```cpp
explicit binning_memory_resource(cuda::mr::any_resource<cuda::mr::device_accessible> upstream);
```

_Source: `cpp/include/rmm/mr/binning_memory_resource.hpp:53`_

### Binning Memory Resource Constructor (binning_memory_resource.hpp:67)

Construct a new binning memory resource object with a range of initial bins.

Constructs a new binning memory resource and adds bins backed by `fixed_size_memory_resource` in the range [2^min_size_exponent, 2^max_size_exponent]. For example if `min_size_exponent==18` and `max_size_exponent==22`, creates bins of sizes 256KiB, 512KiB, 1024KiB, 2048KiB and 4096KiB.

**Parameters:**

- `upstream`: The resource used to allocate bin pools.
- `min_size_exponent`: The minimum base-2 exponent bin size.
- `max_size_exponent`: The maximum base-2 exponent bin size.

```cpp
binning_memory_resource(cuda::mr::any_resource<cuda::mr::device_accessible> upstream, int8_t min_size_exponent, int8_t max_size_exponent);
```

_Source: `cpp/include/rmm/mr/binning_memory_resource.hpp:67`_

### Add Bin

Add a bin allocator to this resource

Adds `bin_resource` if provided; otherwise constructs and adds a fixed_size_memory_resource.

This bin will be used for any allocation smaller than `allocation_size` that is larger than the next smaller bin's allocation size.

If there is already a bin of the specified size nothing is changed.

This function is not thread safe.

**Parameters:**

- `allocation_size`: The maximum size that this bin allocates
- `bin_resource`: The memory resource for the bin

```cpp
void add_bin(std::size_t allocation_size, std::optional<device_async_resource_ref> bin_resource = std::nullopt);
```

_Source: `cpp/include/rmm/mr/binning_memory_resource.hpp:93`_

## `cpp/include/rmm/mr/callback_memory_resource.hpp`

### Allocate Callback T Type Alias

Callback function type used by callback memory resource for allocation.

The signature of the callback function is: `void* allocate_callback_t(std::size_t bytes, cuda_stream_view stream, void* arg);`

* Returns a pointer to an allocation of at least `bytes` usable immediately on `stream`. The stream-ordered behavior requirements are identical to `allocate`.

* The `arg` is provided to the constructor of the `callback_memory_resource` and will be forwarded along to every invocation of the callback function.

```cpp
using allocate_callback_t = std::function<void*(std::size_t, cuda_stream_view, void*)>;
```

_Source: `cpp/include/rmm/mr/callback_memory_resource.hpp:36`_

### Deallocate Callback T Type Alias

Callback function type used by callback_memory_resource for deallocation.

The signature of the callback function is: `void deallocate_callback_t(void* ptr, std::size_t bytes, cuda_stream_view stream, void* arg);`

* Deallocates memory pointed to by `ptr`. `bytes` specifies the size of the allocation in bytes, and must equal the value of `bytes` that was passed to the allocate callback function. The stream-ordered behavior requirements are identical to `deallocate`.

* The `arg` is provided to the constructor of the `callback_memory_resource` and will be forwarded along to every invocation of the callback function.

```cpp
using deallocate_callback_t = std::function<void(void*, std::size_t, cuda_stream_view, void*)>;
```

_Source: `cpp/include/rmm/mr/callback_memory_resource.hpp:52`_

### Callback Memory Resource Class

A device memory resource that uses the provided callbacks for memory allocation and deallocation.

This class is copyable and shares ownership of its internal state via `cuda::mr::shared_resource`.

```cpp
class RMM_EXPORT callback_memory_resource : public cuda::mr::shared_resource<detail::callback_memory_resource_impl>
```

_Source: `cpp/include/rmm/mr/callback_memory_resource.hpp:65`_

### Get Property (callback_memory_resource.hpp:73)

Enables the `cuda::mr::device_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(callback_memory_resource const&, cuda::mr::device_accessible) noexcept
```

_Source: `cpp/include/rmm/mr/callback_memory_resource.hpp:73`_

### Callback Memory Resource Constructor

Construct a new callback memory resource.

Constructs a callback memory resource that uses the user-provided callbacks `allocate_callback` for allocation and `deallocate_callback` for deallocation.

**Parameters:**

- `allocate_callback`: The callback function used for allocation
- `deallocate_callback`: The callback function used for deallocation
- `allocate_callback_arg`: Additional context passed to `allocate_callback`. It is the caller's responsibility to maintain the lifetime of the pointed-to data for the duration of the lifetime of the `callback_memory_resource`.
- `deallocate_callback_arg`: Additional context passed to `deallocate_callback`. It is the caller's responsibility to maintain the lifetime of the pointed-to data for the duration of the lifetime of the `callback_memory_resource`.

```cpp
callback_memory_resource(allocate_callback_t allocate_callback, deallocate_callback_t deallocate_callback, void* allocate_callback_arg = nullptr, void* deallocate_callback_arg = nullptr);
```

_Source: `cpp/include/rmm/mr/callback_memory_resource.hpp:93`_

## `cpp/include/rmm/mr/cuda_async_managed_memory_resource.hpp`

### CUDA Async Managed Memory Resource Class

Memory resource that uses `cudaMallocFromPoolAsync`/`cudaFreeFromPoolAsync` with a managed memory pool for allocation/deallocation.

```cpp
class RMM_EXPORT cuda_async_managed_memory_resource final : public cuda::mr::shared_resource<detail::cuda_async_managed_memory_resource_impl>
```

_Source: `cpp/include/rmm/mr/cuda_async_managed_memory_resource.hpp:28`_

### Get Property (cuda_async_managed_memory_resource.hpp:36)

Enables the `cuda::mr::device_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(cuda_async_managed_memory_resource const&, cuda::mr::device_accessible) noexcept
```

_Source: `cpp/include/rmm/mr/cuda_async_managed_memory_resource.hpp:36`_

### Get Property (cuda_async_managed_memory_resource.hpp:44)

Enables the `cuda::mr::host_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(cuda_async_managed_memory_resource const&, cuda::mr::host_accessible) noexcept
```

_Source: `cpp/include/rmm/mr/cuda_async_managed_memory_resource.hpp:44`_

### CUDA Async Managed Memory Resource Constructor

Constructs a cuda_async_managed_memory_resource with the default managed memory pool for the current device.

The default managed memory pool is the pool that is created when the device is created. Pool properties such as the release threshold are not modified.

**Throws:**

- `rmm::logic_error`: if the CUDA version does not support `cudaMallocFromPoolAsync` with managed memory pool

```cpp
cuda_async_managed_memory_resource();
```

_Source: `cpp/include/rmm/mr/cuda_async_managed_memory_resource.hpp:59`_

## `cpp/include/rmm/mr/cuda_async_memory_resource.hpp`

### CUDA Async Memory Resource Class

Memory resource that uses `cudaMallocAsync`/`cudaFreeAsync` for allocation/deallocation.

```cpp
class RMM_EXPORT cuda_async_memory_resource final : public cuda::mr::shared_resource<detail::cuda_async_memory_resource_impl>
```

_Source: `cpp/include/rmm/mr/cuda_async_memory_resource.hpp:30`_

### Allocation Handle Type Enum

Flags for specifying memory allocation handle types.

> **Note:** These values are exact copies from `cudaMemAllocationHandleType`. We need a placeholder that can be used consistently in the constructor of `cuda_async_memory_resource` with all supported versions of CUDA. See the `cudaMemAllocationHandleType` docs at https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html and ensure the enum values are kept in sync with the CUDA documentation.

> **Note:** cudaMemHandleTypeFabric can be used instead of 0x8 once we require CUDA 12.4+.

```cpp
enum class allocation_handle_type : std::int32_t
```

_Source: `cpp/include/rmm/mr/cuda_async_memory_resource.hpp:47`_

### Mempool Usage Enum

Flags for specifying memory pool usage.

> **Note:** These values are exact copies from the runtime API. See the `cudaMemPoolProps` docs at https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaMemPoolProps.html and ensure the enum values are kept in sync with the CUDA documentation. `cudaMemPoolCreateUsageHwDecompress` is currently the only supported usage flag, introduced in CUDA 12.8 and documented in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html

```cpp
enum class mempool_usage : unsigned short
```

_Source: `cpp/include/rmm/mr/cuda_async_memory_resource.hpp:70`_

### Get Property (cuda_async_memory_resource.hpp:78)

Enables the `cuda::mr::device_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(cuda_async_memory_resource const&, cuda::mr::device_accessible) noexcept
```

_Source: `cpp/include/rmm/mr/cuda_async_memory_resource.hpp:78`_

### CUDA Async Memory Resource Constructor

Constructs a cuda_async_memory_resource with the optionally specified initial pool size and release threshold.

If the pool size grows beyond the release threshold, unused memory held by the pool will be released at the next synchronization event.

**Throws:**

- `rmm::logic_error`: if the CUDA version does not support `cudaMallocAsync`

**Parameters:**

- `initial_pool_size`: Optional initial size in bytes of the pool. If provided, the pool will be primed by allocating and immediately deallocating this amount of memory on the default CUDA stream.
- `release_threshold`: Optional release threshold size in bytes of the pool. If no value is provided, the release threshold is set to the maximum value of `std::uint64_t`, so that the pool retains memory across synchronization events unless the caller specifies otherwise.
- `export_handle_type`: Optional `cudaMemAllocationHandleType` that allocations from this resource should support interprocess communication (IPC). Default is `cudaMemHandleTypeNone` for no IPC support.

```cpp
cuda_async_memory_resource(std::optional<std::size_t> initial_pool_size =
```

_Source: `cpp/include/rmm/mr/cuda_async_memory_resource.hpp:103`_

## `cpp/include/rmm/mr/cuda_async_view_memory_resource.hpp`

### CUDA Async View Memory Resource Class

Memory resource that uses `cudaMallocAsync`/`cudaFreeAsync` for allocation/deallocation.

```cpp
class cuda_async_view_memory_resource final
```

_Source: `cpp/include/rmm/mr/cuda_async_view_memory_resource.hpp:30`_

### CUDA Async View Memory Resource Constructor

Constructs a cuda_async_view_memory_resource which uses an existing CUDA memory pool. The provided pool is not owned by cuda_async_view_memory_resource and must remain valid during the lifetime of the memory resource.

**Throws:**

- `rmm::logic_error`: if the CUDA version does not support `cudaMallocAsync`

**Parameters:**

- `pool_handle`: Handle to a CUDA memory pool which will be used to serve allocation requests.

```cpp
cuda_async_view_memory_resource(cudaMemPool_t pool_handle) : cuda_pool_handle_
```

_Source: `cpp/include/rmm/mr/cuda_async_view_memory_resource.hpp:42`_

### Allocate (cuda_async_view_memory_resource.hpp:81)

Allocates memory of size at least `bytes`.

The returned pointer will have at minimum 256 byte alignment.

**Parameters:**

- `stream`: Stream on which to perform allocation
- `bytes`: The size of the allocation
- `alignment`: The alignment of the allocation

**Returns:** void* Pointer to the newly allocated memory

```cpp
void* allocate(cuda::stream_ref stream, std::size_t bytes, [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
```

_Source: `cpp/include/rmm/mr/cuda_async_view_memory_resource.hpp:81`_

### Deallocate (cuda_async_view_memory_resource.hpp:101)

Deallocate memory pointed to by `ptr`.

**Parameters:**

- `stream`: Stream on which to perform deallocation
- `ptr`: Pointer to be deallocated
- `bytes`: The size in bytes of the allocation. This must be equal to the value of `bytes` that was passed to the `allocate` call that returned `ptr`.
- `alignment`: The alignment that was passed to the `allocate` call that returned `ptr`

```cpp
void deallocate(cuda::stream_ref stream, void* ptr, [[maybe_unused]] std::size_t bytes, [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
```

_Source: `cpp/include/rmm/mr/cuda_async_view_memory_resource.hpp:101`_

### Allocate Sync (cuda_async_view_memory_resource.hpp:116)

Allocates memory of size at least `bytes` synchronously.

**Parameters:**

- `bytes`: The size of the allocation
- `alignment`: The alignment of the allocation

**Returns:** void* Pointer to the newly allocated memory

```cpp
void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
```

_Source: `cpp/include/rmm/mr/cuda_async_view_memory_resource.hpp:116`_

### Deallocate Sync (cuda_async_view_memory_resource.hpp:130)

Deallocate memory pointed to by `ptr` synchronously.

**Parameters:**

- `ptr`: Pointer to be deallocated
- `bytes`: The size in bytes of the allocation
- `alignment`: The alignment that was passed to the `allocate` call that returned `ptr`

```cpp
void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
```

_Source: `cpp/include/rmm/mr/cuda_async_view_memory_resource.hpp:130`_

### Get Property (cuda_async_view_memory_resource.hpp:163)

Enables the `cuda::mr::device_accessible` property

This property declares that a `cuda_async_view_memory_resource` provides device accessible memory

```cpp
RMM_CONSTEXPR_FRIEND void get_property(cuda_async_view_memory_resource const&, cuda::mr::device_accessible) noexcept
```

_Source: `cpp/include/rmm/mr/cuda_async_view_memory_resource.hpp:163`_

## `cpp/include/rmm/mr/cuda_memory_resource.hpp`

### CUDA Memory Resource Class

Memory resource that uses cudaMalloc/Free for allocation/deallocation.

```cpp
class cuda_memory_resource final
```

_Source: `cpp/include/rmm/mr/cuda_memory_resource.hpp:26`_

### Allocate (cuda_memory_resource.hpp:49)

Allocates memory of size at least `bytes`.

The returned pointer will have at minimum 256 byte alignment.

The stream argument is ignored.

**Parameters:**

- `stream`: This argument is ignored
- `bytes`: The size of the allocation
- `alignment`: The alignment of the allocation

**Returns:** void* Pointer to the newly allocated memory

```cpp
void* allocate([[maybe_unused]] cuda::stream_ref stream, std::size_t bytes, [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
```

_Source: `cpp/include/rmm/mr/cuda_memory_resource.hpp:49`_

### Deallocate (cuda_memory_resource.hpp:69)

Deallocate memory pointed to by `ptr`.

The stream argument is ignored.

**Parameters:**

- `stream`: This argument is ignored
- `ptr`: Pointer to be deallocated
- `bytes`: The size in bytes of the allocation. This must be equal to the value of `bytes` that was passed to the `allocate` call that returned `ptr`.
- `alignment`: The alignment that was passed to the `allocate` call that returned `ptr`

```cpp
void deallocate([[maybe_unused]] cuda::stream_ref stream, void* ptr, [[maybe_unused]] std::size_t bytes, [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
```

_Source: `cpp/include/rmm/mr/cuda_memory_resource.hpp:69`_

### Allocate Sync (cuda_memory_resource.hpp:84)

Allocates memory of size at least `bytes` synchronously.

**Parameters:**

- `bytes`: The size of the allocation
- `alignment`: The alignment of the allocation

**Returns:** void* Pointer to the newly allocated memory

```cpp
void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
```

_Source: `cpp/include/rmm/mr/cuda_memory_resource.hpp:84`_

### Deallocate Sync (cuda_memory_resource.hpp:98)

Deallocate memory pointed to by `ptr` synchronously.

**Parameters:**

- `ptr`: Pointer to be deallocated
- `bytes`: The size in bytes of the allocation
- `alignment`: The alignment that was passed to the `allocate` call that returned `ptr`

```cpp
void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
```

_Source: `cpp/include/rmm/mr/cuda_memory_resource.hpp:98`_

### Get Property (cuda_memory_resource.hpp:110)

Enables the `cuda::mr::device_accessible` property

This property declares that a `cuda_memory_resource` provides device accessible memory

```cpp
RMM_CONSTEXPR_FRIEND void get_property(cuda_memory_resource const&, cuda::mr::device_accessible) noexcept
```

_Source: `cpp/include/rmm/mr/cuda_memory_resource.hpp:110`_

## `cpp/include/rmm/mr/fixed_size_memory_resource.hpp`

### Fixed Size Memory Resource Class

A memory resource which allocates memory blocks of a single fixed size.

Supports only allocations of size smaller than the configured block_size.

This class is copyable and shares ownership of its internal state, allowing multiple instances to safely reference the same underlying pool.

```cpp
class RMM_EXPORT fixed_size_memory_resource : public cuda::mr::shared_resource<detail::fixed_size_memory_resource_impl>
```

_Source: `cpp/include/rmm/mr/fixed_size_memory_resource.hpp:31`_

### Get Property (fixed_size_memory_resource.hpp:41)

Enables the `cuda::mr::device_accessible` property

This property declares that a `fixed_size_memory_resource` provides device accessible memory

```cpp
RMM_CONSTEXPR_FRIEND void get_property(fixed_size_memory_resource const&, cuda::mr::device_accessible) noexcept
```

_Source: `cpp/include/rmm/mr/fixed_size_memory_resource.hpp:41`_

### Fixed Size Memory Resource Constructor

Construct a new `fixed_size_memory_resource` that allocates memory from `upstream`.

When the pool of blocks is all allocated, grows the pool by allocating `blocks_to_preallocate` more blocks from `upstream`.

**Parameters:**

- `upstream`: The resource from which to allocate blocks for the pool.
- `block_size`: The size of blocks to allocate.
- `blocks_to_preallocate`: The number of blocks to allocate to initialize the pool.

```cpp
explicit fixed_size_memory_resource( cuda::mr::any_resource<cuda::mr::device_accessible> upstream, std::size_t block_size = default_block_size, std::size_t blocks_to_preallocate = default_blocks_to_preallocate);
```

_Source: `cpp/include/rmm/mr/fixed_size_memory_resource.hpp:64`_

## `cpp/include/rmm/mr/managed_memory_resource.hpp`

### Managed Memory Resource Class

Memory resource that uses cudaMallocManaged/Free for allocation/deallocation.

```cpp
class managed_memory_resource final
```

_Source: `cpp/include/rmm/mr/managed_memory_resource.hpp:26`_

### Allocate (managed_memory_resource.hpp:49)

Allocates memory of size at least `bytes`.

The returned pointer will have at minimum 256 byte alignment.

The stream argument is ignored.

**Parameters:**

- `stream`: This argument is ignored
- `bytes`: The size of the allocation
- `alignment`: The alignment of the allocation

**Returns:** void* Pointer to the newly allocated memory

```cpp
void* allocate([[maybe_unused]] cuda::stream_ref stream, std::size_t bytes, [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
```

_Source: `cpp/include/rmm/mr/managed_memory_resource.hpp:49`_

### Deallocate (managed_memory_resource.hpp:73)

Deallocate memory pointed to by `ptr`.

The stream argument is ignored.

**Parameters:**

- `stream`: This argument is ignored
- `ptr`: Pointer to be deallocated
- `bytes`: The size in bytes of the allocation. This must be equal to the value of `bytes` that was passed to the `allocate` call that returned `ptr`.
- `alignment`: The alignment that was passed to the `allocate` call that returned `ptr`

```cpp
void deallocate([[maybe_unused]] cuda::stream_ref stream, void* ptr, [[maybe_unused]] std::size_t bytes, [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
```

_Source: `cpp/include/rmm/mr/managed_memory_resource.hpp:73`_

### Allocate Sync (managed_memory_resource.hpp:88)

Allocates memory of size at least `bytes` synchronously.

**Parameters:**

- `bytes`: The size of the allocation
- `alignment`: The alignment of the allocation

**Returns:** void* Pointer to the newly allocated memory

```cpp
void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
```

_Source: `cpp/include/rmm/mr/managed_memory_resource.hpp:88`_

### Deallocate Sync (managed_memory_resource.hpp:102)

Deallocate memory pointed to by `ptr` synchronously.

**Parameters:**

- `ptr`: Pointer to be deallocated
- `bytes`: The size in bytes of the allocation
- `alignment`: The alignment that was passed to the `allocate` call that returned `ptr`

```cpp
void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
```

_Source: `cpp/include/rmm/mr/managed_memory_resource.hpp:102`_

### Get Property (managed_memory_resource.hpp:114)

Enables the `cuda::mr::device_accessible` property

This property declares that a `managed_memory_resource` provides device accessible memory

```cpp
RMM_CONSTEXPR_FRIEND void get_property(managed_memory_resource const&, cuda::mr::device_accessible) noexcept
```

_Source: `cpp/include/rmm/mr/managed_memory_resource.hpp:114`_

### Get Property (managed_memory_resource.hpp:124)

Enables the `cuda::mr::host_accessible` property

This property declares that a `managed_memory_resource` provides host accessible memory

```cpp
RMM_CONSTEXPR_FRIEND void get_property(managed_memory_resource const&, cuda::mr::host_accessible) noexcept
```

_Source: `cpp/include/rmm/mr/managed_memory_resource.hpp:124`_

## `cpp/include/rmm/mr/pinned_host_memory_resource.hpp`

### Pinned Host Memory Resource Class

Memory resource class for allocating pinned host memory.

This class uses CUDA's `cudaHostAlloc` to allocate pinned host memory. It satisfies the `cuda::mr::resource` and `cuda::mr::synchronous_resource` concepts, and the `cuda::mr::host_accessible` and `cuda::mr::device_accessible` properties.

```cpp
class pinned_host_memory_resource final
```

_Source: `cpp/include/rmm/mr/pinned_host_memory_resource.hpp:34`_

### Allocate (pinned_host_memory_resource.hpp:63)

Allocates pinned host memory of size at least `bytes` bytes.

**Throws:**

- `rmm::out_of_memory`: if the requested allocation could not be fulfilled due to a CUDA out of memory error.
- `rmm::bad_alloc`: if the requested allocation could not be fulfilled due to any other reason.

The stream argument is ignored.

**Parameters:**

- `stream`: CUDA stream on which to perform the allocation (ignored).
- `bytes`: The size, in bytes, of the allocation.
- `alignment`: The alignment of the allocation

**Returns:** Pointer to the newly allocated memory.

```cpp
void* allocate([[maybe_unused]] cuda::stream_ref stream, std::size_t bytes, [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
```

_Source: `cpp/include/rmm/mr/pinned_host_memory_resource.hpp:63`_

### Deallocate (pinned_host_memory_resource.hpp:89)

Deallocate memory pointed to by `ptr`.

The stream argument is ignored.

**Parameters:**

- `stream`: This argument is ignored.
- `ptr`: Pointer to be deallocated
- `bytes`: The size in bytes of the allocation. This must be equal to the value of `bytes` that was passed to the `allocate` call that returned `ptr`.
- `alignment`: The alignment that was passed to the `allocate` call that returned `ptr`

```cpp
void deallocate([[maybe_unused]] cuda::stream_ref stream, void* ptr, std::size_t bytes, [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
```

_Source: `cpp/include/rmm/mr/pinned_host_memory_resource.hpp:89`_

### Allocate Sync (pinned_host_memory_resource.hpp:107)

Allocates pinned host memory of size at least `bytes` bytes synchronously.

**Parameters:**

- `bytes`: The size, in bytes, of the allocation.
- `alignment`: The alignment of the allocation

**Returns:** Pointer to the newly allocated memory.

```cpp
void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
```

_Source: `cpp/include/rmm/mr/pinned_host_memory_resource.hpp:107`_

### Deallocate Sync (pinned_host_memory_resource.hpp:121)

Deallocate memory pointed to by `ptr` synchronously.

**Parameters:**

- `ptr`: Pointer to be deallocated
- `bytes`: The size in bytes of the allocation
- `alignment`: The alignment that was passed to the `allocate` call that returned `ptr`

```cpp
void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
```

_Source: `cpp/include/rmm/mr/pinned_host_memory_resource.hpp:121`_

### Get Property (pinned_host_memory_resource.hpp:133)

Enables the `cuda::mr::device_accessible` property

This property declares that a `pinned_host_memory_resource` provides device accessible memory

```cpp
RMM_CONSTEXPR_FRIEND void get_property(pinned_host_memory_resource const&, cuda::mr::device_accessible) noexcept
```

_Source: `cpp/include/rmm/mr/pinned_host_memory_resource.hpp:133`_

### Get Property (pinned_host_memory_resource.hpp:143)

Enables the `cuda::mr::host_accessible` property

This property declares that a `pinned_host_memory_resource` provides host accessible memory

```cpp
RMM_CONSTEXPR_FRIEND void get_property(pinned_host_memory_resource const&, cuda::mr::host_accessible) noexcept
```

_Source: `cpp/include/rmm/mr/pinned_host_memory_resource.hpp:143`_

## `cpp/include/rmm/mr/pool_memory_resource.hpp`

### Pool Memory Resource Class

A coalescing best-fit suballocator which uses a pool of memory allocated from an upstream memory_resource.

Allocation and deallocation are thread-safe. Also, this class is compatible with CUDA per-thread default stream.

This class is copyable and shares ownership of its internal state, allowing multiple instances to safely reference the same underlying pool.

```cpp
class RMM_EXPORT pool_memory_resource : public cuda::mr::shared_resource<detail::pool_memory_resource_impl>
```

_Source: `cpp/include/rmm/mr/pool_memory_resource.hpp:34`_

### Get Property (pool_memory_resource.hpp:44)

Enables the `cuda::mr::device_accessible` property

This property declares that a `pool_memory_resource` provides device accessible memory

```cpp
RMM_CONSTEXPR_FRIEND void get_property(pool_memory_resource const&, cuda::mr::device_accessible) noexcept
```

_Source: `cpp/include/rmm/mr/pool_memory_resource.hpp:44`_

### Pool Memory Resource Constructor

Construct a `pool_memory_resource` and allocate the initial device memory pool using `upstream`.

**Throws:**

- `rmm::logic_error`: if `initial_pool_size` is not aligned to a multiple of 256 bytes.
- `rmm::logic_error`: if `maximum_pool_size` is neither the default nor aligned to a multiple of 256 bytes.

**Parameters:**

- `upstream`: The resource from which to allocate blocks for the pool.
- `initial_pool_size`: Minimum size, in bytes, of the initial pool.
- `maximum_pool_size`: Maximum size, in bytes, that the pool can grow to. Defaults to all of the available memory from the upstream resource.

```cpp
explicit pool_memory_resource(cuda::mr::any_resource<cuda::mr::device_accessible> upstream, std::size_t initial_pool_size, std::optional<std::size_t> maximum_pool_size = std::nullopt);
```

_Source: `cpp/include/rmm/mr/pool_memory_resource.hpp:62`_

## `cpp/include/rmm/mr/sam_headroom_memory_resource.hpp`

### Sam Headroom Memory Resource Class

Resource that uses system memory resource to allocate memory with a headroom.

System allocated memory (SAM) can be migrated to the GPU, but is never migrated back the host. If GPU memory is over-subscribed, this can cause other CUDA calls to fail with out-of-memory errors. To work around this problem, when using a system memory resource, we reserve some GPU memory as headroom for other CUDA calls, and only conditionally set its preferred location to the GPU if the allocation would not eat into the headroom.

Since doing this check on every allocation can be expensive, the caller may choose to use other allocators (e.g. `binning_memory_resource`) for small allocations, and use this allocator for large allocations only.

```cpp
class RMM_EXPORT sam_headroom_memory_resource final : public cuda::mr::shared_resource<detail::sam_headroom_memory_resource_impl>
```

_Source: `cpp/include/rmm/mr/sam_headroom_memory_resource.hpp:35`_

### Get Property (sam_headroom_memory_resource.hpp:43)

Enables the `cuda::mr::device_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(sam_headroom_memory_resource const&, cuda::mr::device_accessible) noexcept
```

_Source: `cpp/include/rmm/mr/sam_headroom_memory_resource.hpp:43`_

### Get Property (sam_headroom_memory_resource.hpp:51)

Enables the `cuda::mr::host_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(sam_headroom_memory_resource const&, cuda::mr::host_accessible) noexcept
```

_Source: `cpp/include/rmm/mr/sam_headroom_memory_resource.hpp:51`_

### Sam Headroom Memory Resource Constructor

Construct a headroom memory resource.

**Parameters:**

- `headroom`: Size of the reserved GPU memory as headroom

```cpp
explicit sam_headroom_memory_resource(std::size_t headroom);
```

_Source: `cpp/include/rmm/mr/sam_headroom_memory_resource.hpp:61`_

## `cpp/include/rmm/mr/system_memory_resource.hpp`

### Is System Memory Supported

Check if system allocated memory (SAM) is supported on the specified device.

**Parameters:**

- `device_id`: The device to check

**Returns:** true if SAM is supported on the device, false otherwise

```cpp
static bool is_system_memory_supported(cuda_device_id device_id)
```

_Source: `cpp/include/rmm/mr/system_memory_resource.hpp:31`_

### System Memory Resource Class

Memory resource that uses malloc/free for allocation/deallocation.

There are two flavors of hardware/software environments that support accessing system allocated memory (SAM) from the GPU: HMM and ATS.

Heterogeneous Memory Management (HMM) is a software-based solution for PCIe-connected GPUs on x86 systems. Requirements: - NVIDIA CUDA 12.2 with the open-source r535_00 driver or newer. - A sufficiently recent Linux kernel: 6.1.24+, 6.2.11+, or 6.3+. - A GPU with one of the following supported architectures: NVIDIA Turing, NVIDIA Ampere, NVIDIA Ada Lovelace, NVIDIA Hopper, or newer. - A 64-bit x86 CPU.

For more information, see https://developer.nvidia.com/blog/simplifying-gpu-application-development-with-heterogeneous-memory-management/.

Address Translation Services (ATS) is a hardware/software solution for the Grace Hopper Superchip that uses the NVLink Chip-2-Chip (C2C) interconnect to provide coherent memory. For more information, see https://developer.nvidia.com/blog/nvidia-grace-hopper-superchip-architecture-in-depth/.

```cpp
class system_memory_resource final
```

_Source: `cpp/include/rmm/mr/system_memory_resource.hpp:68`_

### Allocate (system_memory_resource.hpp:95)

Allocates memory of size at least `bytes`.

The returned pointer will have at minimum 256 byte alignment.

The stream argument is ignored.

**Parameters:**

- `stream`: This argument is ignored
- `bytes`: The size of the allocation
- `alignment`: The alignment of the allocation

**Returns:** void* Pointer to the newly allocated memory

```cpp
void* allocate([[maybe_unused]] cuda::stream_ref stream, std::size_t bytes, [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
```

_Source: `cpp/include/rmm/mr/system_memory_resource.hpp:95`_

### Deallocate (system_memory_resource.hpp:122)

Deallocate memory pointed to by `ptr`.

This function synchronizes the stream before deallocating the memory.

**Parameters:**

- `stream`: The stream in which to order this deallocation
- `ptr`: Pointer to be deallocated
- `bytes`: The size in bytes of the allocation. This must be equal to the value of `bytes` that was passed to the `allocate` call that returned `ptr`.
- `alignment`: The alignment that was passed to the `allocate` call that returned `ptr`

```cpp
void deallocate(cuda::stream_ref stream, void* ptr, std::size_t bytes, [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
```

_Source: `cpp/include/rmm/mr/system_memory_resource.hpp:122`_

### Allocate Sync (system_memory_resource.hpp:144)

Allocates memory of size at least `bytes` synchronously.

**Parameters:**

- `bytes`: The size of the allocation
- `alignment`: The alignment of the allocation

**Returns:** void* Pointer to the newly allocated memory

```cpp
void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
```

_Source: `cpp/include/rmm/mr/system_memory_resource.hpp:144`_

### Deallocate Sync (system_memory_resource.hpp:158)

Deallocate memory pointed to by `ptr` synchronously.

**Parameters:**

- `ptr`: Pointer to be deallocated
- `bytes`: The size in bytes of the allocation
- `alignment`: The alignment that was passed to the `allocate` call that returned `ptr`

```cpp
void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
```

_Source: `cpp/include/rmm/mr/system_memory_resource.hpp:158`_

### Get Property (system_memory_resource.hpp:170)

Enables the `cuda::mr::device_accessible` property

This property declares that a `system_memory_resource` provides device-accessible memory

```cpp
RMM_CONSTEXPR_FRIEND void get_property(system_memory_resource const&, cuda::mr::device_accessible) noexcept
```

_Source: `cpp/include/rmm/mr/system_memory_resource.hpp:170`_

### Get Property (system_memory_resource.hpp:180)

Enables the `cuda::mr::host_accessible` property

This property declares that a `system_memory_resource` provides host-accessible memory

```cpp
RMM_CONSTEXPR_FRIEND void get_property(system_memory_resource const&, cuda::mr::host_accessible) noexcept
```

_Source: `cpp/include/rmm/mr/system_memory_resource.hpp:180`_

## `cpp/include/rmm/mr/per_device_resource.hpp`

### Initial Resource

Returns a reference to the initial resource.

Returns a global instance of a `cuda_memory_resource` as a function local static.

**Returns:** Reference to the static cuda_memory_resource used as the initial, default resource

```cpp
RMM_EXPORT inline cuda_memory_resource& initial_resource()
```

_Source: `cpp/include/rmm/mr/per_device_resource.hpp:57`_

### Ref Map Lock

returnReference to the lock

```cpp
RMM_EXPORT inline std::mutex& ref_map_lock()
```

_Source: `cpp/include/rmm/mr/per_device_resource.hpp:66`_

### Get Ref Map

returnReference to the map from device id -> any_resource

```cpp
RMM_EXPORT inline auto& get_ref_map()
```

_Source: `cpp/include/rmm/mr/per_device_resource.hpp:76`_

### Get Per Device Resource Ref

Get the `device_async_resource_ref` for the specified device.

Returns a `device_async_resource_ref` for the specified device. The initial resource_ref references a `cuda_memory_resource`.

`device_id.value()` must be in the range `[0, cudaGetDeviceCount())`, otherwise behavior is undefined.

This function is thread-safe with respect to concurrent calls to `set_per_device_resource`, `get_per_device_resource_ref`, `get_current_device_resource_ref`, `set_current_device_resource`, `reset_per_device_resource`, and `reset_current_device_resource`. Concurrent calls to any of these functions will result in a valid state, but the order of execution is undefined.

> **Note:** The returned `device_async_resource_ref` should only be used when CUDA device `device_id` is the current device (e.g. set using `cudaSetDevice()`). The behavior of a `device_async_resource_ref` is undefined if used while the active CUDA device is a different device from the one that was active when the memory resource was created.

**Parameters:**

- `device_id`: The id of the target device

**Returns:** The current `device_async_resource_ref` for device `device_id`

```cpp
inline device_async_resource_ref get_per_device_resource_ref(cuda_device_id device_id)
```

_Source: `cpp/include/rmm/mr/per_device_resource.hpp:110`_

### Set Per Device Resource

Set the memory resource for the specified device.

Takes ownership of the provided resource by value. The resource is moved into the per-device resource map.

`device_id.value()` must be in the range `[0, cudaGetDeviceCount())`, otherwise behavior is undefined.

This function is thread-safe with respect to concurrent calls to `set_per_device_resource`, `get_per_device_resource_ref`, `get_current_device_resource_ref`, `set_current_device_resource`, `reset_per_device_resource`, and `reset_current_device_resource`. Concurrent calls to any of these functions will result in a valid state, but the order of execution is undefined.

> **Note:** The resource passed in `new_resource` must have been created when device `device_id` was the current CUDA device (e.g. set using `cudaSetDevice()`). The behavior of a memory resource is undefined if used while the active CUDA device is a different device from the one that was active when the memory resource was created.

> **Note:** The per-device resource map keeps the provided resource alive until process exit. Its destructor may therefore run during process termination. If the destructor may call CUDA APIs, it must consult `rmm::process_is_exiting()` and skip those calls when it returns `true`.

**Parameters:**

- `device_id`: The id of the target device
- `new_resource`: New resource to use for `device_id`

**Returns:** An owning `any_resource` holding the previous resource for `device_id`

```cpp
inline cuda::mr::any_resource<cuda::mr::device_accessible> set_per_device_resource( cuda_device_id device_id, cuda::mr::any_resource<cuda::mr::device_accessible> new_resource)
```

_Source: `cpp/include/rmm/mr/per_device_resource.hpp:151`_

### Get Current Device Resource Ref

Get the `device_async_resource_ref` for the current device.

Returns the `device_async_resource_ref` set for the current device. The initial resource_ref references a `cuda_memory_resource`.

The "current device" is the device returned by `cudaGetDevice`.

This function is thread-safe with respect to concurrent calls to `set_per_device_resource`, `get_per_device_resource_ref`, `get_current_device_resource_ref`, `set_current_device_resource`, `reset_per_device_resource`, and `reset_current_device_resource`. Concurrent calls to any of these functions will result in a valid state, but the order of execution is undefined.

> **Note:** The returned `device_async_resource_ref` should only be used with the current CUDA device. Changing the current device (e.g. using `cudaSetDevice()`) and then using the returned `resource_ref` can result in undefined behavior. The behavior of a `device_async_resource_ref` is undefined if used while the active CUDA device is a different device from the one that was active when the memory resource was created.

**Returns:** `device_async_resource_ref` active for the current device

```cpp
inline device_async_resource_ref get_current_device_resource_ref()
```

_Source: `cpp/include/rmm/mr/per_device_resource.hpp:187`_

### Set Current Device Resource

Set the memory resource for the current device.

Takes ownership of the provided resource by value. The "current device" is the device returned by `cudaGetDevice`.

This function is thread-safe with respect to concurrent calls to `set_per_device_resource`, `get_per_device_resource_ref`, `get_current_device_resource_ref`, `set_current_device_resource`, `reset_per_device_resource`, and `reset_current_device_resource`. Concurrent calls to any of these functions will result in a valid state, but the order of execution is undefined.

> **Note:** The resource passed in `new_resource` must have been created for the current CUDA device. The behavior of a memory resource is undefined if used while the active CUDA device is a different device from the one that was active when the memory resource was created.

> **Note:** The per-device resource map keeps the provided resource alive until process exit. Its destructor may therefore run during process termination. If the destructor may call CUDA APIs, it must consult `rmm::process_is_exiting()` and skip those calls when it returns `true`.

**Parameters:**

- `new_resource`: New resource to use for the current device

**Returns:** An owning `any_resource` holding the previous resource for the current device

```cpp
inline cuda::mr::any_resource<cuda::mr::device_accessible> set_current_device_resource( cuda::mr::any_resource<cuda::mr::device_accessible> new_resource)
```

_Source: `cpp/include/rmm/mr/per_device_resource.hpp:215`_

### Reset Per Device Resource

Reset the memory resource for the specified device to the initial resource.

Resets to the initial `cuda_memory_resource`.

`device_id.value()` must be in the range `[0, cudaGetDeviceCount())`, otherwise behavior is undefined.

This function is thread-safe with respect to concurrent calls to `set_per_device_resource`, `get_per_device_resource_ref`, `get_current_device_resource_ref`, `set_current_device_resource`, `reset_per_device_resource`, and `reset_current_device_resource`. Concurrent calls to any of these functions will result in a valid state, but the order of execution is undefined.

**Parameters:**

- `device_id`: The id of the target device

**Returns:** An owning `any_resource` holding the previous resource for `device_id`

```cpp
inline cuda::mr::any_resource<cuda::mr::device_accessible> reset_per_device_resource( cuda_device_id device_id)
```

_Source: `cpp/include/rmm/mr/per_device_resource.hpp:238`_

### Reset Current Device Resource

Reset the memory resource for the current device to the initial resource.

Resets to the initial `cuda_memory_resource`. The "current device" is the device returned by `cudaGetDevice`.

This function is thread-safe with respect to concurrent calls to `set_per_device_resource`, `get_per_device_resource_ref`, `get_current_device_resource_ref`, `set_current_device_resource`, `reset_per_device_resource`, and `reset_current_device_resource`. Concurrent calls to any of these functions will result in a valid state, but the order of execution is undefined.

**Returns:** An owning `any_resource` holding the previous resource for the current device

```cpp
inline cuda::mr::any_resource<cuda::mr::device_accessible> reset_current_device_resource()
```

_Source: `cpp/include/rmm/mr/per_device_resource.hpp:258`_

## `cpp/include/rmm/mr/polymorphic_allocator.hpp`

### Polymorphic Allocator Constructor (polymorphic_allocator.hpp:49)

Construct a `polymorphic_allocator` using the return value of `rmm::mr::get_current_device_resource_ref()` as the underlying memory resource.

```cpp
polymorphic_allocator() = default;
```

_Source: `cpp/include/rmm/mr/polymorphic_allocator.hpp:49`_

### Polymorphic Allocator Constructor (polymorphic_allocator.hpp:58)

Construct a `polymorphic_allocator` using the provided memory resource.

This constructor provides an implicit conversion from `device_async_resource_ref`.

**Parameters:**

- `mr`: The upstream memory resource to use for allocation.

```cpp
polymorphic_allocator(cuda::mr::any_resource<cuda::mr::device_accessible> mr) : mr_(std::move(mr))
```

_Source: `cpp/include/rmm/mr/polymorphic_allocator.hpp:58`_

### Polymorphic Allocator

Construct a `polymorphic_allocator` using the underlying memory resource of `other`.

**Parameters:**

- `other`: The `polymorphic_allocator` whose memory resource will be used as the underlying resource of the new `polymorphic_allocator`.

```cpp
template <typename U> polymorphic_allocator(polymorphic_allocator<U> const& other) noexcept : mr_(other.get_upstream_resource())
```

_Source: `cpp/include/rmm/mr/polymorphic_allocator.hpp:68`_

### Allocate (polymorphic_allocator.hpp:81)

Allocates storage for `num` objects of type `T` using the underlying memory resource.

**Parameters:**

- `num`: The number of objects to allocate storage for
- `stream`: The stream on which to perform the allocation

**Returns:** Pointer to the allocated storage

```cpp
value_type* allocate(std::size_t num, cuda_stream_view stream)
```

_Source: `cpp/include/rmm/mr/polymorphic_allocator.hpp:81`_

### Deallocate (polymorphic_allocator.hpp:97)

Deallocates storage pointed to by `ptr`.

`ptr` must have been allocated from a memory resource `r` that compares equal to `get_upstream_resource()` using `r.allocate(n * sizeof(T))`.

**Parameters:**

- `ptr`: Pointer to memory to deallocate
- `num`: Number of objects originally allocated
- `stream`: Stream on which to perform the deallocation

```cpp
void deallocate(value_type* ptr, std::size_t num, cuda_stream_view stream) noexcept
```

_Source: `cpp/include/rmm/mr/polymorphic_allocator.hpp:97`_

### Stream Allocator Adaptor Constructor (polymorphic_allocator.hpp:179)

< by this allocator

```cpp
stream_allocator_adaptor() = delete;
```

_Source: `cpp/include/rmm/mr/polymorphic_allocator.hpp:179`_

### Stream Allocator Adaptor Constructor (polymorphic_allocator.hpp:190)

Construct a `stream_allocator_adaptor` using `a` as the underlying allocator.

> **Note:** The `stream` must not be destroyed before the `stream_allocator_adaptor`, otherwise behavior is undefined.

**Parameters:**

- `allocator`: The stream ordered allocator to use as the underlying allocator
- `stream`: The stream used with the underlying allocator

```cpp
stream_allocator_adaptor(Allocator const& allocator, cuda_stream_view stream) : alloc_
```

_Source: `cpp/include/rmm/mr/polymorphic_allocator.hpp:190`_

### Stream Allocator Adaptor

Construct a `stream_allocator_adaptor` using `other.underlying_allocator()` and `other.stream()` as the underlying allocator and stream.

**Template Parameters:**

- `OtherAllocator`: Type of `other`'s underlying allocator

**Parameters:**

- `other`: The other `stream_allocator_adaptor` whose underlying allocator and stream will be copied

```cpp
template <typename OtherAllocator> stream_allocator_adaptor(stream_allocator_adaptor<OtherAllocator> const& other) : stream_allocator_adaptor
```

_Source: `cpp/include/rmm/mr/polymorphic_allocator.hpp:203`_

### Allocate (polymorphic_allocator.hpp:227)

Allocates storage for `num` objects of type `T` using the underlying allocator on `stream()`.

**Parameters:**

- `num`: The number of objects to allocate storage for

**Returns:** Pointer to the allocated storage

```cpp
value_type* allocate(std::size_t num)
```

_Source: `cpp/include/rmm/mr/polymorphic_allocator.hpp:227`_

### Deallocate (polymorphic_allocator.hpp:238)

Deallocates storage pointed to by `ptr` using the underlying allocator on `stream()`.

`ptr` must have been allocated from by an allocator `a` that compares equal to `underlying_allocator()` using `a.allocate(n)`.

**Parameters:**

- `ptr`: Pointer to memory to deallocate
- `num`: Number of objects originally allocated

```cpp
void deallocate(value_type* ptr, std::size_t num) noexcept
```

_Source: `cpp/include/rmm/mr/polymorphic_allocator.hpp:238`_
