---
slug: api-reference/cpp-api-memory-resources
---

# Memory Resources

Generated from RMM C++ headers.

## `cpp/include/rmm/mr/arena_memory_resource.hpp`

### Arena Memory Resource Class

A suballocator that emphasizes fragmentation avoidance and scalable concurrency support.

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

```cpp
explicit arena_memory_resource(cuda::mr::any_resource<cuda::mr::device_accessible> upstream, std::optional<std::size_t> arena_size = std::nullopt, bool dump_log_on_failure = false);
```

_Source: `cpp/include/rmm/mr/arena_memory_resource.hpp:84`_

## `cpp/include/rmm/mr/binning_memory_resource.hpp`

### Binning Memory Resource Class

Allocates memory from upstream resources associated with bin sizes.

```cpp
class RMM_EXPORT binning_memory_resource : public cuda::mr::shared_resource<detail::binning_memory_resource_impl>
```

_Source: `cpp/include/rmm/mr/binning_memory_resource.hpp:30`_

### Get Property (binning_memory_resource.hpp:40)

Enables the `cuda::mr::device_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(binning_memory_resource const&, cuda::mr::device_accessible) noexcept
```

_Source: `cpp/include/rmm/mr/binning_memory_resource.hpp:40`_

### Binning Memory Resource Constructor (binning_memory_resource.hpp:53)

Construct a new binning memory resource object.

```cpp
explicit binning_memory_resource(cuda::mr::any_resource<cuda::mr::device_accessible> upstream);
```

_Source: `cpp/include/rmm/mr/binning_memory_resource.hpp:53`_

### Binning Memory Resource Constructor (binning_memory_resource.hpp:67)

Construct a new binning memory resource object with a range of initial bins.

```cpp
binning_memory_resource(cuda::mr::any_resource<cuda::mr::device_accessible> upstream, int8_t min_size_exponent, int8_t max_size_exponent);
```

_Source: `cpp/include/rmm/mr/binning_memory_resource.hpp:67`_

### Add Bin

Add a bin allocator to this resource

```cpp
void add_bin(std::size_t allocation_size, std::optional<device_async_resource_ref> bin_resource = std::nullopt);
```

_Source: `cpp/include/rmm/mr/binning_memory_resource.hpp:93`_

## `cpp/include/rmm/mr/callback_memory_resource.hpp`

### Allocate Callback T Type Alias

Callback function type used by callback memory resource for allocation.

```cpp
using allocate_callback_t = std::function<void*(std::size_t, cuda_stream_view, void*)>;
```

_Source: `cpp/include/rmm/mr/callback_memory_resource.hpp:36`_

### Deallocate Callback T Type Alias

Callback function type used by callback_memory_resource for deallocation.

```cpp
using deallocate_callback_t = std::function<void(void*, std::size_t, cuda_stream_view, void*)>;
```

_Source: `cpp/include/rmm/mr/callback_memory_resource.hpp:52`_

### Callback Memory Resource Class

A device memory resource that uses the provided callbacks for memory allocation and deallocation.

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

```cpp
enum class allocation_handle_type : std::int32_t
```

_Source: `cpp/include/rmm/mr/cuda_async_memory_resource.hpp:47`_

### Mempool Usage Enum

Flags for specifying memory pool usage.

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

```cpp
cuda_async_view_memory_resource(cudaMemPool_t pool_handle) : cuda_pool_handle_
```

_Source: `cpp/include/rmm/mr/cuda_async_view_memory_resource.hpp:42`_

### Allocate (cuda_async_view_memory_resource.hpp:81)

Allocates memory of size at least `bytes`.

```cpp
void* allocate(cuda::stream_ref stream, std::size_t bytes, [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
```

_Source: `cpp/include/rmm/mr/cuda_async_view_memory_resource.hpp:81`_

### Deallocate (cuda_async_view_memory_resource.hpp:101)

Deallocate memory pointed to by `ptr`.

```cpp
void deallocate(cuda::stream_ref stream, void* ptr, [[maybe_unused]] std::size_t bytes, [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
```

_Source: `cpp/include/rmm/mr/cuda_async_view_memory_resource.hpp:101`_

### Allocate Sync (cuda_async_view_memory_resource.hpp:116)

Allocates memory of size at least `bytes` synchronously.

```cpp
void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
```

_Source: `cpp/include/rmm/mr/cuda_async_view_memory_resource.hpp:116`_

### Deallocate Sync (cuda_async_view_memory_resource.hpp:130)

Deallocate memory pointed to by `ptr` synchronously.

```cpp
void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
```

_Source: `cpp/include/rmm/mr/cuda_async_view_memory_resource.hpp:130`_

### Get Property (cuda_async_view_memory_resource.hpp:163)

Enables the `cuda::mr::device_accessible` property

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

```cpp
void* allocate([[maybe_unused]] cuda::stream_ref stream, std::size_t bytes, [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
```

_Source: `cpp/include/rmm/mr/cuda_memory_resource.hpp:49`_

### Deallocate (cuda_memory_resource.hpp:69)

Deallocate memory pointed to by `ptr`.

```cpp
void deallocate([[maybe_unused]] cuda::stream_ref stream, void* ptr, [[maybe_unused]] std::size_t bytes, [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
```

_Source: `cpp/include/rmm/mr/cuda_memory_resource.hpp:69`_

### Allocate Sync (cuda_memory_resource.hpp:84)

Allocates memory of size at least `bytes` synchronously.

```cpp
void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
```

_Source: `cpp/include/rmm/mr/cuda_memory_resource.hpp:84`_

### Deallocate Sync (cuda_memory_resource.hpp:98)

Deallocate memory pointed to by `ptr` synchronously.

```cpp
void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
```

_Source: `cpp/include/rmm/mr/cuda_memory_resource.hpp:98`_

### Get Property (cuda_memory_resource.hpp:110)

Enables the `cuda::mr::device_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(cuda_memory_resource const&, cuda::mr::device_accessible) noexcept
```

_Source: `cpp/include/rmm/mr/cuda_memory_resource.hpp:110`_

## `cpp/include/rmm/mr/fixed_size_memory_resource.hpp`

### Fixed Size Memory Resource Class

A memory resource which allocates memory blocks of a single fixed size.

```cpp
class RMM_EXPORT fixed_size_memory_resource : public cuda::mr::shared_resource<detail::fixed_size_memory_resource_impl>
```

_Source: `cpp/include/rmm/mr/fixed_size_memory_resource.hpp:31`_

### Get Property (fixed_size_memory_resource.hpp:41)

Enables the `cuda::mr::device_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(fixed_size_memory_resource const&, cuda::mr::device_accessible) noexcept
```

_Source: `cpp/include/rmm/mr/fixed_size_memory_resource.hpp:41`_

### Fixed Size Memory Resource Constructor

Construct a new `fixed_size_memory_resource` that allocates memory from `upstream`.

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

```cpp
void* allocate([[maybe_unused]] cuda::stream_ref stream, std::size_t bytes, [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
```

_Source: `cpp/include/rmm/mr/managed_memory_resource.hpp:49`_

### Deallocate (managed_memory_resource.hpp:73)

Deallocate memory pointed to by `ptr`.

```cpp
void deallocate([[maybe_unused]] cuda::stream_ref stream, void* ptr, [[maybe_unused]] std::size_t bytes, [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
```

_Source: `cpp/include/rmm/mr/managed_memory_resource.hpp:73`_

### Allocate Sync (managed_memory_resource.hpp:88)

Allocates memory of size at least `bytes` synchronously.

```cpp
void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
```

_Source: `cpp/include/rmm/mr/managed_memory_resource.hpp:88`_

### Deallocate Sync (managed_memory_resource.hpp:102)

Deallocate memory pointed to by `ptr` synchronously.

```cpp
void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
```

_Source: `cpp/include/rmm/mr/managed_memory_resource.hpp:102`_

### Get Property (managed_memory_resource.hpp:114)

Enables the `cuda::mr::device_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(managed_memory_resource const&, cuda::mr::device_accessible) noexcept
```

_Source: `cpp/include/rmm/mr/managed_memory_resource.hpp:114`_

### Get Property (managed_memory_resource.hpp:124)

Enables the `cuda::mr::host_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(managed_memory_resource const&, cuda::mr::host_accessible) noexcept
```

_Source: `cpp/include/rmm/mr/managed_memory_resource.hpp:124`_

## `cpp/include/rmm/mr/pinned_host_memory_resource.hpp`

### Pinned Host Memory Resource Class

Memory resource class for allocating pinned host memory.

```cpp
class pinned_host_memory_resource final
```

_Source: `cpp/include/rmm/mr/pinned_host_memory_resource.hpp:34`_

### Allocate (pinned_host_memory_resource.hpp:63)

Allocates pinned host memory of size at least `bytes` bytes.

```cpp
void* allocate([[maybe_unused]] cuda::stream_ref stream, std::size_t bytes, [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
```

_Source: `cpp/include/rmm/mr/pinned_host_memory_resource.hpp:63`_

### Deallocate (pinned_host_memory_resource.hpp:89)

Deallocate memory pointed to by `ptr`.

```cpp
void deallocate([[maybe_unused]] cuda::stream_ref stream, void* ptr, std::size_t bytes, [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
```

_Source: `cpp/include/rmm/mr/pinned_host_memory_resource.hpp:89`_

### Allocate Sync (pinned_host_memory_resource.hpp:107)

Allocates pinned host memory of size at least `bytes` bytes synchronously.

```cpp
void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
```

_Source: `cpp/include/rmm/mr/pinned_host_memory_resource.hpp:107`_

### Deallocate Sync (pinned_host_memory_resource.hpp:121)

Deallocate memory pointed to by `ptr` synchronously.

```cpp
void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
```

_Source: `cpp/include/rmm/mr/pinned_host_memory_resource.hpp:121`_

### Get Property (pinned_host_memory_resource.hpp:133)

Enables the `cuda::mr::device_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(pinned_host_memory_resource const&, cuda::mr::device_accessible) noexcept
```

_Source: `cpp/include/rmm/mr/pinned_host_memory_resource.hpp:133`_

### Get Property (pinned_host_memory_resource.hpp:143)

Enables the `cuda::mr::host_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(pinned_host_memory_resource const&, cuda::mr::host_accessible) noexcept
```

_Source: `cpp/include/rmm/mr/pinned_host_memory_resource.hpp:143`_

## `cpp/include/rmm/mr/pool_memory_resource.hpp`

### Pool Memory Resource Class

A coalescing best-fit suballocator which uses a pool of memory allocated from an upstream memory_resource.

```cpp
class RMM_EXPORT pool_memory_resource : public cuda::mr::shared_resource<detail::pool_memory_resource_impl>
```

_Source: `cpp/include/rmm/mr/pool_memory_resource.hpp:34`_

### Get Property (pool_memory_resource.hpp:44)

Enables the `cuda::mr::device_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(pool_memory_resource const&, cuda::mr::device_accessible) noexcept
```

_Source: `cpp/include/rmm/mr/pool_memory_resource.hpp:44`_

### Pool Memory Resource Constructor

Construct a `pool_memory_resource` and allocate the initial device memory pool using `upstream`.

```cpp
explicit pool_memory_resource(cuda::mr::any_resource<cuda::mr::device_accessible> upstream, std::size_t initial_pool_size, std::optional<std::size_t> maximum_pool_size = std::nullopt);
```

_Source: `cpp/include/rmm/mr/pool_memory_resource.hpp:62`_

## `cpp/include/rmm/mr/sam_headroom_memory_resource.hpp`

### Sam Headroom Memory Resource Class

Resource that uses system memory resource to allocate memory with a headroom.

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

```cpp
explicit sam_headroom_memory_resource(std::size_t headroom);
```

_Source: `cpp/include/rmm/mr/sam_headroom_memory_resource.hpp:61`_

## `cpp/include/rmm/mr/system_memory_resource.hpp`

### Is System Memory Supported

Check if system allocated memory (SAM) is supported on the specified device.

```cpp
static bool is_system_memory_supported(cuda_device_id device_id)
```

_Source: `cpp/include/rmm/mr/system_memory_resource.hpp:31`_

### System Memory Resource Class

Memory resource that uses malloc/free for allocation/deallocation.

```cpp
class system_memory_resource final
```

_Source: `cpp/include/rmm/mr/system_memory_resource.hpp:68`_

### Allocate (system_memory_resource.hpp:95)

Allocates memory of size at least `bytes`.

```cpp
void* allocate([[maybe_unused]] cuda::stream_ref stream, std::size_t bytes, [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
```

_Source: `cpp/include/rmm/mr/system_memory_resource.hpp:95`_

### Deallocate (system_memory_resource.hpp:122)

Deallocate memory pointed to by `ptr`.

```cpp
void deallocate(cuda::stream_ref stream, void* ptr, std::size_t bytes, [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
```

_Source: `cpp/include/rmm/mr/system_memory_resource.hpp:122`_

### Allocate Sync (system_memory_resource.hpp:144)

Allocates memory of size at least `bytes` synchronously.

```cpp
void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
```

_Source: `cpp/include/rmm/mr/system_memory_resource.hpp:144`_

### Deallocate Sync (system_memory_resource.hpp:158)

Deallocate memory pointed to by `ptr` synchronously.

```cpp
void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
```

_Source: `cpp/include/rmm/mr/system_memory_resource.hpp:158`_

### Get Property (system_memory_resource.hpp:170)

Enables the `cuda::mr::device_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(system_memory_resource const&, cuda::mr::device_accessible) noexcept
```

_Source: `cpp/include/rmm/mr/system_memory_resource.hpp:170`_

### Get Property (system_memory_resource.hpp:180)

Enables the `cuda::mr::host_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(system_memory_resource const&, cuda::mr::host_accessible) noexcept
```

_Source: `cpp/include/rmm/mr/system_memory_resource.hpp:180`_

## `cpp/include/rmm/mr/per_device_resource.hpp`

### Initial Resource

Returns a reference to the initial resource.

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

```cpp
inline device_async_resource_ref get_per_device_resource_ref(cuda_device_id device_id)
```

_Source: `cpp/include/rmm/mr/per_device_resource.hpp:110`_

### Set Per Device Resource

Set the memory resource for the specified device.

```cpp
inline cuda::mr::any_resource<cuda::mr::device_accessible> set_per_device_resource( cuda_device_id device_id, cuda::mr::any_resource<cuda::mr::device_accessible> new_resource)
```

_Source: `cpp/include/rmm/mr/per_device_resource.hpp:151`_

### Get Current Device Resource Ref

Get the `device_async_resource_ref` for the current device.

```cpp
inline device_async_resource_ref get_current_device_resource_ref()
```

_Source: `cpp/include/rmm/mr/per_device_resource.hpp:187`_

### Set Current Device Resource

Set the memory resource for the current device.

```cpp
inline cuda::mr::any_resource<cuda::mr::device_accessible> set_current_device_resource( cuda::mr::any_resource<cuda::mr::device_accessible> new_resource)
```

_Source: `cpp/include/rmm/mr/per_device_resource.hpp:215`_

### Reset Per Device Resource

Reset the memory resource for the specified device to the initial resource.

```cpp
inline cuda::mr::any_resource<cuda::mr::device_accessible> reset_per_device_resource( cuda_device_id device_id)
```

_Source: `cpp/include/rmm/mr/per_device_resource.hpp:238`_

### Reset Current Device Resource

Reset the memory resource for the current device to the initial resource.

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

```cpp
polymorphic_allocator(cuda::mr::any_resource<cuda::mr::device_accessible> mr) : mr_(std::move(mr))
```

_Source: `cpp/include/rmm/mr/polymorphic_allocator.hpp:58`_

### Polymorphic Allocator

Construct a `polymorphic_allocator` using the underlying memory resource of `other`.

```cpp
template <typename U> polymorphic_allocator(polymorphic_allocator<U> const& other) noexcept : mr_(other.get_upstream_resource())
```

_Source: `cpp/include/rmm/mr/polymorphic_allocator.hpp:68`_

### Allocate (polymorphic_allocator.hpp:81)

Allocates storage for `num` objects of type `T` using the underlying memory resource.

```cpp
value_type* allocate(std::size_t num, cuda_stream_view stream)
```

_Source: `cpp/include/rmm/mr/polymorphic_allocator.hpp:81`_

### Deallocate (polymorphic_allocator.hpp:97)

Deallocates storage pointed to by `ptr`.

```cpp
void deallocate(value_type* ptr, std::size_t num, cuda_stream_view stream) noexcept
```

_Source: `cpp/include/rmm/mr/polymorphic_allocator.hpp:97`_

### Stream Allocator Adaptor Constructor

Construct a `stream_allocator_adaptor` using `a` as the underlying allocator.

```cpp
stream_allocator_adaptor(Allocator const& allocator, cuda_stream_view stream) : alloc_
```

_Source: `cpp/include/rmm/mr/polymorphic_allocator.hpp:190`_

### Stream Allocator Adaptor

Construct a `stream_allocator_adaptor` using `other.underlying_allocator()` and `other.stream()` as the underlying allocator and stream.

```cpp
template <typename OtherAllocator> stream_allocator_adaptor(stream_allocator_adaptor<OtherAllocator> const& other) : stream_allocator_adaptor
```

_Source: `cpp/include/rmm/mr/polymorphic_allocator.hpp:203`_

### Allocate (polymorphic_allocator.hpp:227)

Allocates storage for `num` objects of type `T` using the underlying allocator on `stream()`.

```cpp
value_type* allocate(std::size_t num)
```

_Source: `cpp/include/rmm/mr/polymorphic_allocator.hpp:227`_

### Deallocate (polymorphic_allocator.hpp:238)

Deallocates storage pointed to by `ptr` using the underlying allocator on `stream()`.

```cpp
void deallocate(value_type* ptr, std::size_t num) noexcept
```

_Source: `cpp/include/rmm/mr/polymorphic_allocator.hpp:238`_
