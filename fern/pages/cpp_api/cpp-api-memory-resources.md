---
slug: api-reference/cpp-api-memory-resources
---

# Memory Resources

Generated from RMM C++ headers.

## `cpp/include/rmm/mr/arena_memory_resource.hpp`

### Arena Memory Resource

A suballocator that emphasizes fragmentation avoidance and scalable concurrency support.

```cpp
class RMM_EXPORT arena_memory_resource : public cuda::mr::shared_resource<detail::arena_memory_resource_impl> {
```

_Source: `cpp/include/rmm/mr/arena_memory_resource.hpp:63`_

### Get Property

Enables the `cuda::mr::device_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(arena_memory_resource const&, cuda::mr::device_accessible) noexcept {
```

_Source: `cpp/include/rmm/mr/arena_memory_resource.hpp:71`_

### Arena Memory Resource

Construct an `arena_memory_resource`.

```cpp
explicit arena_memory_resource(cuda::mr::any_resource<cuda::mr::device_accessible> upstream, std::optional<std::size_t> arena_size = std::nullopt, bool dump_log_on_failure = false);
```

_Source: `cpp/include/rmm/mr/arena_memory_resource.hpp:84`_

## `cpp/include/rmm/mr/binning_memory_resource.hpp`

### Binning Memory Resource

Allocates memory from upstream resources associated with bin sizes.

```cpp
class RMM_EXPORT binning_memory_resource : public cuda::mr::shared_resource<detail::binning_memory_resource_impl> {
```

_Source: `cpp/include/rmm/mr/binning_memory_resource.hpp:30`_

### Get Property

Enables the `cuda::mr::device_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(binning_memory_resource const&, cuda::mr::device_accessible) noexcept {
```

_Source: `cpp/include/rmm/mr/binning_memory_resource.hpp:40`_

### Binning Memory Resource

Construct a new binning memory resource object.

```cpp
explicit binning_memory_resource(cuda::mr::any_resource<cuda::mr::device_accessible> upstream);
```

_Source: `cpp/include/rmm/mr/binning_memory_resource.hpp:53`_

### Binning Memory Resource

Construct a new binning memory resource object with a range of initial bins.

```cpp
binning_memory_resource(cuda::mr::any_resource<cuda::mr::device_accessible> upstream, int8_t min_size_exponent, // NOLINT(bugprone-easily-swappable-parameters) int8_t max_size_exponent);
```

_Source: `cpp/include/rmm/mr/binning_memory_resource.hpp:67`_

### Add Bin

Add a bin allocator to this resource

```cpp
void add_bin(std::size_t allocation_size, std::optional<device_async_resource_ref> bin_resource = std::nullopt);
```

_Source: `cpp/include/rmm/mr/binning_memory_resource.hpp:93`_

## `cpp/include/rmm/mr/callback_memory_resource.hpp`

### Allocate Callback T

Callback function type used by callback memory resource for allocation.

```cpp
using allocate_callback_t = std::function<void*(std::size_t, cuda_stream_view, void*)>;
```

_Source: `cpp/include/rmm/mr/callback_memory_resource.hpp:36`_

### Deallocate Callback T

Callback function type used by callback_memory_resource for deallocation.

```cpp
using deallocate_callback_t = std::function<void(void*, std::size_t, cuda_stream_view, void*)>;
```

_Source: `cpp/include/rmm/mr/callback_memory_resource.hpp:52`_

### Callback Memory Resource

A device memory resource that uses the provided callbacks for memory allocation

```cpp
class RMM_EXPORT callback_memory_resource : public cuda::mr::shared_resource<detail::callback_memory_resource_impl> {
```

_Source: `cpp/include/rmm/mr/callback_memory_resource.hpp:65`_

### Get Property

Enables the `cuda::mr::device_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(callback_memory_resource const&, cuda::mr::device_accessible) noexcept {
```

_Source: `cpp/include/rmm/mr/callback_memory_resource.hpp:73`_

### Callback Memory Resource

Construct a new callback memory resource.

```cpp
callback_memory_resource(allocate_callback_t allocate_callback, deallocate_callback_t deallocate_callback, void* allocate_callback_arg = nullptr, void* deallocate_callback_arg = nullptr);
```

_Source: `cpp/include/rmm/mr/callback_memory_resource.hpp:93`_

## `cpp/include/rmm/mr/cuda_async_managed_memory_resource.hpp`

### CUDA Async Managed Memory Resource

Memory resource that uses `cudaMallocFromPoolAsync`/`cudaFreeFromPoolAsync`

```cpp
class RMM_EXPORT cuda_async_managed_memory_resource final : public cuda::mr::shared_resource<detail::cuda_async_managed_memory_resource_impl> {
```

_Source: `cpp/include/rmm/mr/cuda_async_managed_memory_resource.hpp:28`_

### Get Property

Enables the `cuda::mr::device_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(cuda_async_managed_memory_resource const&, cuda::mr::device_accessible) noexcept {
```

_Source: `cpp/include/rmm/mr/cuda_async_managed_memory_resource.hpp:36`_

### Get Property

Enables the `cuda::mr::host_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(cuda_async_managed_memory_resource const&, cuda::mr::host_accessible) noexcept {
```

_Source: `cpp/include/rmm/mr/cuda_async_managed_memory_resource.hpp:44`_

### CUDA Async Managed Memory Resource

Constructs a cuda_async_managed_memory_resource with the default managed memory pool for

```cpp
cuda_async_managed_memory_resource();
```

_Source: `cpp/include/rmm/mr/cuda_async_managed_memory_resource.hpp:59`_

## `cpp/include/rmm/mr/cuda_async_memory_resource.hpp`

### CUDA Async Memory Resource

Memory resource that uses `cudaMallocAsync`/`cudaFreeAsync` for

```cpp
class RMM_EXPORT cuda_async_memory_resource final : public cuda::mr::shared_resource<detail::cuda_async_memory_resource_impl> {
```

_Source: `cpp/include/rmm/mr/cuda_async_memory_resource.hpp:30`_

### Allocation Handle Type

Flags for specifying memory allocation handle types.

```cpp
enum class allocation_handle_type : std::int32_t {
```

_Source: `cpp/include/rmm/mr/cuda_async_memory_resource.hpp:47`_

### Mempool Usage

Flags for specifying memory pool usage.

```cpp
enum class mempool_usage : unsigned short {
```

_Source: `cpp/include/rmm/mr/cuda_async_memory_resource.hpp:70`_

### Get Property

Enables the `cuda::mr::device_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(cuda_async_memory_resource const&, cuda::mr::device_accessible) noexcept {
```

_Source: `cpp/include/rmm/mr/cuda_async_memory_resource.hpp:78`_

### CUDA Async Memory Resource

Constructs a cuda_async_memory_resource with the optionally specified initial pool size

```cpp
cuda_async_memory_resource(std::optional<std::size_t> initial_pool_size = {}, std::optional<std::size_t> release_threshold = {}, std::optional<allocation_handle_type> export_handle_type = {});
```

_Source: `cpp/include/rmm/mr/cuda_async_memory_resource.hpp:103`_

## `cpp/include/rmm/mr/cuda_async_view_memory_resource.hpp`

### CUDA Async View Memory Resource

Memory resource that uses `cudaMallocAsync`/`cudaFreeAsync` for

```cpp
class cuda_async_view_memory_resource final {
```

_Source: `cpp/include/rmm/mr/cuda_async_view_memory_resource.hpp:30`_

### CUDA Async View Memory Resource

Constructs a cuda_async_view_memory_resource which uses an existing CUDA memory pool.

```cpp
cuda_async_view_memory_resource(cudaMemPool_t pool_handle) : cuda_pool_handle_{[pool_handle]() {
```

_Source: `cpp/include/rmm/mr/cuda_async_view_memory_resource.hpp:42`_

### Allocate

Allocates memory of size at least `bytes`.

```cpp
void* allocate(cuda::stream_ref stream, std::size_t bytes, [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) {
```

_Source: `cpp/include/rmm/mr/cuda_async_view_memory_resource.hpp:81`_

### Deallocate

Deallocate memory pointed to by `ptr`.

```cpp
void deallocate(cuda::stream_ref stream, void* ptr, [[maybe_unused]] std::size_t bytes, [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept {
```

_Source: `cpp/include/rmm/mr/cuda_async_view_memory_resource.hpp:101`_

### Allocate Sync

Allocates memory of size at least `bytes` synchronously.

```cpp
void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) {
```

_Source: `cpp/include/rmm/mr/cuda_async_view_memory_resource.hpp:116`_

### Deallocate Sync

Deallocate memory pointed to by `ptr` synchronously.

```cpp
void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept {
```

_Source: `cpp/include/rmm/mr/cuda_async_view_memory_resource.hpp:130`_

### Get Property

Enables the `cuda::mr::device_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(cuda_async_view_memory_resource const&, cuda::mr::device_accessible) noexcept {
```

_Source: `cpp/include/rmm/mr/cuda_async_view_memory_resource.hpp:163`_

## `cpp/include/rmm/mr/cuda_memory_resource.hpp`

### CUDA Memory Resource

Memory resource that uses cudaMalloc/Free for allocation/deallocation.

```cpp
class cuda_memory_resource final {
```

_Source: `cpp/include/rmm/mr/cuda_memory_resource.hpp:26`_

### Allocate

Allocates memory of size at least `bytes`.

```cpp
void* allocate([[maybe_unused]] cuda::stream_ref stream, std::size_t bytes, [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) {
```

_Source: `cpp/include/rmm/mr/cuda_memory_resource.hpp:49`_

### Deallocate

Deallocate memory pointed to by `ptr`.

```cpp
void deallocate([[maybe_unused]] cuda::stream_ref stream, void* ptr, [[maybe_unused]] std::size_t bytes, [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept {
```

_Source: `cpp/include/rmm/mr/cuda_memory_resource.hpp:69`_

### Allocate Sync

Allocates memory of size at least `bytes` synchronously.

```cpp
void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) {
```

_Source: `cpp/include/rmm/mr/cuda_memory_resource.hpp:84`_

### Deallocate Sync

Deallocate memory pointed to by `ptr` synchronously.

```cpp
void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept {
```

_Source: `cpp/include/rmm/mr/cuda_memory_resource.hpp:98`_

### Get Property

Enables the `cuda::mr::device_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(cuda_memory_resource const&, cuda::mr::device_accessible) noexcept {
```

_Source: `cpp/include/rmm/mr/cuda_memory_resource.hpp:110`_

## `cpp/include/rmm/mr/fixed_size_memory_resource.hpp`

### Fixed Size Memory Resource

A memory resource which allocates memory blocks of a single fixed size.

```cpp
class RMM_EXPORT fixed_size_memory_resource : public cuda::mr::shared_resource<detail::fixed_size_memory_resource_impl> {
```

_Source: `cpp/include/rmm/mr/fixed_size_memory_resource.hpp:31`_

### Get Property

Enables the `cuda::mr::device_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(fixed_size_memory_resource const&, cuda::mr::device_accessible) noexcept {
```

_Source: `cpp/include/rmm/mr/fixed_size_memory_resource.hpp:41`_

### Fixed Size Memory Resource

Construct a new `fixed_size_memory_resource` that allocates memory from

```cpp
explicit fixed_size_memory_resource( cuda::mr::any_resource<cuda::mr::device_accessible> upstream, std::size_t block_size = default_block_size, std::size_t blocks_to_preallocate = default_blocks_to_preallocate);
```

_Source: `cpp/include/rmm/mr/fixed_size_memory_resource.hpp:64`_

## `cpp/include/rmm/mr/managed_memory_resource.hpp`

### Managed Memory Resource

Memory resource that uses cudaMallocManaged/Free for allocation/deallocation.

```cpp
class managed_memory_resource final {
```

_Source: `cpp/include/rmm/mr/managed_memory_resource.hpp:26`_

### Allocate

Allocates memory of size at least `bytes`.

```cpp
void* allocate([[maybe_unused]] cuda::stream_ref stream, std::size_t bytes, [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) {
```

_Source: `cpp/include/rmm/mr/managed_memory_resource.hpp:49`_

### Deallocate

Deallocate memory pointed to by `ptr`.

```cpp
void deallocate([[maybe_unused]] cuda::stream_ref stream, void* ptr, [[maybe_unused]] std::size_t bytes, [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept {
```

_Source: `cpp/include/rmm/mr/managed_memory_resource.hpp:73`_

### Allocate Sync

Allocates memory of size at least `bytes` synchronously.

```cpp
void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) {
```

_Source: `cpp/include/rmm/mr/managed_memory_resource.hpp:88`_

### Deallocate Sync

Deallocate memory pointed to by `ptr` synchronously.

```cpp
void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept {
```

_Source: `cpp/include/rmm/mr/managed_memory_resource.hpp:102`_

### Get Property

Enables the `cuda::mr::device_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(managed_memory_resource const&, cuda::mr::device_accessible) noexcept {
```

_Source: `cpp/include/rmm/mr/managed_memory_resource.hpp:114`_

### Get Property

Enables the `cuda::mr::host_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(managed_memory_resource const&, cuda::mr::host_accessible) noexcept {
```

_Source: `cpp/include/rmm/mr/managed_memory_resource.hpp:124`_

## `cpp/include/rmm/mr/pinned_host_memory_resource.hpp`

### Pinned Host Memory Resource

Memory resource class for allocating pinned host memory.

```cpp
class pinned_host_memory_resource final {
```

_Source: `cpp/include/rmm/mr/pinned_host_memory_resource.hpp:34`_

### Allocate

Allocates pinned host memory of size at least `bytes` bytes.

```cpp
void* allocate([[maybe_unused]] cuda::stream_ref stream, std::size_t bytes, [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) {
```

_Source: `cpp/include/rmm/mr/pinned_host_memory_resource.hpp:63`_

### Deallocate

Deallocate memory pointed to by `ptr`.

```cpp
void deallocate([[maybe_unused]] cuda::stream_ref stream, void* ptr, std::size_t bytes, [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept {
```

_Source: `cpp/include/rmm/mr/pinned_host_memory_resource.hpp:89`_

### Allocate Sync

Allocates pinned host memory of size at least `bytes` bytes synchronously.

```cpp
void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) {
```

_Source: `cpp/include/rmm/mr/pinned_host_memory_resource.hpp:107`_

### Deallocate Sync

Deallocate memory pointed to by `ptr` synchronously.

```cpp
void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept {
```

_Source: `cpp/include/rmm/mr/pinned_host_memory_resource.hpp:121`_

### Get Property

Enables the `cuda::mr::device_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(pinned_host_memory_resource const&, cuda::mr::device_accessible) noexcept {
```

_Source: `cpp/include/rmm/mr/pinned_host_memory_resource.hpp:133`_

### Get Property

Enables the `cuda::mr::host_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(pinned_host_memory_resource const&, cuda::mr::host_accessible) noexcept {
```

_Source: `cpp/include/rmm/mr/pinned_host_memory_resource.hpp:143`_

## `cpp/include/rmm/mr/pool_memory_resource.hpp`

### Pool Memory Resource

A coalescing best-fit suballocator which uses a pool of memory allocated from

```cpp
class RMM_EXPORT pool_memory_resource : public cuda::mr::shared_resource<detail::pool_memory_resource_impl> {
```

_Source: `cpp/include/rmm/mr/pool_memory_resource.hpp:34`_

### Get Property

Enables the `cuda::mr::device_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(pool_memory_resource const&, cuda::mr::device_accessible) noexcept {
```

_Source: `cpp/include/rmm/mr/pool_memory_resource.hpp:44`_

### Pool Memory Resource

Construct a `pool_memory_resource` and allocate the initial device memory pool using

```cpp
explicit pool_memory_resource(cuda::mr::any_resource<cuda::mr::device_accessible> upstream, std::size_t initial_pool_size, std::optional<std::size_t> maximum_pool_size = std::nullopt);
```

_Source: `cpp/include/rmm/mr/pool_memory_resource.hpp:62`_

## `cpp/include/rmm/mr/sam_headroom_memory_resource.hpp`

### Sam Headroom Memory Resource

Resource that uses system memory resource to allocate memory with a headroom.

```cpp
class RMM_EXPORT sam_headroom_memory_resource final : public cuda::mr::shared_resource<detail::sam_headroom_memory_resource_impl> {
```

_Source: `cpp/include/rmm/mr/sam_headroom_memory_resource.hpp:35`_

### Get Property

Enables the `cuda::mr::device_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(sam_headroom_memory_resource const&, cuda::mr::device_accessible) noexcept {
```

_Source: `cpp/include/rmm/mr/sam_headroom_memory_resource.hpp:43`_

### Get Property

Enables the `cuda::mr::host_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(sam_headroom_memory_resource const&, cuda::mr::host_accessible) noexcept {
```

_Source: `cpp/include/rmm/mr/sam_headroom_memory_resource.hpp:51`_

### Sam Headroom Memory Resource

Construct a headroom memory resource.

```cpp
explicit sam_headroom_memory_resource(std::size_t headroom);
```

_Source: `cpp/include/rmm/mr/sam_headroom_memory_resource.hpp:61`_

## `cpp/include/rmm/mr/system_memory_resource.hpp`

### Is System Memory Supported

Check if system allocated memory (SAM) is supported on the specified device.

```cpp
static bool is_system_memory_supported(cuda_device_id device_id) {
```

_Source: `cpp/include/rmm/mr/system_memory_resource.hpp:31`_

### System Memory Resource

Memory resource that uses malloc/free for allocation/deallocation.

```cpp
class system_memory_resource final {
```

_Source: `cpp/include/rmm/mr/system_memory_resource.hpp:68`_

### Allocate

Allocates memory of size at least `bytes`.

```cpp
void* allocate([[maybe_unused]] cuda::stream_ref stream, std::size_t bytes, [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) {
```

_Source: `cpp/include/rmm/mr/system_memory_resource.hpp:95`_

### Deallocate

Deallocate memory pointed to by `ptr`.

```cpp
void deallocate(cuda::stream_ref stream, void* ptr, std::size_t bytes, [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept {
```

_Source: `cpp/include/rmm/mr/system_memory_resource.hpp:122`_

### Allocate Sync

Allocates memory of size at least `bytes` synchronously.

```cpp
void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) {
```

_Source: `cpp/include/rmm/mr/system_memory_resource.hpp:144`_

### Deallocate Sync

Deallocate memory pointed to by `ptr` synchronously.

```cpp
void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept {
```

_Source: `cpp/include/rmm/mr/system_memory_resource.hpp:158`_

### Get Property

Enables the `cuda::mr::device_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(system_memory_resource const&, cuda::mr::device_accessible) noexcept {
```

_Source: `cpp/include/rmm/mr/system_memory_resource.hpp:170`_

### Get Property

Enables the `cuda::mr::host_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(system_memory_resource const&, cuda::mr::host_accessible) noexcept {
```

_Source: `cpp/include/rmm/mr/system_memory_resource.hpp:180`_

## `cpp/include/rmm/mr/per_device_resource.hpp`

### Initial Resource

Returns a reference to the initial resource.

```cpp
RMM_EXPORT inline cuda_memory_resource& initial_resource() {
```

_Source: `cpp/include/rmm/mr/per_device_resource.hpp:57`_

### Ref Map Lock

```cpp
RMM_EXPORT inline std::mutex& ref_map_lock() {
```

_Source: `cpp/include/rmm/mr/per_device_resource.hpp:66`_

### Get Ref Map

```cpp
RMM_EXPORT inline auto& get_ref_map() {
```

_Source: `cpp/include/rmm/mr/per_device_resource.hpp:76`_

### Get Per Device Resource Ref

Get the `device_async_resource_ref` for the specified device.

```cpp
inline device_async_resource_ref get_per_device_resource_ref(cuda_device_id device_id) {
```

_Source: `cpp/include/rmm/mr/per_device_resource.hpp:110`_

### Set Per Device Resource

Set the memory resource for the specified device.

```cpp
inline cuda::mr::any_resource<cuda::mr::device_accessible> set_per_device_resource( cuda_device_id device_id, cuda::mr::any_resource<cuda::mr::device_accessible> new_resource) {
```

_Source: `cpp/include/rmm/mr/per_device_resource.hpp:151`_

### Get Current Device Resource Ref

Get the `device_async_resource_ref` for the current device.

```cpp
inline device_async_resource_ref get_current_device_resource_ref() {
```

_Source: `cpp/include/rmm/mr/per_device_resource.hpp:187`_

### Set Current Device Resource

Set the memory resource for the current device.

```cpp
inline cuda::mr::any_resource<cuda::mr::device_accessible> set_current_device_resource( cuda::mr::any_resource<cuda::mr::device_accessible> new_resource) {
```

_Source: `cpp/include/rmm/mr/per_device_resource.hpp:215`_

### Reset Per Device Resource

Reset the memory resource for the specified device to the initial resource.

```cpp
inline cuda::mr::any_resource<cuda::mr::device_accessible> reset_per_device_resource( cuda_device_id device_id) {
```

_Source: `cpp/include/rmm/mr/per_device_resource.hpp:238`_

### Reset Current Device Resource

Reset the memory resource for the current device to the initial resource.

```cpp
inline cuda::mr::any_resource<cuda::mr::device_accessible> reset_current_device_resource() {
```

_Source: `cpp/include/rmm/mr/per_device_resource.hpp:258`_

## `cpp/include/rmm/mr/polymorphic_allocator.hpp`

### Polymorphic Allocator

Construct a `polymorphic_allocator` using the return value of

```cpp
polymorphic_allocator() = default;
```

_Source: `cpp/include/rmm/mr/polymorphic_allocator.hpp:49`_

### Polymorphic Allocator

Construct a `polymorphic_allocator` using the provided memory resource.

```cpp
polymorphic_allocator(cuda::mr::any_resource<cuda::mr::device_accessible> mr) : mr_(std::move(mr)) {
```

_Source: `cpp/include/rmm/mr/polymorphic_allocator.hpp:58`_

### Polymorphic Allocator

Construct a `polymorphic_allocator` using the underlying memory resource of `other`.

```cpp
template <typename U> polymorphic_allocator(polymorphic_allocator<U> const& other) noexcept : mr_(other.get_upstream_resource()) {
```

_Source: `cpp/include/rmm/mr/polymorphic_allocator.hpp:68`_

### Allocate

Allocates storage for `num` objects of type `T` using the underlying memory resource.

```cpp
value_type* allocate(std::size_t num, cuda_stream_view stream) {
```

_Source: `cpp/include/rmm/mr/polymorphic_allocator.hpp:81`_

### Deallocate

Deallocates storage pointed to by `ptr`.

```cpp
void deallocate(value_type* ptr, std::size_t num, cuda_stream_view stream) noexcept {
```

_Source: `cpp/include/rmm/mr/polymorphic_allocator.hpp:97`_

### Stream Allocator Adaptor

Construct a `stream_allocator_adaptor` using `a` as the underlying allocator.

```cpp
stream_allocator_adaptor(Allocator const& allocator, cuda_stream_view stream) : alloc_{allocator}, stream_{stream} {
```

_Source: `cpp/include/rmm/mr/polymorphic_allocator.hpp:190`_

### Stream Allocator Adaptor

Construct a `stream_allocator_adaptor` using `other.underlying_allocator()` and

```cpp
template <typename OtherAllocator> stream_allocator_adaptor(stream_allocator_adaptor<OtherAllocator> const& other) : stream_allocator_adaptor{other.underlying_allocator(), other.stream()} {
```

_Source: `cpp/include/rmm/mr/polymorphic_allocator.hpp:203`_

### Allocate

Allocates storage for `num` objects of type `T` using the underlying allocator on

```cpp
value_type* allocate(std::size_t num) { return alloc_.allocate(num, stream()); } * @brief Deallocates storage pointed to by `ptr` using the underlying allocator on `stream()`. * * `ptr` must have been allocated from by an allocator `a` that compares equal to * `underlying_allocator()` using `a.allocate(n)`. * * @param ptr Pointer to memory to deallocate * @param num Number of objects originally allocated
```

_Source: `cpp/include/rmm/mr/polymorphic_allocator.hpp:229`_

### Deallocate

Deallocates storage pointed to by `ptr` using the underlying allocator on `stream()`.

```cpp
void deallocate(value_type* ptr, std::size_t num) noexcept {
```

_Source: `cpp/include/rmm/mr/polymorphic_allocator.hpp:238`_
