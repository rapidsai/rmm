---
slug: api-reference/cpp-api-memory-resource-adaptors
---

# Memory Resource Adaptors

Generated from RMM C++ headers.

## `cpp/include/rmm/mr/aligned_resource_adaptor.hpp`

### Aligned Resource Adaptor Class

Resource that adapts an upstream resource to allocate memory with a specified alignment.

If the requested alignment is smaller than `CUDA_ALLOCATION_ALIGNMENT` (256 bytes) it is increased to `CUDA_ALLOCATION_ALIGNMENT`. An optional threshold controls the minimum size above which the custom alignment is applied.

This class is copyable and shares ownership of its internal state via `cuda::mr::shared_resource`.

```cpp
class RMM_EXPORT aligned_resource_adaptor : public cuda::mr::shared_resource<detail::aligned_resource_adaptor_impl>
```

_Source: `cpp/include/rmm/mr/aligned_resource_adaptor.hpp:33`_

### Get Property (aligned_resource_adaptor.hpp:41)

Enables the `cuda::mr::device_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(aligned_resource_adaptor const&, cuda::mr::device_accessible) noexcept
```

_Source: `cpp/include/rmm/mr/aligned_resource_adaptor.hpp:41`_

### Aligned Resource Adaptor Constructor

Construct an aligned resource adaptor using `upstream` to satisfy allocation requests.

**Throws:**

- `rmm::logic_error`: if `alignment` is not a power of 2

**Parameters:**

- `upstream`: The resource used for allocating/deallocating device memory.
- `alignment`: The size used for allocation alignment (raised to CUDA_ALLOCATION_ALIGNMENT if smaller).
- `alignment_threshold`: Only allocations >= this size are aligned to `alignment`.

```cpp
explicit aligned_resource_adaptor(cuda::mr::any_resource<cuda::mr::device_accessible> upstream, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT, std::size_t alignment_threshold = default_alignment_threshold);
```

_Source: `cpp/include/rmm/mr/aligned_resource_adaptor.hpp:63`_

## `cpp/include/rmm/mr/failure_callback_resource_adaptor.hpp`

### Get Property (failure_callback_resource_adaptor.hpp:55)

Enables the `cuda::mr::device_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(failure_callback_resource_adaptor const&, cuda::mr::device_accessible) noexcept
```

_Source: `cpp/include/rmm/mr/failure_callback_resource_adaptor.hpp:55`_

### Failure Callback Resource Adaptor Constructor

Construct a new `failure_callback_resource_adaptor` using `upstream` to satisfy allocation requests.

**Parameters:**

- `upstream`: The resource used for allocating/deallocating device memory
- `callback`: Callback function failure_callback_t
- `callback_arg`: Extra argument passed to `callback`

```cpp
failure_callback_resource_adaptor(cuda::mr::any_resource<cuda::mr::device_accessible> upstream, failure_callback_t callback, void* callback_arg) : shared_base(cuda::mr::make_shared_resource< detail::failure_callback_resource_adaptor_impl<ExceptionType>>( std::move(upstream), std::move(callback), callback_arg))
```

_Source: `cpp/include/rmm/mr/failure_callback_resource_adaptor.hpp:68`_

## `cpp/include/rmm/mr/is_resource_adaptor.hpp`

No documented declarations found.

## `cpp/include/rmm/mr/limiting_resource_adaptor.hpp`

### Limiting Resource Adaptor Class

Resource that uses an upstream resource to allocate memory and limits the total allocations possible.

An instance of this resource can be constructed with an existing, upstream resource in order to satisfy allocation requests, but any existing allocations will be untracked. Atomics are used to make this thread-safe, but note that the `get_allocated_bytes` may not include in-flight allocations.

This class is copyable and shares ownership of its internal state via `cuda::mr::shared_resource`.

```cpp
class RMM_EXPORT limiting_resource_adaptor : public cuda::mr::shared_resource<detail::limiting_resource_adaptor_impl>
```

_Source: `cpp/include/rmm/mr/limiting_resource_adaptor.hpp:35`_

### Get Property (limiting_resource_adaptor.hpp:43)

Enables the `cuda::mr::device_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(limiting_resource_adaptor const&, cuda::mr::device_accessible) noexcept
```

_Source: `cpp/include/rmm/mr/limiting_resource_adaptor.hpp:43`_

### Limiting Resource Adaptor Constructor

Construct a new limiting resource adaptor using `upstream` to satisfy allocation requests and limiting the total allocation amount possible.

**Parameters:**

- `upstream`: The resource used for allocating/deallocating device memory
- `allocation_limit`: Maximum memory allowed for this allocator
- `alignment`: Alignment in bytes for the start of each allocated buffer

```cpp
limiting_resource_adaptor(cuda::mr::any_resource<cuda::mr::device_accessible> upstream, std::size_t allocation_limit, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT);
```

_Source: `cpp/include/rmm/mr/limiting_resource_adaptor.hpp:56`_

## `cpp/include/rmm/mr/logging_resource_adaptor.hpp`

### Logging Resource Adaptor Class

Resource that uses an upstream resource to allocate memory and logs information about the requested allocation/deallocations.

An instance of this resource can be constructed with an existing, upstream resource in order to satisfy allocation requests and log allocation/deallocation activity.

This class is copyable and shares ownership of its internal state, allowing multiple instances to safely reference the same underlying resource and logger.

```cpp
class RMM_EXPORT logging_resource_adaptor : public cuda::mr::shared_resource<detail::logging_resource_adaptor_impl>
```

_Source: `cpp/include/rmm/mr/logging_resource_adaptor.hpp:37`_

### Get Property (logging_resource_adaptor.hpp:47)

Enables the `cuda::mr::device_accessible` property

This property declares that a `logging_resource_adaptor` provides device accessible memory

```cpp
RMM_CONSTEXPR_FRIEND void get_property(logging_resource_adaptor const&, cuda::mr::device_accessible) noexcept
```

_Source: `cpp/include/rmm/mr/logging_resource_adaptor.hpp:47`_

### Logging Resource Adaptor Constructor (logging_resource_adaptor.hpp:72)

Construct a new logging resource adaptor using `upstream` to satisfy allocation requests and logging information about each allocation/free to the file specified by `filename`.

The logfile will be written using CSV formatting.

Clears the contents of `filename` if it already exists.

Creating multiple `logging_resource_adaptor`s with the same `filename` will result in undefined behavior.

**Throws:**

- `spdlog::spdlog_ex`: if opening `filename` failed

**Parameters:**

- `upstream`: The resource_ref used for allocating/deallocating device memory.
- `filename`: Name of file to write log info. If not specified, retrieves the file name from the environment variable "RMM_LOG_FILE".
- `auto_flush`: If true, flushes the log for every (de)allocation. Warning, this will degrade performance.

```cpp
logging_resource_adaptor(cuda::mr::any_resource<cuda::mr::device_accessible> upstream, std::string const& filename = get_default_filename(), bool auto_flush = false);
```

_Source: `cpp/include/rmm/mr/logging_resource_adaptor.hpp:72`_

### Logging Resource Adaptor Constructor (logging_resource_adaptor.hpp:88)

Construct a new logging resource adaptor using `upstream` to satisfy allocation requests and logging information about each allocation/free to the ostream specified by `stream`.

The logfile will be written using CSV formatting.

**Parameters:**

- `upstream`: The resource_ref used for allocating/deallocating device memory.
- `stream`: The ostream to write log info.
- `auto_flush`: If true, flushes the log for every (de)allocation. Warning, this will degrade performance.

```cpp
logging_resource_adaptor(cuda::mr::any_resource<cuda::mr::device_accessible> upstream, std::ostream& stream, bool auto_flush = false);
```

_Source: `cpp/include/rmm/mr/logging_resource_adaptor.hpp:88`_

### Logging Resource Adaptor Constructor (logging_resource_adaptor.hpp:104)

Construct a new logging resource adaptor using `upstream` to satisfy allocation requests and logging information about each allocation/free to the sinks specified.

The logfile will be written using CSV formatting.

**Parameters:**

- `upstream`: The resource_ref used for allocating/deallocating device memory.
- `sinks`: A list of logging sinks to which log output will be written.
- `auto_flush`: If true, flushes the log for every (de)allocation. Warning, this will degrade performance.

```cpp
logging_resource_adaptor(cuda::mr::any_resource<cuda::mr::device_accessible> upstream, std::initializer_list<rapids_logger::sink_ptr> sinks, bool auto_flush = false);
```

_Source: `cpp/include/rmm/mr/logging_resource_adaptor.hpp:104`_

### Flush

Flush logger contents.

```cpp
void flush();
```

_Source: `cpp/include/rmm/mr/logging_resource_adaptor.hpp:116`_

### Get Default Filename

Return the value of the environment variable RMM_LOG_FILE.

**Throws:**

- `rmm::logic_error`: if `RMM_LOG_FILE` is not set.

**Returns:** The value of RMM_LOG_FILE as `std::string`.

```cpp
static std::string get_default_filename();
```

_Source: `cpp/include/rmm/mr/logging_resource_adaptor.hpp:132`_

## `cpp/include/rmm/mr/prefetch_resource_adaptor.hpp`

### Prefetch Resource Adaptor Class

Resource that prefetches all memory allocations.

This class is copyable and shares ownership of its internal state via `cuda::mr::shared_resource`.

```cpp
class RMM_EXPORT prefetch_resource_adaptor : public cuda::mr::shared_resource<detail::prefetch_resource_adaptor_impl>
```

_Source: `cpp/include/rmm/mr/prefetch_resource_adaptor.hpp:28`_

### Get Property (prefetch_resource_adaptor.hpp:36)

Enables the `cuda::mr::device_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(prefetch_resource_adaptor const&, cuda::mr::device_accessible) noexcept
```

_Source: `cpp/include/rmm/mr/prefetch_resource_adaptor.hpp:36`_

### Prefetch Resource Adaptor Constructor

Construct a new prefetch resource adaptor using `upstream` to satisfy allocation requests.

**Parameters:**

- `upstream`: The resource_ref used for allocating/deallocating device memory

```cpp
explicit prefetch_resource_adaptor(cuda::mr::any_resource<cuda::mr::device_accessible> upstream);
```

_Source: `cpp/include/rmm/mr/prefetch_resource_adaptor.hpp:47`_

## `cpp/include/rmm/mr/statistics_resource_adaptor.hpp`

### Statistics Resource Adaptor Class

Resource that uses an upstream resource to allocate memory and tracks allocation statistics (current, peak, total bytes and allocation counts).

Supports nested statistics via `push_counters()`/`pop_counters()`. Intended as a debug adaptor.

This class is copyable and shares ownership of its internal state via `cuda::mr::shared_resource`.

```cpp
class RMM_EXPORT statistics_resource_adaptor : public cuda::mr::shared_resource<detail::statistics_resource_adaptor_impl>
```

_Source: `cpp/include/rmm/mr/statistics_resource_adaptor.hpp:32`_

### Counter Type Alias

Counter type tracking current, peak, and total bytes or allocations.

```cpp
using counter = detail::statistics_resource_adaptor_impl::counter;
```

_Source: `cpp/include/rmm/mr/statistics_resource_adaptor.hpp:38`_

### Read Lock T Type Alias (statistics_resource_adaptor.hpp:40)

Shared-reader lock type used to protect the counter stack.

```cpp
using read_lock_t = detail::statistics_resource_adaptor_impl::read_lock_t;
```

_Source: `cpp/include/rmm/mr/statistics_resource_adaptor.hpp:40`_

### Write Lock T Type Alias (statistics_resource_adaptor.hpp:42)

Exclusive-writer lock type used to protect the counter stack.

```cpp
using write_lock_t = detail::statistics_resource_adaptor_impl::write_lock_t;
```

_Source: `cpp/include/rmm/mr/statistics_resource_adaptor.hpp:42`_

### Get Property (statistics_resource_adaptor.hpp:47)

Enables the `cuda::mr::device_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(statistics_resource_adaptor const&, cuda::mr::device_accessible) noexcept
```

_Source: `cpp/include/rmm/mr/statistics_resource_adaptor.hpp:47`_

### Statistics Resource Adaptor Constructor

Construct a statistics resource adaptor using `upstream` to satisfy allocation requests.

**Parameters:**

- `upstream`: The resource used for allocating/deallocating device memory.

```cpp
explicit statistics_resource_adaptor( cuda::mr::any_resource<cuda::mr::device_accessible> upstream);
```

_Source: `cpp/include/rmm/mr/statistics_resource_adaptor.hpp:57`_

## `cpp/include/rmm/mr/thread_safe_resource_adaptor.hpp`

### Thread Safe Resource Adaptor Class

Resource that adapts an upstream resource to be thread safe.

An instance of this resource can be constructed with an existing, upstream resource in order to satisfy allocation requests. This adaptor wraps allocations and deallocations from the upstream in a mutex lock.

This class is copyable and shares ownership of its internal state via `cuda::mr::shared_resource`.

```cpp
class RMM_EXPORT thread_safe_resource_adaptor : public cuda::mr::shared_resource<detail::thread_safe_resource_adaptor_impl>
```

_Source: `cpp/include/rmm/mr/thread_safe_resource_adaptor.hpp:33`_

### Get Property (thread_safe_resource_adaptor.hpp:43)

Enables the `cuda::mr::device_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(thread_safe_resource_adaptor const&, cuda::mr::device_accessible) noexcept
```

_Source: `cpp/include/rmm/mr/thread_safe_resource_adaptor.hpp:43`_

### Thread Safe Resource Adaptor Constructor

Construct a new thread safe resource adaptor using `upstream` to satisfy allocation requests.

**Parameters:**

- `upstream`: The resource used for allocating/deallocating device memory.

```cpp
explicit thread_safe_resource_adaptor( cuda::mr::any_resource<cuda::mr::device_accessible> upstream);
```

_Source: `cpp/include/rmm/mr/thread_safe_resource_adaptor.hpp:54`_

## `cpp/include/rmm/mr/thrust_allocator_adaptor.hpp`

### Thrust Allocator (thrust_allocator_adaptor.hpp:63)

Default constructor creates an allocator using the default memory resource and default stream.

```cpp
RMM_EXEC_CHECK_DISABLE thrust_allocator()
```

_Source: `cpp/include/rmm/mr/thrust_allocator_adaptor.hpp:63`_

### Thrust Allocator (thrust_allocator_adaptor.hpp:72)

Constructs a `thrust_allocator` using the default device memory resource and specified stream.

**Parameters:**

- `stream`: The stream to be used for device memory (de)allocation

```cpp
RMM_EXEC_CHECK_DISABLE explicit thrust_allocator(cuda_stream_view stream) : _stream
```

_Source: `cpp/include/rmm/mr/thrust_allocator_adaptor.hpp:72`_

### Thrust Allocator (thrust_allocator_adaptor.hpp:82)

Constructs a `thrust_allocator` using a device memory resource and stream.

**Parameters:**

- `mr`: The resource to be used for device memory allocation
- `stream`: The stream to be used for device memory (de)allocation

```cpp
RMM_EXEC_CHECK_DISABLE thrust_allocator(cuda_stream_view stream, cuda::mr::any_resource<cuda::mr::device_accessible> mr) : _stream
```

_Source: `cpp/include/rmm/mr/thrust_allocator_adaptor.hpp:82`_

### Thrust Allocator (thrust_allocator_adaptor.hpp:93)

Copy constructor. Copies the resource pointer and stream.

**Parameters:**

- `other`: The `thrust_allocator` to copy

```cpp
RMM_EXEC_CHECK_DISABLE thrust_allocator(thrust_allocator const& other) : Base(other), _stream
```

_Source: `cpp/include/rmm/mr/thrust_allocator_adaptor.hpp:93`_

### Thrust Allocator (thrust_allocator_adaptor.hpp:104)

Move constructor. Moves the resource pointer and stream.

**Parameters:**

- `other`: The `thrust_allocator` to move from

```cpp
RMM_EXEC_CHECK_DISABLE thrust_allocator(thrust_allocator&& other) noexcept : Base(std::move(other)), _stream
```

_Source: `cpp/include/rmm/mr/thrust_allocator_adaptor.hpp:104`_

### Thrust Allocator (thrust_allocator_adaptor.hpp:124)

Copy constructor from a `thrust_allocator` of a different type. Copies the resource pointer and stream.

**Parameters:**

- `other`: The `thrust_allocator` to copy

```cpp
RMM_EXEC_CHECK_DISABLE template <typename U> thrust_allocator(thrust_allocator<U> const& other) : _mr(other.resource()), _stream
```

_Source: `cpp/include/rmm/mr/thrust_allocator_adaptor.hpp:124`_

### Allocate

Allocate objects of type `T`

**Parameters:**

- `num`: The number of elements of type `T` to allocate

**Returns:** pointer Pointer to the newly allocated storage

```cpp
pointer allocate(size_type num)
```

_Source: `cpp/include/rmm/mr/thrust_allocator_adaptor.hpp:137`_

### Deallocate

Deallocates objects of type `T`

**Parameters:**

- `ptr`: Pointer returned by a previous call to `allocate`
- `num`: number of elements, *must* be equal to the argument passed to the prior `allocate` call that produced `ptr`

```cpp
void deallocate(pointer ptr, size_type num) noexcept
```

_Source: `cpp/include/rmm/mr/thrust_allocator_adaptor.hpp:151`_

### Get Property (thrust_allocator_adaptor.hpp:176)

Enables the `cuda::mr::device_accessible` property

This property declares that a `thrust_allocator` provides device accessible memory

```cpp
RMM_CONSTEXPR_FRIEND void get_property(thrust_allocator const&, cuda::mr::device_accessible) noexcept
```

_Source: `cpp/include/rmm/mr/thrust_allocator_adaptor.hpp:176`_

## `cpp/include/rmm/mr/tracking_resource_adaptor.hpp`

### Tracking Resource Adaptor Class

Resource that uses an upstream resource to allocate memory and tracks allocations.

Tracks every allocation (size, pointer, and optionally stack trace). Intended as a debug adaptor; should not be used in performance-sensitive code.

This class is copyable and shares ownership of its internal state via `cuda::mr::shared_resource`.

```cpp
class RMM_EXPORT tracking_resource_adaptor : public cuda::mr::shared_resource<detail::tracking_resource_adaptor_impl>
```

_Source: `cpp/include/rmm/mr/tracking_resource_adaptor.hpp:34`_

### Allocation Info Type Alias

Allocation info type (pointer, size, optional stack trace).

```cpp
using allocation_info = detail::tracking_resource_adaptor_impl::allocation_info;
```

_Source: `cpp/include/rmm/mr/tracking_resource_adaptor.hpp:40`_

### Read Lock T Type Alias (tracking_resource_adaptor.hpp:42)

Shared-reader lock type used to protect the allocations map.

```cpp
using read_lock_t = detail::tracking_resource_adaptor_impl::read_lock_t;
```

_Source: `cpp/include/rmm/mr/tracking_resource_adaptor.hpp:42`_

### Write Lock T Type Alias (tracking_resource_adaptor.hpp:44)

Exclusive-writer lock type used to protect the allocations map.

```cpp
using write_lock_t = detail::tracking_resource_adaptor_impl::write_lock_t;
```

_Source: `cpp/include/rmm/mr/tracking_resource_adaptor.hpp:44`_

### Get Property (tracking_resource_adaptor.hpp:49)

Enables the `cuda::mr::device_accessible` property

```cpp
RMM_CONSTEXPR_FRIEND void get_property(tracking_resource_adaptor const&, cuda::mr::device_accessible) noexcept
```

_Source: `cpp/include/rmm/mr/tracking_resource_adaptor.hpp:49`_

### Tracking Resource Adaptor Constructor

Construct a tracking resource adaptor using `upstream` to satisfy allocation requests.

**Parameters:**

- `upstream`: The resource used for allocating/deallocating device memory.
- `capture_stacks`: If true, capture stacks for each allocation.

```cpp
tracking_resource_adaptor(cuda::mr::any_resource<cuda::mr::device_accessible> upstream, bool capture_stacks = false);
```

_Source: `cpp/include/rmm/mr/tracking_resource_adaptor.hpp:60`_

### Log Outstanding Allocations

Log any outstanding allocations via RMM_LOG_DEBUG.

```cpp
void log_outstanding_allocations() const;
```

_Source: `cpp/include/rmm/mr/tracking_resource_adaptor.hpp:95`_

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

## `cpp/include/rmm/mr/failure_callback_t.hpp`

### Failure Callback T Type Alias

Callback function type used by failure_callback_resource_adaptor

The resource adaptor calls this function when a memory allocation throws a specified exception type. The function decides whether the resource adaptor should try to allocate the memory again or re-throw the exception.

The callback function signature is: `bool failure_callback_t(std::size_t bytes, void* callback_arg)`

The callback function is passed two parameters: `bytes` is the size of the failed memory allocation and `arg` is the extra argument passed to the constructor of the `failure_callback_resource_adaptor`. The callback function returns a bool where true means to retry the memory allocation and false means to re-throw the exception.

```cpp
using failure_callback_t = std::function<bool(std::size_t, void*)>;
```

_Source: `cpp/include/rmm/mr/failure_callback_t.hpp:34`_
