---
slug: api-reference/cpp-api-data-containers
---

# Data Containers

Generated from RMM C++ headers.

## `cpp/include/rmm/device_buffer.hpp`

### Device Buffer Class

RAII construct for device memory allocation

This class allocates untyped and *uninitialized* device memory using a `cuda::mr::any_resource<cuda::mr::device_accessible>`. If not explicitly specified, the memory resource returned from `get_current_device_resource_ref()` is used.

> **Note:** Unlike `std::vector` or `thrust::device_vector`, the device memory allocated by a `device_buffer` is uninitialized. Therefore, it is undefined behavior to read the contents of `data()` before first initializing it.

Examples:

```cpp
// Allocates at least 100 bytes of device memory using the default memory
// resource and default stream.
device_buffer buff(100);

// Allocates at least 100 bytes using the custom memory resource and
// specified stream
custom_memory_resource mr;
cuda_stream_view stream = cuda_stream_view{};
device_buffer custom_buff(100, stream, &mr);

// Deep copies `buff` into a new device buffer using the specified stream
device_buffer buff_copy(buff, stream);

// Moves the memory in `from_buff` to `to_buff`. Deallocates previously allocated
// to_buff memory on `to_buff.stream()`.
device_buffer to_buff(std::move(from_buff));

// Deep copies `buff` into a new device buffer using the specified stream
device_buffer buff_copy(buff, stream);

// Shallow copies `buff` into a new device_buffer, `buff` is now empty
device_buffer buff_move(std::move(buff));

// Default construction. Buffer is empty
device_buffer buff_default{};

// If the requested size is larger than the current size, resizes allocation to the new size and
// deep copies any previous contents. Otherwise, simply updates the value of `size()` to the
// newly requested size without any allocations or copies. Uses the specified stream.
buff_default.resize(100, stream);
```

```cpp
class device_buffer
```

_Source: `cpp/include/rmm/device_buffer.hpp:72`_

### Device Buffer Constructor (device_buffer.hpp:85)

Default constructor creates an empty `device_buffer`

```cpp
device_buffer();
```

_Source: `cpp/include/rmm/device_buffer.hpp:85`_

### Device Buffer Constructor (device_buffer.hpp:101)

Constructs a new device buffer of `size` uninitialized bytes

> **Note:** The buffer is guaranteed to have an alignment of at least `rmm::CUDA_ALLOCATION_ALIGNMENT`. Use the constructor with explicit alignment specification if you need something different.

**Throws:**

- `rmm::bad_alloc`: If allocation fails.

**Parameters:**

- `size`: Size in bytes to allocate in device memory.
- `stream`: CUDA stream on which memory may be allocated if the memory resource supports streams.
- `mr`: Memory resource to use for the device memory allocation.

```cpp
explicit device_buffer( std::size_t size, cuda_stream_view stream, cuda::mr::any_resource<cuda::mr::device_accessible> mr = mr::get_current_device_resource_ref());
```

_Source: `cpp/include/rmm/device_buffer.hpp:101`_

### Device Buffer Constructor (device_buffer.hpp:116)

Constructs a new device buffer of `size` uninitialized bytes

> **Note:** The buffer is guaranteed to have an alignment of at least `rmm::CUDA_ALLOCATION_ALIGNMENT`. Use the constructor with explicit alignment specification if you need something different.

**Throws:**

- `rmm::bad_alloc`: If allocation fails.

**Parameters:**

- `size`: Size in bytes to allocate in device memory.
- `stream`: CUDA stream on which memory may be allocated if the memory resource supports streams.
- `mr`: Memory resource to use for the device memory allocation.

**Throws:**

- `rmm::bad_alloc`: If the requested alignment cannot be satisfied by the provided memory resource.
- `rmm::invalid_argument`: If the requested alignment is not a power of two

**Parameters:**

- `alignment`: Required alignment of the allocation. The actual alignment will be at least the requested alignment.

```cpp
explicit device_buffer( std::size_t size, std::size_t alignment, cuda_stream_view stream, cuda::mr::any_resource<cuda::mr::device_accessible> mr = mr::get_current_device_resource_ref());
```

_Source: `cpp/include/rmm/device_buffer.hpp:116`_

### Device Buffer Constructor (device_buffer.hpp:145)

Construct a new device buffer by copying from a raw pointer to an existing host or device memory allocation.

> **Note:** This function does not synchronize `stream`. `source_data` is copied on `stream`, so the caller is responsible for correct synchronization to ensure that `source_data` is valid when the copy occurs. This includes destroying `source_data` in stream order after this function is called, or synchronizing or waiting on `stream` after this function returns as necessary.

> **Note:** The buffer is guaranteed to have an alignment of at least `rmm::CUDA_ALLOCATION_ALIGNMENT`. Use the constructor with explicit alignment specification if you need something different.

**Throws:**

- `rmm::bad_alloc`: If creating the new allocation fails.
- `rmm::logic_error`: If `source_data` is null, and `size != 0`.
- `rmm::cuda_error`: if copying from the device memory fails.

**Parameters:**

- `source_data`: Pointer to the host or device memory to copy from.
- `size`: Size in bytes to copy.
- `stream`: CUDA stream on which memory may be allocated if the memory resource supports streams.
- `mr`: Memory resource to use for the device memory allocation

```cpp
device_buffer( void const* source_data, std::size_t size, cuda_stream_view stream, cuda::mr::any_resource<cuda::mr::device_accessible> mr = mr::get_current_device_resource_ref());
```

_Source: `cpp/include/rmm/device_buffer.hpp:145`_

### Device Buffer Constructor (device_buffer.hpp:161)

Construct a new device buffer by copying from a raw pointer to an existing host or device memory allocation.

> **Note:** This function does not synchronize `stream`. `source_data` is copied on `stream`, so the caller is responsible for correct synchronization to ensure that `source_data` is valid when the copy occurs. This includes destroying `source_data` in stream order after this function is called, or synchronizing or waiting on `stream` after this function returns as necessary.

> **Note:** The buffer is guaranteed to have an alignment of at least `rmm::CUDA_ALLOCATION_ALIGNMENT`. Use the constructor with explicit alignment specification if you need something different.

**Throws:**

- `rmm::bad_alloc`: If creating the new allocation fails.
- `rmm::logic_error`: If `source_data` is null, and `size != 0`.
- `rmm::cuda_error`: if copying from the device memory fails.

**Parameters:**

- `source_data`: Pointer to the host or device memory to copy from.
- `size`: Size in bytes to copy.
- `stream`: CUDA stream on which memory may be allocated if the memory resource supports streams.
- `mr`: Memory resource to use for the device memory allocation

**Throws:**

- `rmm::bad_alloc`: If the requested alignment cannot be satisfied by the provided memory resource.
- `rmm::invalid_argument`: If the requested alignment is not a power of two

**Parameters:**

- `alignment`: Required alignment of the allocation. The actual alignment will be at least the requested alignment.

```cpp
explicit device_buffer( void const* source_data, std::size_t size, std::size_t alignment, cuda_stream_view stream, cuda::mr::any_resource<cuda::mr::device_accessible> mr = mr::get_current_device_resource_ref());
```

_Source: `cpp/include/rmm/device_buffer.hpp:161`_

### Device Buffer Constructor (device_buffer.hpp:193)

Construct a new `device_buffer` by deep copying the contents of another `device_buffer`, optionally using the specified stream and memory resource.

> **Note:** Only copies `other.size()` bytes from `other`, i.e., if `other.size() != other.capacity()`, then the size and capacity of the newly constructed `device_buffer` will be equal to `other.size()`.

> **Note:** The new buffer has the same alignment guarantees as the copied-from buffer. If you need to control the alignment of the new buffer explicitly, use `device_buffer(void const*, std::size_t, std::size_t, cuda_stream_view, cuda::mr::any_resource<cuda::mr::device_accessible>)`.

> **Note:** This function does not synchronize `stream`. `other` is copied on `stream`, so the caller is responsible for correct synchronization to ensure that `other` is valid when the copy occurs. This includes destroying `other` in stream order after this function is called, or synchronizing or waiting on `stream` after this function returns as necessary.

**Throws:**

- `rmm::bad_alloc`: If creating the new allocation fails.
- `rmm::cuda_error`: if copying from `other` fails.

**Parameters:**

- `other`: The `device_buffer` whose contents will be copied
- `stream`: The stream to use for the allocation and copy
- `mr`: The resource to use for allocating the new `device_buffer`

```cpp
device_buffer( device_buffer const& other, cuda_stream_view stream, cuda::mr::any_resource<cuda::mr::device_accessible> mr = mr::get_current_device_resource_ref());
```

_Source: `cpp/include/rmm/device_buffer.hpp:193`_

### Device Buffer Constructor (device_buffer.hpp:209)

Constructs a new `device_buffer` by moving the contents of another `device_buffer` into the newly constructed one.

After the new `device_buffer` is constructed, `other` is modified to be a valid, empty `device_buffer`, i.e., `data()` returns `nullptr`, and `size()` and `capacity()` are zero.

**Parameters:**

- `other`: The `device_buffer` whose contents will be moved into the newly constructed one.

```cpp
device_buffer(device_buffer&& other) noexcept;
```

_Source: `cpp/include/rmm/device_buffer.hpp:209`_

### Reserve (device_buffer.hpp:258)

Increase the capacity of the device memory allocation

If the requested `new_capacity` is less than or equal to `capacity()`, no action is taken.

If `new_capacity` is larger than `capacity()`, a new allocation is made on `stream` to satisfy `new_capacity`, and the contents of the old allocation are copied on `stream` to the new allocation. The old allocation is then freed. The bytes from `[size(), new_capacity)` are uninitialized.

> **Note:** This function does not synchronize `stream`. `new_capacity` is allocated on `stream`, so the caller is responsible for synchroning the current stream (accessed by `stream()`) before calling this function to ensure that the data is valid when the allocation occurs (if any).

**Throws:**

- `rmm::bad_alloc`: If creating the new allocation fails
- `rmm::cuda_error`: if the copy from the old to new allocation fails

**Parameters:**

- `new_capacity`: The requested new capacity, in bytes
- `stream`: The stream to use for allocation and copy

```cpp
void reserve(std::size_t new_capacity, cuda_stream_view stream);
```

_Source: `cpp/include/rmm/device_buffer.hpp:258`_

### Resize (device_buffer.hpp:289)

Resize the device memory allocation

If the requested `new_size` is less than or equal to `capacity()`, no action is taken other than updating the value that is returned from `size()`. Specifically, no memory is allocated nor copied. The value `capacity()` remains the actual size of the device memory allocation.

> **Note:** `shrink_to_fit()` may be used to force the deallocation of unused `capacity()`.

If `new_size` is larger than `capacity()`, a new allocation is made on `stream` to satisfy `new_size`, and the contents of the old allocation are copied on `stream` to the new allocation. The old allocation is then freed. The bytes from `[old_size, new_size)` are uninitialized.

The invariant `size() <= capacity()` holds.

> **Note:** This function does not synchronize `stream`. `new_size` is allocated on `stream`, so the caller is responsible for synchroning the current stream (accessed by `stream()`) before calling this function to ensure that the data is valid when the allocation occurs (if any).

**Throws:**

- `rmm::bad_alloc`: If creating the new allocation fails
- `rmm::cuda_error`: if the copy from the old to new allocation fails

**Parameters:**

- `new_size`: The requested new size, in bytes
- `stream`: The stream to use for allocation and copy

```cpp
void resize(std::size_t new_size, cuda_stream_view stream);
```

_Source: `cpp/include/rmm/device_buffer.hpp:289`_

### Shrink To Fit (device_buffer.hpp:308)

Forces the deallocation of unused memory.

Reallocates and copies on stream `stream` the contents of the device memory allocation to reduce `capacity()` to `size()`.

If `size() == capacity()`, no allocations or copies occur.

> **Note:** This function does not synchronize `stream`. The new buffer is allocated on `stream`, so the caller is responsible for synchroning the current stream (accessed by `stream()`) before calling this function to ensure that the data is valid when the allocation occurs (if any).

**Throws:**

- `rmm::bad_alloc`: If creating the new allocation fails
- `rmm::cuda_error`: If the copy from the old to new allocation fails

**Parameters:**

- `stream`: The stream on which the allocation and copy are performed

```cpp
void shrink_to_fit(cuda_stream_view stream);
```

_Source: `cpp/include/rmm/device_buffer.hpp:308`_

### Data

returnPointer to the device memory allocation

```cpp
void* data() noexcept
```

_Source: `cpp/include/rmm/device_buffer.hpp:318`_

### Set Stream (device_buffer.hpp:371)

Sets the stream to be used for deallocation

If no other rmm::device_buffer method that allocates memory is called after this call with a different stream argument, then `stream` will be used for deallocation in the `rmm::device_uvector` destructor. However, if either of `resize()` or `shrink_to_fit()` is called after this, the later stream parameter will be stored and used in the destructor.

**Parameters:**

- `stream`: The stream to use for deallocation

```cpp
void set_stream(cuda_stream_view stream) noexcept
```

_Source: `cpp/include/rmm/device_buffer.hpp:371`_

### Allocate Async

Allocates the specified amount of memory and updates the size/capacity accordingly.

Allocates on `stream()` using the memory resource passed to the constructor.

If `bytes == 0`, sets `_data = nullptr`.

**Parameters:**

- `bytes`: The amount of memory to allocate

```cpp
void allocate_async(std::size_t bytes);
```

_Source: `cpp/include/rmm/device_buffer.hpp:398`_

### Deallocate Async

Deallocate any memory held by this `device_buffer` and clear the size/capacity/data members.

If the buffer doesn't hold any memory, i.e., `capacity() == 0`, doesn't call the resource deallocation.

Deallocates on `stream()` using the memory resource passed to the constructor.

```cpp
void deallocate_async() noexcept;
```

_Source: `cpp/include/rmm/device_buffer.hpp:409`_

### Copy Async

Copies the specified number of `bytes` from `source` into the internal device allocation.

`source` can point to either host or device memory.

This function assumes `_data` already points to an allocation large enough to hold `bytes` bytes.

**Parameters:**

- `source`: The pointer to copy from
- `bytes`: The number of bytes to copy

```cpp
void copy_async(void const* source, std::size_t bytes);
```

_Source: `cpp/include/rmm/device_buffer.hpp:423`_

## `cpp/include/rmm/device_scalar.hpp`

### Device Scalar Constructor (device_scalar.hpp:60)

Copy ctor is deleted as it doesn't allow a stream argument

```cpp
device_scalar(device_scalar const&) = delete;
```

_Source: `cpp/include/rmm/device_scalar.hpp:60`_

### Device Scalar Constructor (device_scalar.hpp:70)

Default constructor is deleted as it doesn't allow a stream argument

```cpp
device_scalar() = delete;
```

_Source: `cpp/include/rmm/device_scalar.hpp:70`_

### Device Scalar Constructor (device_scalar.hpp:86)

Construct a new uninitialized `device_scalar`.

Does not synchronize the stream.

> **Note:** This device_scalar is only safe to access in kernels and copies on the specified CUDA stream, or on another stream only if a dependency is enforced (e.g. using `cudaStreamWaitEvent()`).

**Throws:**

- `rmm::bad_alloc`: if allocating the device memory fails.

**Parameters:**

- `stream`: Stream on which to perform asynchronous allocation.
- `mr`: Optional, resource with which to allocate.

```cpp
explicit device_scalar( cuda_stream_view stream, cuda::mr::any_resource<cuda::mr::device_accessible> mr = mr::get_current_device_resource_ref()) : _storage
```

_Source: `cpp/include/rmm/device_scalar.hpp:86`_

### Device Scalar Constructor (device_scalar.hpp:111)

Construct a new `device_scalar` with an initial value.

Does not synchronize the stream.

> **Note:** This device_scalar is only safe to access in kernels and copies on the specified CUDA stream, or on another stream only if a dependency is enforced (e.g. using `cudaStreamWaitEvent()`).

**Throws:**

- `rmm::bad_alloc`: if allocating the device memory for `initial_value` fails.
- `rmm::bad_alloc`: If the provided memory resource cannot allocate with alignment to satisfy the alignment requirements of the value type.
- `rmm::cuda_error`: if copying `initial_value` to device memory fails.

**Parameters:**

- `initial_value`: The initial value of the object in device memory.
- `stream`: Optional, stream on which to perform allocation and copy.
- `mr`: Optional, resource with which to allocate.

```cpp
explicit device_scalar( value_type const& initial_value, cuda_stream_view stream, cuda::mr::any_resource<cuda::mr::device_accessible> mr = mr::get_current_device_resource_ref()) : _storage
```

_Source: `cpp/include/rmm/device_scalar.hpp:111`_

### Device Scalar Constructor (device_scalar.hpp:132)

Construct a new `device_scalar` by deep copying the contents of another `device_scalar`, using the specified stream and memory resource.

**Throws:**

- `rmm::bad_alloc`: If creating the new allocation fails.
- `rmm::cuda_error`: if copying from `other` fails.

**Parameters:**

- `other`: The `device_scalar` whose contents will be copied
- `stream`: The stream to use for the allocation and copy
- `mr`: The resource to use for allocating the new `device_scalar`

```cpp
device_scalar( device_scalar const& other, cuda_stream_view stream, cuda::mr::any_resource<cuda::mr::device_accessible> mr = mr::get_current_device_resource_ref()) : _storage
```

_Source: `cpp/include/rmm/device_scalar.hpp:132`_

### Set Value Async

Sets the value of the `device_scalar` to the value of `v`.

> **Note:** If the stream specified to this function is different from the stream specified to the constructor, then appropriate dependencies must be inserted between the streams (e.g. using `cudaStreamWaitEvent()` or `cudaStreamSynchronize()`) before and after calling this function, otherwise there may be a race condition.

This function does not synchronize `stream` before returning. Therefore, the object referenced by `v` should not be destroyed or modified until `stream` has been synchronized. Otherwise, behavior is undefined.

> **Note:** This function incurs a host to device memcpy and should be used carefully.

Example:

```cpp
rmm::device_scalar<int32_t> s;

int v{42};

// Copies 42 to device storage on `stream`. Does _not_ synchronize
vec.set_value_async(v, stream);
...
cudaStreamSynchronize(stream);
// Synchronization is required before `v` can be modified
v = 13;
```

**Throws:**

- `rmm::cuda_error`: if copying `value` to device memory fails.

**Parameters:**

- `value`: The host value which will be copied to device
- `stream`: CUDA stream on which to perform the copy

```cpp
void set_value_async(value_type const& value, cuda_stream_view stream)
```

_Source: `cpp/include/rmm/device_scalar.hpp:194`_

### Set Value To Zero Async

Sets the value of the `device_scalar` to zero on the specified stream.

> **Note:** If the stream specified to this function is different from the stream specified to the constructor, then appropriate dependencies must be inserted between the streams (e.g. using `cudaStreamWaitEvent()` or `cudaStreamSynchronize()`) before and after calling this function, otherwise there may be a race condition.

This function does not synchronize `stream` before returning.

> **Note:** This function incurs a device memset and should be used carefully.

**Parameters:**

- `stream`: CUDA stream on which to perform the copy

```cpp
void set_value_to_zero_async(cuda_stream_view stream)
```

_Source: `cpp/include/rmm/device_scalar.hpp:217`_

### Set Stream (device_scalar.hpp:264)

Sets the stream to be used for deallocation

**Parameters:**

- `stream`: Stream to be used for deallocation

```cpp
void set_stream(cuda_stream_view stream) noexcept
```

_Source: `cpp/include/rmm/device_scalar.hpp:264`_

## `cpp/include/rmm/device_uvector.hpp`

### Device Uvector Constructor (device_uvector.hpp:101)

Copy ctor is deleted as it doesn't allow a stream argument

```cpp
device_uvector(device_uvector const&) = delete;
```

_Source: `cpp/include/rmm/device_uvector.hpp:101`_

### Device Uvector Constructor (device_uvector.hpp:111)

Default constructor is deleted as it doesn't allow a stream argument

```cpp
device_uvector() = delete;
```

_Source: `cpp/include/rmm/device_uvector.hpp:111`_

### Device Uvector Constructor (device_uvector.hpp:127)

Construct a new `device_uvector` with sufficient uninitialized storage for `size` elements.

Elements are uninitialized. Reading an element before it is initialized results in undefined behavior.

**Throws:**

- `rmm::bad_alloc`: If the provided memory resource cannot allocate with alignment to satisfy the alignment requirements of the value type.

**Parameters:**

- `size`: The number of elements to allocate storage for
- `stream`: The stream on which to perform the allocation
- `mr`: The resource used to allocate the device storage

```cpp
explicit device_uvector( size_type size, cuda_stream_view stream, cuda::mr::any_resource<cuda::mr::device_accessible> mr = mr::get_current_device_resource_ref()) : _storage
```

_Source: `cpp/include/rmm/device_uvector.hpp:127`_

### Device Uvector Constructor (device_uvector.hpp:144)

Construct a new device_uvector by deep copying the contents of another `device_uvector`.

Elements are copied as if by `memcpy`, i.e., `T`'s copy constructor is not invoked.

**Parameters:**

- `other`: The vector to copy from
- `stream`: The stream on which to perform the copy
- `mr`: The resource used to allocate device memory for the new vector

```cpp
explicit device_uvector( device_uvector const& other, cuda_stream_view stream, cuda::mr::any_resource<cuda::mr::device_accessible> mr = mr::get_current_device_resource_ref()) : _storage
```

_Source: `cpp/include/rmm/device_uvector.hpp:144`_

### Set Element Async

Performs an asynchronous copy of `v` to the specified element in device memory.

This function does not synchronize stream `s` before returning. Therefore, the object referenced by `v` should not be destroyed or modified until `stream` has been synchronized. Otherwise, behavior is undefined.

> **Note:** This function incurs a host to device memcpy and should be used sparingly.

> **Note:** Calling this function with a literal or other r-value reference for `v` is disallowed to prevent the implementation from asynchronously copying from a literal or other implicit temporary after it is deleted or goes out of scope.

Example:

```cpp
rmm::device_uvector<int32_t> vec(100, stream);

int v{42};

// Copies 42 to element 0 on `stream`. Does _not_ synchronize
vec.set_element_async(0, v, stream);
...
cudaStreamSynchronize(stream);
// Synchronization is required before `v` can be modified
v = 13;
```

**Throws:**

- `rmm::out_of_range`: exception if `element_index >= size()`

**Parameters:**

- `element_index`: Index of the target element
- `value`: The value to copy to the specified element
- `stream`: The stream on which to perform the copy

```cpp
void set_element_async(size_type element_index, value_type const& value, cuda_stream_view stream)
```

_Source: `cpp/include/rmm/device_uvector.hpp:213`_

### Set Element To Zero Async

Asynchronously sets the specified element to zero in device memory.

This function does not synchronize stream `s` before returning

> **Note:** This function incurs a device memset and should be used sparingly.

Example:

```cpp
rmm::device_uvector<int32_t> vec(100, stream);

int v{42};

// Sets element at index 42 to 0 on `stream`. Does _not_ synchronize
vec.set_element_to_zero_async(42, stream);
```

**Throws:**

- `rmm::out_of_range`: exception if `element_index >= size()`

**Parameters:**

- `element_index`: Index of the target element
- `stream`: The stream on which to perform the copy

```cpp
void set_element_to_zero_async(size_type element_index, cuda_stream_view stream)
```

_Source: `cpp/include/rmm/device_uvector.hpp:247`_

### Set Element

Performs a synchronous copy of `v` to the specified element in device memory.

Because this function synchronizes the stream `s`, it is safe to destroy or modify the object referenced by `v` after this function has returned.

> **Note:** This function incurs a host to device memcpy and should be used sparingly.

> **Note:** This function synchronizes `stream`.

Example:

```cpp
rmm::device_uvector<int32_t> vec(100, stream);

int v{42};

// Copies 42 to element 0 on `stream` and synchronizes the stream
vec.set_element(0, v, stream);

// It is safe to destroy or modify `v`
v = 13;
```

**Throws:**

- `rmm::out_of_range`: exception if `element_index >= size()`

**Parameters:**

- `element_index`: Index of the target element
- `value`: The value to copy to the specified element
- `stream`: The stream on which to perform the copy

```cpp
void set_element(size_type element_index, T const& value, cuda_stream_view stream)
```

_Source: `cpp/include/rmm/device_uvector.hpp:284`_

### Reserve (device_uvector.hpp:357)

Increases the capacity of the vector to `new_capacity` elements.

If `new_capacity <= capacity()`, no action is taken.

If `new_capacity > capacity()`, a new allocation of size `new_capacity` is created, and the first `size()` elements from the current allocation are copied there as if by memcpy. Finally, the old allocation is freed and replaced by the new allocation.

**Parameters:**

- `new_capacity`: The desired capacity (number of elements)
- `stream`: The stream on which to perform the allocation/copy (if any)

```cpp
void reserve(size_type new_capacity, cuda_stream_view stream)
```

_Source: `cpp/include/rmm/device_uvector.hpp:357`_

### Resize (device_uvector.hpp:378)

Resizes the vector to contain `new_size` elements.

If `new_size > size()`, the additional elements are uninitialized.

If `new_size < capacity()`, no action is taken other than updating the value of `size()`. No memory is allocated nor copied. `shrink_to_fit()` may be used to force deallocation of unused memory.

If `new_size > capacity()`, elements are copied as if by memcpy to a new allocation.

The invariant `size() <= capacity()` holds.

**Parameters:**

- `new_size`: The desired number of elements
- `stream`: The stream on which to perform the allocation/copy (if any)

```cpp
void resize(size_type new_size, cuda_stream_view stream)
```

_Source: `cpp/include/rmm/device_uvector.hpp:378`_

### Shrink To Fit (device_uvector.hpp:390)

Forces deallocation of unused device memory.

If `capacity() > size()`, reallocates and copies vector contents to eliminate unused memory.

**Parameters:**

- `stream`: Stream on which to perform allocation and copy

```cpp
void shrink_to_fit(cuda_stream_view stream)
```

_Source: `cpp/include/rmm/device_uvector.hpp:390`_

### Release

Release ownership of device memory storage.

**Returns:** The `device_buffer` used to store the vector elements

```cpp
device_buffer release() noexcept
```

_Source: `cpp/include/rmm/device_uvector.hpp:397`_

### Set Stream (device_uvector.hpp:615)

Sets the stream to be used for deallocation

If no other rmm::device_uvector method that allocates memory is called after this call with a different stream argument, then `stream` will be used for deallocation in the `rmm::device_uvector destructor. However, if either of `resize()` or `shrink_to_fit()` is called after this, the later stream parameter will be stored and used in the destructor.

**Parameters:**

- `stream`: The stream to use for deallocation

```cpp
void set_stream(cuda_stream_view stream) noexcept
```

_Source: `cpp/include/rmm/device_uvector.hpp:615`_

## `cpp/include/rmm/device_vector.hpp`

No documented declarations found.

## `cpp/include/rmm/resource_ref.hpp`

### Device Resource Ref Type Alias

Alias for a `cuda::mr::synchronous_resource_ref` with the property `cuda::mr::device_accessible`.

```cpp
using device_resource_ref = cuda::mr::synchronous_resource_ref<cuda::mr::device_accessible>;
```

_Source: `cpp/include/rmm/resource_ref.hpp:24`_

### Device Async Resource Ref Type Alias

Alias for a `cuda::mr::resource_ref` with the property `cuda::mr::device_accessible`.

```cpp
using device_async_resource_ref = cuda::mr::resource_ref<cuda::mr::device_accessible>;
```

_Source: `cpp/include/rmm/resource_ref.hpp:30`_

### Host Resource Ref Type Alias

Alias for a `cuda::mr::synchronous_resource_ref` with the property `cuda::mr::host_accessible`.

```cpp
using host_resource_ref = cuda::mr::synchronous_resource_ref<cuda::mr::host_accessible>;
```

_Source: `cpp/include/rmm/resource_ref.hpp:36`_

### Host Async Resource Ref Type Alias

Alias for a `cuda::mr::resource_ref` with the property `cuda::mr::host_accessible`.

```cpp
using host_async_resource_ref = cuda::mr::resource_ref<cuda::mr::host_accessible>;
```

_Source: `cpp/include/rmm/resource_ref.hpp:42`_

### Host Device Resource Ref Type Alias

Alias for a `cuda::mr::synchronous_resource_ref` with the properties `cuda::mr::host_accessible` and `cuda::mr::device_accessible`.

```cpp
using host_device_resource_ref = cuda::mr::synchronous_resource_ref<cuda::mr::host_accessible, cuda::mr::device_accessible>;
```

_Source: `cpp/include/rmm/resource_ref.hpp:48`_

### Host Device Async Resource Ref Type Alias

Alias for a `cuda::mr::resource_ref` with the properties `cuda::mr::host_accessible` and `cuda::mr::device_accessible`.

```cpp
using host_device_async_resource_ref = cuda::mr::resource_ref<cuda::mr::host_accessible, cuda::mr::device_accessible>;
```

_Source: `cpp/include/rmm/resource_ref.hpp:55`_

### To Device Async Resource Ref Checked

Convert pointer to memory resource into `device_async_resource_ref`, checking for `nullptr`

**Template Parameters:**

- `Resource`: The type of the memory resource.

**Parameters:**

- `res`: A pointer to the memory resource.

**Returns:** A `device_async_resource_ref` to the memory resource.

**Throws:**

- `std::logic_error`: if the memory resource pointer is null.

```cpp
template <class Resource> device_async_resource_ref to_device_async_resource_ref_checked(Resource* res)
```

_Source: `cpp/include/rmm/resource_ref.hpp:67`_
