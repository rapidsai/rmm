---
slug: api-reference/cpp-api-data-containers
---

# Data Containers

Generated from RMM C++ headers.

## `cpp/include/rmm/device_buffer.hpp`

### Device Buffer

RAII construct for device memory allocation

```cpp
class device_buffer {
```

_Source: `cpp/include/rmm/device_buffer.hpp:72`_

### Device Buffer

Default constructor creates an empty `device_buffer`

```cpp
device_buffer();
```

_Source: `cpp/include/rmm/device_buffer.hpp:85`_

### Device Buffer

Constructs a new device buffer of `size` uninitialized bytes

```cpp
explicit device_buffer( std::size_t size, cuda_stream_view stream, cuda::mr::any_resource<cuda::mr::device_accessible> mr = mr::get_current_device_resource_ref());
```

_Source: `cpp/include/rmm/device_buffer.hpp:101`_

### Device Buffer

Construct a new device buffer by copying from a raw pointer to an existing host or

```cpp
device_buffer( void const* source_data, std::size_t size, cuda_stream_view stream, cuda::mr::any_resource<cuda::mr::device_accessible> mr = mr::get_current_device_resource_ref());
```

_Source: `cpp/include/rmm/device_buffer.hpp:145`_

### Device Buffer

Construct a new `device_buffer` by deep copying the contents of

```cpp
device_buffer( device_buffer const& other, cuda_stream_view stream, cuda::mr::any_resource<cuda::mr::device_accessible> mr = mr::get_current_device_resource_ref());
```

_Source: `cpp/include/rmm/device_buffer.hpp:193`_

### Device Buffer

Constructs a new `device_buffer` by moving the contents of another

```cpp
device_buffer(device_buffer&& other) noexcept;
```

_Source: `cpp/include/rmm/device_buffer.hpp:209`_

### Reserve

Increase the capacity of the device memory allocation

```cpp
void reserve(std::size_t new_capacity, cuda_stream_view stream);
```

_Source: `cpp/include/rmm/device_buffer.hpp:258`_

### Resize

Resize the device memory allocation

```cpp
void resize(std::size_t new_size, cuda_stream_view stream);
```

_Source: `cpp/include/rmm/device_buffer.hpp:289`_

### Shrink To Fit

Forces the deallocation of unused memory.

```cpp
void shrink_to_fit(cuda_stream_view stream);
```

_Source: `cpp/include/rmm/device_buffer.hpp:308`_

### Data

```cpp
void* data() noexcept { return _data; } * @briefreturn{The number of bytes} */ [[nodiscard]] std::size_t size() const noexcept { return _size; } * @briefreturn{The signed number of bytes} */ [[nodiscard]] std::int64_t ssize() const noexcept {
```

_Source: `cpp/include/rmm/device_buffer.hpp:322`_

### Set Stream

Sets the stream to be used for deallocation

```cpp
void set_stream(cuda_stream_view stream) noexcept { _stream = stream; } * @briefreturn{The resource used to allocate and deallocate} */ [[nodiscard]] rmm::device_async_resource_ref memory_resource() noexcept { return _mr; } private: void* _data{nullptr}; ///< Pointer to device memory allocation std::size_t _size{}; ///< Requested size of the device memory allocation std::size_t _alignment{rmm::CUDA_ALLOCATION_ALIGNMENT}; ///< The alignment of the allocation
```

_Source: `cpp/include/rmm/device_buffer.hpp:374`_

### Allocate Async

Allocates the specified amount of memory and updates the size/capacity accordingly.

```cpp
void allocate_async(std::size_t bytes);
```

_Source: `cpp/include/rmm/device_buffer.hpp:398`_

### Deallocate Async

Deallocate any memory held by this `device_buffer` and clear the

```cpp
void deallocate_async() noexcept;
```

_Source: `cpp/include/rmm/device_buffer.hpp:409`_

### Copy Async

Copies the specified number of `bytes` from `source` into the

```cpp
void copy_async(void const* source, std::size_t bytes);
```

_Source: `cpp/include/rmm/device_buffer.hpp:423`_

## `cpp/include/rmm/device_scalar.hpp`

### Device Scalar

Copy ctor is deleted as it doesn't allow a stream argument

```cpp
device_scalar(device_scalar const&) = delete;
```

_Source: `cpp/include/rmm/device_scalar.hpp:60`_

### Device Scalar

Default constructor is deleted as it doesn't allow a stream argument

```cpp
device_scalar() = delete;
```

_Source: `cpp/include/rmm/device_scalar.hpp:70`_

### Device Scalar

Construct a new uninitialized `device_scalar`.

```cpp
explicit device_scalar( cuda_stream_view stream, cuda::mr::any_resource<cuda::mr::device_accessible> mr = mr::get_current_device_resource_ref()) : _storage{1, stream, std::move(mr)} {
```

_Source: `cpp/include/rmm/device_scalar.hpp:86`_

### Device Scalar

Construct a new `device_scalar` with an initial value.

```cpp
explicit device_scalar( value_type const& initial_value, cuda_stream_view stream, cuda::mr::any_resource<cuda::mr::device_accessible> mr = mr::get_current_device_resource_ref()) : _storage{1, stream, std::move(mr)} {
```

_Source: `cpp/include/rmm/device_scalar.hpp:111`_

### Device Scalar

Construct a new `device_scalar` by deep copying the contents of

```cpp
device_scalar( device_scalar const& other, cuda_stream_view stream, cuda::mr::any_resource<cuda::mr::device_accessible> mr = mr::get_current_device_resource_ref()) : _storage{other._storage, stream, std::move(mr)} {
```

_Source: `cpp/include/rmm/device_scalar.hpp:132`_

### Set Value Async

Sets the value of the `device_scalar` to the value of `v`.

```cpp
void set_value_async(value_type const& value, cuda_stream_view stream) {
```

_Source: `cpp/include/rmm/device_scalar.hpp:194`_

### Set Value To Zero Async

Sets the value of the `device_scalar` to zero on the specified stream.

```cpp
void set_value_to_zero_async(cuda_stream_view stream) {
```

_Source: `cpp/include/rmm/device_scalar.hpp:217`_

### Set Stream

Sets the stream to be used for deallocation

```cpp
void set_stream(cuda_stream_view stream) noexcept { _storage.set_stream(stream); } private: rmm::device_uvector<T> _storage;
```

_Source: `cpp/include/rmm/device_scalar.hpp:265`_

## `cpp/include/rmm/device_uvector.hpp`

### Device Uvector

Copy ctor is deleted as it doesn't allow a stream argument

```cpp
device_uvector(device_uvector const&) = delete;
```

_Source: `cpp/include/rmm/device_uvector.hpp:101`_

### Device Uvector

Default constructor is deleted as it doesn't allow a stream argument

```cpp
device_uvector() = delete;
```

_Source: `cpp/include/rmm/device_uvector.hpp:111`_

### Device Uvector

Construct a new `device_uvector` with sufficient uninitialized storage for `size`

```cpp
explicit device_uvector( size_type size, cuda_stream_view stream, cuda::mr::any_resource<cuda::mr::device_accessible> mr = mr::get_current_device_resource_ref()) : _storage{elements_to_bytes(size), std::alignment_of_v<T>, stream, std::move(mr)} {
```

_Source: `cpp/include/rmm/device_uvector.hpp:127`_

### Device Uvector

Construct a new device_uvector by deep copying the contents of another `device_uvector`.

```cpp
explicit device_uvector( device_uvector const& other, cuda_stream_view stream, cuda::mr::any_resource<cuda::mr::device_accessible> mr = mr::get_current_device_resource_ref()) : _storage{other._storage, stream, std::move(mr)} {
```

_Source: `cpp/include/rmm/device_uvector.hpp:144`_

### Set Element Async

Performs an asynchronous copy of `v` to the specified element in device memory.

```cpp
void set_element_async(size_type element_index, value_type const& value, cuda_stream_view stream) {
```

_Source: `cpp/include/rmm/device_uvector.hpp:213`_

### Set Element To Zero Async

Asynchronously sets the specified element to zero in device memory.

```cpp
void set_element_to_zero_async(size_type element_index, cuda_stream_view stream) {
```

_Source: `cpp/include/rmm/device_uvector.hpp:247`_

### Set Element

Performs a synchronous copy of `v` to the specified element in device memory.

```cpp
void set_element(size_type element_index, T const& value, cuda_stream_view stream) {
```

_Source: `cpp/include/rmm/device_uvector.hpp:284`_

### Reserve

Increases the capacity of the vector to `new_capacity` elements.

```cpp
void reserve(size_type new_capacity, cuda_stream_view stream) {
```

_Source: `cpp/include/rmm/device_uvector.hpp:357`_

### Resize

Resizes the vector to contain `new_size` elements.

```cpp
void resize(size_type new_size, cuda_stream_view stream) {
```

_Source: `cpp/include/rmm/device_uvector.hpp:378`_

### Shrink To Fit

Forces deallocation of unused device memory.

```cpp
void shrink_to_fit(cuda_stream_view stream) { _storage.shrink_to_fit(stream); } * @brief Release ownership of device memory storage. * * @return The `device_buffer` used to store the vector elements */ device_buffer release() noexcept { return std::move(_storage); } * @brief Returns the number of elements that can be held in currently allocated storage. *
```

_Source: `cpp/include/rmm/device_uvector.hpp:394`_

### Release

Release ownership of device memory storage.

```cpp
device_buffer release() noexcept { return std::move(_storage); } * @brief Returns the number of elements that can be held in currently allocated storage. * * @return size_type The number of elements that can be stored without requiring a new * allocation. */ [[nodiscard]] size_type capacity() const noexcept {
```

_Source: `cpp/include/rmm/device_uvector.hpp:399`_

### Set Stream

Sets the stream to be used for deallocation

```cpp
void set_stream(cuda_stream_view stream) noexcept { _storage.set_stream(stream); } private: device_buffer _storage{}; ///< Device memory storage for vector elements [[nodiscard]] size_type constexpr elements_to_bytes(size_type num_elements) const noexcept {
```

_Source: `cpp/include/rmm/device_uvector.hpp:617`_

## `cpp/include/rmm/device_vector.hpp`

No documented declarations found.

## `cpp/include/rmm/resource_ref.hpp`

### Device Resource Ref

Alias for a `cuda::mr::synchronous_resource_ref` with the property

```cpp
using device_resource_ref = cuda::mr::synchronous_resource_ref<cuda::mr::device_accessible>;
```

_Source: `cpp/include/rmm/resource_ref.hpp:24`_

### Device Async Resource Ref

Alias for a `cuda::mr::resource_ref` with the property

```cpp
using device_async_resource_ref = cuda::mr::resource_ref<cuda::mr::device_accessible>;
```

_Source: `cpp/include/rmm/resource_ref.hpp:30`_

### Host Resource Ref

Alias for a `cuda::mr::synchronous_resource_ref` with the property

```cpp
using host_resource_ref = cuda::mr::synchronous_resource_ref<cuda::mr::host_accessible>;
```

_Source: `cpp/include/rmm/resource_ref.hpp:36`_

### Host Async Resource Ref

Alias for a `cuda::mr::resource_ref` with the property

```cpp
using host_async_resource_ref = cuda::mr::resource_ref<cuda::mr::host_accessible>;
```

_Source: `cpp/include/rmm/resource_ref.hpp:42`_

### Host Device Resource Ref

Alias for a `cuda::mr::synchronous_resource_ref` with the properties

```cpp
using host_device_resource_ref = cuda::mr::synchronous_resource_ref<cuda::mr::host_accessible, cuda::mr::device_accessible>;
```

_Source: `cpp/include/rmm/resource_ref.hpp:48`_

### Host Device Async Resource Ref

Alias for a `cuda::mr::resource_ref` with the properties

```cpp
using host_device_async_resource_ref = cuda::mr::resource_ref<cuda::mr::host_accessible, cuda::mr::device_accessible>;
```

_Source: `cpp/include/rmm/resource_ref.hpp:55`_

### To Device Async Resource Ref Checked

Convert pointer to memory resource into `device_async_resource_ref`, checking for

```cpp
template <class Resource> device_async_resource_ref to_device_async_resource_ref_checked(Resource* res) {
```

_Source: `cpp/include/rmm/resource_ref.hpp:67`_
