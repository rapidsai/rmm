/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>
#include <cuda_runtime_api.h>

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace rmm {
/**
 * @addtogroup data_containers
 * @{
 * @file
 */
/**
 * @brief RAII construct for device memory allocation
 *
 * This class allocates untyped and *uninitialized* device memory using a
 * `device_async_resource_ref`. If not explicitly specified, the memory resource
 * returned from `get_current_device_resource()` is used.
 *
 * @note Unlike `std::vector` or `thrust::device_vector`, the device memory
 * allocated by a `device_buffer` is uninitialized. Therefore, it is undefined
 * behavior to read the contents of `data()` before first initializing it.
 *
 * Examples:
 * ```
 * //Allocates at least 100 bytes of device memory using the default memory
 * //resource and default stream.
 * device_buffer buff(100);
 *
 * // allocates at least 100 bytes using the custom memory resource and
 * // specified stream
 * custom_memory_resource mr;
 * cuda_stream_view stream = cuda_stream_view{};
 * device_buffer custom_buff(100, stream, &mr);
 *
 * // deep copies `buff` into a new device buffer using the specified stream
 * device_buffer buff_copy(buff, stream);
 *
 * // moves the memory in `from_buff` to `to_buff`. Deallocates previously allocated
 * // to_buff memory on `to_buff.stream()`.
 * device_buffer to_buff(std::move(from_buff));
 *
 * // deep copies `buff` into a new device buffer using the specified stream
 * device_buffer buff_copy(buff, stream);
 *
 * // shallow copies `buff` into a new device_buffer, `buff` is now empty
 * device_buffer buff_move(std::move(buff));
 *
 * // Default construction. Buffer is empty
 * device_buffer buff_default{};
 *
 * // If the requested size is larger than the current size, resizes allocation to the new size and
 * // deep copies any previous contents. Otherwise, simply updates the value of `size()` to the
 * // newly requested size without any allocations or copies. Uses the specified stream.
 * buff_default.resize(100, stream);
 *```
 */
class device_buffer {
 public:
  // The copy constructor and copy assignment operator without a stream are deleted because they
  // provide no way to specify an explicit stream
  device_buffer(device_buffer const& other)            = delete;
  device_buffer& operator=(device_buffer const& other) = delete;

  /**
   * @brief Default constructor creates an empty `device_buffer`
   */
  // Note: we cannot use `device_buffer() = default;` because nvcc implicitly adds
  // `__host__ __device__` specifiers to the defaulted constructor when it is called within the
  // context of both host and device functions. Specifically, the `cudf::type_dispatcher` is a host-
  // device function. This causes warnings/errors because this ctor invokes host-only functions.
  device_buffer() : _mr{rmm::mr::get_current_device_resource()} {}

  /**
   * @brief Constructs a new device buffer of `size` uninitialized bytes
   *
   * @throws rmm::bad_alloc If allocation fails.
   *
   * @param size Size in bytes to allocate in device memory.
   * @param stream CUDA stream on which memory may be allocated if the memory
   * resource supports streams.
   * @param mr Memory resource to use for the device memory allocation.
   */
  explicit device_buffer(std::size_t size,
                         cuda_stream_view stream,
                         device_async_resource_ref mr = mr::get_current_device_resource())
    : _stream{stream}, _mr{mr}
  {
    cuda_set_device_raii dev{_device};
    allocate_async(size);
  }

  /**
   * @brief Construct a new device buffer by copying from a raw pointer to an existing host or
   * device memory allocation.
   *
   * @note This function does not synchronize `stream`. `source_data` is copied on `stream`, so the
   * caller is responsible for correct synchronization to ensure that `source_data` is valid when
   * the copy occurs. This includes destroying `source_data` in stream order after this function is
   * called, or synchronizing or waiting on `stream` after this function returns as necessary.
   *
   * @throws rmm::bad_alloc If creating the new allocation fails.
   * @throws rmm::logic_error If `source_data` is null, and `size != 0`.
   * @throws rmm::cuda_error if copying from the device memory fails.
   *
   * @param source_data Pointer to the host or device memory to copy from.
   * @param size Size in bytes to copy.
   * @param stream CUDA stream on which memory may be allocated if the memory
   * resource supports streams.
   * @param mr Memory resource to use for the device memory allocation
   */
  device_buffer(void const* source_data,
                std::size_t size,
                cuda_stream_view stream,
                device_async_resource_ref mr = mr::get_current_device_resource())
    : _stream{stream}, _mr{mr}
  {
    cuda_set_device_raii dev{_device};
    allocate_async(size);
    copy_async(source_data, size);
  }

  /**
   * @brief Construct a new `device_buffer` by deep copying the contents of
   * another `device_buffer`, optionally using the specified stream and memory
   * resource.
   *
   * @note Only copies `other.size()` bytes from `other`, i.e., if
   *`other.size() != other.capacity()`, then the size and capacity of the newly
   * constructed `device_buffer` will be equal to `other.size()`.
   *
   * @note This function does not synchronize `stream`. `other` is copied on `stream`, so the
   * caller is responsible for correct synchronization to ensure that `other` is valid when
   * the copy occurs. This includes destroying `other` in stream order after this function is
   * called, or synchronizing or waiting on `stream` after this function returns as necessary.
   *
   * @throws rmm::bad_alloc If creating the new allocation fails.
   * @throws rmm::cuda_error if copying from `other` fails.
   *
   * @param other The `device_buffer` whose contents will be copied
   * @param stream The stream to use for the allocation and copy
   * @param mr The resource to use for allocating the new `device_buffer`
   */
  device_buffer(device_buffer const& other,
                cuda_stream_view stream,
                device_async_resource_ref mr = mr::get_current_device_resource())
    : device_buffer{other.data(), other.size(), stream, mr}
  {
  }

  /**
   * @brief Constructs a new `device_buffer` by moving the contents of another
   * `device_buffer` into the newly constructed one.
   *
   * After the new `device_buffer` is constructed, `other` is modified to be a
   * valid, empty `device_buffer`, i.e., `data()` returns `nullptr`, and
   * `size()` and `capacity()` are zero.
   *
   * @param other The `device_buffer` whose contents will be moved into the
   * newly constructed one.
   */
  device_buffer(device_buffer&& other) noexcept
    : _data{other._data},
      _size{other._size},
      _capacity{other._capacity},
      _stream{other.stream()},
      _mr{other._mr},
      _device{other._device}
  {
    other._data     = nullptr;
    other._size     = 0;
    other._capacity = 0;
    other.set_stream(cuda_stream_view{});
    other._device = cuda_device_id{-1};
  }

  /**
   * @brief Move assignment operator moves the contents from `other`.
   *
   * This `device_buffer`'s current device memory allocation will be deallocated
   * on `stream()`.
   *
   * If a different stream is required, call `set_stream()` on
   * the instance before assignment. After assignment, this instance's stream is
   * replaced by the `other.stream()`.
   *
   * @param other The `device_buffer` whose contents will be moved.
   *
   * @return A reference to this `device_buffer`
   */
  device_buffer& operator=(device_buffer&& other) noexcept
  {
    if (&other != this) {
      cuda_set_device_raii dev{_device};
      deallocate_async();

      _data     = other._data;
      _size     = other._size;
      _capacity = other._capacity;
      set_stream(other.stream());
      _mr     = other._mr;
      _device = other._device;

      other._data     = nullptr;
      other._size     = 0;
      other._capacity = 0;
      other.set_stream(cuda_stream_view{});
      other._device = cuda_device_id{-1};
    }
    return *this;
  }

  /**
   * @brief Destroy the device buffer object
   *
   * @note If the memory resource supports streams, this destructor deallocates
   * using the stream most recently passed to any of this device buffer's
   * methods.
   */
  ~device_buffer() noexcept
  {
    cuda_set_device_raii dev{_device};
    deallocate_async();
    _stream = cuda_stream_view{};
  }

  /**
   * @brief Increase the capacity of the device memory allocation
   *
   * If the requested `new_capacity` is less than or equal to `capacity()`, no
   * action is taken.
   *
   * If `new_capacity` is larger than `capacity()`, a new allocation is made on
   * `stream` to satisfy `new_capacity`, and the contents of the old allocation are
   * copied on `stream` to the new allocation. The old allocation is then freed.
   * The bytes from `[size(), new_capacity)` are uninitialized.
   *
   * @throws rmm::bad_alloc If creating the new allocation fails
   * @throws rmm::cuda_error if the copy from the old to new allocation
   * fails
   *
   * @param new_capacity The requested new capacity, in bytes
   * @param stream The stream to use for allocation and copy
   */
  void reserve(std::size_t new_capacity, cuda_stream_view stream)
  {
    set_stream(stream);
    if (new_capacity > capacity()) {
      cuda_set_device_raii dev{_device};
      auto tmp            = device_buffer{new_capacity, stream, _mr};
      auto const old_size = size();
      RMM_CUDA_TRY(cudaMemcpyAsync(tmp.data(), data(), size(), cudaMemcpyDefault, stream.value()));
      *this = std::move(tmp);
      _size = old_size;
    }
  }

  /**
   * @brief Resize the device memory allocation
   *
   * If the requested `new_size` is less than or equal to `capacity()`, no
   * action is taken other than updating the value that is returned from
   * `size()`. Specifically, no memory is allocated nor copied. The value
   * `capacity()` remains the actual size of the device memory allocation.
   *
   * @note `shrink_to_fit()` may be used to force the deallocation of unused
   * `capacity()`.
   *
   * If `new_size` is larger than `capacity()`, a new allocation is made on
   * `stream` to satisfy `new_size`, and the contents of the old allocation are
   * copied on `stream` to the new allocation. The old allocation is then freed.
   * The bytes from `[old_size, new_size)` are uninitialized.
   *
   * The invariant `size() <= capacity()` holds.
   *
   * @throws rmm::bad_alloc If creating the new allocation fails
   * @throws rmm::cuda_error if the copy from the old to new allocation
   * fails
   *
   * @param new_size The requested new size, in bytes
   * @param stream The stream to use for allocation and copy
   */
  void resize(std::size_t new_size, cuda_stream_view stream)
  {
    set_stream(stream);
    // If the requested size is smaller than the current capacity, just update
    // the size without any allocations
    if (new_size <= capacity()) {
      _size = new_size;
    } else {
      cuda_set_device_raii dev{_device};
      auto tmp = device_buffer{new_size, stream, _mr};
      RMM_CUDA_TRY(cudaMemcpyAsync(tmp.data(), data(), size(), cudaMemcpyDefault, stream.value()));
      *this = std::move(tmp);
    }
  }

  /**
   * @brief Forces the deallocation of unused memory.
   *
   * Reallocates and copies on stream `stream` the contents of the device memory
   * allocation to reduce `capacity()` to `size()`.
   *
   * If `size() == capacity()`, no allocations or copies occur.
   *
   * @throws rmm::bad_alloc If creating the new allocation fails
   * @throws rmm::cuda_error If the copy from the old to new allocation fails
   *
   * @param stream The stream on which the allocation and copy are performed
   */
  void shrink_to_fit(cuda_stream_view stream)
  {
    set_stream(stream);
    if (size() != capacity()) {
      cuda_set_device_raii dev{_device};
      // Invoke copy ctor on self which only copies `[0, size())` and swap it
      // with self. The temporary `device_buffer` will hold the old contents
      // which will then be destroyed
      auto tmp = device_buffer{*this, stream, _mr};
      std::swap(tmp, *this);
    }
  }

  /**
   * @briefreturn{Const pointer to the device memory allocation}
   */
  [[nodiscard]] void const* data() const noexcept { return _data; }

  /**
   * @briefreturn{Pointer to the device memory allocation}
   */
  void* data() noexcept { return _data; }

  /**
   * @briefreturn{The number of bytes}
   */
  [[nodiscard]] std::size_t size() const noexcept { return _size; }

  /**
   * @briefreturn{The signed number of bytes}
   */
  [[nodiscard]] std::int64_t ssize() const noexcept
  {
    assert(size() < static_cast<std::size_t>(std::numeric_limits<int64_t>::max()) &&
           "Size overflows signed integer");
    return static_cast<int64_t>(size());
  }

  /**
   * @briefreturn{Whether or not the buffer currently holds any data}
   *
   * If `is_empty() == true`, the `device_buffer` may still hold an allocation
   * if `capacity() > 0`.
   */
  [[nodiscard]] bool is_empty() const noexcept { return 0 == size(); }

  /**
   * @brief Returns actual size in bytes of device memory allocation.
   *
   * The invariant `size() <= capacity()` holds.
   *
   * @return The actual size in bytes of the device memory allocation
   */
  [[nodiscard]] std::size_t capacity() const noexcept { return _capacity; }

  /**
   * @briefreturn{The stream most recently specified for allocation/deallocation}
   */
  [[nodiscard]] cuda_stream_view stream() const noexcept { return _stream; }

  /**
   * @brief Sets the stream to be used for deallocation
   *
   * If no other rmm::device_buffer method that allocates memory is called
   * after this call with a different stream argument, then @p stream
   * will be used for deallocation in the `rmm::device_uvector` destructor.
   * However, if either of `resize()` or `shrink_to_fit()` is called after this,
   * the later stream parameter will be stored and used in the destructor.
   *
   * @param stream The stream to use for deallocation
   */
  void set_stream(cuda_stream_view stream) noexcept { _stream = stream; }

  /**
   * @briefreturn{The resource used to allocate and deallocate}
   */
  [[nodiscard]] rmm::device_async_resource_ref memory_resource() const noexcept { return _mr; }

 private:
  void* _data{nullptr};        ///< Pointer to device memory allocation
  std::size_t _size{};         ///< Requested size of the device memory allocation
  std::size_t _capacity{};     ///< The actual size of the device memory allocation
  cuda_stream_view _stream{};  ///< Stream to use for device memory deallocation

  rmm::device_async_resource_ref _mr{
    rmm::mr::get_current_device_resource()};  ///< The memory resource used to
                                              ///< allocate/deallocate device memory
  cuda_device_id _device{get_current_cuda_device()};

  /**
   * @brief Allocates the specified amount of memory and updates the size/capacity accordingly.
   *
   * Allocates on `stream()` using the memory resource passed to the constructor.
   *
   * If `bytes == 0`, sets `_data = nullptr`.
   *
   * @param bytes The amount of memory to allocate
   */
  void allocate_async(std::size_t bytes)
  {
    _size     = bytes;
    _capacity = bytes;
    _data     = (bytes > 0) ? _mr.allocate_async(bytes, stream()) : nullptr;
  }

  /**
   * @brief Deallocate any memory held by this `device_buffer` and clear the
   * size/capacity/data members.
   *
   * If the buffer doesn't hold any memory, i.e., `capacity() == 0`, doesn't
   * call the resource deallocation.
   *
   * Deallocates on `stream()` using the memory resource passed to the constructor.
   */
  void deallocate_async() noexcept
  {
    if (capacity() > 0) { _mr.deallocate_async(data(), capacity(), stream()); }
    _size     = 0;
    _capacity = 0;
    _data     = nullptr;
  }

  /**
   * @brief Copies the specified number of `bytes` from `source` into the
   * internal device allocation.
   *
   * `source` can point to either host or device memory.
   *
   * This function assumes `_data` already points to an allocation large enough
   * to hold `bytes` bytes.
   *
   * @param source The pointer to copy from
   * @param bytes The number of bytes to copy
   */
  void copy_async(void const* source, std::size_t bytes)
  {
    if (bytes > 0) {
      RMM_EXPECTS(nullptr != source, "Invalid copy from nullptr.");
      RMM_EXPECTS(nullptr != _data, "Invalid copy to nullptr.");

      RMM_CUDA_TRY(cudaMemcpyAsync(_data, source, bytes, cudaMemcpyDefault, stream().value()));
    }
  }
};

/** @} */  // end of group
}  // namespace rmm
