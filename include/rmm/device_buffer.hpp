/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <rmm/mr/default_memory_resource.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <cuda_runtime_api.h>
#include <cassert>
#include <stdexcept>

namespace rmm {
/**---------------------------------------------------------------------------*
 * @file device_buffer.hpp
 * @brief RAII construct for device memory allocation
 *
 * This class allocates untyped and *uninitialized* device memory using a
 * `device_memory_resource`. If not explicitly specified, the memory resource
 * returned from `get_default_resource()` is used.
 *
 * @note Unlike `std::vector` or `thrust::device_vector`, the device memory
 * allocated by a `device_buffer` is unitialized. Therefore, it is undefined
 * behavior to read the contents of `data()` before first initializing it.
 *
 * Examples:
 * ```
 * //Allocates at least 100 bytes of device memory using the default memory
 * //resource
 * device_buffer buff(100);
 *
 * // allocates at least 100 bytes using the custom memory resource
 * custom_memory_resource mr;
 * device_buffer custom_buff(100, &mr);
 *
 * // deep copies `buff` into a new device buffer
 * device_buffer buff_copy(buff);
 *
 * // shallow copies `buff` into a new device_buffer, `buff` is now empty
 * device_buffer buff_move(std::move(buff));
 *
 * // Default construction. Buffer is empty
 * device_buffer buff_default{};
 *
 * // If the requested sized is larger than the current
 * // size, resizes allocation to the new size and deep copies any previous
 * // contents. Otherwise, simply updates the value of `size()` to the newly
 * // requested size without any allocations or copies
 * buff_default.resize(100);
 *```
 *---------------------------------------------------------------------------**/

class device_buffer {
 public:
  /**---------------------------------------------------------------------------*
   * @brief Constructs an empty `device_buffer` of size 0
   *---------------------------------------------------------------------------**/
  device_buffer() = default;

  /**---------------------------------------------------------------------------*
   * @brief Constructs a new device buffer of `size` unitialized bytes
   *
   * @throws std::bad_alloc If creating the new allocation fails
   *
   * @param size Size in bytes to allocate in device memory
   * @param stream CUDA stream on which memory may be allocated if the memory
   * resource supports streams, otherwise the default stream is used.
   * @param mr Memory resource to use for the device memory allocation
   *---------------------------------------------------------------------------**/
  explicit device_buffer(
      std::size_t size, cudaStream_t stream = 0,
      mr::device_memory_resource* mr = mr::get_default_resource())
      : _size{size}, _capacity{size}, _stream{stream}, _mr{mr} {
    _data = _mr->allocate(size, stream);
  }

  /**---------------------------------------------------------------------------*
   * @brief Construct a new device buffer by copying from a raw pointer to an
   * existing host or device memory allocation.
   *
   * @throws std::bad_alloc If creating the new allocation fails
   * @throws std::runtime_error If `source_data` is null, and `size != 0`
   * @throws std::runtime_error if copying from the device memory fails
   *
   * @param source_data Pointer to the host or device memory to copy from
   * @param size Size in bytes to copy
   * @param stream CUDA stream on which memory may be allocated if the memory
   * resource supports streams, otherwise the default stream is used.
   * @param mr Memory resource to use for the device memory allocation
   *---------------------------------------------------------------------------**/
  device_buffer(void const* source_data, std::size_t size,
                cudaStream_t stream = 0,
                mr::device_memory_resource* mr = mr::get_default_resource())
      : _size{size}, _capacity{size}, _stream{stream}, _mr{mr} {
    if (nullptr == source_data and (0 != size)) {
      throw std::runtime_error{"Invalid size."};
    }

    _data = _mr->allocate(_size, stream);
    auto status =
        cudaMemcpyAsync(_data, source_data, _size, cudaMemcpyDefault, _stream);
    if (cudaSuccess != status) {
      throw std::runtime_error{"Device memcopy failed."};
    }
  }

  /**---------------------------------------------------------------------------*
   * @brief Constructs a new `device_buffer` by deep copying the contents of
   * another `device_buffer`.
   *
   * Uses `other.stream()` and `other.memory_resource()` for allocation.
   *
   * @note Only copies `other.size()` bytes from `other`, i.e., if
   *`other.size()
   * != other.capacity()`, then the size and capacity of the newly constructed
   *`device_buffer` will be equal to `other.size()`.
   *
   * @throws std::bad_alloc If creating the new allocation fails
   * @throws std::runtime_error if copying from `other` fails
   *
   * @param other The other `device_buffer` to deep copy
   *---------------------------------------------------------------------------**/
  device_buffer(device_buffer const& other)
      : _size{other._size},
        _capacity{other._size},
        _stream{other._stream},
        _mr{other._mr} {
    _data = _mr->allocate(_size, _stream);
    auto status =
        cudaMemcpyAsync(_data, other._data, _size, cudaMemcpyDefault, _stream);

    if (cudaSuccess != status) {
      throw std::runtime_error{"Device memory copy failed."};
    }
  }

  /**---------------------------------------------------------------------------*
   * @brief Construct a new `device_buffer` by deep copying the contents of
   * another `device_buffer` using the specified stream and memory resource.
   *
   * @note Only copies `other.size()` bytes from `other`, i.e., if
   *`other.size()
   * != other.capacity()`, then the size and capacity of the newly constructed
   *`device_buffer` will be equal to `other.size()`.

   * @throws std::bad_alloc If creating the new allocation fails
   * @throws std::runtime_error if copying from `other` fails
   *
   * @param other The `device_buffer` whose contents will be copied
   * @param stream The stream to use for the allocation and copy
   * @param mr The resource to use for allocating the new `device_buffer`
   *---------------------------------------------------------------------------**/
  device_buffer(
      device_buffer const& other, cudaStream_t stream,
      rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource())
      : _size{other.size()}, _capacity{other.size()}, _stream{stream}, _mr{mr} {
    _data = memory_resource()->allocate(size(), _stream);

    auto status = cudaMemcpyAsync(data(), other.data(), size(),
                                  cudaMemcpyDefault, _stream);

    if (cudaSuccess != status) {
      throw std::runtime_error{"Device memory copy failed."};
    }
  }

  /**---------------------------------------------------------------------------*
   * @brief Constructs a new `device_buffer` by moving the contents of another
   * `device_buffer` into the newly constructed one.
   *
   * After the new `device_buffer` is constructed, `other` is modified to be
   * empty, i.e., `data()` returns `nullptr`, and `size()` is zero.
   *
   * @throws Nothing
   *
   * @param other The `device_buffer` whose contents will be moved into the
   * newly constructed one
   *---------------------------------------------------------------------------**/
  device_buffer(device_buffer&& other) noexcept
      : _data{other._data},
        _size{other._size},
        _capacity{other._capacity},
        _stream{other._stream},
        _mr{other._mr} {
    other._data = nullptr;
    other._size = 0;
    other._capacity = 0;
    other._stream = 0;
  }

  /**---------------------------------------------------------------------------*
   * @brief Copy assignment operator copies the contents of `other`
   *
   * @param other The `device_buffer` to copy
   *---------------------------------------------------------------------------**/
  device_buffer& operator=(device_buffer const& other) {
    if (&other != this) {
      _mr->deallocate(_data, _capacity, _stream);
      _size = other._size;
      _capacity = other._size;  // only allocate _size bytes
      _stream = other._stream;
      _mr = other._mr;
      _data = _mr->allocate(_size, _stream);
      auto status = cudaMemcpyAsync(_data, other._data, _size,
                                    cudaMemcpyDefault, _stream);

      if (cudaSuccess != status) {
        throw std::runtime_error{"Device memory copy failed."};
      }
    }
    return *this;
  }

  /**---------------------------------------------------------------------------*
   * @brief Move assignment operator moves the contents from `other`
   *
   * @param other The `device_buffer` whose contents will be moved
   *---------------------------------------------------------------------------**/
  device_buffer& operator=(device_buffer&& other) {
    if (&other != this) {
      _mr->deallocate(_data, _capacity, _stream);
      _data = other._data;
      _size = other._size;
      _capacity = other._capacity;
      _stream = other._stream;
      _mr = other._mr;

      other._data = nullptr;
      other._size = 0;
      other._capacity = 0;
      other._stream = 0;
    }
    return *this;
  }

  ~device_buffer() noexcept {
    _mr->deallocate(_data, _capacity, _stream);
    _data = nullptr;
    _size = 0;
    _capacity = 0;
    _stream = 0;
    _mr = nullptr;
  }

  /**---------------------------------------------------------------------------*
   * @brief Resize the device memory allocation
   *
   * If the requested `new_size` is less than or equal to the current size, no
   * action is taken other than updating the value that is returned from
   * `size()`. I.e., no memory is allocated nor copied. The value `capacity()`
   * will remain the actual size of the device memory allocation.
   *
   * @note `shrink_to_fit()` may be used to force the deallocation of unused
   * `capacity()`.
   *
   * If `new_size` is larger than the current size, a new allocation is made
   *to satisfy `new_size`, and the contents of the old allocation are copied
   *to the new allocation. The old allocation is then freed.
   *
   * The invariant `size() <= capacity()` will always be true.
   *
   * The `stream` returned by `stream()` is used for the allocation and
   *copying of the new memory.
   *
   * @throws std::bad_alloc If creating the new allocation fails
   * @throws std::runtime_error if the copy from the old to new allocation
   *fails
   *
   * @param new_size The requested new size, in bytes
   *---------------------------------------------------------------------------**/
  void resize(std::size_t new_size) {
    // If the requested size is smaller, just update the size without any
    // allocations
    if (new_size <= _size) {
      _size = new_size;
    } else {
      void* const new_data = _mr->allocate(new_size, _stream);
      if (_size > 0) {
        auto status =
            cudaMemcpyAsync(new_data, _data, _size, cudaMemcpyDefault, _stream);

        if (cudaSuccess != status) {
          throw std::runtime_error{"Device memory copy failed."};
        }
      }
      _mr->deallocate(_data, _size, _stream);
      _data = new_data;
      _size = new_size;
      _capacity = new_size;
    }
  }

  /**---------------------------------------------------------------------------*
   * @brief Forces the deallocation of unused memory.
   *
   * Reallocates and copies the contents of the device memory allocation to
   * reduce `capacity()` to `size()`.
   *
   * If `size() == capacity()` this function has no effect.
   *
   * @throws std::bad_alloc If creating the new allocation fails
   * @throws std::runtime_error If the copy from the old to new allocation
   *fails
   *
   *---------------------------------------------------------------------------**/
  void shrink_to_fit() {
    if (size() != capacity()) {
      void* const new_data = _mr->allocate(size(), stream());
      if (size() > 0) {
        auto status = cudaMemcpyAsync(new_data, _data, size(),
                                      cudaMemcpyDefault, stream());
        if (cudaSuccess != status) {
          throw std::runtime_error{"Device memory copy failed."};
        }
      }
      _mr->deallocate(_data, size(), stream());
      _data = new_data;
      _capacity = size();
    }
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns raw pointer to underlying device memory allocation
   *---------------------------------------------------------------------------**/
  void const* data() const noexcept { return _data; }

  /**---------------------------------------------------------------------------*
   * @brief Returns raw pointer to underlying device memory allocation
   *---------------------------------------------------------------------------**/
  void* data() noexcept { return _data; }

  /**---------------------------------------------------------------------------*
   * @brief Returns size in bytes that was requested for the device memory
   * allocation
   *---------------------------------------------------------------------------**/
  std::size_t size() const noexcept { return _size; }

  /**---------------------------------------------------------------------------*
   * @brief Returns actual size in bytes of device memory allocation.
   *
   * The following invariant will always be true:
   * ```
   * size() <= capacity()
   * ```
   *---------------------------------------------------------------------------**/
  std::size_t capacity() const noexcept { return _capacity; }

  /**---------------------------------------------------------------------------*
   * @brief Returns stream used for allocation/deallocation
   *---------------------------------------------------------------------------**/
  cudaStream_t stream() const noexcept { return _stream; }

  /**---------------------------------------------------------------------------*
   * @brief Returns pointer to the memory resource used to allocate and
   * deallocate the device memory
   *---------------------------------------------------------------------------**/
  mr::device_memory_resource* memory_resource() const noexcept { return _mr; }

 private:
  void* _data{nullptr};     ///< Pointer to device memory allocation
  std::size_t _size{};      ///< Requested size of the device memory allocation
  std::size_t _capacity{};  ///< The actual size of the device memory allocation
  cudaStream_t _stream{};   ///< Stream which may be used for
                            ///< allocation/deallocation of device memory
  mr::device_memory_resource* _mr{
      mr::get_default_resource()};  ///< The memory resource used to
                                    ///< allocate/deallocate device memory
};                                  // namespace rmm
}  // namespace rmm
