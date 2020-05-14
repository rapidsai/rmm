/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <rmm/detail/error.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/default_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include "rmm/thrust_rmm_allocator.h"

#include <thrust/uninitialized_fill.h>

namespace rmm {

/**
 * @brief An *uninitialized* vector of elements in device memory.
 *
 * Similar to a `thrust::device_vector`, `device_uvector` is a random access container of elements
 * stored contiguously in device memory. However, unlike `thrust::device_vector`, `device_uvector`
 * does *not* default initialize the vector elements. Initialization is only performed when
 * explicitly requested via appropriate constructors.
 *
 * Avoiding default initialization improves performance by eliminating the kernel launch required to
 * default initialize the elements. This initialization is often unnecessary, e.g., when the vector
 * is created to hold some output from some operation.
 *
 * However, this restricts the element type `T` to only trivially copyable types. In short,
 * trivially copyable types can be safely copied with `memcpy`. For more information, see
 * https://en.cppreference.com/w/cpp/types/is_trivially_copyable.
 *
 * Another key difference over `thrust::device_vector` is that all operations that invoke
 * allocation, kernels, or memcpys take a CUDA stream parameter to indicate on which stream the
 * operation will be performed.
 *
 * @tparam T Trivially copyable element type
 */
template <typename T>
class device_uvector {
  static_assert(std::is_trivially_copyable<T>::value,
                "device_uvector only supports types that are trivially copyable.");

 public:
  using value_type      = T;
  using size_type       = std::size_t;
  using reference       = value_type&;
  using const_reference = value_type const&;
  using pointer         = value_type*;
  using const_pointer   = value_type const*;
  using iterator        = pointer;
  using const_iterator  = const_pointer;

  device_uvector()                      = default;
  ~device_uvector()                     = default;
  device_uvector(device_uvector const&) = default;
  device_uvector(device_uvector&&)      = default;
  device_uvector& operator=(device_uvector const&) = default;
  device_uvector& operator=(device_uvector&&) = default;

  /**
   * @brief Construct a new `device_uvector` with sufficient uninitialized storage for `size`
   * elements.
   *
   * Elements are uninitialized. Reading an element before it is initialized results in undefined
   * behavior.
   *
   * @param size The number of elements to allocate storage for
   * @param stream The stream on which to perform the allocation
   * @param mr The resource used to allocate the device storage
   */
  explicit device_uvector(std::size_t size,
                          cudaStream_t stream                 = 0,
                          rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource())
    : _storage{elements_to_bytes(size), stream, mr}
  {
  }

  /**
   * @brief Construct a new `device_uvector` with the specified number of elements initialized to
   * the specified value.
   *
   * @param size The number of elements
   * @param v The value used to initialize all elements
   * @param stream The stream on which to perform the allocation and initialization
   * @param mr The resource used to allocate the device storage
   */
  device_uvector(std::size_t size,
                 value_type const& v,
                 cudaStream_t stream                 = 0,
                 rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource())
    : device_uvector(size, stream, mr)
  {
    if (std::is_arithmetic<T>::value && (value_type{0} == v)) {
      // This is ~20% faster for sizes less than 1M
      RMM_CUDA_TRY(cudaMemsetAsync(data(), 0, _storage.size(), stream));
    } else {
      thrust::uninitialized_fill(rmm::exec_policy(stream)->on(stream), begin(), end(), v);
    }
  }

  /**
   * @brief Construct a new device_uvector by deep copying the contents of another `device_uvector`.
   *
   * Elements are copied as if it were performed by `memcpy`, i.e., `T`'s copy constructor is not
   * invoked.
   *
   * @param other The vector to copy from
   * @param stream The stream on which to perform the copy
   * @param mr The resource used to allocate device memory for the new vector
   */
  explicit device_uvector(device_uvector const& other,
                          cudaStream_t stream                 = 0,
                          rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource())
    : _storage{other.storage, stream, mr}
  {
  }

  /**
   * @brief Resizes the vector to contain `new_size` elements.
   *
   * If `new_size > size()`, the additional elements are uninitialized.
   *
   * If `new_size < capacity()`, no action is taken other than updating the value of `size()`. No
   * memory is allocated nor copied. `shrink_to_fit()` may be used to force deallocation of unused
   * memory.
   *
   * If `new_size > capacity()`, elements are copied as if by mempcy to a new allocation.
   *
   * The invariant `size() <= capacity()` holds.
   *
   * @param new_size The desired number of elements
   * @param stream The stream on which to perform the allocation/copy (if any)
   */
  void resize(std::size_t new_size, cudaStream_t stream = 0)
  {
    _storage.resize(elements_to_bytes(new_size), stream);
  }

  /**
   * @brief Forces deallocation of unused device memory.
   *
   * If `capacity() > size()`, reallocates and copies vector contents to eliminate unused memory.
   *
   * @param stream Stream on which to perform allocation and copy
   */
  void shrink_to_fit(cudaStream_t stream = 0) { _storage.shrink_to_fit(stream); }

  /**
   * @brief Release ownership of device memory storage.
   *
   * @return The `device_buffer` used to store the vector elements
   */
  device_buffer release() noexcept { return std::move(_storage); }

  /**
   * @brief Returns the number of elements that can be held in currently allocated storage.
   *
   * @return std::size_t The number of elements that can be stored without requiring a new
   * allocation.
   */
  std::size_t capacity() const noexcept { return bytes_to_elements(_storage.capacity()); }

  /**
   * @brief Returns pointer to underlying device storage.
   *
   * @return Raw pointer to element storage in device memory.
   */
  pointer data() noexcept { return static_cast<pointer>(_storage.data()); }

  /**
   * @brief Returns const pointer to underlying device storage.
   *
   * @return const_pointer Raw const pointer to element storage in device memory.
   */
  const_pointer data() const noexcept { return static_cast<const_pointer>(_storage.data()); }

  /**
   * @brief Returns an iterator to the first element.
   *
   * @return Iterator to the first element.
   */
  iterator begin() noexcept { return data(); }

  /**
   * @brief Returns an iterator to the first element.
   *
   * @return Iterator to the first element.
   */
  const_iterator begin() const noexcept { return data(); }

  /**
   * @brief Returns an iterator to the element following the last element of the vector.
   *
   * The element referenced by `end()` is a placeholder and dereferencing it results in undefined
   * behavior.
   *
   * @return Iterator to one past the last element.
   */
  iterator end() noexcept { return data() + size(); }

  /**
   * @brief Returns an iterator to the element following the last element of the vector.
   *
   * The element referenced by `end()` is a placeholder and dereferencing it results in undefined
   * behavior.
   *
   * @return Iterator to one past the last element.
   */
  const_iterator end() const noexcept { return data() + size(); }

  /**
   * @brief Returns the number of elements in the vector.
   *
   * @return The number of elements.
   */
  std::size_t size() const noexcept { return bytes_to_elements(_storage.size()); }

  /**
   * @brief Returns if the vector contains no elements, i.e., `size() == 0`.
   *
   * @return true The vector is empty
   * @return false The vector is not empty
   */
  bool is_empty() const noexcept { return size() == 0; }

  /**
   * @brief Returns pointer to the resource used to allocate and deallocate the device storage.
   *
   * @return Pointer to underlying resource
   */
  mr::device_memory_resource* memory_resource() const noexcept
  {
    return _storage.memory_resource();
  }

 private:
  device_buffer _storage{};  ///< Device memory storage for vector elements

  std::size_t constexpr elements_to_bytes(std::size_t num_elements) const noexcept
  {
    return num_elements * sizeof(value_type);
  }

  std::size_t constexpr bytes_to_elements(std::size_t num_bytes) const noexcept
  {
    return num_bytes / sizeof(value_type);
  }
};
}  // namespace rmm
