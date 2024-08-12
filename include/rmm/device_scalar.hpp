/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <type_traits>

namespace rmm {
/**
 * @addtogroup data_containers
 * @{
 * @file
 */

/**
 * @brief Container for a single object of type `T` in device memory.
 *
 * `T` must be trivially copyable.
 *
 * @tparam T The object's type
 */
template <typename T>
class device_scalar {
 public:
  static_assert(std::is_trivially_copyable<T>::value, "Scalar type must be trivially copyable");

  using value_type = typename device_uvector<T>::value_type;  ///< T, the type of the scalar element
  using size_type  = typename device_uvector<T>::size_type;   ///< The type used for the size
  using reference  = typename device_uvector<T>::reference;   ///< value_type&
  using const_reference = typename device_uvector<T>::const_reference;  ///< const value_type&
  using pointer =
    typename device_uvector<T>::pointer;  ///< The type of the pointer returned by data()
  using const_pointer = typename device_uvector<T>::const_pointer;  ///< The type of the iterator
                                                                    ///< returned by data() const

  RMM_EXEC_CHECK_DISABLE
  ~device_scalar() = default;

  RMM_EXEC_CHECK_DISABLE
  device_scalar(device_scalar&&) noexcept = default;  ///< Default move constructor

  /**
   * @brief Default move assignment operator
   *
   * @return device_scalar& A reference to the assigned-to object
   */
  device_scalar& operator=(device_scalar&&) noexcept = default;

  /**
   * @brief Copy ctor is deleted as it doesn't allow a stream argument
   */
  device_scalar(device_scalar const&) = delete;

  /**
   * @brief Copy assignment is deleted as it doesn't allow a stream argument
   */
  device_scalar& operator=(device_scalar const&) = delete;

  /**
   * @brief Default constructor is deleted as it doesn't allow a stream argument
   */
  device_scalar() = delete;

  /**
   * @brief Construct a new uninitialized `device_scalar`.
   *
   * Does not synchronize the stream.
   *
   * @note This device_scalar is only safe to access in kernels and copies on the specified CUDA
   * stream, or on another stream only if a dependency is enforced (e.g. using
   * `cudaStreamWaitEvent()`).
   *
   * @throws rmm::bad_alloc if allocating the device memory fails.
   *
   * @param stream Stream on which to perform asynchronous allocation.
   * @param mr Optional, resource with which to allocate.
   */
  explicit device_scalar(cuda_stream_view stream,
                         device_async_resource_ref mr = mr::get_current_device_resource())
    : _storage{1, stream, mr}
  {
  }

  /**
   * @brief Construct a new `device_scalar` with an initial value.
   *
   * Does not synchronize the stream.
   *
   * @note This device_scalar is only safe to access in kernels and copies on the specified CUDA
   * stream, or on another stream only if a dependency is enforced (e.g. using
   * `cudaStreamWaitEvent()`).
   *
   * @throws rmm::bad_alloc if allocating the device memory for `initial_value` fails.
   * @throws rmm::cuda_error if copying `initial_value` to device memory fails.
   *
   * @param initial_value The initial value of the object in device memory.
   * @param stream Optional, stream on which to perform allocation and copy.
   * @param mr Optional, resource with which to allocate.
   */
  explicit device_scalar(value_type const& initial_value,
                         cuda_stream_view stream,
                         device_async_resource_ref mr = mr::get_current_device_resource())
    : _storage{1, stream, mr}
  {
    set_value_async(initial_value, stream);
  }

  /**
   * @brief Construct a new `device_scalar` by deep copying the contents of
   * another `device_scalar`, using the specified stream and memory
   * resource.
   *
   * @throws rmm::bad_alloc If creating the new allocation fails.
   * @throws rmm::cuda_error if copying from `other` fails.
   *
   * @param other The `device_scalar` whose contents will be copied
   * @param stream The stream to use for the allocation and copy
   * @param mr The resource to use for allocating the new `device_scalar`
   */
  device_scalar(device_scalar const& other,
                cuda_stream_view stream,
                device_async_resource_ref mr = mr::get_current_device_resource())
    : _storage{other._storage, stream, mr}
  {
  }

  /**
   * @brief Copies the value from device to host, synchronizes, and returns the value.
   *
   * Synchronizes `stream` after copying the data from device to host.
   *
   * @note If the stream specified to this function is different from the stream specified
   * to the constructor, then an appropriate dependency must be inserted between the streams
   * (e.g. using `cudaStreamWaitEvent()` or `cudaStreamSynchronize()`) before calling this function,
   * otherwise there may be a race condition.
   *
   * @throws rmm::cuda_error If the copy fails.
   * @throws rmm::cuda_error If synchronizing `stream` fails.
   *
   * @return T The value of the scalar.
   * @param stream CUDA stream on which to perform the copy and synchronize.
   */
  [[nodiscard]] value_type value(cuda_stream_view stream) const
  {
    return _storage.front_element(stream);
  }

  /**
   * @brief Sets the value of the `device_scalar` to the value of `v`.
   *
   * This specialization for fundamental types is optimized to use `cudaMemsetAsync` when
   * `v` is zero.
   *
   * @note If the stream specified to this function is different from the stream specified
   * to the constructor, then appropriate dependencies must be inserted between the streams
   * (e.g. using `cudaStreamWaitEvent()` or `cudaStreamSynchronize()`) before and after calling
   * this function, otherwise there may be a race condition.
   *
   * This function does not synchronize `stream` before returning. Therefore, the object
   * referenced by `v` should not be destroyed or modified until `stream` has been
   * synchronized. Otherwise, behavior is undefined.
   *
   * @note: This function incurs a host to device memcpy or device memset and should be used
   * carefully.
   *
   * Example:
   * \code{cpp}
   * rmm::device_scalar<int32_t> s;
   *
   * int v{42};
   *
   * // Copies 42 to device storage on `stream`. Does _not_ synchronize
   * vec.set_value_async(v, stream);
   * ...
   * cudaStreamSynchronize(stream);
   * // Synchronization is required before `v` can be modified
   * v = 13;
   * \endcode
   *
   * @throws rmm::cuda_error if copying @p value to device memory fails.
   *
   * @param value The host value which will be copied to device
   * @param stream CUDA stream on which to perform the copy
   */
  void set_value_async(value_type const& value, cuda_stream_view stream)
  {
    _storage.set_element_async(0, value, stream);
  }

  // Disallow passing literals to set_value to avoid race conditions where the memory holding the
  // literal can be freed before the async memcpy / memset executes.
  void set_value_async(value_type&&, cuda_stream_view) = delete;

  /**
   * @brief Sets the value of the `device_scalar` to zero on the specified stream.
   *
   * @note If the stream specified to this function is different from the stream specified
   * to the constructor, then appropriate dependencies must be inserted between the streams
   * (e.g. using `cudaStreamWaitEvent()` or `cudaStreamSynchronize()`) before and after calling
   * this function, otherwise there may be a race condition.
   *
   * This function does not synchronize `stream` before returning.
   *
   * @note: This function incurs a device memset and should be used carefully.
   *
   * @param stream CUDA stream on which to perform the copy
   */
  void set_value_to_zero_async(cuda_stream_view stream)
  {
    _storage.set_element_to_zero_async(value_type{0}, stream);
  }

  /**
   * @brief Returns pointer to object in device memory.
   *
   * @note If the returned device pointer is used on a CUDA stream different from the stream
   * specified to the constructor, then appropriate dependencies must be inserted between the
   * streams (e.g. using `cudaStreamWaitEvent()` or `cudaStreamSynchronize()`), otherwise there may
   * be a race condition.
   *
   * @return Pointer to underlying device memory
   */
  [[nodiscard]] pointer data() noexcept { return static_cast<pointer>(_storage.data()); }

  /**
   * @brief Returns const pointer to object in device memory.
   *
   * @note If the returned device pointer is used on a CUDA stream different from the stream
   * specified to the constructor, then appropriate dependencies must be inserted between the
   * streams (e.g. using `cudaStreamWaitEvent()` or `cudaStreamSynchronize()`), otherwise there may
   * be a race condition.
   *
   * @return Const pointer to underlying device memory
   */
  [[nodiscard]] const_pointer data() const noexcept
  {
    return static_cast<const_pointer>(_storage.data());
  }

  /**
   * @briefreturn{The size of the scalar: always 1}
   */
  [[nodiscard]] constexpr size_type size() const noexcept { return 1; }

  /**
   * @briefreturn{Stream associated with the device memory allocation}
   */
  [[nodiscard]] cuda_stream_view stream() const noexcept { return _storage.stream(); }

  /**
   * @brief Sets the stream to be used for deallocation
   *
   * @param stream Stream to be used for deallocation
   */
  void set_stream(cuda_stream_view stream) noexcept { _storage.set_stream(stream); }

 private:
  rmm::device_uvector<T> _storage;
};

/** @} */  // end of group
}  // namespace rmm
