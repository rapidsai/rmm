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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <type_traits>

namespace rmm {

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

  /**
   * @brief Construct a new uninitialized `device_scalar`.
   *
   * Does not synchronize the stream.
   *
   * @note This device_scalar is only safe to access in kernels and copies on the specified CUDA
   * stream, or on another stream only if a dependency is enforced (e.g. using
   * `cudaStreamWaitEvent()`).
   *
   * @throws `rmm::bad_alloc` if allocating the device memory fails.
   *
   * @param stream Stream on which to perform asynchronous allocation.
   * @param mr Optional, resource with which to allocate.
   */
  explicit device_scalar(
    cuda_stream_view stream,
    rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource())
    : buffer{sizeof(T), stream, mr}
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
   * @throws `rmm::bad_alloc` if allocating the device memory for `initial_value` fails.
   * @throws `rmm::cuda_error` if copying `initial_value` to device memory fails.
   *
   * @param initial_value The initial value of the object in device memory.
   * @param stream Optional, stream on which to perform allocation and copy.
   * @param mr Optional, resource with which to allocate.
   */
  explicit device_scalar(
    T const &initial_value,
    cuda_stream_view stream             = cuda_stream_view{},
    rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource())
    : buffer{sizeof(T), stream, mr}
  {
    set_value(initial_value, stream);
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
  device_scalar(device_scalar const &other,
                cuda_stream_view stream             = {},
                rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource())
    : buffer{other.buffer, stream, mr}
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
   * @throws `rmm::cuda_error` If the copy fails.
   * @throws `rmm::cuda_error` If synchronizing `stream` fails.
   *
   * @return T The value of the scalar.
   * @param stream CUDA stream on which to perform the copy and synchronize.
   */
  T value(cuda_stream_view stream = cuda_stream_view{}) const
  {
    T host_value{};
    _memcpy(&host_value, buffer.data(), stream);
    stream.synchronize();
    return host_value;
  }

  /**
   * @brief Sets the value of the `device_scalar` to the given `host_value`.
   *
   * This specialization for fundamental types is optimized to use `cudaMemsetAsync` when
   * `host_value` is zero.
   *
   * @note If the stream specified to this function is different from the stream specified
   * to the constructor, then appropriate dependencies must be inserted between the streams
   * (e.g. using `cudaStreamWaitEvent()` or `cudaStreamSynchronize()`) before and after calling
   * this function, otherwise there may be a race condition.
   *
   * This function does not synchronize `stream` before returning. Therefore, the object
   * referenced by `host_value` should not be destroyed or modified until `stream` has been
   * synchronized. Otherwise, behavior is undefined.
   *
   * @note: This function incurs a host to device memcpy or device memset and should be used
   * sparingly.
   *
   * Example:
   * \code{cpp}
   * rmm::device_scalar<int32_t> s;
   *
   * int v{42};
   *
   * // Copies 42 to device storage on `stream`. Does _not_ synchronize
   * vec.set_value(v, stream);
   * ...
   * cudaStreamSynchronize(stream);
   * // Synchronization is required before `v` can be modified
   * v = 13;
   * \endcode
   *
   * @throws `rmm::cuda_error` if copying `host_value` to device memory fails.
   *
   * @param host_value The host value which will be copied to device
   * @param stream CUDA stream on which to perform the copy
   */
  template <typename U = T>
  auto set_value(U const &host_value, cuda_stream_view stream = cuda_stream_view{})
    -> std::enable_if_t<std::is_fundamental<U>::value && not std::is_same<U, bool>::value, void>
  {
    if (host_value == U{0}) {
      set_value_zero(stream);
    } else {
      _memcpy(buffer.data(), &host_value, stream);
    }
  }

  /**
   * @brief Sets the value of the `device_scalar` to the given `host_value`.
   *
   * This specialization for `bool` is optimized to always use `cudaMemsetAsync`.
   *
   * @note If the stream specified to this function is different from the stream specified
   * to the constructor, then appropriate dependencies must be inserted between the streams
   * (e.g. using `cudaStreamWaitEvent()` or `cudaStreamSynchronize()`) before and after calling
   * this function, otherwise there may be a race condition.
   *
   * This function does not synchronize `stream` before returning. `host_value` is passed by value
   * so a host-side copy may be performed before calling a device memset.
   *
   * @note: This function incurs a device memset.
   *
   * Example:
   * \code{cpp}
   * rmm::device_scalar<bool> s;
   *
   * bool v{true};
   *
   * // Copies `true` to device storage on `stream`. Does _not_ synchronize
   * vec.set_value(v, stream);
   * ...
   * cudaStreamSynchronize(stream);
   * // Synchronization is required before `v` can be modified
   * v = false;
   * \endcode
   *
   * @throws `rmm::cuda_error` if the device memset fails.
   *
   * @param host_value The host value which the scalar will be set to (true or false)
   * @param stream CUDA stream on which to perform the device memset
   */
  template <typename U = T>
  auto set_value(U const &host_value, cuda_stream_view stream = cuda_stream_view{})
    -> std::enable_if_t<std::is_same<U, bool>::value, void>
  {
    RMM_CUDA_TRY(cudaMemsetAsync(buffer.data(), host_value, sizeof(bool), stream.value()));
  }

  /**
   * @brief Sets the value of the `device_scalar` to the given `host_value`.
   *
   * Specialization for non-fundamental types.
   *
   * @note If the stream specified to this function is different from the stream specified
   * to the constructor, then appropriate dependencies must be inserted between the streams
   * (e.g. using `cudaStreamWaitEvent()` or `cudaStreamSynchronize()`) before and after calling
   * this function, otherwise there may be a race condition.
   *
   * This function does not synchronize `stream` before returning. Therefore, the object
   * referenced by `host_value` should not be destroyed or modified until `stream` has been
   * synchronized. Otherwise, behavior is undefined.
   *
   * @note: This function incurs a host to device memcpy and should be used sparingly.

   * Example:
   * \code{cpp}
   * rmm::device_scalar<my_type> s;
   *
   * my_type v{42, "text"};
   *
   * // Copies 42 to device storage on `stream`. Does _not_ synchronize
   * vec.set_value(v, stream);
   * ...
   * cudaStreamSynchronize(stream);
   * // Synchronization is required before `v` can be modified
   * v.value = 21;
   * \endcode
   *
   * @throws `rmm::cuda_error` if copying `host_value` to device memory fails
   * @throws `rmm::cuda_error` if synchronizing `stream` fails
   *
   * @param host_value The host value which will be copied to device
   * @param stream CUDA stream on which to perform the copy
   */
  template <typename U = T>
  auto set_value(T const &host_value, cuda_stream_view stream = cuda_stream_view{})
    -> std::enable_if_t<not std::is_fundamental<U>::value, void>
  {
    _memcpy(buffer.data(), &host_value, stream);
  }

  // Disallow passing literals to set_value to avoid race conditions where the memory holding the
  // literal can be freed before the async memcpy / memset executes.
  void set_value(T &&host_value, cuda_stream_view stream = cuda_stream_view{}) = delete;

  /**
   * @brief Sets the value of the `device_scalar` to zero.
   *
   * Only supported for fundamental types.
   *
   * @note If the stream specified to this function is different from the stream specified
   * to the constructor, then appropriate dependencies must be inserted between the streams
   * (e.g. using `cudaStreamWaitEvent()` or `cudaStreamSynchronize()`) before and after calling
   * this function, otherwise there may be a race condition.
   *
   * This function does not synchronize `stream` before returning.
   *
   * @note: This function incurs a device memset and should be used sparingly.
   *
   * @throws `rmm::cuda_error` if the device memset fails.
   *
   * @param stream CUDA stream on which to perform the device memset
   */
  template <typename U = T>
  auto set_value_zero(cuda_stream_view stream = cuda_stream_view{})
    -> std::enable_if_t<std::is_fundamental<U>::value, void>
  {
    RMM_CUDA_TRY(cudaMemsetAsync(buffer.data(), 0, sizeof(U), stream.value()));
  }

  /**
   * @brief Returns pointer to object in device memory.
   *
   * @note If the returned device pointer is used on a CUDA stream different from the stream
   * specified to the constructor, then appropriate dependencies must be inserted between the
   * streams (e.g. using `cudaStreamWaitEvent()` or `cudaStreamSynchronize()`), otherwise there may
   * be a race condition.
   */
  T *data() noexcept { return static_cast<T *>(buffer.data()); }

  /**
   * @brief Returns const pointer to object in device memory.
   *
   * @note If the returned device pointer is used on a CUDA stream different from the stream
   * specified to the constructor, then appropriate dependencies must be inserted between the
   * streams (e.g. using `cudaStreamWaitEvent()` or `cudaStreamSynchronize()`), otherwise there may
   * be a race condition.
   */
  T const *data() const noexcept { return static_cast<T const *>(buffer.data()); }

  device_scalar()                 = default;
  ~device_scalar()                = default;
  device_scalar(device_scalar &&) = default;
  device_scalar &operator=(device_scalar const &) = delete;
  device_scalar &operator=(device_scalar &&) = delete;

 private:
  rmm::device_buffer buffer{sizeof(T)};

  inline void _memcpy(void *dst, const void *src, cuda_stream_view stream) const
  {
    RMM_CUDA_TRY(cudaMemcpyAsync(dst, src, sizeof(T), cudaMemcpyDefault, stream.value()));
  }
};
}  // namespace rmm
