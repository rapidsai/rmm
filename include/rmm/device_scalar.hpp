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

#include <rmm/rmm.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/default_memory_resource.hpp>
#include <rmm/mr/device_memory_resource.hpp>

namespace rmm {

/**---------------------------------------------------------------------------*
 * @brief Container for a single object of type `T` in device memory.
 *
 * `T` must be trivially copyable.
 *
 * @tparam T The object's type
 *---------------------------------------------------------------------------**/
template <typename T>
class device_scalar {
 public:
  static_assert(std::is_trivially_copyable<T>::value,
                "Scalar type must be trivially copyable");

  /**---------------------------------------------------------------------------*
   * @brief Construct a new `device_scalar`
   *
   * @param initial_value The initial value of the object in device memory
   * @param stream_ Optional, stream on which to perform allocation and copy
   * @param mr_ Optional, resource with which to allocate
   *---------------------------------------------------------------------------**/
  explicit device_scalar(
      T const &initial_value, cudaStream_t stream_ = 0,
      rmm::mr::device_memory_resource *mr_ = rmm::mr::get_default_resource())
      : buff{sizeof(T), stream_, mr_} {
    
    _set_value<false>(initial_value, buff.stream());
  }

  /**---------------------------------------------------------------------------*
   * @brief Copies the value from device to host and returns the value.
   *
   * @return T The value of the scalar after synchronizing its stream
   *---------------------------------------------------------------------------**/
  T value() const {
    return _value<true>(buff.stream());
  }

  /**---------------------------------------------------------------------------*
   * @brief Copies the value from device to host and returns the value.
   *
   * @return T The value of the scalar after synchronizing its stream
   * @param stream CUDA stream on which to perform the copy
   *---------------------------------------------------------------------------**/
  T value(cudaStream_t stream) const {
    return _value<true>(stream);
  }

  /**---------------------------------------------------------------------------*
   * @brief Copies the value from device to host and returns the value.
   *
   * @return T The value of the scalar
   *---------------------------------------------------------------------------**/
  T value_async() const {
    return _value<false>(buff.stream());
  }

  /**---------------------------------------------------------------------------*
   * @brief Copies the value from device to host and returns the value.
   *
   * @return T The value of the scalar
   * @param stream CUDA stream on which to perform the copy
   *---------------------------------------------------------------------------**/
  T value_async(cudaStream_t stream) const {
    return _value<false>(stream);
  }

  /**---------------------------------------------------------------------------*
   * @brief Copies the value from host to device and synchronizes.
   *
   * @param host_value The host value which will be copied to device
   *---------------------------------------------------------------------------**/
  void set_value(T host_value) {
    _set_value<true>(host_value, buff.stream());
  }


  /**---------------------------------------------------------------------------*
   * @brief Copies the value from host to device and synchronizes.
   *
   * @param host_value The host value which will be copied to device
   * @param stream CUDA stream on which to perform the copy
   *---------------------------------------------------------------------------**/
  void set_value(T host_value, cudaStream_t stream) {
    _set_value<true>(host_value, stream);
  }

  /**---------------------------------------------------------------------------*
   * @brief Copies the value from host to device.
   *
   * @param host_value The host value which will be copied to device
   *---------------------------------------------------------------------------**/
  void set_value_async(T host_value) {
    _set_value<false>(host_value, buff.stream());
  }

  /**---------------------------------------------------------------------------*
   * @brief Copies the value from host to device.
   *
   * @param host_value The host value which will be copied to device
   * @param stream CUDA stream on which to perform the copy
   *---------------------------------------------------------------------------**/
  void set_value_async(T host_value, cudaStream_t stream) {
    _set_value<false>(host_value, stream);
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns pointer to object in device memory.
   *---------------------------------------------------------------------------**/
  T *data() noexcept { return static_cast<T *>(buff.data()); }

  /**---------------------------------------------------------------------------*
   * @brief Returns pointer to object in device memory.
   *---------------------------------------------------------------------------**/
  T const *data() const noexcept { return static_cast<T const *>(buff.data()); }

  device_scalar() = default;
  ~device_scalar() = default;
  device_scalar(device_scalar const &) = default;
  device_scalar(device_scalar &&) = default;
  device_scalar &operator=(device_scalar const &) = delete;
  device_scalar &operator=(device_scalar &&) = delete;

 private:
  rmm::device_buffer buff{sizeof(T)};

  template<bool synchronize>
  inline T _value(cudaStream_t stream) const {
    T host_value{};
    _memcpy<synchronize>(&host_value, buff.data(), sizeof(T), stream);
    return host_value;
  }

  template<bool synchronize>
  inline void _set_value(T host_value, cudaStream_t stream) {
    _memcpy<synchronize>(buff.data(), &host_value, sizeof(T), stream);
  }

  template<bool synchronize>
  inline void _memcpy(void *dst, const void *src, size_t count, cudaStream_t stream) const {
    auto status = _memcpy_copy(dst, src, count, stream);

    if (RMM_SUCCESS != status) {
      throw std::runtime_error{"Device memcpy failed."};
    }

    if (false == synchronize) {
      return;
    }

    status = _memcpy_sync(stream);

    if (RMM_SUCCESS != status) {
      throw std::runtime_error{"Stream sync failed."};
    }
  }

  inline rmmError_t _memcpy_copy(void *dst, const void *src, size_t count, cudaStream_t stream) const {
    RMM_CHECK_CUDA(cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, stream));
    return RMM_SUCCESS;
  }

  inline rmmError_t _memcpy_sync(cudaStream_t stream) const {
    RMM_CHECK_CUDA(cudaStreamSynchronize(stream));
    return RMM_SUCCESS;
  }
};

}  // namespace rmm
