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
    auto status = cudaMemcpyAsync(buff.data(), &initial_value, sizeof(T),
                                  cudaMemcpyDefault, buff.stream());

    if (cudaSuccess != status) {
      throw std::runtime_error{"Device memcpy failed."};
    }
  }

  /**---------------------------------------------------------------------------*
   * @brief Copies the value from device to host and returns the value.
   *
   * @return T The value of the scalar after synchronizing its stream
   *---------------------------------------------------------------------------**/
  T value() const {
    T host_value{};
    auto status = cudaMemcpyAsync(&host_value, buff.data(), sizeof(T),
                                  cudaMemcpyDefault, buff.stream());
    if (cudaSuccess != status) {
      throw std::runtime_error{"Device memcpy failed."};
    }
    status = cudaStreamSynchronize(buff.stream());
    if (cudaSuccess != status) {
      throw std::runtime_error{"Stream sync failed."};
    }
    return host_value;
  }

  /**---------------------------------------------------------------------------*
   * @brief Copies the value from hostto device.
   *
   * @param host_value The value of the scalar after synchronizing its stream
   *---------------------------------------------------------------------------**/
  void set_value(T host_value) {
    auto status = cudaMemcpyAsync(buff.data(), &host_value, sizeof(T),
                                  cudaMemcpyDefault, buff.stream());
    if (cudaSuccess != status) {
      throw std::runtime_error{"Device memcpy failed."};
    }
    status = cudaStreamSynchronize(buff.stream());
    if (cudaSuccess != status) {
      throw std::runtime_error{"Stream sync failed."};
    }
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
};

}  // namespace rmm
