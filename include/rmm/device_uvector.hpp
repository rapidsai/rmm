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

namespace rmm {

template <typename T>
class device_uvector {
  static_assert(std::is_trivially_copyable<T>::value,
                "device_uvector only supports types that are trivially copyable.");

 public:
  using value_type     = T;
  using size_type      = std::size_t;
  using reference      = value_type&;
  using pointer        = value_type*;
  using const_pointer  = value_type const*;
  using iterator       = pointer;
  using const_iterator = const_pointer;

  device_uvector()                      = default;
  ~device_uvector()                     = default;
  device_uvector(device_uvector const&) = default;
  device_uvector(device_uvector&&)      = default;
  device_uvector& operator=(device_uvector const&) = default;
  device_uvector& operator=(device_uvector&&) = default;

  explicit device_uvector(device_uvector const& other,
                          cudaStream_t stream                 = 0,
                          rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource())
    : _storage{other.storage, stream, mr}, _size{other.size()} {}

  explicit device_uvector(std::size_t size,
                          cudaStream_t stream                 = 0,
                          rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource())
    : _storage{size * sizeof(T), stream, mr}, _size{size} {}

  pointer
  data() noexcept {
    return static_cast<pointer>(_storage.data());
  }

  const_pointer
  data() const noexcept {
    return static_cast<const_pointer>(_storage.data());
  }

  iterator
  begin() noexcept {
    return data();
  }

  const_iterator
  begin() const noexcept {
    return data();
  }

  iterator
  end() noexcept {
    return data() + size();
  }

  const_iterator
  end() const noexcept {
    return data() + size();
  }

  std::size_t
  size() const noexcept {
    return _size;
  }

  bool
  is_empty() const noexcept {
    return size() == 0;
  }

 private:
  device_buffer _storage{};
  std::size_t _size{};
};

}  // namespace rmm
