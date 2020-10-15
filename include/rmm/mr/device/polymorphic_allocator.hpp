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

#include <memory>
#include <type_traits>

#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include "rmm/mr/device/device_memory_resource.hpp"

namespace rmm {
namespace mr {

/**
 * @brief A stream ordered Allocator using a `rmm::mr::device_memory_resource` to satisfy
 * (de)allocations.
 *
 * Similar to `std::pmr::polymorphic_allocator`, uses the runtime polymorphism of
 * `device_memory_resource` to allow containers with `polymorphic_allocator` as their static
 * allocator type to be interoperable, but exhibit different behavior depending on resource used.
 *
 * Unlike STL allocators, `polymorphic_allocator`'s `allocate` and `deallocate` functions are stream
 * ordered. To allow interoperability with interfaces that do not
 *
 * @tparam T The allocators value type.
 */
template <typename T>
class polymorphic_allocator {
 public:
  using value_type = T;

  polymorphic_allocator()                                = default;
  polymorphic_allocator(polymorphic_allocator<T> const&) = default;

  polymorphic_allocator(device_memory_resource* mr) : mr_{mr} {}

  template <typename U>
  polymorphic_allocator(polymorphic_allocator<U> const& other) noexcept : mr_{other.mr_}
  {
  }

  value_type* allocate(std::size_t n, cudaStream_t stream)
  {
    resource()->allocate(n * sizeof(T), stream);
  }

  void deallocate(value_type* p, std::size_t n, cudaStream_t stream)
  {
    resource()->deallocate(p, n * sizeof(T), stream);
  }

  device_memory_resource* resource() const noexcept { return mr_; }

 private:
  device_memory_resource* mr_{get_current_device_resource()};
};

template <typename T, typename U>
bool operator==(polymorphic_allocator<T> const& lhs, polymorphic_allocator<U> const& rhs)
{
  return lhs.resource()->is_equal(*rhs.resource());
}

template <typename T, typename U>
bool operator!=(polymorphic_allocator<T> const& lhs, polymorphic_allocator<U> const& rhs)
{
  return not(lhs == rhs);
}
