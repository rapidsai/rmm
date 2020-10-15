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

template <typename Allocator>
class stream_allocator_adaptor {
 public:
  using value_type = typename std::allocator_traits<Allocator>::value_type;

  stream_allocator_adaptor() = delete;

  stream_allocator_adaptor(Allocator const& a, cudaStream_t stream) : alloc_{a}, stream_{stream} {}

  template <typename OtherAllocator>
  stream_allocator_adaptor(stream_allocator_adaptor<OtherAllocator> const& other)
    : stream_allocator_adaptor{other.underlying_allocator(), other.stream()}
  {
  }

  template <typename T>
  struct rebind {
    using other =
      stream_allocator_adaptor<typename std::allocator_traits<Allocator>::template rebind_alloc<T>>;
  };

  value_type* allocate(std::size_t n) { return alloc_.allocate(n, stream()); }

  void deallocate(value_type* p, std::size_t n) { alloc_.deallocate(p, n, stream()); }

  cudaStream_t stream() const noexcept { return stream_; }

  Allocator underlying_allocator() const noexcept { return alloc_; }

 private:
  Allocator alloc_;
  cudaStream_t stream_;
};

template <typename A, typename O>
bool operator==(stream_allocator_adaptor<A> const& lhs, stream_allocator_adaptor<O> const& rhs)
{
  return lhs.underlying_allocator() == rhs.underlying_allocator();
}

template <typename A, typename O>
bool operator!=(stream_allocator_adaptor<A> const& lhs, stream_allocator_adaptor<O> const& rhs)
{
  return not(lhs == rhs);
}

template <typename Allocator>
auto make_stream_allocator_adaptor(Allocator const& allocator, cudaStream_t s){
    return stream_allocator_adaptor<Allocator>{allocator,s};
}

}  // namespace mr
}  // namespace rmm
