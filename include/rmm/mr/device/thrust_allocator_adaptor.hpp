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

#include <thrust/detail/type_traits/pointer_traits.h>
#include <thrust/device_malloc_allocator.h>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

namespace rmm {
namespace mr {
/**
 * @brief An `allocator` compatible with Thrust containers and algorithms using
 * a `device_memory_resource` for memory (de)allocation.
 *
 * Unlike a `device_memory_resource`, `thrust_allocator` is typed and bound to
 * allocate objects of a specific type `T`, but can be freely rebound to other
 * types.
 *
 * @tparam T The type of the objects that will be allocated by this allocator
 */
template <typename T>
class thrust_allocator : public thrust::device_malloc_allocator<T> {
 public:
  using Base      = thrust::device_malloc_allocator<T>;
  using pointer   = typename Base::pointer;
  using size_type = typename Base::size_type;

  /**
   * @brief Provides the type of a `thrust_allocator` instantiated with another
   * type.
   *
   * @tparam U the other type to use for instantiation
   */
  template <typename U>
  struct rebind {
    using other = thrust_allocator<U>;
  };

  /**
   * @brief Default constructor creates an allocator using the default memory
   * resource and default stream.
   */
  thrust_allocator() = default;

  /**
   * @brief Constructs a `thrust_allocator` using the default device memory
   * resource and specified stream.
   *
   * @param stream The stream to be used for device memory (de)allocation
   */
  explicit thrust_allocator(cuda_stream_view stream) : _stream{stream} {}

  /**
   * @brief Constructs a `thrust_allocator` using a device memory resource and
   * stream.
   *
   * @param mr The resource to be used for device memory allocation
   * @param stream The stream to be used for device memory (de)allocation
   */
  thrust_allocator(cuda_stream_view stream, device_memory_resource* mr) : _stream{stream}, _mr(mr)
  {
  }

  /**
   * @brief Copy constructor. Copies the resource pointer and stream.
   *
   * @param other The `thrust_allocator` to copy
   */
  template <typename U>
  thrust_allocator(thrust_allocator<U> const& other)
    : _mr(other.resource()), _stream{other.stream()}
  {
  }

  /**
   * @brief Allocate objects of type `T`
   *
   * @param n  The number of elements of type `T` to allocate
   * @return pointer Pointer to the newly allocated storage
   */
  pointer allocate(size_type n)
  {
    return thrust::device_pointer_cast(static_cast<T*>(_mr->allocate_async(n * sizeof(T), _stream)));
  }

  /**
   * @brief Deallocates objects of type `T`
   *
   * @param p Pointer returned by a previous call to `allocate`
   * @param n number of elements, *must* be equal to the argument passed to the
   * prior `allocate` call that produced `p`
   */
  void deallocate(pointer p, size_type n)
  {
    return _mr->deallocate_async(thrust::raw_pointer_cast(p), n * sizeof(T), _stream);
  }

  /**
   * @brief Returns the device memory resource used by this allocator.
   */
  device_memory_resource* resource() const noexcept { return _mr; }

  /**
   * @brief Returns the stream used by this allocator.
   */
  cuda_stream_view stream() const noexcept { return _stream; }

 private:
  cuda_stream_view _stream{};
  device_memory_resource* _mr{rmm::mr::get_current_device_resource()};
};
}  // namespace mr
}  // namespace rmm
