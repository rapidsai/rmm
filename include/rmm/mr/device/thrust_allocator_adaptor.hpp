/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <rmm/cuda_device.hpp>
#include <rmm/detail/thrust_namespace.h>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>
#include <thrust/device_malloc_allocator.h>
#include <thrust/device_ptr.h>
#include <thrust/memory.h>

namespace rmm::mr {
/**
 * @addtogroup device_resource_adaptors
 * @{
 * @file
 */
/**
 * @brief An `allocator` compatible with Thrust containers and algorithms using
 * a `device_async_resource_ref` for memory (de)allocation.
 *
 * Unlike a `device_async_resource_ref`, `thrust_allocator` is typed and bound to
 * allocate objects of a specific type `T`, but can be freely rebound to other
 * types.
 *
 * The allocator records the current cuda device and may only be used with a backing
 * `device_async_resource_ref` valid for the same device.
 *
 * @tparam T The type of the objects that will be allocated by this allocator
 */
template <typename T>
class thrust_allocator : public thrust::device_malloc_allocator<T> {
 public:
  using Base      = thrust::device_malloc_allocator<T>;  ///< The base type of this allocator
  using pointer   = typename Base::pointer;              ///< The pointer type
  using size_type = typename Base::size_type;            ///< The size type

  /**
   * @brief Provides the type of a `thrust_allocator` instantiated with another
   * type.
   *
   * @tparam U the other type to use for instantiation
   */
  template <typename U>
  struct rebind {
    using other = thrust_allocator<U>;  ///< The type to bind to
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
  thrust_allocator(cuda_stream_view stream, rmm::device_async_resource_ref mr)
    : _stream{stream}, _mr(mr)
  {
  }

  /**
   * @brief Copy constructor. Copies the resource pointer and stream.
   *
   * @param other The `thrust_allocator` to copy
   */
  template <typename U>
  thrust_allocator(thrust_allocator<U> const& other)
    : _mr(other.resource()), _stream{other.stream()}, _device{other._device}
  {
  }

  /**
   * @brief Allocate objects of type `T`
   *
   * @param num  The number of elements of type `T` to allocate
   * @return pointer Pointer to the newly allocated storage
   */
  pointer allocate(size_type num)
  {
    cuda_set_device_raii dev{_device};
    return thrust::device_pointer_cast(
      static_cast<T*>(_mr.allocate_async(num * sizeof(T), _stream)));
  }

  /**
   * @brief Deallocates objects of type `T`
   *
   * @param ptr Pointer returned by a previous call to `allocate`
   * @param num number of elements, *must* be equal to the argument passed to the
   * prior `allocate` call that produced `p`
   */
  void deallocate(pointer ptr, size_type num)
  {
    cuda_set_device_raii dev{_device};
    return _mr.deallocate_async(thrust::raw_pointer_cast(ptr), num * sizeof(T), _stream);
  }

  /**
   * @briefreturn{rmm::device_async_resource_ref to the upstream resource}
   */
  [[nodiscard]] rmm::device_async_resource_ref get_upstream_resource() const noexcept
  {
    return _mr;
  }

  /**
   * @briefreturn{The stream used by this allocator}
   */
  [[nodiscard]] cuda_stream_view stream() const noexcept { return _stream; }

  /**
   * @brief Enables the `cuda::mr::device_accessible` property
   *
   * This property declares that a `thrust_allocator` provides device accessible memory
   */
  friend void get_property(thrust_allocator const&, cuda::mr::device_accessible) noexcept {}

 private:
  cuda_stream_view _stream{};
  rmm::device_async_resource_ref _mr{rmm::mr::get_current_device_resource()};
  cuda_device_id _device{get_current_cuda_device()};
};
/** @} */  // end of group
}  // namespace rmm::mr
