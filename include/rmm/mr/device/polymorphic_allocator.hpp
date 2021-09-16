/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <cstddef>
#include <memory>
#include <type_traits>

namespace rmm::mr {

/**
 * @brief A stream ordered Allocator using a `rmm::mr::device_memory_resource` to satisfy
 * (de)allocations.
 *
 * Similar to `std::pmr::polymorphic_allocator`, uses the runtime polymorphism of
 * `device_memory_resource` to allow containers with `polymorphic_allocator` as their static
 * allocator type to be interoperable, but exhibit different behavior depending on resource used.
 *
 * Unlike STL allocators, `polymorphic_allocator`'s `allocate` and `deallocate` functions are stream
 * ordered. Use `stream_allocator_adaptor` to allow interoperability with interfaces that require
 * standard, non stream-ordered `Allocator` interfaces.
 *
 * @tparam T The allocators value type.
 */
template <typename T>
class polymorphic_allocator {
 public:
  using value_type = T;
  /**
   * @brief Construct a `polymorphic_allocator` using the return value of
   * `rmm::mr::get_current_device_resource()` as the underlying memory resource.
   *
   */
  polymorphic_allocator() = default;

  /**
   * @brief Construct a `polymorphic_allocator` using the provided memory resource.
   *
   * This constructor provides an implicit conversion from `memory_resource*`.
   *
   * @param mr The `device_memory_resource` to use as the underlying resource.
   */
  polymorphic_allocator(device_memory_resource* mr) : mr_{mr} {}

  /**
   * @brief Construct a `polymorphic_allocator` using `other.resource()` as the underlying memory
   * resource.
   *
   * @param other The `polymorphic_resource` whose `resource()` will be used as the underlying
   * resource of the new `polymorphic_allocator`.
   */
  template <typename U>
  polymorphic_allocator(polymorphic_allocator<U> const& other) noexcept : mr_{other.resource()}
  {
  }

  /**
   * @brief Allocates storage for `num` objects of type `T` using the underlying memory resource.
   *
   * @param num The number of objects to allocate storage for
   * @param stream The stream on which to perform the allocation
   * @return Pointer to the allocated storage
   */
  value_type* allocate(std::size_t num, cuda_stream_view stream)
  {
    return static_cast<value_type*>(resource()->allocate(num * sizeof(T), stream));
  }

  /**
   * @brief Deallocates storage pointed to by `ptr`.
   *
   * `ptr` must have been allocated from a `rmm::mr::device_memory_resource` `r` that compares equal
   * to `*resource()` using `r.allocate(n * sizeof(T))`.
   *
   * @param ptr Pointer to memory to deallocate
   * @param num Number of objects originally allocated
   * @param stream Stream on which to perform the deallocation
   */
  void deallocate(value_type* ptr, std::size_t num, cuda_stream_view stream)
  {
    resource()->deallocate(ptr, num * sizeof(T), stream);
  }

  /**
   * @brief Returns pointer to the underlying `rmm::mr::device_memory_resource`.
   *
   * @return Pointer to the underlying resource.
   */
  [[nodiscard]] device_memory_resource* resource() const noexcept { return mr_; }

 private:
  device_memory_resource* mr_{
    get_current_device_resource()};  ///< Underlying resource used for (de)allocation
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

/**
 * @brief Adapts a stream ordered allocator to provide a standard `Allocator` interface
 *
 * A stream-ordered allocator (i.e., `allocate/deallocate` use a `cuda_stream_view`) cannot be used
 * in an interface that expects a standard C++ `Allocator` interface. `stream_allocator_adaptor`
 * wraps a stream-ordered allocator and a stream to provide a standard `Allocator` interface. The
 * adaptor uses the wrapped stream in calls to the underlying allocator's `allocate` and
 *`deallocate` functions.
 *
 * Example:
 *\code{c++}
 * my_stream_ordered_allocator<int> a{...};
 * cuda_stream_view s = // create stream;
 *
 * auto adapted = make_stream_allocator_adaptor(a, s);
 *
 * // Allocates storage for `n` int's on stream `s`
 * int * p = std::allocator_traits<decltype(adapted)>::allocate(adapted, n);
 *\endcode
 *
 * @tparam Allocator Stream ordered allocator type to adapt
 */
template <typename Allocator>
class stream_allocator_adaptor {
 public:
  using value_type = typename std::allocator_traits<Allocator>::value_type;

  stream_allocator_adaptor() = delete;

  /**
   * @brief Construct a `stream_allocator_adaptor` using `a` as the underlying allocator.
   *
   * @note: The `stream` must not be destroyed before the `stream_allocator_adaptor`, otherwise
   * behavior is undefined.
   *
   * @param allocator The stream ordered allocator to use as the underlying allocator
   * @param stream The stream used with the underlying allocator
   */
  stream_allocator_adaptor(Allocator const& allocator, cuda_stream_view stream)
    : alloc_{allocator}, stream_{stream}
  {
  }

  /**
   * @brief Construct a `stream_allocator_adaptor` using `other.underlying_allocator()` and
   * `other.stream()` as the underlying allocator and stream.
   *
   * @tparam OtherAllocator Type of `other`'s underlying allocator
   * @param other The other `stream_allocator_adaptor` whose underlying allocator and stream will be
   * copied
   */
  template <typename OtherAllocator>
  stream_allocator_adaptor(stream_allocator_adaptor<OtherAllocator> const& other)
    : stream_allocator_adaptor{other.underlying_allocator(), other.stream()}
  {
  }

  /**
   * @brief Rebinds the allocator to the specified type.
   *
   * @tparam T The desired `value_type` of the rebound allocator type
   */
  template <typename T>
  struct rebind {
    using other =
      stream_allocator_adaptor<typename std::allocator_traits<Allocator>::template rebind_alloc<T>>;
  };

  /**
   * @brief Allocates storage for `num` objects of type `T` using the underlying allocator on
   * `stream()`.
   *
   * @param num The number of objects to allocate storage for
   * @return Pointer to the allocated storage
   */
  value_type* allocate(std::size_t num) { return alloc_.allocate(num, stream()); }

  /**
   * @brief Deallocates storage pointed to by `ptr` using the underlying allocator on `stream()`.
   *
   * `ptr` must have been allocated from by an allocator `a` that compares equal to
   * `underlying_allocator()` using `a.allocate(n)`.
   *
   * @param ptr Pointer to memory to deallocate
   * @param num Number of objects originally allocated
   */
  void deallocate(value_type* ptr, std::size_t num) { alloc_.deallocate(ptr, num, stream()); }

  /**
   * @brief Returns the underlying stream on which calls to the underlying allocator are made.
   *
   */
  [[nodiscard]] cuda_stream_view stream() const noexcept { return stream_; }

  /**
   * @brief Returns the underlying stream-ordered allocator
   *
   */
  [[nodiscard]] Allocator underlying_allocator() const noexcept { return alloc_; }

 private:
  Allocator alloc_;          ///< Underlying allocator used for (de)allocation
  cuda_stream_view stream_;  ///< Stream on which (de)allocations are performed
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

/**
 * @brief Factory to construct a `stream_allocator_adaptor` from an existing stream-ordered
 * allocator.
 *
 * @tparam Allocator Type of the stream-ordered allocator
 * @param allocator The allocator to use as the underlying allocator of the
 * `stream_allocator_adaptor`
 * @param stream The stream on which the `stream_allocator_adaptor` will perform (de)allocations
 * @return A `stream_allocator_adaptor` wrapping `allocator` and `s`
 */
template <typename Allocator>
auto make_stream_allocator_adaptor(Allocator const& allocator, cuda_stream_view stream)
{
  return stream_allocator_adaptor<Allocator>{allocator, stream};
}

}  // namespace rmm::mr
