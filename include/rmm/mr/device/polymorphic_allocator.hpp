/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cstddef>
#include <memory>
#include <type_traits>

namespace rmm::mr {
/**
 * @addtogroup device_memory_resources
 * @{
 * @file
 */
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
  using value_type = T;  ///< T, the value type of objects allocated by this allocator
  /**
   * @brief Construct a `polymorphic_allocator` using the return value of
   * `rmm::mr::get_current_device_resource()` as the underlying memory resource.
   *
   */
  polymorphic_allocator() = default;

  /**
   * @brief Construct a `polymorphic_allocator` using the provided memory resource.
   *
   * This constructor provides an implicit conversion from `device_async_resource_ref`.
   *
   * @param mr The upstream memory resource to use for allocation.
   */
  polymorphic_allocator(device_async_resource_ref mr) : mr_{mr} {}

  /**
   * @brief Construct a `polymorphic_allocator` using the underlying memory resource of `other`.
   *
   * @param other The `polymorphic_allocator` whose memory resource will be used as the underlying
   * resource of the new `polymorphic_allocator`.
   */
  template <typename U>
  polymorphic_allocator(polymorphic_allocator<U> const& other) noexcept
    : mr_{other.get_upstream_resource()}
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
    return static_cast<value_type*>(
      get_upstream_resource().allocate_async(num * sizeof(T), stream));
  }

  /**
   * @brief Deallocates storage pointed to by `ptr`.
   *
   * `ptr` must have been allocated from a memory resource `r` that compares equal
   * to `get_upstream_resource()` using `r.allocate(n * sizeof(T))`.
   *
   * @param ptr Pointer to memory to deallocate
   * @param num Number of objects originally allocated
   * @param stream Stream on which to perform the deallocation
   */
  void deallocate(value_type* ptr, std::size_t num, cuda_stream_view stream)
  {
    get_upstream_resource().deallocate_async(ptr, num * sizeof(T), stream);
  }

  /**
   * @briefreturn{rmm::device_async_resource_ref to the upstream resource}
   */
  [[nodiscard]] rmm::device_async_resource_ref get_upstream_resource() const noexcept
  {
    return mr_;
  }

 private:
  rmm::device_async_resource_ref mr_{
    get_current_device_resource()};  ///< Underlying resource used for (de)allocation
};

/**
 * @brief Compare two `polymorphic_allocator`s for equality.
 *
 * Two `polymorphic_allocator`s are equal if their underlying memory resources compare equal.
 *
 * @tparam T Type of the first allocator
 * @tparam U Type of the second allocator
 * @param lhs The first allocator to compare
 * @param rhs The second allocator to compare
 * @return true if the two allocators are equal, false otherwise
 */
template <typename T, typename U>
bool operator==(polymorphic_allocator<T> const& lhs, polymorphic_allocator<U> const& rhs)
{
  return lhs.get_upstream_resource() == rhs.get_upstream_resource();
}

/**
 * @brief Compare two `polymorphic_allocator`s for inequality.
 *
 * Two `polymorphic_allocator`s are not equal if their underlying memory resources compare not
 * equal.
 *
 * @tparam T Type of the first allocator
 * @tparam U Type of the second allocator
 * @param lhs The first allocator to compare
 * @param rhs The second allocator to compare
 * @return true if the two allocators are not equal, false otherwise
 */
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
 *\code{.cpp}
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
  using value_type =
    typename std::allocator_traits<Allocator>::value_type;  ///< The value type of objects allocated
                                                            ///< by this allocator

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
    using other = stream_allocator_adaptor<typename std::allocator_traits<
      Allocator>::template rebind_alloc<T>>;  ///< The type to bind to
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
   * @briefreturn{The stream on which calls to the underlying allocator are made}
   */
  [[nodiscard]] cuda_stream_view stream() const noexcept { return stream_; }

  /**
   * @briefreturn{The underlying allocator}
   */
  [[nodiscard]] Allocator underlying_allocator() const noexcept { return alloc_; }

 private:
  Allocator alloc_;          ///< Underlying allocator used for (de)allocation
  cuda_stream_view stream_;  ///< Stream on which (de)allocations are performed
};

/**
 * @brief Compare two `stream_allocator_adaptor`s for equality.
 *
 * Two `stream_allocator_adaptor`s are equal if their underlying allocators compare equal.
 *
 * @tparam A Type of the first allocator
 * @tparam O Type of the second allocator
 * @param lhs The first allocator to compare
 * @param rhs The second allocator to compare
 * @return true if the two allocators are equal, false otherwise
 */
template <typename A, typename O>
bool operator==(stream_allocator_adaptor<A> const& lhs, stream_allocator_adaptor<O> const& rhs)
{
  return lhs.underlying_allocator() == rhs.underlying_allocator();
}

/**
 * @brief Compare two `stream_allocator_adaptor`s for inequality.
 *
 * Two `stream_allocator_adaptor`s are not equal if their underlying allocators compare not equal.
 *
 * @tparam A Type of the first allocator
 * @tparam O Type of the second allocator
 * @param lhs The first allocator to compare
 * @param rhs The second allocator to compare
 * @return true if the two allocators are not equal, false otherwise
 */
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
/** @} */  // end of group
}  // namespace rmm::mr
