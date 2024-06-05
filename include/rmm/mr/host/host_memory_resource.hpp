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

#include <rmm/detail/nvtx/ranges.hpp>

#include <cuda/memory_resource>

#include <cstddef>
#include <utility>

namespace rmm::mr {
/**
 * @addtogroup host_memory_resources
 * @{
 * @file
 */

/**
 * @brief Base class for host memory allocation.
 *
 * This is based on `std::pmr::memory_resource`:
 * https://en.cppreference.com/w/cpp/memory/memory_resource
 *
 * When C++17 is available for use in RMM, `rmm::host_memory_resource` should
 * inherit from `std::pmr::memory_resource`.
 *
 * This class serves as the interface that all host memory resource
 * implementations must satisfy.
 *
 * There are two private, pure virtual functions that all derived classes must
 * implement: `do_allocate` and `do_deallocate`. Optionally, derived classes may
 * also override `is_equal`. By default, `is_equal` simply performs an identity
 * comparison.
 *
 * The public, non-virtual functions `allocate`, `deallocate`, and `is_equal`
 * simply call the private virtual functions. The reason for this is to allow
 * implementing shared, default behavior in the base class. For example, the
 * base class' `allocate` function may log every allocation, no matter what
 * derived class implementation is used.
 *
 */
class host_memory_resource {
 public:
  host_memory_resource()                                = default;
  virtual ~host_memory_resource()                       = default;
  host_memory_resource(host_memory_resource const&)     = default;  ///< @default_copy_constructor
  host_memory_resource(host_memory_resource&&) noexcept = default;  ///< @default_move_constructor
  host_memory_resource& operator=(host_memory_resource const&) =
    default;  ///< @default_copy_assignment{host_memory_resource}
  host_memory_resource& operator=(host_memory_resource&&) noexcept =
    default;  ///< @default_move_assignment{host_memory_resource}

  /**
   * @brief Allocates memory on the host of size at least `bytes` bytes.
   *
   * The returned storage is aligned to the specified `alignment` if supported, and to
   * `alignof(std::max_align_t)` otherwise.
   *
   * @throws std::bad_alloc When the requested `bytes` and `alignment` cannot be allocated.
   *
   * @param bytes The size of the allocation
   * @param alignment Alignment of the allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* allocate(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
  {
    RMM_FUNC_RANGE();
    return do_allocate(bytes, alignment);
  }

  /**
   * @brief Deallocate memory pointed to by `ptr`.
   *
   * `ptr` must have been returned by a prior call to `allocate(bytes,alignment)` on a
   * `host_memory_resource` that compares equal to `*this`, and the storage it points to must not
   * yet have been deallocated, otherwise behavior is undefined.
   *
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the value of `bytes`
   *              that was passed to the `allocate` call that returned `ptr`.
   * @param alignment Alignment of the allocation. This must be equal to the value of `alignment`
   *                  that was passed to the `allocate` call that returned `ptr`.
   */
  void deallocate(void* ptr, std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
  {
    RMM_FUNC_RANGE();
    do_deallocate(ptr, bytes, alignment);
  }

  /**
   * @brief Compare this resource to another.
   *
   * Two `host_memory_resource`s compare equal if and only if memory allocated from one
   * `host_memory_resource` can be deallocated from the other and vice versa.
   *
   * By default, simply checks if \p *this and \p other refer to the same object, i.e., does not
   * check if they are two objects of the same class.
   *
   * @param other The other resource to compare to
   * @returns true if the two resources are equivalent
   */
  [[nodiscard]] bool is_equal(host_memory_resource const& other) const noexcept
  {
    return do_is_equal(other);
  }

  /**
   * @brief Comparison operator with another device_memory_resource
   *
   * @param other The other resource to compare to
   * @return true If the two resources are equivalent
   * @return false If the two resources are not equivalent
   */
  [[nodiscard]] bool operator==(host_memory_resource const& other) const noexcept
  {
    return do_is_equal(other);
  }

  /**
   * @brief Comparison operator with another device_memory_resource
   *
   * @param other The other resource to compare to
   * @return false If the two resources are equivalent
   * @return true If the two resources are not equivalent
   */
  [[nodiscard]] bool operator!=(host_memory_resource const& other) const noexcept
  {
    return !do_is_equal(other);
  }

  /**
   * @brief Enables the `cuda::mr::host_accessible` property
   *
   * This property declares that a `host_memory_resource` provides host accessible memory
   */
  friend void get_property(host_memory_resource const&, cuda::mr::host_accessible) noexcept {}

 private:
  /**
   * @brief Allocates memory on the host of size at least `bytes` bytes.
   *
   * The returned storage is aligned to the specified `alignment` if supported, and to
   * `alignof(std::max_align_t)` otherwise.
   *
   * @throws std::bad_alloc When the requested `bytes` and `alignment` cannot be allocated.
   *
   * @param bytes The size of the allocation
   * @param alignment Alignment of the allocation
   * @return void* Pointer to the newly allocated memory
   */
  virtual void* do_allocate(std::size_t bytes,
                            std::size_t alignment = alignof(std::max_align_t)) = 0;

  /**
   * @brief Deallocate memory pointed to by `ptr`.
   *
   * `ptr` must have been returned by a prior call to `allocate(bytes,alignment)` on a
   * `host_memory_resource` that compares equal to `*this`, and the storage it points to must not
   * yet have been deallocated, otherwise behavior is undefined.
   *
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the value of `bytes`
   *              that was passed to the `allocate` call that returned `ptr`.
   * @param alignment Alignment of the allocation. This must be equal to the value of `alignment`
   *                  that was passed to the `allocate` call that returned `ptr`.
   */
  virtual void do_deallocate(void* ptr,
                             std::size_t bytes,
                             std::size_t alignment = alignof(std::max_align_t)) = 0;

  /**
   * @brief Compare this resource to another.
   *
   * Two host_memory_resources compare equal if and only if memory allocated from one
   * host_memory_resource can be deallocated from the other and vice versa.
   *
   * By default, simply checks if `*this` and `other` refer to the same object, i.e., does not check
   * whether they are two objects of the same class.
   *
   * @param other The other resource to compare to
   * @return true If the two resources are equivalent
   */
  [[nodiscard]] virtual bool do_is_equal(host_memory_resource const& other) const noexcept
  {
    return this == &other;
  }
};
static_assert(cuda::mr::resource_with<host_memory_resource, cuda::mr::host_accessible>);
/** @} */  // end of group

}  // namespace rmm::mr
