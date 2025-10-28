/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/aligned.hpp>
#include <rmm/detail/aligned.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/host/host_memory_resource.hpp>

#include <cstddef>

namespace RMM_NAMESPACE {
namespace mr {
/**
 * @addtogroup host_memory_resources
 * @{
 * @file
 */

/**
 * @brief A `host_memory_resource` that uses the global `operator new` and `operator delete` to
 * allocate host memory.
 *
 * @deprecated This class is deprecated in 25.12 and will be removed in 26.02.
 */
class [[deprecated("new_delete_resource is deprecated in 25.12 and will be removed in 26.02.")]]
new_delete_resource final : public host_memory_resource {
 public:
  new_delete_resource()                           = default;
  ~new_delete_resource() override                 = default;
  new_delete_resource(new_delete_resource const&) = default;  ///< @default_copy_constructor
  new_delete_resource(new_delete_resource&&)      = default;  ///< @default_move_constructor
  new_delete_resource& operator=(new_delete_resource const&) =
    default;  ///< @default_copy_assignment{new_delete_resource}
  new_delete_resource& operator=(new_delete_resource&&) =
    default;  ///< @default_move_assignment{new_delete_resource}

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
   * @return Pointer to the newly allocated memory
   */
  void* do_allocate(std::size_t bytes,
                    std::size_t alignment = rmm::RMM_DEFAULT_HOST_ALIGNMENT) override
  {
    RMM_EXPECTS(rmm::is_supported_alignment(alignment),
                "Allocation alignment is not a power of 2.",
                rmm::bad_alloc);

    return rmm::detail::aligned_host_allocate(
      bytes, alignment, [](std::size_t size) { return ::operator new(size); });
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
  void do_deallocate(void* ptr,
                     std::size_t bytes,
                     std::size_t alignment = rmm::RMM_DEFAULT_HOST_ALIGNMENT) noexcept override
  {
    rmm::detail::aligned_host_deallocate(
      ptr, bytes, alignment, [](void* ptr) { ::operator delete(ptr); });
  }
};

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
