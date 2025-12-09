/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <cuda/memory_resource>

#include <cstddef>
#include <type_traits>

namespace RMM_NAMESPACE {
namespace mr::detail {

/**
 * @brief A copyable view wrapping a `device_memory_resource*` pointer.
 *
 * This class serves as a temporary bridge to enable compatibility with CCCL 3.2's memory resource
 * design, which requires resource types to be copyable but no longer accepts raw pointers directly.
 * Since `device_memory_resource` is a virtual base class that cannot be copied, this view provides
 * a copyable wrapper around a `device_memory_resource*` pointer.
 *
 * This is an internal implementation detail and should not be used directly by users. It will be
 * removed once RMM fully migrates away from the `device_memory_resource` virtual base class.
 *
 * @note This class does NOT manage the lifetime of the wrapped pointer. The caller is responsible
 * for ensuring the pointed-to resource remains valid for the lifetime of this view.
 */
class device_memory_resource_view {
 public:
  /**
   * @brief Constructs a view wrapping the given `device_memory_resource` pointer.
   *
   * @throws rmm::logic_error if `ptr` is null
   *
   * @param ptr Non-null pointer to a `device_memory_resource`
   */
  device_memory_resource_view(device_memory_resource* ptr) : resource_ptr_{ptr}
  {
    RMM_EXPECTS(ptr != nullptr, "device_memory_resource_view cannot wrap a null pointer");
  }

  /**
   * @brief Synchronously allocates memory of size at least `bytes`.
   *
   * @param bytes The size of the allocation
   * @param alignment The alignment of the allocation
   * @return void* Pointer to the newly allocated memory
   */
  [[nodiscard]] void* allocate_sync(std::size_t bytes,
                                    std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return resource_ptr_->allocate_sync(bytes, alignment);
  }

  /**
   * @brief Synchronously deallocates memory pointed to by `ptr`.
   *
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation
   * @param alignment The alignment of the allocation
   */
  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    resource_ptr_->deallocate_sync(ptr, bytes, alignment);
  }

  /**
   * @brief Asynchronously allocates memory of size at least `bytes` on the specified stream.
   *
   * @param stream The stream on which to perform the allocation
   * @param bytes The size of the allocation
   * @param alignment The alignment of the allocation
   * @return void* Pointer to the newly allocated memory
   */
  [[nodiscard]] void* allocate(cuda_stream_view stream,
                               std::size_t bytes,
                               std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return resource_ptr_->allocate(stream, bytes, alignment);
  }

  /**
   * @brief Asynchronously deallocates memory pointed to by `ptr` on the specified stream.
   *
   * @param stream The stream on which to perform the deallocation
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation
   * @param alignment The alignment of the allocation
   */
  void deallocate(cuda_stream_view stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    resource_ptr_->deallocate(stream, ptr, bytes, alignment);
  }

  /**
   * @brief Returns the wrapped `device_memory_resource` pointer.
   *
   * @return device_memory_resource* The wrapped pointer
   */
  [[nodiscard]] device_memory_resource* get() const noexcept { return resource_ptr_; }

  /**
   * @brief Compares this view to another for equality.
   *
   * Two views are equal if they wrap pointers to resources that compare equal.
   *
   * @param other The other view to compare to
   * @return true If the wrapped resources are equivalent
   */
  [[nodiscard]] bool operator==(device_memory_resource_view const& other) const noexcept
  {
    // If both pointers are null, they're equal
    if (resource_ptr_ == nullptr && other.resource_ptr_ == nullptr) { return true; }
    // If only one is null, they're not equal
    if (resource_ptr_ == nullptr || other.resource_ptr_ == nullptr) { return false; }
    // Otherwise, compare the resources they point to
    return resource_ptr_->is_equal(*other.resource_ptr_);
  }

  /**
   * @brief Compares this view to another for inequality.
   *
   * @param other The other view to compare to
   * @return true If the wrapped resources are not equivalent
   */
  [[nodiscard]] bool operator!=(device_memory_resource_view const& other) const noexcept
  {
    return !(*this == other);
  }

  /**
   * @brief Enables the `cuda::mr::device_accessible` property
   *
   * This property declares that the wrapped `device_memory_resource` provides device accessible
   * memory.
   */
  friend void get_property(device_memory_resource_view const&, cuda::mr::device_accessible) noexcept
  {
  }

  /**
   * @brief Enables the `cuda::mr::host_accessible` property
   *
   * This property declares that the wrapped `device_memory_resource` may provide host accessible
   * memory. This is needed for resources like pinned_host_memory_resource that are both host and
   * device accessible.
   */
  friend void get_property(device_memory_resource_view const&, cuda::mr::host_accessible) noexcept
  {
  }

 private:
  device_memory_resource* resource_ptr_{nullptr};
};

// Static assertions to verify that device_memory_resource_view satisfies CCCL resource concepts
static_assert(cuda::mr::resource_with<device_memory_resource_view, cuda::mr::device_accessible>,
              "device_memory_resource_view must satisfy async resource concept");
static_assert(
  cuda::mr::synchronous_resource_with<device_memory_resource_view, cuda::mr::device_accessible>,
  "device_memory_resource_view must satisfy synchronous resource concept");

// Verify copyability - required for resource_ref construction
static_assert(cuda::std::copyable<device_memory_resource_view>,
              "device_memory_resource_view must satisfy copyable concept");
static_assert(cuda::std::copy_constructible<device_memory_resource_view>,
              "device_memory_resource_view must be copy constructible");

}  // namespace mr::detail
}  // namespace RMM_NAMESPACE
