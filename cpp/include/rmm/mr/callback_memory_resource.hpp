/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/detail/export.hpp>
#include <rmm/mr/detail/callback_memory_resource_impl.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <cstddef>
#include <functional>

namespace RMM_NAMESPACE {
namespace mr {
/**
 * @addtogroup memory_resources
 * @{
 */

/**
 * @brief Callback function type used by callback memory resource for allocation.
 *
 * The signature of the callback function is:
 *   `void* allocate_callback_t(std::size_t bytes, cuda_stream_view stream, void* arg);`
 *
 * * Returns a pointer to an allocation of at least `bytes` usable immediately on
 *   `stream`. The stream-ordered behavior requirements are identical to
 *   `allocate`.
 *
 * * The `arg` is provided to the constructor of the `callback_memory_resource`
 *   and will be forwarded along to every invocation of the callback function.
 */
using allocate_callback_t = std::function<void*(std::size_t, cuda_stream_view, void*)>;

/**
 * @brief Callback function type used by callback_memory_resource for deallocation.
 *
 * The signature of the callback function is:
 *   `void deallocate_callback_t(void* ptr, std::size_t bytes, cuda_stream_view stream, void* arg);`
 *
 * * Deallocates memory pointed to by `ptr`. `bytes` specifies the size of the allocation
 *   in bytes, and must equal the value of `bytes` that was passed to the allocate callback
 *   function. The stream-ordered behavior requirements are identical to
 *   `deallocate`.
 *
 * * The `arg` is provided to the constructor of the `callback_memory_resource`
 *   and will be forwarded along to every invocation of the callback function.
 */
using deallocate_callback_t = std::function<void(void*, std::size_t, cuda_stream_view, void*)>;

namespace detail {
class callback_memory_resource_impl;
}

/**
 * @brief A device memory resource that uses the provided callbacks for memory allocation
 * and deallocation.
 *
 * This class is copyable and shares ownership of its internal state via
 * `cuda::mr::shared_resource`.
 */
class RMM_EXPORT callback_memory_resource
  : public cuda::mr::shared_resource<detail::callback_memory_resource_impl> {
  using shared_base = cuda::mr::shared_resource<detail::callback_memory_resource_impl>;

 public:
  /**
   * @brief Enables the `cuda::mr::device_accessible` property
   */
  RMM_CONSTEXPR_FRIEND void get_property(callback_memory_resource const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

  /**
   * @brief Construct a new callback memory resource.
   *
   * Constructs a callback memory resource that uses the user-provided callbacks
   * `allocate_callback` for allocation and `deallocate_callback` for deallocation.
   *
   * @param allocate_callback The callback function used for allocation
   * @param deallocate_callback The callback function used for deallocation
   * @param allocate_callback_arg Additional context passed to `allocate_callback`.
   * It is the caller's responsibility to maintain the lifetime of the pointed-to data
   * for the duration of the lifetime of the `callback_memory_resource`.
   * @param deallocate_callback_arg Additional context passed to `deallocate_callback`.
   * It is the caller's responsibility to maintain the lifetime of the pointed-to data
   * for the duration of the lifetime of the `callback_memory_resource`.
   */
  callback_memory_resource(allocate_callback_t allocate_callback,
                           deallocate_callback_t deallocate_callback,
                           void* allocate_callback_arg   = nullptr,
                           void* deallocate_callback_arg = nullptr);

  callback_memory_resource()  = delete;
  ~callback_memory_resource() = default;
};

static_assert(cuda::mr::resource_with<callback_memory_resource, cuda::mr::device_accessible>,
              "callback_memory_resource does not satisfy the cuda::mr::resource concept");

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
