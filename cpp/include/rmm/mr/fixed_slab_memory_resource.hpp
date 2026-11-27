/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/detail/export.hpp>
#include <rmm/mr/detail/fixed_slab_memory_resource_impl.hpp>

#include <cuda/memory_resource>
#include <cuda/stream>

#include <cstddef>

RMM_NAMESPACE_BEGIN
namespace mr {
/**
 * @addtogroup memory_resources
 * @{
 * @file
 */

/**
 * @brief A memory resource that carves allocations from a single fixed-size slab.
 *
 * **Single-stream contract:** The slab is stream-ordered on its construction stream, and every
 * `allocate`/`deallocate` call must pass that recorded stream (`allocate` throws `rmm::logic_error`
 * on mismatch; `noexcept` `deallocate` terminates fail-closed; null stream references normalize to
 * `cudaStreamLegacy`, while `cudaStreamPerThread` remains distinct). No cross-stream ordering is
 * provided. Cross-stream use would require per-stream tracking (a stream registry), PTDS ownership
 * checks, and per-call event waits/teardown fencing. These are not independent features: adding
 * any one means adding all of them, so none can be added piecemeal.
 *
 * **Zero-CUDA-call hot path and fixed capacity:** Construction makes one upstream allocation of
 * `slab_bytes` on the recorded stream. `allocate` carves aligned slices in O(1) via host pointer
 * arithmetic with zero CUDA calls; requests exceeding remaining capacity (including alignment
 * padding) throw
 * `rmm::out_of_memory`, and the resource never grows. `allocate_sync` is a host-only carve that
 * performs no wait: the caller must synchronize the recorded stream before host-synchronous use.
 *
 * **Lifetime, ownership, and thread safety:** Derives `cuda::mr::shared_resource` so refcounted
 * copies and carved buffers holding owning handles share one slab and keep it alive after the
 * constructing scope dies. A bare non-owning view (`rmm::device_async_resource_ref`) must never
 * outlive the last owner. `deallocate` is a zero-CUDA-call memory no-op because lifetime belongs
 * to the refcounted resource rather than individual slices; the slab is freed once at last-owner
 * destruction, stream-ordered on the recorded stream (not necessarily complete on return; upstream
 * owns completion). That stream must outlive the last owner on the same active CUDA
 * device/context; stream/graph capture is not supported. Shared ownership is not thread safety:
 * carves are unsynchronized (wrap in `thread_safe_resource_adaptor` for concurrent access).
 * Intended for single-stream producers that know all buffer sizes up front, not general-purpose
 * allocation.
 */
class RMM_EXPORT fixed_slab_memory_resource
  : public cuda::mr::shared_resource<detail::fixed_slab_memory_resource_impl> {
  using shared_base = cuda::mr::shared_resource<detail::fixed_slab_memory_resource_impl>;

 public:
  /**
   * @brief Construct the slab: one upstream allocation of `slab_bytes` on `stream`.
   *
   * @param upstream The upstream resource from which the slab is allocated exactly once.
   * @param stream The stream on which the slab allocation is ordered.
   *   When constructed on the per-thread default stream (`cudaStreamPerThread`), use the
   *   resource only from that thread: the per-thread stream handle is a sentinel shared by
   *   every thread, so the stream check cannot distinguish another thread's stream.
   * @param slab_bytes Exact total size of the slab; allocations are carved from it.
   *
   * @throws rmm::logic_error if `slab_bytes` is zero or not a multiple of
   *   `rmm::CUDA_ALLOCATION_ALIGNMENT` (256 bytes).
   * @throws rmm::out_of_memory if `upstream` throws `rmm::out_of_memory` or returns `nullptr`.
   */
  fixed_slab_memory_resource(cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
                             cuda::stream_ref stream,
                             std::size_t slab_bytes);

  /**
   * @brief O(1) carve of an aligned slice from the slab on the recorded stream.
   *
   * The returned slice is aligned to `std::max(alignment, rmm::CUDA_ALLOCATION_ALIGNMENT)`
   * and is ordered on the recorded stream behind all prior slab work. Pure host
   * arithmetic: no CUDA calls are made. If `bytes == 0`, returns `nullptr` without
   * validating `alignment` or advancing the cursor.
   *
   * @param stream The recorded stream the slab was allocated on; any other stream throws.
   * @param bytes Size in bytes of the requested slice.
   * @param alignment Alignment in bytes of the requested slice (must be a power of two).
   * @return void* Pointer to the carved slice, or `nullptr` if `bytes == 0`.
   *
   * @throws rmm::logic_error if `stream` differs from the recorded stream, or if
   * `bytes > 0` and `alignment` is zero or not a power of two.
   * @throws rmm::out_of_memory if fewer than `bytes` (plus alignment padding) remain; the
   * cursor is unchanged on failure, so a subsequent smaller request still succeeds.
   */
  [[nodiscard]] void* allocate(cuda::stream_ref stream,
                               std::size_t bytes,
                               std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT);

  /**
   * @brief No-op on the memory. Carved memory is owned by the slab and released only with
   * the resource. No CUDA calls are made.
   *
   * The stream is validated against the recorded stream. Calls with `ptr == nullptr` or
   * `bytes == 0` are ignored without validation.
   *
   * This function is `noexcept`, so a mismatching stream cannot throw: it terminates the
   * process (fail-closed), consistent with the throwing paths' logic_error contract.
   *
   * @param stream The recorded stream the slab was allocated on; any other stream terminates.
   * @param ptr Pointer to the carved slice (unused except for the null check).
   * @param bytes Size in bytes of the carved slice (unused except for the zero check).
   * @param alignment Alignment of the carved slice (unused).
   */
  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept;

  /**
   * @brief Host-only carve: pure bump arithmetic, NO wait is performed.
   *
   * The caller must guarantee the recorded stream's queued work (including the slab
   * allocation itself) has completed before host-synchronous use of the returned slice,
   * e.g. by synchronizing the recorded stream the resource was constructed with. If `bytes == 0`,
   * returns `nullptr` without advancing the cursor.
   *
   * @param bytes Size in bytes of the requested slice.
   * @param alignment Alignment in bytes of the requested slice (must be a power of two).
   * @return void* Pointer to the carved slice, or `nullptr` if `bytes == 0`.
   *
   * @throws rmm::logic_error if `bytes > 0` and `alignment` is zero or not a power of two.
   * @throws rmm::out_of_memory if fewer than `bytes` (plus alignment padding) remain.
   */
  [[nodiscard]] void* allocate_sync(std::size_t bytes,
                                    std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT);

  /**
   * @brief Synchronous no-op deallocation: no stream work is issued and no validation
   * is performed, so there is nothing to complete before returning.
   *
   * @param ptr Unused.
   * @param bytes Unused.
   * @param alignment Unused.
   */
  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept;

  /**
   * @brief Total slab capacity in bytes.
   *
   * @return std::size_t Total size in bytes of the slab allocated at construction.
   */
  [[nodiscard]] std::size_t slab_bytes() const noexcept;

  /**
   * @brief Base pointer of the slab allocated at construction.
   *
   * The pointer is valid while any refcounted owner of the resource is alive (the
   * refcount guarantees the upstream allocation outlives the last handle). Memory in the
   * slab may only be accessed on the recorded stream (or after that stream's queued work
   * has completed). This accessor never waits.
   *
   * @return void* Non-null base pointer to the underlying slab allocation.
   */
  [[nodiscard]] void* data() const noexcept;

  /**
   * @brief Bytes carved so far, including alignment padding but not the prospective padding
   * of failed requests.
   *
   * @return std::size_t Number of slab bytes committed so far.
   */
  [[nodiscard]] std::size_t carved_bytes() const noexcept;

  /**
   * @brief Enables the `cuda::mr::device_accessible` property.
   */
  RMM_CONSTEXPR_FRIEND void get_property(fixed_slab_memory_resource const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

  ~fixed_slab_memory_resource() = default;
  fixed_slab_memory_resource(fixed_slab_memory_resource const&) =
    default;  ///< @default_copy_constructor
  fixed_slab_memory_resource(fixed_slab_memory_resource&&) =
    default;  ///< @default_move_constructor
  fixed_slab_memory_resource& operator=(fixed_slab_memory_resource const&) =
    default;  ///< @default_copy_assignment{fixed_slab_memory_resource}
  fixed_slab_memory_resource& operator=(fixed_slab_memory_resource&&) =
    default;  ///< @default_move_assignment{fixed_slab_memory_resource}
};

// static property checks
static_assert(cuda::mr::synchronous_resource<fixed_slab_memory_resource>);
static_assert(cuda::mr::resource<fixed_slab_memory_resource>);
static_assert(
  cuda::mr::synchronous_resource_with<fixed_slab_memory_resource, cuda::mr::device_accessible>);
static_assert(cuda::mr::resource_with<fixed_slab_memory_resource, cuda::mr::device_accessible>);

/** @} */  // end of group
}  // namespace mr
RMM_NAMESPACE_END
