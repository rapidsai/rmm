/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/aligned.hpp>
#include <rmm/detail/export.hpp>

#include <cuda/memory_resource>
#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>

RMM_NAMESPACE_BEGIN
namespace mr {
namespace detail {

/**
 * @brief Implementation class for fixed_slab_memory_resource.
 *
 * Owns one upstream allocation of an exact, caller-specified size (the "slab") and carves
 * aligned slices from it by advancing a byte offset. Carved memory is never reused or
 * returned individually: `deallocate` performs no memory operation, and the whole slab is
 * returned to the upstream only when the last owner of the implementation is destroyed. The
 * implementation never grows; an allocation that does not fit throws `rmm::out_of_memory`.
 *
 * Single-stream design: the slab is allocated on the construction stream, and every
 * `allocate`/`deallocate` call must pass that same stream (`allocate` throws `rmm::logic_error` on
 * mismatch; `noexcept` `deallocate` terminates fail-closed). Because carving is pointer arithmetic
 * and all slab work stays on one stream, no per-call CUDA calls are needed for ordering.
 *
 * Design intent: cheap bulk carves for single-stream producers that know all sizes up front.
 * Cross-stream use is not part of this design — it would require per-stream tracking (a
 * stream registry), PTDS ownership checks, and per-call event waits/teardown fencing, which
 * are coupled features that cannot be added piecemeal. `deallocate` is a no-op because
 * lifetime belongs to the refcounted resource, not individual slices.
 *
 * The construction stream must still be live when the last owner is destroyed; the slab is
 * freed stream-ordered on it. No thread safety is provided; shared ownership is not
 * synchronization of mutable state.
 *
 * This class satisfies the CCCL `cuda::mr::resource` concept and is held by
 * `fixed_slab_memory_resource` via `cuda::mr::shared_resource` for reference-counted
 * ownership.
 */
class fixed_slab_memory_resource_impl final {
 public:
  RMM_HIDDEN fixed_slab_memory_resource_impl(
    cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
    cuda::stream_ref stream,
    std::size_t slab_bytes);

  RMM_EXPORT ~fixed_slab_memory_resource_impl() noexcept;

  fixed_slab_memory_resource_impl(fixed_slab_memory_resource_impl const&)            = delete;
  fixed_slab_memory_resource_impl(fixed_slab_memory_resource_impl&&)                 = delete;
  fixed_slab_memory_resource_impl& operator=(fixed_slab_memory_resource_impl const&) = delete;
  fixed_slab_memory_resource_impl& operator=(fixed_slab_memory_resource_impl&&)      = delete;

  [[nodiscard]] bool operator==(fixed_slab_memory_resource_impl const& other) const noexcept
  {
    return this == std::addressof(other);
  }

  [[nodiscard]] bool operator!=(fixed_slab_memory_resource_impl const& other) const noexcept
  {
    return !(*this == other);
  }

  /**
   * @brief Carves an aligned slice from the slab on the recorded stream.
   *
   * The returned slice is ordered on that stream behind all prior slab work. Pure host
   * arithmetic: no CUDA calls are made on this path.
   *
   * @throws rmm::logic_error if `stream` differs from the recorded stream, or `alignment`
   * is zero or not a power of two
   * @throws rmm::out_of_memory if fewer than `bytes` (plus alignment padding) remain
   */
  RMM_EXPORT [[nodiscard]] void* allocate(cuda::stream_ref stream,
                                          std::size_t bytes,
                                          std::size_t alignment);

  /**
   * @brief No-op on the memory: carved slices are owned by the slab and released only with
   * the resource itself.
   *
   * The stream is only validated against the recorded stream. Null/zero calls return
   * before any validation and are ignored entirely. No CUDA calls are made.
   *
   * `noexcept` boundary: a mismatching stream cannot throw and terminates the process
   * (fail-closed).
   */
  RMM_EXPORT void deallocate(cuda::stream_ref stream,
                             void* ptr,
                             std::size_t bytes,
                             std::size_t alignment) noexcept;

  /**
   * @brief Host-only carve from the slab: pure bump arithmetic, NO wait is performed.
   *
   * The caller must guarantee the recorded stream's queued work (including the slab
   * allocation itself) has completed before host-synchronous use of the returned slice,
   * e.g. by synchronizing the recorded stream (retrievable via `stream()`).
   *
   * @throws rmm::logic_error if `bytes > 0` and `alignment` is zero or not a power of two
   * @throws rmm::out_of_memory if fewer than `bytes` (plus alignment padding) remain
   */
  RMM_EXPORT [[nodiscard]] void* allocate_sync(std::size_t bytes, std::size_t alignment);

  /**
   * @brief Synchronous deallocation: no-op; no stream work is issued, so there is nothing
   * to complete before returning.
   */
  RMM_EXPORT void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment) noexcept;

  /**
   * @brief The stream the slab was allocated on; every allocate/deallocate call must use it.
   */
  [[nodiscard]] cudaStream_t stream() const noexcept { return construction_stream_; }

  /**
   * @brief Total slab capacity in bytes; never changes after construction.
   */
  [[nodiscard]] std::size_t slab_bytes() const noexcept { return slab_bytes_; }

  /**
   * @brief Base pointer of the slab allocated at construction; never changes after
   * construction.
   *
   * Valid while any refcounted owner is alive. Memory in the slab may only be accessed on
   * the recorded stream (or after that stream's queued work has completed); this accessor
   * never waits.
   */
  [[nodiscard]] void* data() const noexcept { return slab_; }

  /**
   * @brief Bytes handed out so far, including alignment padding, excluding prospective
   * padding of failed requests.
   */
  [[nodiscard]] std::size_t carved_bytes() const noexcept { return offset_; }

  RMM_CONSTEXPR_FRIEND void get_property(fixed_slab_memory_resource_impl const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

 private:
  // Hidden: only the constructor, destructor and four resource methods cross the ABI.
  [[nodiscard]] RMM_HIDDEN static cudaStream_t normalize_stream(cuda::stream_ref stream) noexcept
  {
    // Null is normalized to the explicit legacy stream at the library boundary; the
    // per-thread default stream stays distinct and is accepted as the recorded stream.
    return stream.get() == nullptr ? cudaStream_t{cudaStreamLegacy} : stream.get();
  }

  /**
   * @brief Throws unless `stream` is the recorded construction stream.
   *
   * The single-stream contract: the slab lives on the recorded stream, and all work
   * referencing it must use that stream.
   *
   * @throws rmm::logic_error on any other stream
   */
  RMM_HIDDEN void check_stream(cudaStream_t stream) const;

  /**
   * @brief Computes the start offset of a nonzero carve without mutating the cursor.
   *
   * The padding is derived from the low bits of the actual address `slab_ + offset_`, so a
   * base aligned to the allocation alignment but not to a larger requested alignment still
   * yields correctly aligned results. Modular arithmetic bounds every intermediate to
   * `2 * (alignment - 1)`.
   *
   * @throws rmm::out_of_memory if `bytes` plus padding does not fit in the remaining slab
   */
  RMM_HIDDEN [[nodiscard]] std::size_t carve_offset(std::size_t bytes, std::size_t alignment) const;

  RMM_HIDDEN void release() noexcept;

  cuda::mr::any_resource<cuda::mr::device_accessible> upstream_mr_;
  std::size_t const slab_bytes_;
  std::size_t offset_{0};
  void* slab_{nullptr};
  cudaStream_t const construction_stream_;
};

}  // namespace detail
}  // namespace mr
RMM_NAMESPACE_END
