/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/mr/detail/arena.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <thread>

RMM_NAMESPACE_BEGIN
namespace mr {
namespace detail {

/// The type used to key per-stream arenas by CUDA stream ID.
using stream_id_type = unsigned long long;

/**
 * @brief Return the CUDA stream ID used to key per-stream arenas.
 *
 * Normalizes the default stream to `cudaStreamLegacy` for consistency between
 * PTDS and non-PTDS mode, then returns the stream's ID from `cudaStreamGetId`.
 * Keying by ID (rather than the raw `cudaStream_t` handle) prevents a newly
 * created stream that reuses a destroyed stream's handle value from inheriting
 * stale arena state (ABA reuse).
 *
 * @param stream The stream for which to get the ID.
 * @return The CUDA stream ID.
 * @throws rmm::cuda_error if `cudaStreamGetId` fails.
 */
stream_id_type get_stream_id(cuda_stream_view stream);

/**
 * @brief Try to return the CUDA stream ID used to key per-stream arenas.
 *
 * Normalizes the default stream to `cudaStreamLegacy` for consistency between
 * PTDS and non-PTDS mode, then returns the stream's ID from `cudaStreamGetId`.
 * Keying by ID (rather than the raw `cudaStream_t` handle) prevents a newly
 * created stream that reuses a destroyed stream's handle value from inheriting
 * stale arena state (ABA reuse).
 *
 * @param stream The stream for which to get the ID.
 * @return The CUDA stream ID, or `std::nullopt` if `cudaStreamGetId` fails.
 */
std::optional<stream_id_type> try_get_stream_id(cuda_stream_view stream) noexcept;

/**
 * @brief Implementation class for arena_memory_resource.
 *
 * Holds the global arena, per-thread arenas, per-stream arenas, and all
 * associated mutexes. This class satisfies the CCCL `cuda::mr::resource`
 * concept and is held by `arena_memory_resource` via
 * `cuda::mr::shared_resource` for reference-counted ownership.
 */
class arena_memory_resource_impl {
 public:
  arena_memory_resource_impl(cuda::mr::any_resource<cuda::mr::device_accessible> upstream_mr,
                             std::optional<std::size_t> arena_size,
                             bool dump_log_on_failure);

  ~arena_memory_resource_impl() = default;

  arena_memory_resource_impl(arena_memory_resource_impl const&)            = delete;
  arena_memory_resource_impl(arena_memory_resource_impl&&)                 = delete;
  arena_memory_resource_impl& operator=(arena_memory_resource_impl const&) = delete;
  arena_memory_resource_impl& operator=(arena_memory_resource_impl&&)      = delete;

  bool operator==(arena_memory_resource_impl const& other) const noexcept
  {
    return this == std::addressof(other);
  }

  bool operator!=(arena_memory_resource_impl const& other) const noexcept
  {
    return !(*this == other);
  }

  [[nodiscard]] void* allocate(cuda::stream_ref stream,
                               std::size_t bytes,
                               std::size_t alignment = alignof(std::max_align_t));

  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = alignof(std::max_align_t)) noexcept;

  [[nodiscard]] void* allocate_sync(std::size_t bytes,
                                    std::size_t alignment = alignof(std::max_align_t));

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept;

  RMM_CONSTEXPR_FRIEND void get_property(arena_memory_resource_impl const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

 private:
  using global_arena = rmm::mr::detail::arena::global_arena;
  using arena        = rmm::mr::detail::arena::arena;

  void defragment();

  void deallocate_from_other_arena(cuda_stream_view stream, void* ptr, std::size_t bytes) noexcept;

  arena& get_arena(cuda_stream_view stream);
  arena& get_thread_arena();
  arena& get_stream_arena(stream_id_type stream_id);

  void dump_memory_log(std::size_t bytes);

  static bool use_per_thread_arena(cuda_stream_view stream);

  global_arena global_arena_;
  std::map<std::thread::id, std::shared_ptr<arena>> thread_arenas_;
  /// Arenas keyed by CUDA stream ID. Stream IDs are unique for the lifetime of the process and
  /// entries are never erased, so this map grows by one entry per distinct stream ever used with
  /// this resource. Applications that continually create and destroy streams will see unbounded
  /// growth; evicting entries for destroyed streams is planned follow-up work.
  std::map<stream_id_type, arena> stream_arenas_;
  bool dump_log_on_failure_{};
  std::shared_ptr<rapids_logger::logger> logger_{};
  mutable std::shared_mutex map_mtx_;
  mutable std::shared_mutex mtx_;
};

}  // namespace detail
}  // namespace mr
RMM_NAMESPACE_END
