/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/detail/coalescing_free_list.hpp>
#include <rmm/mr/detail/stream_ordered_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <cstddef>
#include <mutex>
#include <optional>
#include <set>

namespace RMM_NAMESPACE {
namespace mr {

/**
 * @brief Policy used after an upstream allocation fails while growing a caching memory resource.
 */
enum class caching_memory_resource_oom_fallback_policy {
  release_all,                 ///< Release all cached whole segments, then retry.
  release_oversized_then_all,  ///< Release oversized whole segments first, then all segments.
};

/**
 * @brief Policy controlling whether small and large cached segments can satisfy each other's
 * allocation requests.
 */
enum class caching_memory_resource_pool_policy {
  separate,  ///< Small requests use small segments and large requests use large segments.
  unified,   ///< Any cached segment large enough for a request may be reused.
};

/**
 * @brief Policy controlling reuse of cached blocks from streams other than the allocation stream.
 */
enum class caching_memory_resource_stream_reuse_policy {
  same_stream,   ///< Reuse cached blocks from the same stream only.
  cross_stream,  ///< Reuse cached blocks from other streams by inserting CUDA event waits.
};

namespace detail {

class caching_memory_resource_impl final
  : public stream_ordered_memory_resource<caching_memory_resource_impl, coalescing_free_list> {
 public:
  friend class stream_ordered_memory_resource<caching_memory_resource_impl, coalescing_free_list>;

  static constexpr std::size_t minimum_block_size{512};
  static constexpr std::size_t small_size_threshold{1UL << 20};
  static constexpr std::size_t small_segment_size{2UL << 20};
  static constexpr std::size_t large_segment_size{20UL << 20};
  static constexpr std::size_t minimum_large_allocation{10UL << 20};
  static constexpr std::size_t large_rounding{2UL << 20};

  caching_memory_resource_impl(
    cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
    std::optional<std::size_t> max_split_size = std::nullopt,
    caching_memory_resource_oom_fallback_policy oom_fallback_policy =
      caching_memory_resource_oom_fallback_policy::release_oversized_then_all,
    caching_memory_resource_pool_policy pool_policy = caching_memory_resource_pool_policy::separate,
    caching_memory_resource_stream_reuse_policy stream_reuse_policy =
      caching_memory_resource_stream_reuse_policy::same_stream);

  ~caching_memory_resource_impl();

  bool operator==(caching_memory_resource_impl const& other) const noexcept
  {
    return this == std::addressof(other);
  }

  bool operator!=(caching_memory_resource_impl const& other) const noexcept
  {
    return !(*this == other);
  }

  [[nodiscard]] device_async_resource_ref get_upstream_resource() const noexcept;

  [[nodiscard]] std::size_t release() noexcept;

  [[nodiscard]] std::size_t release_small_blocks() noexcept;

  [[nodiscard]] std::size_t release_large_blocks() noexcept;

  [[nodiscard]] std::size_t cached_small_bytes() const noexcept;

  [[nodiscard]] std::size_t cached_large_bytes() const noexcept;

  [[nodiscard]] std::size_t upstream_allocation_size(std::size_t bytes) const noexcept;

  [[nodiscard]] bool supports_cross_stream_reuse() const noexcept;

  RMM_CONSTEXPR_FRIEND void get_property(caching_memory_resource_impl const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

 protected:
  using free_list  = coalescing_free_list;
  using block_type = free_list::block_type;
  using typename stream_ordered_memory_resource<caching_memory_resource_impl,
                                                coalescing_free_list>::split_block;
  using lock_guard = std::lock_guard<std::mutex>;

  [[nodiscard]] std::size_t get_maximum_allocation_size() const;

  block_type expand_pool(std::size_t size, free_list& blocks, cuda_stream_view stream);

  block_type get_block_from_free_list(free_list& blocks, std::size_t size);

  split_block allocate_from_block(block_type const& block, std::size_t size);

  block_type free_block(void* ptr, std::size_t size) noexcept;

  std::pair<std::size_t, std::size_t> free_list_summary(free_list const& blocks);

 private:
  struct segment {
    char* pointer{};
    std::size_t size{};
    bool is_small{};

    bool operator<(segment const& other) const noexcept { return pointer < other.pointer; }
  };

  [[nodiscard]] bool is_small_allocation(std::size_t size) const noexcept;

  [[nodiscard]] bool should_split(block_type const& block, std::size_t size) const noexcept;

  [[nodiscard]] bool block_is_small(block_type const& block) const noexcept;

  [[nodiscard]] std::size_t release_pool(bool is_small) noexcept;

  [[nodiscard]] std::size_t release_all_pools() noexcept;

  [[nodiscard]] std::size_t release_oversized_blocks(std::size_t size) noexcept;

  block_type block_from_upstream(std::size_t size, cuda_stream_view stream, bool is_small);

  cuda::mr::any_resource<cuda::mr::device_accessible> upstream_mr_;
  std::optional<std::size_t> max_split_size_{};
  caching_memory_resource_oom_fallback_policy oom_fallback_policy_{};
  caching_memory_resource_pool_policy pool_policy_{};
  caching_memory_resource_stream_reuse_policy stream_reuse_policy_{};
  std::set<segment> upstream_blocks_{};
  std::set<block_type, compare_blocks<block_type>> allocated_blocks_{};
  std::size_t cached_small_bytes_{};
  std::size_t cached_large_bytes_{};
};

}  // namespace detail
}  // namespace mr
}  // namespace RMM_NAMESPACE
