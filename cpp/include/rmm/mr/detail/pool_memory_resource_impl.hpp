/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
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
namespace detail {

/**
 * @brief Implementation class for pool_memory_resource.
 *
 * A coalescing best-fit suballocator which uses a pool of memory allocated from
 * an upstream memory resource. This class satisfies the CCCL `cuda::mr::resource`
 * concept and is held by `pool_memory_resource` via `cuda::mr::shared_resource`
 * for reference-counted ownership.
 */
class pool_memory_resource_impl final
  : public stream_ordered_memory_resource<pool_memory_resource_impl, coalescing_free_list> {
 public:
  friend class stream_ordered_memory_resource<pool_memory_resource_impl, coalescing_free_list>;

  pool_memory_resource_impl(device_async_resource_ref upstream,
                            std::size_t initial_pool_size,
                            std::optional<std::size_t> maximum_pool_size);

  ~pool_memory_resource_impl() override;

  bool operator==(pool_memory_resource_impl const& other) const noexcept { return this == &other; }

  bool operator!=(pool_memory_resource_impl const& other) const noexcept
  {
    return !(*this == other);
  }

  [[nodiscard]] device_async_resource_ref get_upstream_resource() const noexcept;

  [[nodiscard]] std::size_t pool_size() const noexcept;

  friend void get_property(pool_memory_resource_impl const&, cuda::mr::device_accessible) noexcept
  {
  }

 protected:
  using free_list  = coalescing_free_list;
  using block_type = free_list::block_type;
  using typename stream_ordered_memory_resource<pool_memory_resource_impl,
                                                coalescing_free_list>::split_block;
  using lock_guard = std::lock_guard<std::mutex>;

  [[nodiscard]] std::size_t get_maximum_allocation_size() const;
  block_type try_to_expand(std::size_t try_size, std::size_t min_size, cuda_stream_view stream);
  void initialize_pool(std::size_t initial_size, std::optional<std::size_t> maximum_size);
  block_type expand_pool(std::size_t size, free_list& blocks, cuda_stream_view stream);
  [[nodiscard]] std::size_t size_to_grow(std::size_t size) const;
  block_type block_from_upstream(std::size_t size, cuda_stream_view stream);
  split_block allocate_from_block(block_type const& block, std::size_t size);
  block_type free_block(void* ptr, std::size_t size) noexcept;
  void release();
  std::pair<std::size_t, std::size_t> free_list_summary(free_list const& blocks);

#ifdef RMM_DEBUG_PRINT
  void print();
#endif

 private:
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream_mr_;
  std::size_t current_pool_size_{};
  std::optional<std::size_t> maximum_pool_size_{};
  std::set<block_type, compare_blocks<block_type>> upstream_blocks_;
#ifdef RMM_POOL_TRACK_ALLOCATIONS
  std::set<block_type, compare_blocks<block_type>> allocated_blocks_;
#endif
};

}  // namespace detail
}  // namespace mr
}  // namespace RMM_NAMESPACE
