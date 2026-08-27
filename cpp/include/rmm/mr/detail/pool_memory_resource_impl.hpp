/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/detail/coalescing_free_list.hpp>
#include <rmm/mr/detail/indexed_coalescing_free_list.hpp>
#include <rmm/mr/detail/indexed_stream_ordered_memory_resource.hpp>
#include <rmm/mr/detail/stream_ordered_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <type_traits>

RMM_NAMESPACE_BEGIN
namespace mr {
namespace detail {

template <typename FreeListType>
inline constexpr bool is_indexed_free_list =
  std::is_same_v<FreeListType, indexed_coalescing_free_list>;

template <typename PoolResource, typename FreeListType>
using pool_stream_ordered_resource =
  std::conditional_t<is_indexed_free_list<FreeListType>,
                     indexed_stream_ordered_memory_resource<PoolResource, FreeListType>,
                     stream_ordered_memory_resource<PoolResource, FreeListType>>;

/**
 * @brief Implementation class for pool memory resources.
 *
 * A coalescing best-fit suballocator that uses a pool of memory allocated from an upstream memory
 * resource. `FreeListType` selects both the free-list implementation and its stream-ordered
 * recovery behavior. This class satisfies the CCCL `cuda::mr::resource` concept and is held by a
 * public pool resource via `cuda::mr::shared_resource` for reference-counted ownership.
 *
 * @tparam FreeListType Free-list implementation used by the pool.
 */
template <typename FreeListType>
class pool_memory_resource_impl final
  : public pool_stream_ordered_resource<pool_memory_resource_impl<FreeListType>, FreeListType> {
  using stream_ordered_base =
    pool_stream_ordered_resource<pool_memory_resource_impl<FreeListType>, FreeListType>;
  friend stream_ordered_base;
  friend class stream_ordered_memory_resource<pool_memory_resource_impl<FreeListType>,
                                              FreeListType>;

 public:
  pool_memory_resource_impl(cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
                            std::size_t initial_pool_size,
                            std::optional<std::size_t> maximum_pool_size);

  ~pool_memory_resource_impl();

  bool operator==(pool_memory_resource_impl const& other) const noexcept
  { return this == std::addressof(other); }

  bool operator!=(pool_memory_resource_impl const& other) const noexcept
  { return !(*this == other); }

  [[nodiscard]] device_async_resource_ref get_upstream_resource() const noexcept;

  [[nodiscard]] std::size_t pool_size() const noexcept;

  RMM_CONSTEXPR_FRIEND void get_property(pool_memory_resource_impl const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

 protected:
  using free_list  = FreeListType;
  using block_type = typename free_list::block_type;
  using typename stream_ordered_base::split_block;
  using lock_guard = std::lock_guard<std::mutex>;

  [[nodiscard]] std::size_t get_maximum_allocation_size() const;
  block_type try_to_expand(std::size_t try_size, std::size_t min_size, cuda_stream_view stream);
  void initialize_pool(std::size_t initial_size, std::optional<std::size_t> maximum_size);
  block_type expand_pool(std::size_t size, free_list& blocks, cuda_stream_view stream);
  void reclaim_free_blocks(std::size_t size, free_list& blocks, cuda_stream_view stream);
  template <typename BeforeReclaim>
  std::size_t reclaim_upstream_blocks_from_owner(std::size_t size,
                                                 std::size_t max_pool_size,
                                                 free_list& blocks,
                                                 cuda_stream_view stream,
                                                 BeforeReclaim&& before_reclaim);
  [[nodiscard]] std::size_t size_to_grow(std::size_t size) const;
  block_type block_from_upstream(std::size_t size, cuda_stream_view stream);
  split_block allocate_from_block(block_type const& block, std::size_t size);
  struct prepared_allocation_tracking {
#ifdef RMM_POOL_TRACK_ALLOCATIONS
    typename std::set<block_type, compare_blocks<block_type>>::node_type node{};
#endif
  };
  prepared_allocation_tracking prepare_allocation_tracking(block_type const& block);
  void commit_allocation_tracking(prepared_allocation_tracking&& prepared) noexcept;
  block_type free_block(void* ptr, std::size_t size) noexcept;
  block_type prepare_free_block(void* ptr, std::size_t size) const noexcept;
  void commit_free_block(block_type const& block) noexcept;
  void release();
  std::pair<std::size_t, std::size_t> free_list_summary(free_list const& blocks);

#ifdef RMM_DEBUG_PRINT
  void print();
#endif

 private:
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream_mr_;
  std::size_t current_pool_size_{};
  std::optional<std::size_t> maximum_pool_size_{};
  std::set<block_type, compare_blocks<block_type>> upstream_blocks_;  ///< Upstream allocations.
#ifdef RMM_POOL_TRACK_ALLOCATIONS
  std::set<block_type, compare_blocks<block_type>> allocated_blocks_;  ///< Live suballocations.
#endif
};

}  // namespace detail
}  // namespace mr
RMM_NAMESPACE_END
