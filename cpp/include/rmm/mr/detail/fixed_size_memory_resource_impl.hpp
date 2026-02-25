/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/detail/fixed_size_free_list.hpp>
#include <rmm/mr/detail/stream_ordered_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <cstddef>
#include <mutex>
#include <utility>
#include <vector>

namespace RMM_NAMESPACE {
namespace mr {
namespace detail {

/**
 * @brief Implementation class for fixed_size_memory_resource.
 *
 * Allocates fixed-size blocks from an upstream resource. This class satisfies
 * the CCCL `cuda::mr::resource` concept and is held by `fixed_size_memory_resource`
 * via `cuda::mr::shared_resource` for reference-counted ownership.
 */
class fixed_size_memory_resource_impl final
  : public stream_ordered_memory_resource<fixed_size_memory_resource_impl, fixed_size_free_list> {
 public:
  friend class stream_ordered_memory_resource<fixed_size_memory_resource_impl,
                                              fixed_size_free_list>;

  static constexpr std::size_t default_block_size            = 1 << 20;
  static constexpr std::size_t default_blocks_to_preallocate = 128;

  fixed_size_memory_resource_impl(device_async_resource_ref upstream,
                                  std::size_t block_size,
                                  std::size_t blocks_to_preallocate);

  ~fixed_size_memory_resource_impl() override;

  bool operator==(fixed_size_memory_resource_impl const& other) const noexcept
  {
    return this == std::addressof(other);
  }

  bool operator!=(fixed_size_memory_resource_impl const& other) const noexcept
  {
    return !(*this == other);
  }

  [[nodiscard]] device_async_resource_ref get_upstream_resource() const noexcept;

  [[nodiscard]] std::size_t get_block_size() const noexcept;

  RMM_CONSTEXPR_FRIEND void get_property(fixed_size_memory_resource_impl const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

 protected:
  using free_list  = fixed_size_free_list;
  using block_type = free_list::block_type;
  using typename stream_ordered_memory_resource<fixed_size_memory_resource_impl,
                                                fixed_size_free_list>::split_block;
  using lock_guard = std::lock_guard<std::mutex>;

  [[nodiscard]] std::size_t get_maximum_allocation_size() const;

  block_type expand_pool(std::size_t size, free_list& blocks, cuda_stream_view stream);

  split_block allocate_from_block(block_type const& block, std::size_t size);

  block_type free_block(void* ptr, std::size_t size) noexcept;

  std::pair<std::size_t, std::size_t> free_list_summary(free_list const& blocks);

 private:
  free_list blocks_from_upstream(cuda_stream_view stream);

  void release();

  cuda::mr::any_resource<cuda::mr::device_accessible> upstream_mr_;
  std::size_t block_size_;
  std::size_t upstream_chunk_size_;
  std::vector<block_type> upstream_blocks_;
};

}  // namespace detail
}  // namespace mr
}  // namespace RMM_NAMESPACE
