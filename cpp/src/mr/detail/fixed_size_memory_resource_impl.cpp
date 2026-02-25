/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/logging_assert.hpp>
#include <rmm/mr/detail/fixed_size_memory_resource_impl.hpp>

#include <cuda/iterator>

#include <cstddef>
#include <mutex>
#include <utility>
#include <vector>

#ifdef RMM_DEBUG_PRINT
#include <rmm/cuda_device.hpp>

#include <iostream>
#endif

namespace RMM_NAMESPACE {
namespace mr {
namespace detail {

fixed_size_memory_resource_impl::fixed_size_memory_resource_impl(device_async_resource_ref upstream,
                                                                 std::size_t block_size,
                                                                 std::size_t blocks_to_preallocate)
  : upstream_mr_{upstream},
    block_size_{align_up(block_size, CUDA_ALLOCATION_ALIGNMENT)},
    upstream_chunk_size_{block_size_ * blocks_to_preallocate}
{
  this->insert_blocks(std::move(blocks_from_upstream(cuda_stream_legacy)), cuda_stream_legacy);
}

fixed_size_memory_resource_impl::~fixed_size_memory_resource_impl() { release(); }

device_async_resource_ref fixed_size_memory_resource_impl::get_upstream_resource() const noexcept
{
  return device_async_resource_ref{
    const_cast<cuda::mr::any_resource<cuda::mr::device_accessible>&>(upstream_mr_)};
}

std::size_t fixed_size_memory_resource_impl::get_block_size() const noexcept { return block_size_; }

std::size_t fixed_size_memory_resource_impl::get_maximum_allocation_size() const
{
  return get_block_size();
}

fixed_size_memory_resource_impl::block_type fixed_size_memory_resource_impl::expand_pool(
  std::size_t size, free_list& blocks, cuda_stream_view stream)
{
  blocks.insert(std::move(blocks_from_upstream(stream)));
  return blocks.get_block(size);
}

fixed_size_memory_resource_impl::free_list fixed_size_memory_resource_impl::blocks_from_upstream(
  cuda_stream_view stream)
{
  void* ptr = upstream_mr_.allocate(stream, upstream_chunk_size_);
  block_type block{ptr};
  upstream_blocks_.push_back(block);

  auto num_blocks = upstream_chunk_size_ / block_size_;

  auto block_gen = [ptr, this](int index) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return block_type{static_cast<char*>(ptr) + index * block_size_};
  };
  auto first =
    cuda::make_transform_iterator(cuda::make_counting_iterator(std::size_t{0}), block_gen);
  return free_list(first, first + num_blocks);
}

fixed_size_memory_resource_impl::split_block fixed_size_memory_resource_impl::allocate_from_block(
  block_type const& block, [[maybe_unused]] std::size_t size)
{
  return {block, block_type{nullptr}};
}

fixed_size_memory_resource_impl::block_type fixed_size_memory_resource_impl::free_block(
  void* ptr, [[maybe_unused]] std::size_t size) noexcept
{
  RMM_LOGGING_ASSERT(align_up(size, CUDA_ALLOCATION_ALIGNMENT) <= block_size_);
  return block_type{ptr};
}

void fixed_size_memory_resource_impl::release()
{
  lock_guard lock(this->get_mutex());

  for (auto block : upstream_blocks_) {
    upstream_mr_.deallocate_sync(block.pointer(), upstream_chunk_size_);
  }
  upstream_blocks_.clear();
}

std::pair<std::size_t, std::size_t> fixed_size_memory_resource_impl::free_list_summary(
  free_list const& blocks)
{
  return blocks.is_empty() ? std::make_pair(std::size_t{0}, std::size_t{0})
                           : std::make_pair(block_size_, blocks.size() * block_size_);
}

#ifdef RMM_DEBUG_PRINT
void fixed_size_memory_resource_impl::print()
{
  lock_guard lock(this->get_mutex());

  auto const [free, total] = rmm::available_device_memory();
  std::cout << "GPU free memory: " << free << " total: " << total << "\n";

  std::cout << "upstream_blocks: " << upstream_blocks_.size() << "\n";
  std::size_t upstream_total{0};

  for (auto blocks : upstream_blocks_) {
    blocks.print();
    upstream_total += upstream_chunk_size_;
  }
  std::cout << "total upstream: " << upstream_total << " B\n";

  this->print_free_blocks();
}
#endif

}  // namespace detail
}  // namespace mr
}  // namespace RMM_NAMESPACE
