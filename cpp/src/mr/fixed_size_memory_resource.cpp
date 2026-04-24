/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/detail/logging_assert.hpp>
#include <rmm/mr/fixed_size_memory_resource.hpp>

#include <cuda/cmath>

#include <cstddef>
#include <mutex>
#include <vector>

namespace RMM_NAMESPACE {
namespace mr {

fixed_size_memory_resource::fixed_size_memory_resource(
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
  std::size_t block_size,
  std::size_t blocks_to_preallocate)
  : shared_base(cuda::mr::make_shared_resource<detail::fixed_size_memory_resource_impl>(
      std::move(upstream), block_size, blocks_to_preallocate))
{
}

device_async_resource_ref fixed_size_memory_resource::get_upstream_resource() const noexcept
{
  return get().get_upstream_resource();
}

std::size_t fixed_size_memory_resource::get_block_size() const noexcept
{
  return get().get_block_size();
}

// multiple_blocks_allocation

multiple_blocks_allocation::multiple_blocks_allocation(std::size_t size,
                                                       std::vector<std::byte*> buffers,
                                                       cuda_stream_view stream,
                                                       fixed_size_memory_resource mr)
  : blocks_(std::move(buffers)), size_(size), stream_(stream), mr_(std::move(mr))
{
  RMM_LOGGING_ASSERT(size_ <= mr_->get_block_size() * blocks_.size());
  RMM_LOGGING_ASSERT(blocks_.empty() ||
                     blocks_.size() == cuda::ceil_div(size_, mr_->get_block_size()));
}

multiple_blocks_allocation::~multiple_blocks_allocation()
{
  if (!blocks_.empty()) {
    std::lock_guard<std::mutex> lock(mr_->get_mutex());
    mr_->deallocate_blocks_async_unsafe(std::move(blocks_), stream_);
  }
}

std::unique_ptr<multiple_blocks_allocation> multiple_blocks_allocation::make_async(
  fixed_size_memory_resource mr, std::size_t size, cuda_stream_view stream)
{
  if (size == 0) {
    return std::unique_ptr<multiple_blocks_allocation>(
      new multiple_blocks_allocation(0, {}, stream, std::move(mr)));
  }

  auto& self = *mr;
  std::lock_guard<std::mutex> lock(self.get_mutex());

  auto stream_event            = self.get_event(stream);
  std::size_t const num_blocks = cuda::ceil_div(size, self.get_block_size());
  std::vector<std::byte*> blocks;
  blocks.reserve(num_blocks);
  try {
    for (std::size_t i = 0; i < num_blocks; ++i) {
      blocks.push_back(
        static_cast<std::byte*>(self.get_block(self.get_block_size(), stream_event).pointer()));
    }
  } catch (...) {
    self.deallocate_blocks_async_unsafe(std::move(blocks), stream);
    throw;
  }

  return std::unique_ptr<multiple_blocks_allocation>(
    new multiple_blocks_allocation(size, std::move(blocks), stream, std::move(mr)));
}

}  // namespace mr
}  // namespace RMM_NAMESPACE
