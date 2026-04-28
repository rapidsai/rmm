/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/logging_assert.hpp>
#include <rmm/logger.hpp>
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
                                                       cuda::stream_ref stream,
                                                       fixed_size_memory_resource mr) noexcept
  : blocks_(std::move(buffers)), size_(size), stream_(stream), mr_(std::move(mr))
{
}

multiple_blocks_allocation::multiple_blocks_allocation(multiple_blocks_allocation&& other) noexcept
  : blocks_(std::move(other.blocks_)),
    size_(other.size_),
    stream_(other.stream_),
    mr_(std::move(other.mr_))
{
  other.size_ = 0;
}

void multiple_blocks_allocation::clear()
{
  if (!blocks_.empty()) {
    std::lock_guard<std::mutex> lock(mr_->get_mutex());
    RMM_CUDA_TRY(mr_->deallocate_blocks_async_unsafe(std::move(blocks_), stream_));
  }
  size_ = 0;
}

multiple_blocks_allocation& multiple_blocks_allocation::operator=(
  multiple_blocks_allocation&& other)
{
  if (this != &other) {
    clear();
    blocks_     = std::move(other.blocks_);
    size_       = other.size_;
    stream_     = other.stream_;
    mr_         = std::move(other.mr_);
    other.size_ = 0;
  }
  return *this;
}

multiple_blocks_allocation::~multiple_blocks_allocation() noexcept
{
  try {
    clear();
  } catch (...) {
    RMM_LOG_ERROR(
      "multiple_blocks_allocation: exception while releasing device blocks in destructor");
  }
}

std::unique_ptr<multiple_blocks_allocation> multiple_blocks_allocation::make_async(
  fixed_size_memory_resource mr, std::size_t size, cuda::stream_ref stream)
{
  RMM_EXPECTS(!cuda_stream_view{stream}.is_per_thread_default(),
              "stream must not be a per-thread default stream",
              rmm::invalid_argument);

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
    RMM_CUDA_TRY(self.deallocate_blocks_async_unsafe(std::move(blocks), stream));
    throw;
  }

  return std::unique_ptr<multiple_blocks_allocation>(
    new multiple_blocks_allocation(size, std::move(blocks), stream, std::move(mr)));
}

}  // namespace mr
}  // namespace RMM_NAMESPACE
