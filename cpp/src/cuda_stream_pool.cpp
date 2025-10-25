/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>

namespace rmm {

cuda_stream_pool::cuda_stream_pool(std::size_t pool_size, cuda_stream::flags flags)
{
  RMM_EXPECTS(pool_size > 0, "Stream pool size must be greater than zero");
  streams_.reserve(pool_size);
  std::generate_n(
    std::back_inserter(streams_), pool_size, [flags]() { return cuda_stream(flags); });
}

rmm::cuda_stream_view cuda_stream_pool::get_stream() const noexcept
{
  return streams_[(next_stream.fetch_add(1, std::memory_order_relaxed)) % streams_.size()].view();
}

rmm::cuda_stream_view cuda_stream_pool::get_stream(std::size_t stream_id) const
{
  return streams_[stream_id % streams_.size()].view();
}

std::size_t cuda_stream_pool::get_pool_size() const noexcept { return streams_.size(); }

}  // namespace rmm
