/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include "../byte_literals.hpp"

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/fixed_size_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace rmm::test {
namespace {

using cuda_mr       = rmm::mr::cuda_memory_resource;
using cuda_async_mr = rmm::mr::cuda_async_memory_resource;

/// Test-only adaptor that forwards to an upstream ref and tracks bytes with an atomic counter.
class byte_count_resource_adaptor final : public rmm::mr::device_memory_resource {
 public:
  byte_count_resource_adaptor(rmm::device_async_resource_ref upstream)
    : upstream_{upstream}, bytes_allocated_{0}
  {
  }

  ~byte_count_resource_adaptor() override = default;

  [[nodiscard]] std::size_t bytes_count() const noexcept
  {
    return bytes_allocated_.load(std::memory_order_relaxed);
  }

 private:
  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override
  {
    void* ptr = upstream_.allocate(stream, bytes);
    bytes_allocated_.fetch_add(bytes, std::memory_order_relaxed);
    return ptr;
  }

  void do_deallocate(void* ptr, std::size_t bytes, rmm::cuda_stream_view stream) noexcept override
  {
    bytes_allocated_.fetch_sub(bytes, std::memory_order_relaxed);
    upstream_.deallocate(stream, ptr, bytes);
  }

  [[nodiscard]] bool do_is_equal(
    rmm::mr::device_memory_resource const& other) const noexcept override
  {
    return this == std::addressof(other);
  }

  rmm::device_async_resource_ref upstream_;
  std::atomic<std::size_t> bytes_allocated_;
};

enum class UpstreamMR { Cuda, CudaAsync };

inline std::string to_string(UpstreamMR k)
{
  switch (k) {
    case UpstreamMR::Cuda: return "Cuda";
    case UpstreamMR::CudaAsync: return "CudaAsync";
    default: return "Unknown";
  }
}

std::shared_ptr<rmm::mr::device_memory_resource> make_upstream_resource(UpstreamMR kind)
{
  switch (kind) {
    case UpstreamMR::Cuda: return std::make_shared<cuda_mr>();
    case UpstreamMR::CudaAsync: return std::make_shared<cuda_async_mr>();
    default: return nullptr;
  }
}

struct FixedSizeMRTestParam {
  UpstreamMR upstream_kind{UpstreamMR::Cuda};
  std::size_t block_size{0};
  std::size_t size{0};
};

std::vector<FixedSizeMRTestParam> make_fixed_size_mr_test_params()
{
  std::vector<FixedSizeMRTestParam> params;
  for (UpstreamMR kind : {UpstreamMR::Cuda, UpstreamMR::CudaAsync}) {
    for (std::size_t block_sz : std::vector<std::size_t>{64, 256, 1_KiB}) {
      for (std::size_t size : std::vector<std::size_t>{0, block_sz / 2, block_sz, 3_KiB}) {
        params.push_back({kind, block_sz, size});
      }
    }
  }
  return params;
}

class FixedSizeMRTest : public ::testing::TestWithParam<FixedSizeMRTestParam> {};

TEST_P(FixedSizeMRTest, AllocateBlocksAsyncUpstreamCountedDeallocateDoesNotReturnToUpstream)
{
  auto const& param                           = GetParam();
  auto upstream                               = make_upstream_resource(param.upstream_kind);
  constexpr std::size_t blocks_to_preallocate = 1;

  auto counting_mr = cuda::mr::make_shared_resource<byte_count_resource_adaptor>(*upstream);

  using fixed_size_mr = rmm::mr::fixed_size_memory_resource<byte_count_resource_adaptor>;

  {
    auto fixed_mr = fixed_size_mr(counting_mr, param.block_size, blocks_to_preallocate);

    rmm::cuda_stream_pool stream_pool{4};
    std::vector<std::unique_ptr<fixed_size_mr::multiple_blocks_allocation>> handles;

    std::size_t const alloc_size  = param.size;
    constexpr int num_allocations = 4;

    std::size_t const expected_blocks = [&]() {
      if (alloc_size == 0) { return std::size_t{0}; }
      std::size_t const aligned    = rmm::align_up(alloc_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
      std::size_t const block_size = param.block_size;
      return cuda::ceil_div(aligned, block_size);
    }();

    for (int i = 0; i < num_allocations; ++i) {
      rmm::cuda_stream_view stream = stream_pool.get_stream();
      const auto& handle = handles.emplace_back(fixed_mr.allocate_blocks_async(alloc_size, stream));

      EXPECT_EQ(handle->size(), alloc_size);
      EXPECT_EQ(handle->capacity(), expected_blocks * param.block_size);
    }
    std::size_t const bytes_after_alloc = counting_mr->bytes_count();

    if (expected_blocks > 0) {
      EXPECT_GE(bytes_after_alloc, expected_blocks * num_allocations * param.block_size);
    }
    handles.clear();

    EXPECT_EQ(counting_mr->bytes_count(), bytes_after_alloc)
      << "After deallocate, upstream value must be unchanged until fixed_size_mr is destroyed";
  }

  EXPECT_EQ(counting_mr->bytes_count(), 0)
    << "After fixed_size_mr destruction, upstream should have released all memory";
}

INSTANTIATE_TEST_SUITE_P(FixedSizeMRTests,
                         FixedSizeMRTest,
                         ::testing::ValuesIn(make_fixed_size_mr_test_params()),
                         [](testing::TestParamInfo<FixedSizeMRTestParam> const& info) {
                           return to_string(info.param.upstream_kind) + "_bs" +
                                  std::to_string(info.param.block_size) + "_sz" +
                                  std::to_string(info.param.size);
                         });

}  // namespace
}  // namespace rmm::test
