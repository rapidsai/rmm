/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/cuda_stream.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/caching_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/limiting_resource_adaptor.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <gtest/gtest.h>

namespace rmm::test {
namespace {

using caching_mr    = rmm::mr::caching_memory_resource;
using cuda_mr       = rmm::mr::cuda_memory_resource;
using limiting_mr   = rmm::mr::limiting_resource_adaptor;
using oom_policy    = rmm::mr::caching_memory_resource_oom_fallback_policy;
using pool_policy   = rmm::mr::caching_memory_resource_pool_policy;
using stream_policy = rmm::mr::caching_memory_resource_stream_reuse_policy;

TEST(CachingMemoryResourceTest, UpstreamSizingPolicy)
{
  caching_mr mr{rmm::mr::get_current_device_resource_ref()};
  EXPECT_EQ(mr.upstream_allocation_size(1), 2UL << 20);
  EXPECT_EQ(mr.upstream_allocation_size(1UL << 20), 2UL << 20);
  EXPECT_EQ(mr.upstream_allocation_size((1UL << 20) + 1), 20UL << 20);
  EXPECT_EQ(mr.upstream_allocation_size((10UL << 20) + 1), 12UL << 20);
}

TEST(CachingMemoryResourceTest, ReusesSmallAllocation)
{
  caching_mr mr{rmm::mr::get_current_device_resource_ref()};
  auto* ptr1 = mr.allocate_sync(4096);
  ASSERT_NE(ptr1, nullptr);
  mr.deallocate_sync(ptr1, 4096);
  auto* ptr2 = mr.allocate_sync(4096);
  EXPECT_EQ(ptr1, ptr2);
  mr.deallocate_sync(ptr2, 4096);
}

TEST(CachingMemoryResourceTest, ReleasesCachedBlocksOnFailure)
{
  cuda_mr cuda;
  limiting_mr limiter{cuda, 41UL << 20};
  caching_mr mr{limiter};

  auto* first = mr.allocate_sync(2UL << 20);
  mr.deallocate_sync(first, 2UL << 20);

  EXPECT_EQ(mr.cached_large_bytes(), 20UL << 20);
  void* large{};
  EXPECT_NO_THROW(large = mr.allocate_sync(22UL << 20));
  EXPECT_EQ(limiter.get_allocated_bytes(), 22UL << 20);
  mr.deallocate_sync(large, 22UL << 20);
}

TEST(CachingMemoryResourceTest, ReleaseAllFallbackReleasesBothPools)
{
  cuda_mr cuda;
  limiting_mr limiter{cuda, 22UL << 20};
  caching_mr mr{limiter, 4UL << 20, oom_policy::release_all};

  auto* small = mr.allocate_sync(4096);
  auto* large = mr.allocate_sync(2UL << 20);
  mr.deallocate_sync(small, 4096);
  mr.deallocate_sync(large, 2UL << 20);

  void* ptr{};
  EXPECT_NO_THROW(ptr = mr.allocate_sync(3UL << 20));
  EXPECT_EQ(mr.cached_small_bytes(), 0);
  EXPECT_EQ(limiter.get_allocated_bytes(), 20UL << 20);
  mr.deallocate_sync(ptr, 3UL << 20);
}

TEST(CachingMemoryResourceTest, OversizedThenAllFallbackPreservesOtherPool)
{
  cuda_mr cuda;
  limiting_mr limiter{cuda, 22UL << 20};
  caching_mr mr{limiter, 4UL << 20, oom_policy::release_oversized_then_all};

  auto* small = mr.allocate_sync(4096);
  auto* large = mr.allocate_sync(2UL << 20);
  mr.deallocate_sync(small, 4096);
  mr.deallocate_sync(large, 2UL << 20);

  void* ptr{};
  EXPECT_NO_THROW(ptr = mr.allocate_sync(3UL << 20));
  EXPECT_EQ(mr.cached_small_bytes(), 2UL << 20);
  EXPECT_EQ(limiter.get_allocated_bytes(), 22UL << 20);
  mr.deallocate_sync(ptr, 3UL << 20);
}

TEST(CachingMemoryResourceTest, SeparatePoolPolicyDoesNotReuseLargeBlockForSmallRequest)
{
  cuda_mr cuda;
  limiting_mr limiter{cuda, 22UL << 20};
  caching_mr mr{limiter};

  auto* large = mr.allocate_sync(2UL << 20);
  mr.deallocate_sync(large, 2UL << 20);

  auto* small = mr.allocate_sync(4096);
  EXPECT_NE(large, small);
  EXPECT_EQ(limiter.get_allocated_bytes(), 22UL << 20);
  mr.deallocate_sync(small, 4096);
}

TEST(CachingMemoryResourceTest, UnifiedPoolPolicyReusesLargeBlockForSmallRequest)
{
  cuda_mr cuda;
  limiting_mr limiter{cuda, 20UL << 20};
  caching_mr mr{
    limiter, std::nullopt, oom_policy::release_oversized_then_all, pool_policy::unified};

  auto* large = mr.allocate_sync(2UL << 20);
  mr.deallocate_sync(large, 2UL << 20);

  auto* small = mr.allocate_sync(4096);
  EXPECT_EQ(large, small);
  EXPECT_EQ(limiter.get_allocated_bytes(), 20UL << 20);
  mr.deallocate_sync(small, 4096);
}

TEST(CachingMemoryResourceTest, SeparatePoolPolicyDoesNotReuseSmallBlockForLargeRequest)
{
  cuda_mr cuda;
  limiting_mr limiter{cuda, 22UL << 20};
  caching_mr mr{limiter};

  auto* small = mr.allocate_sync(1UL << 20);
  mr.deallocate_sync(small, 1UL << 20);

  auto* large = mr.allocate_sync(2UL << 20);
  EXPECT_NE(small, large);
  EXPECT_EQ(limiter.get_allocated_bytes(), 22UL << 20);
  mr.deallocate_sync(large, 2UL << 20);
}

TEST(CachingMemoryResourceTest, UnifiedPoolPolicyReusesSmallBlockForLargeRequest)
{
  cuda_mr cuda;
  limiting_mr limiter{cuda, 2UL << 20};
  caching_mr mr{
    limiter, std::nullopt, oom_policy::release_oversized_then_all, pool_policy::unified};

  auto* small = mr.allocate_sync(1UL << 20);
  mr.deallocate_sync(small, 1UL << 20);

  auto* large = mr.allocate_sync(2UL << 20);
  EXPECT_EQ(small, large);
  EXPECT_EQ(limiter.get_allocated_bytes(), 2UL << 20);
  mr.deallocate_sync(large, 2UL << 20);
}

TEST(CachingMemoryResourceTest, DoesNotReleaseLiveSmallSegment)
{
  cuda_mr cuda;
  limiting_mr limiter{cuda, 2UL << 20};
  caching_mr mr{limiter};

  auto* ptr = mr.allocate_sync(4096);
  EXPECT_EQ(mr.release_small_blocks(), 0);
  EXPECT_EQ(limiter.get_allocated_bytes(), 2UL << 20);
  mr.deallocate_sync(ptr, 4096);
  EXPECT_EQ(mr.release_small_blocks(), 2UL << 20);
  EXPECT_EQ(limiter.get_allocated_bytes(), 0);
}

TEST(CachingMemoryResourceTest, DoesNotReleasePartiallyFreeSmallSegment)
{
  cuda_mr cuda;
  limiting_mr limiter{cuda, 2UL << 20};
  caching_mr mr{limiter};

  auto* ptr1 = mr.allocate_sync(4096);
  auto* ptr2 = mr.allocate_sync(4096);
  mr.deallocate_sync(ptr1, 4096);

  EXPECT_EQ(mr.release_small_blocks(), 0);
  EXPECT_EQ(limiter.get_allocated_bytes(), 2UL << 20);

  mr.deallocate_sync(ptr2, 4096);
  EXPECT_EQ(mr.release_small_blocks(), 2UL << 20);
  EXPECT_EQ(limiter.get_allocated_bytes(), 0);
}

TEST(CachingMemoryResourceTest, ReleaseMergesCrossStreamFreeBlocks)
{
  cuda_mr cuda;
  limiting_mr limiter{cuda, 2UL << 20};
  caching_mr mr{limiter};
  rmm::cuda_stream stream{};

  auto* ptr1 = mr.allocate_sync(4096);
  auto* ptr2 = mr.allocate_sync(4096);
  mr.deallocate_sync(ptr1, 4096);
  mr.deallocate(rmm::cuda_stream_view{stream}, ptr2, 4096, rmm::CUDA_ALLOCATION_ALIGNMENT);
  stream.synchronize();

  EXPECT_EQ(mr.release_small_blocks(), 2UL << 20);
  EXPECT_EQ(limiter.get_allocated_bytes(), 0);
}

TEST(CachingMemoryResourceTest, MediumAllocationUsesLargeSegment)
{
  cuda_mr cuda;
  limiting_mr limiter{cuda, 20UL << 20};
  caching_mr mr{limiter};

  auto* ptr = mr.allocate_sync(2UL << 20);
  EXPECT_EQ(mr.cached_large_bytes(), 20UL << 20);
  mr.deallocate_sync(ptr, 2UL << 20);
  EXPECT_EQ(mr.release_large_blocks(), 20UL << 20);
}

TEST(CachingMemoryResourceTest, LargeRemainderTooSmallConsumesWholeBlock)
{
  caching_mr mr{rmm::mr::get_current_device_resource_ref()};

  auto* large = mr.allocate_sync(4UL << 20);
  mr.deallocate_sync(large, 4UL << 20);

  auto* small = mr.allocate_sync(19UL << 20);
  EXPECT_EQ(large, small);
  EXPECT_EQ(mr.release_large_blocks(), 0);
  mr.deallocate_sync(small, 19UL << 20);
  EXPECT_EQ(mr.release_large_blocks(), 20UL << 20);
}

TEST(CachingMemoryResourceTest, MaxSplitSizePreventsSplittingLargeBlock)
{
  cuda_mr cuda;
  limiting_mr limiter{cuda, 40UL << 20};
  caching_mr mr{limiter, 4UL << 20};

  auto* large = mr.allocate_sync(6UL << 20);
  mr.deallocate_sync(large, 6UL << 20);

  auto* small = mr.allocate_sync(2UL << 20);
  EXPECT_NE(large, small);
  mr.deallocate_sync(small, 2UL << 20);
  EXPECT_EQ(mr.release_large_blocks(), 40UL << 20);
}

TEST(CachingMemoryResourceTest, ReusesOtherStreamAllocation)
{
  caching_mr mr{rmm::mr::get_current_device_resource_ref(),
                std::nullopt,
                oom_policy::release_oversized_then_all,
                pool_policy::separate,
                stream_policy::cross_stream};
  rmm::cuda_stream stream{};

  auto* ptr1 = mr.allocate(rmm::cuda_stream_view{stream}, 4096, rmm::CUDA_ALLOCATION_ALIGNMENT);
  mr.deallocate(rmm::cuda_stream_view{stream}, ptr1, 4096, rmm::CUDA_ALLOCATION_ALIGNMENT);
  auto* ptr2 = mr.allocate_sync(4096);

  EXPECT_EQ(ptr1, ptr2);
  mr.deallocate_sync(ptr2, 4096);
}

TEST(CachingMemoryResourceTest, SameStreamPolicyDoesNotReuseOtherStreamAllocation)
{
  cuda_mr cuda;
  limiting_mr limiter{cuda, 4UL << 20};
  caching_mr mr{limiter};
  rmm::cuda_stream stream{};

  auto* ptr1 = mr.allocate(rmm::cuda_stream_view{stream}, 4096, rmm::CUDA_ALLOCATION_ALIGNMENT);
  mr.deallocate(rmm::cuda_stream_view{stream}, ptr1, 4096, rmm::CUDA_ALLOCATION_ALIGNMENT);
  stream.synchronize();

  auto* ptr2 = mr.allocate_sync(4096);
  EXPECT_NE(ptr1, ptr2);
  EXPECT_EQ(limiter.get_allocated_bytes(), 4UL << 20);
  mr.deallocate_sync(ptr2, 4096);
}

TEST(CachingMemoryResourceTest, SameStreamPolicyReusesSameStreamAllocation)
{
  caching_mr mr{rmm::mr::get_current_device_resource_ref()};
  rmm::cuda_stream stream{};

  auto* ptr1 = mr.allocate(rmm::cuda_stream_view{stream}, 4096, rmm::CUDA_ALLOCATION_ALIGNMENT);
  mr.deallocate(rmm::cuda_stream_view{stream}, ptr1, 4096, rmm::CUDA_ALLOCATION_ALIGNMENT);
  auto* ptr2 = mr.allocate(rmm::cuda_stream_view{stream}, 4096, rmm::CUDA_ALLOCATION_ALIGNMENT);

  EXPECT_EQ(ptr1, ptr2);
  mr.deallocate(rmm::cuda_stream_view{stream}, ptr2, 4096, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

TEST(CachingMemoryResourceTest, ReusesDeletedStreamAllocation)
{
  caching_mr mr{rmm::mr::get_current_device_resource_ref(),
                std::nullopt,
                oom_policy::release_oversized_then_all,
                pool_policy::separate,
                stream_policy::cross_stream};

  void* ptr1{};
  {
    rmm::cuda_stream stream{};
    ptr1 = mr.allocate(rmm::cuda_stream_view{stream}, 4096, rmm::CUDA_ALLOCATION_ALIGNMENT);
    mr.deallocate(rmm::cuda_stream_view{stream}, ptr1, 4096, rmm::CUDA_ALLOCATION_ALIGNMENT);
    stream.synchronize();
  }

  auto* ptr2 = mr.allocate_sync(4096);
  EXPECT_EQ(ptr1, ptr2);
  mr.deallocate_sync(ptr2, 4096);
}

TEST(CachingMemoryResourceTest, DestructorReleasesCachedBlocks)
{
  cuda_mr cuda;
  limiting_mr limiter{cuda, 2UL << 20};

  {
    caching_mr mr{limiter};
    auto* ptr = mr.allocate_sync(4096);
    mr.deallocate_sync(ptr, 4096);
    EXPECT_EQ(limiter.get_allocated_bytes(), 2UL << 20);
  }

  EXPECT_EQ(limiter.get_allocated_bytes(), 0);
}

TEST(CachingMemoryResourceTest, SmallRemainderTooSmallConsumesWholeBlock)
{
  cuda_mr cuda;
  limiting_mr limiter{cuda, 2UL << 20};
  caching_mr mr{limiter};

  auto* first  = mr.allocate_sync(1UL << 20);
  auto* second = mr.allocate_sync((1UL << 20) - 256);

  EXPECT_THROW((void)mr.allocate_sync(256), rmm::out_of_memory);

  mr.deallocate_sync(second, (1UL << 20) - 256);
  mr.deallocate_sync(first, 1UL << 20);
  EXPECT_EQ(mr.release_small_blocks(), 2UL << 20);
}

TEST(CachingMemoryResourceTest, SplitsReusableLargeBlock)
{
  caching_mr mr{rmm::mr::get_current_device_resource_ref()};

  auto* large = mr.allocate_sync(4UL << 20);
  mr.deallocate_sync(large, 4UL << 20);

  auto* smaller = mr.allocate_sync(2UL << 20);
  EXPECT_EQ(large, smaller);
  EXPECT_EQ(mr.release_large_blocks(), 0);
  mr.deallocate_sync(smaller, 2UL << 20);
  EXPECT_EQ(mr.release_large_blocks(), 20UL << 20);
}

TEST(CachingMemoryResourceTest, ReleaseDropsCachedBytes)
{
  caching_mr mr{rmm::mr::get_current_device_resource_ref()};
  auto* ptr = mr.allocate_sync(4096);
  mr.deallocate_sync(ptr, 4096);
  EXPECT_GT(mr.cached_small_bytes(), 0);
  EXPECT_GT(mr.release_small_blocks(), 0);
  EXPECT_EQ(mr.cached_small_bytes(), 0);
}

}  // namespace

namespace test_properties {

static_assert(
  cuda::mr::resource_with<rmm::mr::caching_memory_resource, cuda::mr::device_accessible>);

}  // namespace test_properties

}  // namespace rmm::test
