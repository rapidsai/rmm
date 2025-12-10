/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/detail/error.hpp>
#include <rmm/detail/runtime_capabilities.hpp>
#include <rmm/mr/cuda_async_pinned_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

namespace rmm::test {
namespace {

using cuda_async_pinned_mr = rmm::mr::cuda_async_pinned_memory_resource;

class AsyncPinnedMRTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    if (!rmm::detail::runtime_async_pinned_alloc::is_supported()) {
      GTEST_SKIP() << "Skipping tests because cuda_async_pinned_memory_resource "
                   << "requires CUDA 12.6 or higher and memory pool support.";
    }
  }
};

TEST_F(AsyncPinnedMRTest, BasicAllocateDeallocate)
{
  const auto alloc_size{100};
  cuda_async_pinned_mr mr{};
  void* ptr = mr.allocate_sync(alloc_size);
  ASSERT_NE(nullptr, ptr);
  mr.deallocate_sync(ptr, alloc_size);
}

TEST_F(AsyncPinnedMRTest, EqualityWithSamePool)
{
  // Two instances wrapping the same default pinned pool should compare equal if they
  // ultimately refer to the same underlying pool handle. Construct two and compare.
  cuda_async_pinned_mr mr1{};
  cuda_async_pinned_mr mr2{};
  EXPECT_TRUE(mr1.is_equal(mr2));
}

TEST_F(AsyncPinnedMRTest, AllocatedPointerIsAccessibleFromHost)
{
  const auto alloc_size{sizeof(int) * 100};
  cuda_async_pinned_mr mr{};
  auto* ptr = static_cast<int*>(mr.allocate_sync(alloc_size));
  ASSERT_NE(nullptr, ptr);

  // Pinned memory should be accessible from host
  // Write from host
  EXPECT_NO_THROW({
    for (int i = 0; i < 100; ++i) {
      ptr[i] = i;
    }
  });

  // Verify we can read back
  EXPECT_EQ(ptr[0], 0);
  EXPECT_EQ(ptr[50], 50);
  EXPECT_EQ(ptr[99], 99);

  mr.deallocate_sync(ptr, alloc_size);
}

TEST_F(AsyncPinnedMRTest, MultipleAllocationsAccessible)
{
  const auto alloc_size{512};
  cuda_async_pinned_mr mr{};

  void* ptr1 = mr.allocate_sync(alloc_size);
  void* ptr2 = mr.allocate_sync(alloc_size * 2);
  void* ptr3 = mr.allocate_sync(alloc_size / 2);

  ASSERT_NE(nullptr, ptr1);
  ASSERT_NE(nullptr, ptr2);
  ASSERT_NE(nullptr, ptr3);

  // Verify all pointers are accessible from host
  auto* typed_ptr1 = static_cast<char*>(ptr1);
  auto* typed_ptr2 = static_cast<char*>(ptr2);
  auto* typed_ptr3 = static_cast<char*>(ptr3);

  EXPECT_NO_THROW({
    typed_ptr1[0] = 'a';
    typed_ptr2[0] = 'b';
    typed_ptr3[0] = 'c';
  });

  EXPECT_EQ(typed_ptr1[0], 'a');
  EXPECT_EQ(typed_ptr2[0], 'b');
  EXPECT_EQ(typed_ptr3[0], 'c');

  mr.deallocate_sync(ptr1, alloc_size);
  mr.deallocate_sync(ptr2, alloc_size * 2);
  mr.deallocate_sync(ptr3, alloc_size / 2);
}

TEST_F(AsyncPinnedMRTest, PoolHandleIsValid)
{
  cuda_async_pinned_mr mr{};
  cudaMemPool_t pool_handle = mr.pool_handle();
  EXPECT_NE(pool_handle, nullptr);
}

TEST_F(AsyncPinnedMRTest, AllocatedPointerIsAccessibleFromDevice)
{
  const auto alloc_size{sizeof(int) * 100};
  cuda_async_pinned_mr mr{};
  auto* ptr = static_cast<int*>(mr.allocate_sync(alloc_size));
  ASSERT_NE(nullptr, ptr);

  // Initialize from host
  for (int i = 0; i < 100; ++i) {
    ptr[i] = i;
  }

  // Allocate device memory and copy from pinned -> device -> back to verify device access
  int* d_ptr{};
  EXPECT_EQ(cudaMalloc(&d_ptr, alloc_size), cudaSuccess);
  EXPECT_EQ(cudaMemcpy(d_ptr, ptr, alloc_size, cudaMemcpyDefault), cudaSuccess);

  int result[100];
  EXPECT_EQ(cudaMemcpy(result, d_ptr, alloc_size, cudaMemcpyDefault), cudaSuccess);

  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(result[i], i);
  }

  cudaFree(d_ptr);
  mr.deallocate_sync(ptr, alloc_size);
}

}  // namespace
}  // namespace rmm::test
