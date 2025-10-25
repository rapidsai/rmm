/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/detail/error.hpp>
#include <rmm/detail/runtime_capabilities.hpp>
#include <rmm/mr/device/cuda_async_managed_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

namespace rmm::test {
namespace {

using cuda_async_managed_mr = rmm::mr::cuda_async_managed_memory_resource;

class AsyncManagedMRTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    if (!rmm::detail::runtime_async_managed_alloc::is_supported()) {
      GTEST_SKIP() << "Skipping tests because cuda_async_managed_memory_resource "
                   << "requires CUDA 13.0 or higher and concurrent managed "
                   << "access support.";
    }
  }
};

TEST_F(AsyncManagedMRTest, BasicAllocateDeallocate)
{
  const auto alloc_size{100};
  cuda_async_managed_mr mr{};
  void* ptr = mr.allocate_sync(alloc_size);
  ASSERT_NE(nullptr, ptr);
  mr.deallocate_sync(ptr, alloc_size);
  RMM_CUDA_TRY(cudaDeviceSynchronize());
}

TEST_F(AsyncManagedMRTest, EqualityWithSamePool)
{
  // Two instances wrapping the same default managed pool should compare equal if they
  // ultimately refer to the same underlying pool handle. Construct two and compare.
  cuda_async_managed_mr mr1{};
  cuda_async_managed_mr mr2{};
  EXPECT_TRUE(mr1.is_equal(mr2));
}

TEST_F(AsyncManagedMRTest, AllocatedPointerIsManaged)
{
  const auto alloc_size{1024};
  cuda_async_managed_mr mr{};
  void* ptr = mr.allocate_sync(alloc_size);
  ASSERT_NE(nullptr, ptr);

  // Verify the pointer is managed memory using cudaPointerGetAttributes
  cudaPointerAttributes attrs{};
  RMM_CUDA_TRY(cudaPointerGetAttributes(&attrs, ptr));
  EXPECT_EQ(attrs.type, cudaMemoryTypeManaged);

  mr.deallocate_sync(ptr, alloc_size);
  RMM_CUDA_TRY(cudaDeviceSynchronize());
}

TEST_F(AsyncManagedMRTest, AllocatedPointerIsAccessibleFromHost)
{
  const auto alloc_size{sizeof(int) * 100};
  cuda_async_managed_mr mr{};
  auto* ptr = static_cast<int*>(mr.allocate_sync(alloc_size));
  ASSERT_NE(nullptr, ptr);

  // Synchronize to ensure allocation is complete
  RMM_CUDA_TRY(cudaDeviceSynchronize());

  // Managed memory should be accessible from host
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
  RMM_CUDA_TRY(cudaDeviceSynchronize());
}

TEST_F(AsyncManagedMRTest, MultipleAllocationsAreManaged)
{
  const auto alloc_size{512};
  cuda_async_managed_mr mr{};

  void* ptr1 = mr.allocate_sync(alloc_size);
  void* ptr2 = mr.allocate_sync(alloc_size * 2);
  void* ptr3 = mr.allocate_sync(alloc_size / 2);

  ASSERT_NE(nullptr, ptr1);
  ASSERT_NE(nullptr, ptr2);
  ASSERT_NE(nullptr, ptr3);

  // Verify all pointers are managed memory
  cudaPointerAttributes attrs1{};
  cudaPointerAttributes attrs2{};
  cudaPointerAttributes attrs3{};

  RMM_CUDA_TRY(cudaPointerGetAttributes(&attrs1, ptr1));
  RMM_CUDA_TRY(cudaPointerGetAttributes(&attrs2, ptr2));
  RMM_CUDA_TRY(cudaPointerGetAttributes(&attrs3, ptr3));

  EXPECT_EQ(attrs1.type, cudaMemoryTypeManaged);
  EXPECT_EQ(attrs2.type, cudaMemoryTypeManaged);
  EXPECT_EQ(attrs3.type, cudaMemoryTypeManaged);

  mr.deallocate_sync(ptr1, alloc_size);
  mr.deallocate_sync(ptr2, alloc_size * 2);
  mr.deallocate_sync(ptr3, alloc_size / 2);
  RMM_CUDA_TRY(cudaDeviceSynchronize());
}

}  // namespace
}  // namespace rmm::test
