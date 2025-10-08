/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <rmm/detail/error.hpp>
#include <rmm/detail/runtime_capabilities.hpp>
#include <rmm/mr/device/cuda_async_managed_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

namespace rmm::test {
namespace {

using cuda_async_managed_mr = rmm::mr::cuda_async_managed_memory_resource;

// static property checks
static_assert(
  rmm::detail::polyfill::resource_with<cuda_async_managed_mr, cuda::mr::device_accessible>);
static_assert(
  rmm::detail::polyfill::async_resource_with<cuda_async_managed_mr, cuda::mr::device_accessible>);

class AsyncManagedMRTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    if (!rmm::detail::runtime_async_managed_alloc::is_supported()) {
      GTEST_SKIP() << "Skipping tests since cuda_async_managed_memory_resource "
                   << "requires CUDA 13.0 or higher";
    }
  }
};

TEST_F(AsyncManagedMRTest, BasicAllocateDeallocate)
{
  const auto alloc_size{100};
  cuda_async_managed_mr mr{};
  void* ptr = mr.allocate(alloc_size);
  ASSERT_NE(nullptr, ptr);
  mr.deallocate(ptr, alloc_size);
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

}  // namespace
}  // namespace rmm::test
